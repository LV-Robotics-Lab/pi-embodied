# Copyright 2026 The pi-embodied Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Code mode on the Metaworld server (``code.run`` over its registry), without a simulator: a
fake env whose 39-D observation and info carry object state, which no program may receive."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from pi_embodied_services.robots.metaworld import env_server as mw

#: What Metaworld reports that is object state (the obs, the shaped reward and its metrics).
PRIVILEGED = {
    "obs",
    "reward",
    "near_object",
    "obj_to_target",
    "grasp_reward",
    "in_place_reward",
    "unscaled_reward",
}


class _Env:
    """A Metaworld-like env: the hand follows the action at 1 cm per unit; the puck sits at
    ``puck`` (obs[4:7], as Metaworld's), success when the TCP is within 2 cm of ``goal``."""

    def __init__(self, goal=(0.0, 0.7, 0.2)):
        self.goal = np.asarray(goal, dtype=np.float64)
        self.puck = np.array([0.05, 0.65, 0.02])
        self.mocap_low = np.array([-0.2, 0.5, 0.05])
        self.mocap_high = np.array([0.2, 0.75, 0.3])
        self.hand = np.array([0.0, 0.6, 0.2])
        self.data = SimpleNamespace(mocap_pos=np.array([self.hand + [0, 0, 0.045]]))
        self.model = None
        self._target_pos = self.goal

    @property
    def tcp_center(self):
        return self.hand.copy()

    def get_body_com(self, name):
        return self.hand + [(-1 if name == "leftpad" else 1) * 0.047, 0, 0]

    def step(self, a):
        a = np.clip(np.asarray(a, dtype=np.float64), -1, 1)
        self.hand = np.clip(self.hand + a[:3] * 0.01, self.mocap_low, self.mocap_high)
        self.data.mocap_pos = np.array([self.hand + [0, 0, 0.045]])
        obs = np.zeros(39)
        obs[:3], obs[4:7], obs[-3:] = self.hand, self.puck, self.goal
        d = float(np.linalg.norm(self.hand - self.goal))
        info = {
            "success": float(d < 0.02),
            "near_object": float(np.linalg.norm(self.hand - self.puck) < 0.03),
            "grasp_success": 0.0,
            "grasp_reward": 0.1,
            "in_place_reward": 0.2,
            "obj_to_target": float(np.linalg.norm(self.puck - self.goal)),
            "unscaled_reward": 1.0 - d,
        }
        return obs, 1.0 - d, False, False, info


def facade() -> mw.MetaworldEnvFacade:
    f = object.__new__(mw.MetaworldEnvFacade)
    mw.BaseEnvFacade.__init__(f)
    f._env = _Env()
    f._seed, f._view_size, f._renderers = 0, 4, {}
    f._obs, f._info, f._success_once = np.zeros(39), {}, False
    f._gripper_effort, f._closed, f._box = mw.OPEN, False, None
    f._meta = {"task": "reach-v3", "seed": 0}
    f._render = lambda camera, h, w, depth=False: np.zeros((4, 4, 3), np.uint8)
    return f


def keys(value) -> set[str]:
    """Every dict key anywhere in a program's result."""
    if isinstance(value, dict):
        return set(value) | {k for v in value.values() for k in keys(v)}
    if isinstance(value, (list, tuple)):
        return {k for v in value for k in keys(v)}
    return set()


def test_the_server_serves_code_run_with_a_token_and_exclusivity():
    f = facade()
    assert mw.MetaworldEnvFacade.REQUIRE_TOKEN and f._rpc_token
    assert "code.run" in f._rpc and "code.helpers" in f._rpc


def test_a_high_tier_run_reaches_the_goal_and_the_program_sees_no_object_state():
    f = facade()
    out = f._rpc["code.run"](
        "a = state()\n"
        "b = move_delta([0, 0.1, 0])\n"
        "c = set_gripper(False)\n"
        "RESULT = [a, b, c, state()]\n",
        timeout_s=30,
    )
    assert out["status"] == "ran", out
    seen = keys(out["result"])
    assert not seen & PRIVILEGED, seen & PRIVILEGED
    assert "frames" not in seen, "a motion's frames go to the run's video"
    assert out["result"][1]["info"] == {
        "success": True,
        "grasp_success": False,
        "success_once": True,
    }
    # The move stops at success, 2 cm short of the goal.
    assert out["result"][1]["final_tcp_pos"] == pytest.approx([0, 0.68, 0.2], abs=1e-3)
    # The run's effect for pi: its steps, the latched success, the tools' obs, the video.
    assert out["success"] is True and out["steps"] == f._steps > 0
    assert out["obs"]["obs"].shape == (39,), "pi's own observation is the tools' pack"
    assert set(out["info"]) <= set(mw.PROGRAM_INFO)
    assert out["gripper"] == "close"
    assert len(out["frames"]) == out["steps"]
    assert out["frames"][0].shape == (4, 8, 3), "agentview and wrist side by side"
    assert out["calls"][1]["move_m"] == pytest.approx(0.1)


def test_the_low_tier_raw_steps_answer_with_the_robot_state_and_flags_only():
    f = facade()
    out = f._rpc["code.run"](
        "a = step([0, 0, -1, 1])\n"
        "b = chunk_step([[1, 0, 0, 1]] * 3)\n"
        "c = chunk_step([[0, 1, 0, 1]] * 2, return_all_frames=True)\n"
        "RESULT = [a, b, c]\n",
        timeout_s=30,
        tier="low",
    )
    assert out["status"] == "ran", out
    a, b, c = out["result"]
    assert not keys(out["result"]) & PRIVILEGED
    assert set(a) == {"success", "truncated", "info", "state"}
    assert a["state"]["tcp_pos"] == pytest.approx([0, 0.6, 0.19])
    assert b["steps"] == 3 and "states" not in b
    assert len(c["states"]) == 2
    assert [x["move_m"] for x in out["calls"]] == pytest.approx([0.01, 0.03, 0.02])
    # A chunk renders its last frame only, unless it returns them all.
    assert out["steps"] == 6 and len(out["frames"]) == 1 + 1 + 2
    # A raw step's gripper effort is held like a move's.
    assert f._gripper_effort == mw.CLOSE and out["gripper"] == "close"


def test_the_privileged_tier_alone_reaches_the_ground_truth():
    f = facade()
    f._rpc["env.ground_truth_poses"] = lambda names=None: {"poses": {"goal": {}}}
    low = f._rpc["code.run"]("ground_truth_poses()\n", timeout_s=30, tier="low")
    assert low["status"] == "error", low
    ok = f._rpc["code.run"](
        "RESULT = ground_truth_poses()\n", timeout_s=30, tier="privileged"
    )
    assert ok["status"] == "ran" and "goal" in ok["result"]["poses"], ok


def test_oversized_calls_are_refused_before_they_run():
    f = facade()
    for code, why in [
        ("chunk_step([[0, 0, 0, 0]] * 500)\n", "at most 200 actions"),
        ('render_camera("agentview", 4096, 4096)\n', "at most 1024"),
    ]:
        out = f._rpc["code.run"](code, timeout_s=30, tier="low")
        assert out["status"] == "error" and why in out["error"], out
    assert f._steps == 0


def test_the_run_video_is_bounded():
    f = facade()
    f._begin_run()
    pack = {"agentview": np.zeros((1, 1, 3), np.uint8), "wrist": np.zeros((1, 1, 3))}
    f._keep_frames([pack] * (mw.CODE_MAX_FRAMES + 5))
    assert len(f._run_frames) <= mw.CODE_MAX_FRAMES


def test_the_low_tier_shows_examples_and_s4_drops_them():
    f = facade()
    low = f._rpc["code.api"]("low")["primitives"]
    s4 = f._rpc["code.api"]("low-noexamples")["primitives"]
    assert [p["name"] for p in s4] == [p["name"] for p in low]
    named = {p["name"]: p for p in low}
    for name in ("move_delta", "set_gripper", "step", "chunk_step", "render_camera"):
        assert "example" in named[name], name
    assert all("example" not in p for p in s4)
