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

"""Code mode on the ManiSkill server (``code.run`` over its registry), without a simulator: a
fake env whose info and reward carry object state, which no program may receive."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from pi_embodied_services.robots.maniskill import env_server as ms

#: Object state some ManiSkill tasks report in their info, and the dense reward.
PRIVILEGED = {"reward", "obj_to_goal_dist", "peg_head_pos_at_hole", "tcp_to_obj_pos"}


class _Env:
    """A ManiSkill-like env in pd_ee_delta_pos: the TCP moves 0.1 m per action unit; success
    once the TCP is above z = 0.2."""

    def __init__(self):
        self.tcp = np.array([0.0, 0.0, 0.1])
        # The one Panda agent: its base at the identity, its TCP where the env keeps it.
        env = self
        tcp = SimpleNamespace(
            pose=type("P", (), {"p": property(lambda _: env.tcp.reshape(1, 3))})()
        )
        robot = SimpleNamespace(
            pose=SimpleNamespace(q=np.array([[1.0, 0.0, 0.0, 0.0]]))
        )
        agent = SimpleNamespace(robot=robot, tcp=tcp)
        self.unwrapped = SimpleNamespace(evaluate=self.evaluate, agent=agent)

    def evaluate(self):
        return {
            "success": np.array([self.tcp[2] > 0.2]),
            "is_grasped": np.array([False]),
            "obj_to_goal_dist": np.array([0.3]),
            "peg_head_pos_at_hole": np.zeros((1, 3)),
        }

    def step(self, a):
        a = np.clip(np.asarray(a, dtype=np.float64).reshape(-1), -1, 1)
        self.tcp = self.tcp + a[:3] * ms.DELTA_BOUND_M
        info = {**self.evaluate(), "tcp_to_obj_pos": np.array([[0.1, 0.0, 0.0]])}
        return {"step": 1}, np.array([0.7]), np.array([False]), np.array([False]), info


def facade(wrist: bool = True) -> ms.ManiskillEnvFacade:
    f = object.__new__(ms.ManiskillEnvFacade)
    ms.BaseEnvFacade.__init__(f)
    f._env = _Env()
    f._robot = ms.ROBOTS["panda"]
    f._rig = None
    f._cameras = ms.CAMERAS
    f._grip = [1.0]
    f._obs = {}
    f._meta = {"robot": "panda", "wrist": wrist, "env_id": "PickCube-v1"}
    f._rgb = lambda obs, name: np.zeros((4, 4, 3), np.uint8)
    f._state = lambda: {
        "tcp_pos": f._env.tcp.astype(np.float32),
        "gripper_width": 0.08,
        "qpos": np.zeros(9, np.float32),
    }
    return f


def keys(value) -> set[str]:
    if isinstance(value, dict):
        return set(value) | {k for v in value.values() for k in keys(v)}
    if isinstance(value, (list, tuple)):
        return {k for v in value for k in keys(v)}
    return set()


def test_the_server_serves_code_run_with_a_token():
    f = facade()
    assert ms.ManiskillEnvFacade.REQUIRE_TOKEN and f._rpc_token
    assert "code.run" in f._rpc and "code.helpers" in f._rpc


def test_a_servo_run_reports_steps_success_obs_and_frames_and_hides_object_state():
    f = facade()
    out = f._rpc["code.run"](
        "x, y, z = state()['tcp_pos']\n"
        "frames, info = servo([x, y, z + 0.15], -1, max_steps=20)\n"
        "RESULT = [state(), frames, info]\n",
        timeout_s=30,
    )
    assert out["status"] == "ran", out
    seen = keys(out["result"])
    assert not seen & PRIVILEGED, seen & PRIVILEGED
    assert not seen & {"agentview", "wrist"}, "a motion's images go to the run's video"
    st, frames, info = out["result"]
    assert set(st["info"]) == {"success", "is_grasped"}
    assert info == {"success": True, "is_grasped": False}
    assert frames[-1]["tcp_pos"][2] > 0.2
    assert out["success"] is True and out["steps"] == len(frames) == f._steps
    assert out["gripper"] == -1, "the servo held the Panda's close action"
    assert out["obs"]["agentview"].shape == (4, 4, 3), "pi's own observation"
    assert out["info"] == {"success": True, "is_grasped": False}
    assert len(out["frames"]) == out["steps"]
    assert out["frames"][0].shape == (4, 8, 3), "agentview and wrist side by side"
    assert out["calls"][1]["move_m"] == pytest.approx(0.15, abs=1e-6)


def test_the_low_tier_raw_steps_answer_without_reward_or_images():
    f = facade(wrist=False)
    out = f._rpc["code.run"](
        "a = step([0, 0, 0.1, 1])\n"
        "b = chunk_step([[0.1, 0, 0, 1]] * 3, return_all_frames=True)\n"
        "RESULT = [a, b]\n",
        timeout_s=30,
        tier="low",
    )
    assert out["status"] == "ran", out
    a, b = out["result"]
    assert not keys(out["result"]) & (PRIVILEGED | {"agentview", "wrist"})
    assert set(a) == {"terminated", "truncated", "info", "state"}
    assert a["state"]["tcp_pos"] == pytest.approx([0, 0, 0.11])
    assert b["steps"] == 3 and len(b["states"]) == 3
    assert [c["move_m"] for c in out["calls"]] == pytest.approx([0.01, 0.03])
    assert out["frames"][0].shape == (4, 4, 3), "the agentview alone without a wrist"
    assert out["gripper"] == 1


def test_oversized_calls_are_refused_before_they_run():
    f = facade()
    for code, tier, why in [
        ("servo([0, 0, 0.1], 1, max_steps=1000)\n", "high", "at most 100"),
        ("chunk_step([[0, 0, 0, 1]] * 500)\n", "low", "at most 200 actions"),
    ]:
        out = f._rpc["code.run"](code, timeout_s=30, tier=tier)
        assert out["status"] == "error" and why in out["error"], out
    assert f._steps == 0


def test_the_run_video_is_bounded():
    f = facade()
    f._begin_run()
    f._keep_frames([{"agentview": np.zeros((1, 1, 3), np.uint8)}] * 300)
    assert len(f._run_frames) <= ms.CODE_MAX_FRAMES


def test_the_low_tier_shows_examples_and_s4_drops_them():
    f = facade()
    low = f._rpc["code.api"]("low")["primitives"]
    s4 = f._rpc["code.api"]("low-noexamples")["primitives"]
    assert [p["name"] for p in s4] == [p["name"] for p in low]
    named = {p["name"]: p for p in low}
    for name in ("state", "servo", "step", "chunk_step", "render_camera"):
        assert "example" in named[name], name
    assert all("example" not in p for p in s4)
