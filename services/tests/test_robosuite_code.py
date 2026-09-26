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

"""Code mode on the robosuite server (``code.run`` over its registry), without a simulator: the
facade methods behind the registry are the tools' own, replaced here by recording fakes."""

from __future__ import annotations

import numpy as np
import pytest

from pi_embodied_services.robots.robosuite import tasks
from pi_embodied_services.robots.robosuite.env_server import (
    CODE_MAX_FRAMES,
    OPEN,
    RobosuiteEnvFacade,
)
from pi_embodied_services.utils.rpc import RpcFacade


def facade(task: str = "Lift") -> RobosuiteEnvFacade:
    """A registered robosuite facade whose motion methods are fakes that step a counter."""
    f = object.__new__(RobosuiteEnvFacade)
    RpcFacade.__init__(f)
    f._task_name, f._task = task, tasks.TASKS[task]
    f._grip = {arm: OPEN for arm in f._task.arms}
    f._steps, f._success_step, f._grasp = 0, None, None
    f._run_start, f._run_frames = 0, []
    f._cameras = {"agentview": "robot0_robotview", "wrist": "robot0_eye_in_hand"}
    f.eef = np.array([0.0, 0.0, 1.0])
    f._eef = lambda i: (f.eef.copy(), np.array([1.0, 0, 0, 0]))
    f._render = lambda camera, size, depth=False: np.zeros((2, 2, 3), np.uint8)
    f._pack = lambda: {
        "agentview": np.zeros((2, 2, 3), np.uint8),
        "env_steps": f._steps,
        "success": f._success_step is not None,
    }
    RobosuiteEnvFacade._register_rpc(f)
    f.moves = []

    def move_to(target_xyz, **kw):
        f.moves.append(("move_to", list(target_xyz), kw))
        f.eef = np.asarray(target_xyz, dtype=np.float64)
        f._steps += 10
        if f.eef[2] > 1.05:
            f._success_step = f._success_step or f._steps
        return {
            "obs": f._pack(),
            "info": {
                "ok": True,
                "final_eef_pos": f.eef.tolist(),
                "steps_used": 10,
                "frames": [np.full((2, 2, 3), 7, np.uint8)] * 3,
            },
        }

    def set_gripper(close, **kw):
        f.moves.append(("set_gripper", close, kw))
        f._steps += 5
        return {
            "obs": f._pack(),
            "info": {"gripper": "close" if close else "open", "frames": []},
        }

    f._rpc["env.move_to"] = move_to
    f._rpc["env.set_gripper"] = set_gripper
    f._rpc["env.get_state"] = lambda: {
        "robot0_eef_pos": f.eef.tolist(),
        "success": f._success_step is not None,
    }
    return f


def test_the_server_serves_code_run_with_a_token_and_exclusivity():
    f = facade()
    assert RobosuiteEnvFacade.REQUIRE_TOKEN and f._rpc_token
    assert "code.run" in f._rpc and "code.helpers" in f._rpc
    with pytest.raises(PermissionError):
        f._serve_dispatch("env.get_state", (), {})


def test_a_run_reports_steps_success_the_new_obs_and_the_motion_frames():
    f = facade()
    out = f._rpc["code.run"](
        "set_gripper(True)\nr = move_to([0.0, 0.0, 1.1])\nRESULT = sorted(r)\n",
        timeout_s=30,
    )
    assert out["status"] == "ran", out
    assert "frames" not in out["result"], (
        "the frames go to the run's video, not the program"
    )
    assert out["steps"] == 15 and out["success"] is True and out["success_step"] == 15
    assert out["obs"]["env_steps"] == 15
    assert len(out["frames"]) == 3
    assert out["calls"][1]["move_m"] == pytest.approx(0.1)
    assert [m[0] for m in f.moves] == ["set_gripper", "move_to"]


def test_the_move_cap_counts_move_to_from_the_tcp_and_refuses():
    f = facade()
    out = f._rpc["code.run"](
        "move_to([0.0, 0.0, 1.2])\nmove_to([0.0, 0.0, 0.8])\n",
        timeout_s=30,
        max_move_m=0.3,
    )
    assert out["status"] == "error" and out["limit"] == "max_move_m", out
    assert len(f.moves) == 1


def test_the_low_tier_raw_step_is_estimated_and_answers_without_images():
    f = facade()
    f._rpc["env.step"] = lambda action: (
        {"agentview": np.zeros((512, 512, 3), np.uint8)},
        0.0,
        False,
        False,
        {"robot0_eef_pos": [0, 0, 1]},
    )
    out = f._rpc["code.run"](
        "RESULT = sorted(step([0, 0, 1, 0, 0, 0, -1]))\n", timeout_s=30, tier="low"
    )
    assert out["status"] == "ran", out
    assert out["result"] == ["reward", "state", "success", "truncated"]
    assert out["calls"][0]["move_m"] == pytest.approx(tasks.OSC_POS_MAX_M)
    assert len(out["frames"]) == 1


def test_oversized_calls_are_refused_before_they_run():
    f = facade()
    out = f._rpc["code.run"]("move_to([0, 0, 1.1], max_steps=5000)\n", timeout_s=30)
    assert out["status"] == "error" and "at most 400" in out["error"], out
    assert f.moves == []


def test_the_run_video_is_bounded():
    f = facade()
    f._keep_frames([np.zeros((1, 1, 3), np.uint8)] * (CODE_MAX_FRAMES + 5))
    assert len(f._run_frames) <= CODE_MAX_FRAMES


def test_the_s4_tier_drops_the_examples_the_low_tier_shows():
    f = facade()
    low = {p["name"]: p for p in f._rpc["code.api"]("low")["primitives"]}
    s4 = f._rpc["code.api"]("low-noexamples")
    assert s4["tier"] == "low-noexamples"
    assert [p["name"] for p in s4["primitives"]] == list(low)
    assert "example" in low["move_delta"] and all(
        "example" not in p for p in s4["primitives"]
    )
    docs = {p["name"]: p["doc"] for p in f._code.api("low")}
    assert "Example:" in docs["move_delta"]
    assert all("Example:" not in p["doc"] for p in f._code.api("low-noexamples"))
