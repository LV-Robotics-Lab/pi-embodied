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

"""Code mode on the RoboLab server (``code.run`` over its registry), without Isaac Sim: the
facade methods behind the registry are the tools' own, replaced here by recording fakes."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from pi_embodied_services.robots.robolab.env_server import (
    CODE_MAX_CHUNK,
    CODE_MAX_FRAMES,
    RobolabEnvFacade,
)
from pi_embodied_services.utils.rpc import RpcFacade

IK_SCALE = 0.5
SUBTASK = {"completed": 1, "total": 2, "score": 0.5, "info": "banana grasped"}


def facade() -> RobolabEnvFacade:
    """A registered RoboLab facade whose motion methods are fakes that step a counter."""
    f = object.__new__(RobolabEnvFacade)
    RpcFacade.__init__(f)
    f._h = SimpleNamespace(ik_scale=IK_SCALE, instruction="put the banana in the bowl")
    f._steps, f._terminated, f._truncated = 0, False, False
    f._run_start, f._run_frames = 0, []
    f.eef = np.array([0.4, 0.0, 0.3])
    f.solved = False
    f._state = lambda: {
        "subtask": SUBTASK,
        "eef_pos": f.eef.astype(np.float32),
        "gripper_width": 0.08,
        "success": f.solved,
        "terminated": f._terminated,
        "truncated": f._truncated,
        "env_steps": f._steps,
    }
    f._pack = lambda: {
        "agentview": np.zeros((2, 2, 3), np.uint8),
        "wrist": np.zeros((2, 2, 3), np.uint8),
        **f._state(),
    }
    RobolabEnvFacade._register_rpc(f)
    f.moves = []

    def move_delta(delta_xyz, gripper=None, return_frames=False):
        f.moves.append(("move_delta", list(delta_xyz), gripper))
        f.eef = f.eef + np.asarray(delta_xyz)
        f._steps += 8
        f.solved = f.solved or f.eef[2] > 0.35
        f._terminated = f.solved
        out = {**f._pack(), "moved_m": list(delta_xyz), "decisions": 1}
        if return_frames:
            out["frames"] = [np.full((2, 2, 3), 7, np.uint8)] * 3
        return out

    def step(action):
        f.moves.append(("step", list(action)))
        f._steps += 1
        return f._pack(), 0.0, False, False, {"success": False}

    def chunk_step(actions, return_all_frames=False):
        n = len(actions)
        f.moves.append(("chunk_step", n))
        f._steps += n
        obs = [f._pack() for _ in range(n)] if return_all_frames else f._pack()
        return obs, False, False, {"success": False}

    f._rpc["env.move_delta"] = move_delta
    f._rpc["env.step"] = step
    f._rpc["env.chunk_step"] = chunk_step
    return f


def test_the_server_serves_code_run_with_a_token_and_exclusivity():
    f = facade()
    assert RobolabEnvFacade.REQUIRE_TOKEN and f._rpc_token
    assert "code.run" in f._rpc and "code.helpers" in f._rpc
    with pytest.raises(PermissionError):
        f._serve_dispatch("env.state", (), {})


def test_a_run_reports_steps_success_the_new_obs_and_the_motion_frames():
    f = facade()
    f._steps = 5
    out = f._rpc["code.run"](
        "move_delta([0, 0, 0.02], return_frames=True)\n"
        "r = move_delta([0, 0, 0.05])\nRESULT = sorted(r)\n",
        timeout_s=30,
    )
    assert out["status"] == "ran", out
    assert out["steps"] == 16 and out["success"] is True and out["terminated"] is True
    assert out["truncated"] is False
    assert out["obs"]["env_steps"] == 21 and out["obs"]["agentview"].shape == (2, 2, 3)
    # Three frames of the first motion, the last front image of the second.
    assert len(out["frames"]) == 4
    assert out["calls"][1]["move_m"] == pytest.approx(0.05)
    assert [m[0] for m in f.moves] == ["move_delta", "move_delta"]


def test_a_program_receives_no_images_and_no_subtask_judgement():
    f = facade()
    out = f._rpc["code.run"](
        "a = move_delta([0.01, 0, 0])\nb = state()\n"
        "c = step([0, 0, 0, 0, 0, 0, 0])\nd = chunk_step([[0] * 7] * 2, return_all_frames=True)\n"
        "RESULT = [sorted(a), sorted(b), sorted(c), sorted(c['state']), sorted(d['obs'][0])]\n",
        timeout_s=30,
        tier="low",
    )
    assert out["status"] == "ran", out
    for keys in out["result"]:
        assert not {"agentview", "wrist", "frames", "subtask"} & set(keys), keys
    assert "eef_pos" in out["result"][1]
    assert out["result"][2] == ["reward", "state", "success", "terminated", "truncated"]
    # The finish still carries the evaluator's subtask to pi (details, never the planner).
    assert out["obs"]["subtask"] == SUBTASK


def test_the_move_cap_counts_raw_actions_by_the_ik_scale_and_refuses():
    f = facade()
    out = f._rpc["code.run"](
        "step([0.2, 0, 0, 0, 0, 0, 0])\nchunk_step([[0.2, 0, 0, 0, 0, 0, 0]] * 4)\n",
        timeout_s=30,
        tier="low",
        max_move_m=0.3,
    )
    assert out["status"] == "error" and out["limit"] == "max_move_m", out
    assert out["calls"][0]["move_m"] == pytest.approx(0.2 * IK_SCALE)
    assert [m[0] for m in f.moves] == ["step"]


def test_oversized_chunks_and_raw_steps_after_the_end_are_refused_before_they_run():
    f = facade()
    out = f._rpc["code.run"](
        f"chunk_step([[0] * 7] * {CODE_MAX_CHUNK + 1})\n", timeout_s=30, tier="low"
    )
    assert out["status"] == "error" and f"at most {CODE_MAX_CHUNK}" in out["error"], out
    f._terminated = True
    out = f._rpc["code.run"]("step([0] * 7)\n", timeout_s=30, tier="low")
    assert out["status"] == "error" and "the episode is over" in out["error"], out
    assert f.moves == []


def test_the_run_video_is_bounded():
    f = facade()
    f._keep_frames([np.zeros((1, 1, 3), np.uint8)] * (CODE_MAX_FRAMES + 5))
    assert len(f._run_frames) <= CODE_MAX_FRAMES


def test_the_low_tier_shows_examples_and_the_s4_tier_drops_them():
    f = facade()
    low = f._rpc["code.api"]("low")["primitives"]
    assert all(p.get("example") for p in low if p["name"] != "get_task_language")
    assert all(
        "example" not in p for p in f._rpc["code.api"]("low-noexamples")["primitives"]
    )
