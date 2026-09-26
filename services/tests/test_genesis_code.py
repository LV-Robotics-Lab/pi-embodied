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

"""Code mode on the Genesis server (``code.run`` over its registry), without a simulator: the
facade's own motion methods (move_delta, set_gripper, step, chunk_step, state) run over a fake
servo, so a program reaches exactly what the tools reach."""

from __future__ import annotations

import numpy as np
import pytest

from pi_embodied_services.robots.genesis import env_server as g
from pi_embodied_services.utils.rpc import RpcFacade


def facade() -> g.GenesisEnvFacade:
    """A registered Genesis facade whose servo moves the TCP to its target in 5 control steps;
    the cube rises with the TCP once the gripper closed on it."""
    f = object.__new__(g.GenesisEnvFacade)
    RpcFacade.__init__(f)
    f._task, f._seed, f._view_size = "cube_pick", 0, 4
    f._steps, f._success, f._hold, f._gripper_open = 0, False, 0, True
    f._offset = np.zeros(3)
    f._run_start, f._run_frames = 0, []
    f.pos = np.array([0.5, 0.0, 0.2])
    f.cube_z = g.CUBE_SIZE_M / 2
    f.target = f.pos.copy()

    def servo(target, max_steps=g.SERVO["max_steps"]):
        f.target = np.asarray(target, dtype=np.float64)
        for _ in range(5):
            f._step()
        return 5, False

    def step():
        if not f._gripper_open and f.pos[2] < 0.05:
            f.grasped = True
        if getattr(f, "grasped", False):
            f.cube_z += f.target[2] - f.pos[2]
        f.pos = f.target.copy()
        f._steps += 1
        f._success = f._success or g.lifted(f.cube_z)

    f._tcp = lambda: f.pos.copy()
    f._servo = servo
    f._step = step
    f._grip = lambda: (step(), (1, False))[1]
    f._command_arm = lambda target: setattr(f, "target", np.asarray(target))
    f._command_gripper = lambda: None
    f._width = lambda: 0.04 if f._gripper_open else 0.03
    f._render = lambda name, depth=False: np.full(
        (4, 4, 3), 1 if name == "agentview" else 2, np.uint8
    )
    f._state = lambda: {
        "tcp_pos": f.pos.astype(np.float32),
        "gripper_width": f._width(),
        "gripper_command": "open" if f._gripper_open else "close",
        "success": bool(f._success),
        "is_grasped": getattr(f, "grasped", False),
        "lift_m": round(max(0.0, f.cube_z - g.CUBE_SIZE_M / 2), 4),
        "env_steps": f._steps,
    }
    g.GenesisEnvFacade._register_rpc(f)
    return f


def test_the_server_serves_code_run_with_a_token_and_exclusivity():
    f = facade()
    assert g.GenesisEnvFacade.REQUIRE_TOKEN and f._rpc_token
    assert "code.run" in f._rpc and "code.helpers" in f._rpc
    with pytest.raises(PermissionError):
        f._admit("env.state", None)
    f._code._active = True
    with pytest.raises(RuntimeError, match="run_code program is running"):
        f._admit("env.state", f._rpc_token)


def test_a_run_moves_through_the_tools_methods_and_reports_steps_success_obs_frames():
    f = facade()
    out = f._rpc["code.run"](
        "move_delta([0, 0, -0.18], gripper='open')\n"
        "set_gripper(False)\n"
        "r = move_delta([0, 0, 0.15])\n"
        "RESULT = sorted(r)\n",
        timeout_s=30,
    )
    assert out["status"] == "ran", out
    assert out["success"] is True and out["steps"] == f._steps > 0
    for hidden in ("agentview", "wrist", "lift_m", "frames"):
        assert hidden not in out["result"], hidden
    assert {"tcp_pos", "moved_m", "is_grasped", "success"} <= set(out["result"])
    # The new observation in the shape the tools return it (images and the full state).
    assert out["obs"]["agentview"].shape == (4, 4, 3) and "lift_m" in out["obs"]
    assert out["obs"]["env_steps"] == f._steps
    # One run-video frame per primitive call: the two views side by side.
    assert len(out["frames"]) == 3 and out["frames"][0].shape == (4, 8, 3)
    assert [c.get("move_m", 0) for c in out["calls"]] == pytest.approx(
        [0.18, 0.0, 0.15]
    )


def test_the_low_tier_step_and_state_carry_no_object_state():
    f = facade()
    out = f._rpc["code.run"](
        "a = step([0, 0, 0.01, 1])\n"
        "c = chunk_step([[0, 0, 0.01, 1]] * 3)\n"
        "s = state()\n"
        "RESULT = [sorted(a), sorted(a['state']), sorted(c), sorted(s)]\n",
        timeout_s=30,
        tier="low",
    )
    assert out["status"] == "ran", out
    step, step_state, chunk, state = out["result"]
    assert step == ["reward", "state", "success", "truncated"]
    assert "chunk" not in chunk and "state" in chunk and "success" in chunk
    for keys in (step_state, state):
        assert "lift_m" not in keys and "agentview" not in keys and "tcp_pos" in keys
    assert out["calls"][1].get("move_m") == pytest.approx(0.03)
    assert f._steps == 4 and len(out["frames"]) == 2


def test_oversized_chunks_and_moves_past_the_cap_are_refused_before_they_run():
    f = facade()
    out = f._rpc["code.run"](
        f"chunk_step([[0, 0, 0, 1]] * {g.CODE_MAX_CHUNK + 1})\n",
        timeout_s=30,
        tier="low",
    )
    assert out["status"] == "error" and "at most" in out["error"], out
    out = f._rpc["code.run"](
        "move_delta([0, 0, 0.2])\nmove_delta([0, 0, 0.2])\n",
        timeout_s=30,
        max_move_m=0.3,
    )
    assert out["status"] == "error" and out["limit"] == "max_move_m", out
    assert f.pos[2] == pytest.approx(0.4), "only the first move ran"


def test_the_run_video_is_bounded_and_s4_drops_the_examples():
    f = facade()
    for _ in range(g.CODE_MAX_FRAMES + 5):
        f._keep_frame({"agentview": np.zeros((1, 1, 3)), "wrist": np.zeros((1, 1, 3))})
    assert len(f._run_frames) <= g.CODE_MAX_FRAMES
    low = {p["name"]: p for p in f._rpc["code.api"]("low")["primitives"]}
    assert all(
        "example" in low[n]
        for n in ("state", "move_delta", "set_gripper", "step", "chunk_step")
    )
    assert all(
        "example" not in p for p in f._rpc["code.api"]("low-noexamples")["primitives"]
    )
