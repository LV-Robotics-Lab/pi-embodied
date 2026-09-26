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

"""Code mode on the Piper server (``--code``, utils/code_real.py) against the mocked ROS arm of
test_piper.py: programs step through the facade's guarded ``step`` under pi's limits."""

from __future__ import annotations

import numpy as np
import pytest
from test_piper import HOME, FakeArm, FakeCamera

from pi_embodied_services.robots.piper.env_server import PiperEnvFacade


def cfg(**extra):
    return {
        "arm": "left",
        "cameras": {"front": "/f", "wrist": "/w", "image_size": 32},
        "calibration": {
            "z_floor_m": 0.0,
            "begin_joints": HOME.tolist(),
            "arm_id": "usb-A",
        },
        "limits": {"speed_mps": 0.3, "reset_time_s": 0.1},
        "gripper": {"settle_s": 0.0},
        "motion": {"settle_steps": 1, "settle_dt_s": 0.0},
        "smooth": {"dt_s": 0.001},
        **extra,
    }


def facade(code=True, limits=True) -> tuple[PiperEnvFacade, FakeArm]:
    arm = FakeArm()
    f = PiperEnvFacade(
        cfg(), arm, {"front": FakeCamera(), "wrist": FakeCamera()}, code=code
    )
    if code and limits:
        f._rpc["code.set_limits"](max_move_m=0.03, max_yaw_rad=0.1)
    return f, arm


def run(f, program, **kw):
    return f._rpc["code.run"](program, timeout_s=30, **kw)


def test_code_mode_needs_code_and_then_the_token():
    plain, _ = facade(code=False)
    assert "code.run" not in plain._rpc and plain._rpc_token is None
    f, _ = facade()
    assert f._rpc_token and {"code.run", "code.set_limits"} <= set(f._rpc)
    with pytest.raises(PermissionError):
        f._serve_dispatch("env.get_robot_state", (), {})


def test_a_program_steps_the_arm_and_the_run_reports_it():
    f, _ = facade()
    z0 = f._arm_state("left")["eef_pos"][2]
    out = run(f, "r = step([0, 0, 0.02])\nRESULT = r['ok']\n")
    assert out["status"] == "ran" and out["result"] is True, out
    assert f._arm_state("left")["eef_pos"][2] == pytest.approx(z0 + 0.02, abs=2e-3)
    assert out["motions"] == 1 and out["calls"][0]["move_m"] == pytest.approx(0.02)
    assert len(out["frames"]) == 1 and out["frames"][0].shape == (32, 32, 3)


def test_pi_limits_are_applied_before_the_arm_moves():
    f, arm = facade()
    q0 = np.array(arm.q)
    for program, why in [
        ("step([0.04, 0, 0])", "limit is 0.03 m per call"),
        ("step(yaw=0.15)", "limit of 0.1 rad"),
        ("move_joints('rest')", "not available in code mode"),
        ("halt_arm(reason='done')", "this server drives one arm"),
    ]:
        out = run(f, program)
        assert out["status"] == "error" and why in out["error"], (program, out)
    np.testing.assert_allclose(arm.q, q0)
    out = run(f, "for _ in range(4):\n    step([0.02, 0, 0])\n", max_move_m=0.05)
    assert out["status"] == "error" and out["limit"] == "max_move_m", out
    assert out["motions"] == 2


def test_without_pi_limits_a_program_cannot_step():
    f, arm = facade(limits=False)
    q0 = np.array(arm.q)
    out = run(f, "step([0, 0, 0.01])")
    assert out["status"] == "error" and "code.set_limits" in out["error"], out
    np.testing.assert_allclose(arm.q, q0)
    # A gripper-only step moves nothing, but it still goes through pi's limits.
    out = run(f, "step(gripper='close')")
    assert out["status"] == "error" and "code.set_limits" in out["error"], out


def test_the_low_tier_shows_examples():
    f, _ = facade()
    low = {p["name"]: p for p in f._rpc["code.api"]("low")["primitives"]}
    assert "example" in low["step"] and "example" in low["get_observation"]
