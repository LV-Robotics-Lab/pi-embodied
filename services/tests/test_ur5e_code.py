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

"""Code mode on the UR5e server (``--code``, utils/code_real.py) against the mock arm, gripper and
cameras of test_ur5e.py: programs move through the facade's guarded methods under pi's limits."""

from __future__ import annotations

import numpy as np
import pytest
from test_ur5e import facade

from pi_embodied_services.components.cameras.mock import MOCK_ENV


@pytest.fixture(autouse=True)
def _mock_env(monkeypatch):
    monkeypatch.setenv(MOCK_ENV, "1")


def coded(limits=True):
    f = facade(code=True)
    if limits:
        f._rpc["code.set_limits"](max_move_m=0.05, max_rotate_rad=0.1)
    return f


def run(f, program, **kw):
    return f._rpc["code.run"](program, timeout_s=30, **kw)


def tcp(f) -> np.ndarray:
    return np.asarray(f.controller.state()["tcp_pose"][:3])


def test_code_mode_needs_code_and_then_the_token():
    plain = facade()
    assert "code.run" not in plain._rpc and plain._rpc_token is None
    f = coded()
    assert f._rpc_token and {"code.run", "code.set_limits"} <= set(f._rpc)
    with pytest.raises(PermissionError):
        f._serve_dispatch("env.get_robot_state", (), {})


def test_a_program_moves_through_the_facade_and_the_run_reports_it():
    f = coded()
    start = tcp(f)
    out = run(
        f,
        "move_delta([0, 0, 0.03])\n"
        "p = get_robot_state()['raw_base_state']['tcp_pose']\n"
        "move_pose([p[0] + 0.02, p[1], p[2]])\n"
        "set_gripper(False)\n",
    )
    assert out["status"] == "ran", out
    np.testing.assert_allclose(tcp(f), start + [0.02, 0, 0.03], atol=1e-3)
    assert out["motions"] == 3 and len(out["frames"]) == 3
    moves = [
        c.get("move_m", 0.0) for c in out["calls"] if c["name"] != "get_robot_state"
    ]
    assert moves == pytest.approx([0.03, 0.02, 0.0], abs=1e-3)


def test_pi_limits_refuse_before_the_arm_moves():
    f = coded()
    start = tcp(f)
    for program, why in [
        ("move_delta([0.06, 0, 0])", "limit is 0.05 m per call"),
        ("move_pose([0.45, 0.06, 0.30])", "limit is 0.05 m per call"),
        ("rotate_delta([0, 0, 0.15])", "limit is 0.1 rad per call"),
    ]:
        out = run(f, program)
        assert out["status"] == "error" and why in out["error"], (program, out)
    np.testing.assert_allclose(tcp(f), start, atol=1e-9)
    # The server's own limits still apply under pi's (its floor at z 0.14).
    out = run(f, "for _ in range(5):\n    move_delta([0, 0, -0.04])\n")
    assert out["status"] == "error" and tcp(f)[2] >= 0.14 - 1e-9, out


def test_without_pi_limits_a_program_cannot_move_and_the_run_cap_holds():
    f = coded(limits=False)
    out = run(f, "move_delta([0, 0, 0.01])")
    assert out["status"] == "error" and "code.set_limits" in out["error"], out
    f = coded()
    out = run(f, "for _ in range(4):\n    move_delta([0, 0, 0.04])\n", max_move_m=0.1)
    assert out["status"] == "error" and out["limit"] == "max_move_m", out
    assert out["motions"] == 2


def test_the_low_tier_shows_examples():
    f = coded()
    low = {p["name"]: p for p in f._rpc["code.api"]("low")["primitives"]}
    assert "example" in low["move_delta"] and "example" in low["get_observation"]
