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

"""Code mode on the real Franka servers (robots/franka/code_mode.py) without hardware: the RLinf
facade over a fake worker, the Polymetis facade over its mock NUC and cameras, the dual rig over a
fake two-arm worker. Programs run through the facades' own motion methods; pi's per-call limits
come from ``code.set_limits``."""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("omegaconf")  # the franka servers import their runtime config

from pi_embodied_services.robots.franka.env_server import FrankaEnvFacade  # noqa: E402
from pi_embodied_services.robots.franka_polymetis.env_server import (  # noqa: E402
    FrankaPolymetisFacade,
)
from pi_embodied_services.robots.franka_polymetis.mock import (  # noqa: E402
    MOCK_ENV,
    MockPolymetisRobot,
    MockRGBD,
)
from pi_embodied_services.utils.code_real import CODE_MAX_FRAMES  # noqa: E402

DOWN = (1.0, 0.0, 0.0, 0.0)
LIMITS = {
    "max_move_m": 0.05,
    "max_rotate_rad": 0.3,
    "z_floor_m": 0.14,
    "workspace_xy": [0.3, 0.7, -0.3, 0.3],
}


class FakeWorker:
    """The RLinf worker's env.* methods: a TCP that moves by the commanded delta."""

    def __init__(self):
        self.tcp = np.array([0.5, 0.0, 0.3, *DOWN])
        self.moves: list = []
        self.stops: list = []

    def get_env_meta(self):
        return {}

    def reset(self):
        return {}

    def get_robot_state(self):
        return {"raw_base_state": {"tcp_pose": self.tcp.copy()}}

    def get_observation(self):
        return {
            "main_images": np.zeros((4, 4, 3), np.uint8),
            "extra_view_images": np.full((1, 4, 4, 3), 5, np.uint8),
        }

    def get_camera_meta(self):
        return {}

    def move_delta(self, delta_xyz):
        self.moves.append(("move", list(delta_xyz)))
        self.tcp[:3] += np.asarray(delta_xyz)
        return {"ok": True, "states": np.arange(3.0) + len(self.moves)}

    def rotate_delta(self, delta_rpy):
        self.moves.append(("rotate", list(delta_rpy)))
        return {"ok": True, "states": None}

    def set_gripper(self, *, open):
        self.moves.append(("gripper", open))
        return {"ok": True}

    def chunk_step(self, actions, **kw):
        raise AssertionError("not in the registry")

    def request_stop(self, generation):
        self.stops.append(generation)


def rlinf(code=True, limits=LIMITS) -> tuple[FrankaEnvFacade, FakeWorker]:
    w = FakeWorker()
    f = FrankaEnvFacade(w, code=code)
    if code and limits:
        f._rpc["code.set_limits"](**limits)
    return f, w


def run(f, program: str, **kw):
    return f._rpc["code.run"](program, timeout_s=30, **kw)


def test_code_mode_is_served_only_with_code_and_then_behind_the_token():
    plain, _ = rlinf(code=False)
    assert "code.run" not in plain._rpc and "code.set_limits" not in plain._rpc
    assert plain._rpc_token is None, "without --code the server answers as before"
    f, _ = rlinf()
    assert f._rpc_token and {"code.run", "code.set_limits"} <= set(f._rpc)
    with pytest.raises(PermissionError):
        f._serve_dispatch("env.get_robot_state", (), {})


def test_a_run_moves_through_the_facade_and_reports_motions_states_and_frames():
    f, w = rlinf()
    out = run(
        f,
        "move_delta([0.02, 0, 0])\nrotate_delta([0, 0, 0.2])\nset_gripper(False)\n"
        "RESULT = get_robot_state()['raw_base_state']['tcp_pose'][:3]\n",
    )
    assert out["status"] == "ran", out
    assert np.allclose(out["result"], [0.52, 0.0, 0.3])
    assert [m[0] for m in w.moves] == ["move", "rotate", "gripper"]
    assert out["motions"] == 3
    assert np.allclose(out["states"], [1.0, 2.0, 3.0]), (
        "the last motion's wrapped states"
    )
    assert len(out["frames"]) == 3 and out["frames"][0][0, 0, 0] == 5, "external camera"
    assert [c.get("move_m", 0.0) for c in out["calls"][:3]] == pytest.approx(
        [0.02, 0.0, 0.0]
    )


def test_pi_limits_refuse_a_program_motion_before_it_is_commanded():
    f, w = rlinf()
    for program, why in [
        ("move_delta([0.06, 0, 0])", "limit is 0.05 m per call"),
        ("rotate_delta([0, 0, 0.4])", "limit is 0.3 rad per call"),
        (
            "move_delta([0, 0, -0.05])\nmove_delta([0, 0, -0.05])\nmove_delta([0, 0, -0.05])\nmove_delta([0, 0, -0.05])",
            "outside the workspace",
        ),
        (
            "move_delta([0, 0.05, 0])\nmove_delta([0, 0.05, 0])\nmove_delta([0, 0.05, 0])\n"
            "move_delta([0, 0.05, 0])\nmove_delta([0, 0.05, 0])\nmove_delta([0, 0.05, 0])\n"
            "move_delta([0, 0.05, 0])",
            "outside the workspace",
        ),
    ]:
        w.moves.clear()
        w.tcp = np.array([0.5, 0.0, 0.3, *DOWN])
        out = run(f, program)
        assert out["status"] == "error" and why in out["error"], (program, out)
    assert len(w.moves) == 6, "only the in-box moves ran"
    # Outside the box, a move back toward it is allowed.
    w.tcp = np.array([0.5, 0.0, 0.1, *DOWN])
    assert run(f, "move_delta([0, 0, 0.03])")["status"] == "ran"


def test_without_pi_limits_a_program_cannot_move_and_the_run_cap_counts_translation():
    f, w = rlinf(limits=None)
    out = run(f, "move_delta([0.01, 0, 0])")
    assert out["status"] == "error" and "code.set_limits" in out["error"], out
    assert w.moves == []
    f, w = rlinf()
    out = run(f, "for _ in range(5):\n    move_delta([0.03, 0, 0])\n", max_move_m=0.1)
    assert out["status"] == "error" and out["limit"] == "max_move_m", out
    assert len(w.moves) == 3


def test_a_stop_reaches_the_worker_and_the_program():
    f, w = rlinf()
    f._on_stop(7)
    assert w.stops == [7]


def test_the_run_video_is_bounded():
    f, _ = rlinf()
    f._begin_run()
    for _ in range(CODE_MAX_FRAMES + 3):
        f._code_after(None)
    assert len(f._run_frames) <= CODE_MAX_FRAMES
    assert f._run_motions == CODE_MAX_FRAMES + 3


def test_the_low_tier_shows_examples():
    f, _ = rlinf()
    low = {p["name"]: p for p in f._rpc["code.api"]("low")["primitives"]}
    assert "example" in low["move_delta"] and "example" in low["get_observation"]
    assert all(
        "example" not in p for p in f._rpc["code.api"]("low-noexamples")["primitives"]
    )


# ---- Polymetis ---------------------------------------------------------------


def polymetis(monkeypatch) -> FrankaPolymetisFacade:
    monkeypatch.setenv(MOCK_ENV, "1")
    cfg = {
        "robot": {"nuc_ip": "127.0.0.1", "tcp_offset_m": [0.0, 0.0, 0.0]},
        "cameras": {
            "image_size": 64,
            "devices": {
                "wrist": {"serial": "1", "main": True},
                "third_person": {"serial": "2"},
            },
        },
        "limits": {
            "z_floor_m": 0.14,
            "workspace_min": [0.30, -0.35, 0.10],
            "workspace_max": [0.75, 0.35, 0.60],
            "max_move_m": 0.08,
            "max_rotate_rad": 0.2,
            "settle_dt_s": 0.0,
        },
        "gripper": {"settle_s": 0.0, "min_settle_s": 0.0},
        "reset": {
            "begin_joints": [0.0, -0.5, 0.0, -2.6, 0.0, 2.07, 0.86],
            "lift_m": 0.05,
        },
    }
    robot = MockPolymetisRobot((0.5, 0.0, 0.3, *DOWN))
    cams = {"wrist": MockRGBD("1"), "third_person": MockRGBD("2", depth_m=0.9)}
    return FrankaPolymetisFacade(cfg, robot, cams, sleep=lambda s: None, code=True)


def test_polymetis_runs_a_program_under_both_its_own_and_pi_limits(monkeypatch):
    f = polymetis(monkeypatch)
    assert f._rpc_token and "code.run" in f._rpc
    f._rpc["code.set_limits"](**{**LIMITS, "max_move_m": 0.1})
    out = run(f, "r = move_delta([0.0, 0.0, 0.03])\nRESULT = r['ok']\n")
    assert out["status"] == "ran" and out["result"] is True, out
    assert out["motions"] == 1 and len(out["frames"]) == 1
    tcp = f.controller.state()["tcp_pose"]
    assert tcp[2] == pytest.approx(0.33, abs=2e-3)
    # pi allows 0.1 m, the controller's own per-call cap (0.08 m) still refuses.
    out = run(f, "move_delta([0.09, 0, 0])")
    assert out["status"] == "error" and "limits.max_move_m" in out["error"], out


# ---- dual Franka ---------------------------------------------------------------


class FakeDualWorker(FakeWorker):
    def __init__(self):
        super().__init__()
        self.arms = {
            "left": np.array([0.5, 0.3, 0.3, *DOWN]),
            "right": np.array([0.5, -0.3, 0.3, *DOWN]),
        }

    def get_robot_state(self):
        return {f"{a}_arm": {"tcp_pose": p.copy()} for a, p in self.arms.items()}

    def move_delta(self, arm, delta_xyz):
        self.moves.append(("move", arm, list(delta_xyz)))
        self.arms[arm][:3] += np.asarray(delta_xyz)
        return {"ok": True, "states": None}

    def rotate_delta(self, arm, delta_rpy):
        self.moves.append(("rotate", arm, list(delta_rpy)))
        return {"ok": True}

    def set_gripper(self, arm, *, open):
        self.moves.append(("gripper", arm, open))
        return {"ok": True}

    def recover_joint_posture(self, reason="", return_to_start=True):
        raise AssertionError("refused in code mode")


def test_dual_checks_the_arm_it_moves_and_refuses_the_posture_reset():
    from pi_embodied_services.robots.dual_franka.env_server import DualFrankaEnvFacade

    w = FakeDualWorker()
    f = DualFrankaEnvFacade(w, code=True)
    f._rpc["code.set_limits"](
        max_move_m=0.05,
        max_rotate_rad=0.3,
        z_floor_m=0.14,
        workspace_xy=[0.1, 1.15, -0.35, 0.35],
    )
    out = run(f, "move_delta('left', [0, 0.04, 0])\nset_gripper('right', True)\n")
    assert out["status"] == "ran", out
    assert out["frames"] == [], "pi picks the dual rig's video camera"
    out = run(f, "move_delta('left', [0, 0.04, 0])")
    assert out["status"] == "error" and "outside the workspace" in out["error"], out
    assert run(f, "move_delta('right', [0, 0.04, 0])")["status"] == "ran"
    out = run(f, "recover_joint_posture('stuck')")
    assert out["status"] == "error" and "not available in code mode" in out["error"]
    assert [m[0] for m in w.moves] == ["move", "gripper", "move"]
