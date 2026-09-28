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

"""The real Franka servers' limits, manifest methods and code mode (robots/franka/code_mode.py)
without hardware: the RLinf facade over a fake worker, the Polymetis facade over its mock NUC and
cameras, the dual rig over a fake two-arm worker. pi's limits come at spawn (``limits``) and hold
in the motion methods for pi's tools and programs alike."""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("omegaconf")  # the franka servers import their runtime config

from pi_embodied_services.components.manifest import ManifestError  # noqa: E402
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
from pi_embodied_services.utils.reach import ReachPreview  # noqa: E402

DOWN = (1.0, 0.0, 0.0, 0.0)
DOWN_WXYZ = [0.0, 1.0, 0.0, 0.0]
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
    return FrankaEnvFacade(w, code=code, limits=limits), w


def run(f, program: str, **kw):
    return f._rpc["code.run"](program, timeout_s=30, **kw)


def test_code_run_is_served_only_with_code_and_then_behind_the_token():
    plain, _ = rlinf(code=False)
    assert "code.run" not in plain._rpc and "code.set_limits" not in plain._rpc
    assert plain._rpc_token is None, "without --code the server answers as before"
    assert plain._rpc["code.api"]("high")["manifest_digest"], (
        "code.api answers without --code"
    )
    f, _ = rlinf()
    assert f._rpc_token and "code.run" in f._rpc and "code.set_limits" not in f._rpc
    with pytest.raises(PermissionError):
        f._serve_dispatch("env.get_robot_state", (), {})


def test_the_startup_self_check_passes_and_refuses_an_undeclared_method():
    f, _ = rlinf()
    f._manifest_ready()
    assert f.manifest["robot"] == "franka"
    f, _ = rlinf()
    f._rpc["env.nudge"] = lambda: None
    with pytest.raises(ManifestError, match="env.nudge"):
        f._manifest_ready()


def test_the_server_reports_the_limits_it_enforces():
    f, _ = rlinf()
    assert f._rpc["env.get_env_meta"]()["motion_limits"] == LIMITS
    plain, _ = rlinf(limits=None)
    assert plain.motion_limits() == {
        "max_move_m": 0.1,
        "max_rotate_rad": 0.5,
        "z_floor_m": None,
        "workspace_xy": None,
    }
    with pytest.raises(ValueError, match="workspace-xy"):
        rlinf(limits={"workspace_xy": [1, 0, 0, 1]})


def test_a_run_moves_through_the_facade_and_reports_motions_states_and_frames():
    f, w = rlinf()
    out = run(
        f,
        "move_delta([0.02, 0, 0])\nrotate_delta([0, 0, 0.2])\nclose_gripper()\n"
        "RESULT = get_robot_state()['raw_base_state']['tcp_pose'][:3]\n",
        tier="low",
    )
    assert out["status"] == "error", "close_gripper is the high tier's"
    w.moves.clear()
    w.tcp = np.array([0.5, 0.0, 0.3, *DOWN])
    out = run(
        f,
        "move_delta([0.02, 0, 0])\nrotate_delta([0, 0, 0.2])\nset_gripper(False)\n"
        "RESULT = get_robot_state()['raw_base_state']['tcp_pose'][:3]\n",
        tier="low",
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


def test_pi_limits_refuse_a_tool_call_and_a_program_alike_before_anything_moves():
    f, w = rlinf()
    move = f._rpc["env.move_delta"]
    with pytest.raises(ValueError, match="limit is 0.05 m per call"):
        move(delta_xyz=[0.06, 0, 0])
    with pytest.raises(ValueError, match="limit is 0.3 rad per call"):
        f._rpc["env.rotate_delta"](delta_rpy=[0, 0, 0.4])
    with pytest.raises(ValueError, match="outside the workspace"):
        move(delta_xyz=[0, 0, -0.05])  # well above the floor, but ...
        move(delta_xyz=[0, 0, -0.05])
        move(delta_xyz=[0, 0, -0.05])
        move(delta_xyz=[0, 0, -0.05])
    assert len(w.moves) == 3, "only the moves above the floor ran"
    for program, why in [
        ("move_delta([0.06, 0, 0])", "limit is 0.05 m per call"),
        ("rotate_delta([0, 0, 0.4])", "limit is 0.3 rad per call"),
        ("for _ in range(7):\n    move_delta([0, 0.05, 0])", "outside the workspace"),
    ]:
        w.moves.clear()
        w.tcp = np.array([0.5, 0.0, 0.3, *DOWN])
        out = run(f, program, tier="low")
        assert out["status"] == "error" and why in out["error"], (program, out)
    assert len(w.moves) == 6, "only the in-box moves ran"
    # Outside the box, a move back toward it is allowed.
    w.tcp = np.array([0.5, 0.0, 0.1, *DOWN])
    assert run(f, "move_delta([0, 0, 0.03])", tier="low")["status"] == "ran"


def test_the_run_cap_counts_translation():
    f, w = rlinf()
    out = run(
        f,
        "for _ in range(5):\n    move_delta([0.03, 0, 0])\n",
        max_move_m=0.1,
        tier="low",
    )
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


def test_code_api_lists_the_tiers_this_server_can_serve():
    f, _ = rlinf()
    high = f._rpc["code.api"]("high")["available"]
    assert sorted(high) == [
        "close_gripper",
        "goto_pose",
        "home_pose",
        "open_gripper",
    ], "without --sam3 the perception functions are out"
    low = f._rpc["code.api"]("low")["available"]
    assert {"move_delta", "rotate_delta", "set_gripper", "get_observation"} <= set(low)
    assert not {"solve_ik", "move_to_joints", "segment"} & set(low), "no --ik / --sam3"


# ---- CaP-X's high tier (single arm) -------------------------------------------


def test_goto_pose_runs_bounded_legs_rotation_first_and_checks_the_route_before_moving():
    f, w = rlinf()
    f._rpc["env.reset"]()
    out = f.goto_pose([0.5, 0.0, 0.2], DOWN_WXYZ, z_approach=0.05)
    # 5 cm down to the approach point is two legs within 0.95 x --max-move, then 5 cm more.
    assert out["reached"] and out["legs"] == 4, out
    assert [m[0] for m in w.moves] == ["move"] * 4
    assert np.allclose(w.tcp[:3], [0.5, 0.0, 0.2])
    for m in w.moves:
        assert np.linalg.norm(m[1]) <= 0.05 + 1e-9, "every leg within --max-move"
    # Too far for MAX_LEGS legs, or through the floor: refused before anything moves.
    w.moves.clear()
    with pytest.raises(ValueError, match="move closer first"):
        f.goto_pose([0.5, 0.29, 0.6], DOWN_WXYZ)
    with pytest.raises(ValueError, match="refused before moving"):
        f.goto_pose([0.5, 0.0, 0.1], DOWN_WXYZ)
    assert w.moves == []
    # A turn about z runs as bounded rotate_delta legs (each within --max-rotate).
    yaw = [0.0, np.cos(0.25), np.sin(0.25), 0.0]  # 0.5 rad about z from pointing down
    out = f.goto_pose([0.5, 0.0, 0.2], yaw)
    rotations = [m for m in w.moves if m[0] == "rotate"]
    assert len(rotations) == 2 and out["reached"], out
    assert all(np.linalg.norm(r[1]) <= 0.3 + 1e-9 for r in rotations)
    # home_pose: back to the TCP pose of the reset.
    w.moves.clear()
    f.home_pose()
    assert np.allclose(w.tcp[:3], [0.5, 0.0, 0.3], atol=1e-6)


def test_home_pose_needs_a_reset_and_the_grippers_are_set_gripper():
    f, w = rlinf()
    with pytest.raises(ValueError, match="no reset"):
        f.home_pose()
    f._rpc["env.open_gripper"]()
    f._rpc["env.close_gripper"]()
    assert w.moves == [("gripper", True), ("gripper", False)]


class FakePerception:
    """SAM3 that finds a 4x4 block in the middle of the third-person image."""

    class Book:
        def __init__(self):
            self.items = {}

        def get(self, id):
            return self.items[id]

    def __init__(self, found_on=("third_person",)):
        self.book = self.Book()
        self.found_on = found_on

    def capabilities(self):
        return {"segment": True, "enhance_depth": False}

    def segment(self, camera="wrist", *, prompt=None, **kw):
        if camera not in self.found_on:
            return {"found": False, "reason": "no mask", "ids": []}
        mask = np.zeros((8, 8), bool)
        mask[2:6, 2:6] = True
        self.book.items["d1"] = {"mask": mask}
        return {"found": True, "ids": ["d1"]}


def perceiving(monkeypatch, found_on=("third_person",)):
    from pi_embodied_services.robots.franka import perception as franka_perception

    class Worker(FakeWorker):
        def get_observation(self):
            return {
                "main_images": np.zeros((8, 8, 3), np.uint8),
                "main_depths": np.full((8, 8), 0.5, np.float32),
                "extra_view_images": np.zeros((1, 8, 8, 3), np.uint8),
                "extra_view_depths": np.full((1, 8, 8), 1.0, np.float32),
            }

        def get_camera_meta(self):
            K = [[10.0, 0.0, 4.0], [0.0, 10.0, 4.0], [0.0, 0.0, 1.0]]
            return {
                "observation_camera_map": {"main": "wrist", "extra_0": "ext"},
                "cameras": {"wrist": {"intrinsic_K": K}, "ext": {"intrinsic_K": K}},
            }

    # The external camera looks straight down from 1 m above the base origin + (0.5, 0, 0).
    ext = np.eye(4)
    ext[:3, :3] = np.diag([1.0, -1.0, -1.0])
    ext[:3, 3] = [0.5, 0.0, 1.0]
    monkeypatch.setattr(
        franka_perception,
        "load_calibration_bundle",
        lambda: {"external": {"matrix": ext}, "wrist": {"matrix": np.eye(4)}},
    )
    w = Worker()
    f = FrankaEnvFacade(w, code=True, limits=LIMITS)
    fake = FakePerception(found_on)
    f._perception = fake
    f._rpc["env.segment"] = fake.segment
    return f, w


def test_get_object_pose_back_projects_the_sam3_mask_through_the_calibration(
    monkeypatch,
):
    f, _ = perceiving(monkeypatch)
    pos, quat, extent = f.get_object_pose("red block", return_bbox_extent=True)
    # Pixels 2..5 around the principal point 4 at 1 m depth, fx 10: x -0.2..0.1, y mirrored.
    assert pos[2] == pytest.approx(0.0, abs=1e-6), "the table under a camera 1 m up"
    assert pos[0] == pytest.approx(0.5 - 0.05, abs=1e-6)
    assert quat == [1.0, 0.0, 0.0, 0.0]
    assert extent[0] == pytest.approx(0.3 * 0.9, abs=0.05)
    assert f.get_object_pose("red block")[2] is None
    # Found on neither camera: an error naming both.
    f, _ = perceiving(monkeypatch, found_on=())
    with pytest.raises(ValueError, match="third_person: no mask; wrist: no mask"):
        f.get_object_pose("ghost")


def test_sample_grasp_pose_without_a_grasp_server_keeps_the_current_orientation(
    monkeypatch,
):
    f, _ = perceiving(monkeypatch)
    pos, quat = f.sample_grasp_pose("red block")
    assert pos[2] == pytest.approx(0.0, abs=1e-6)
    assert quat == DOWN_WXYZ, "the TCP's xyzw [1, 0, 0, 0] as wxyz"
    # With a grasp planner: the best candidate, quaternion converted to wxyz.
    f._grasp = object()
    f._rpc["env.plan_grasp"] = lambda object: {
        "candidates": [{"eef_position": [0.4, 0.1, 0.2], "eef_quat_xyzw": [0, 0, 0, 1]}]
    }
    assert f.sample_grasp_pose("red block") == [[0.4, 0.1, 0.2], [1.0, 0.0, 0.0, 0.0]]


# ---- Polymetis ---------------------------------------------------------------


class FakeIk:
    """The ik service: ``ik.solve`` answers the seed + 0.1; ``ik.plan`` with goal_q a TCP path
    that moves 0.25 m in z per rad of joint 0 (a stand-in for forward kinematics)."""

    def __init__(self, tcp):
        self.tcp = np.asarray(tcp, dtype=float)
        self.calls = []

    def call(self, method, args, kwargs, timeout_s=None):
        self.calls.append(method)
        if method == "ik.solve":
            return {"ok": True, "q": list(np.asarray(kwargs["seed_q"]) + 0.1)}
        if method == "ik.plan":
            q0, q1 = np.asarray(kwargs["start_q"]), np.asarray(kwargs["goal_q"])
            n = kwargs["waypoints"]
            path = []
            for t in np.linspace(0, 1, n):
                p = self.tcp.copy()
                p[2] += 0.25 * (q1[0] - q0[0]) * t
                path.append(p.tolist())
            return {"ok": True, "tcp_path": path}
        raise AssertionError(method)


def polymetis(monkeypatch, ik=False, limits=LIMITS) -> FrankaPolymetisFacade:
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
    robot = MockPolymetisRobot((0.5, 0.0, 0.3, *DOWN), home_pose=(0.5, 0.0, 0.3, *DOWN))
    cams = {"wrist": MockRGBD("1"), "third_person": MockRGBD("2", depth_m=0.9)}
    reach = (
        ReachPreview(None, "panda", client=FakeIk((0.5, 0.0, 0.3, *DOWN)))
        if ik
        else None
    )
    return FrankaPolymetisFacade(
        cfg, robot, cams, sleep=lambda s: None, code=True, limits=limits, ik_reach=reach
    )


def test_polymetis_runs_a_program_under_both_its_own_and_pi_limits(monkeypatch):
    f = polymetis(monkeypatch, limits={**LIMITS, "max_move_m": 0.1})
    assert f._rpc_token and "code.run" in f._rpc
    f._manifest_ready()  # the self-check passes on this backend too
    out = run(f, "r = move_delta([0.0, 0.0, 0.03])\nRESULT = r['ok']\n", tier="low")
    assert out["status"] == "ran" and out["result"] is True, out
    assert out["motions"] == 1 and len(out["frames"]) == 1
    tcp = f.controller.state()["tcp_pose"]
    assert tcp[2] == pytest.approx(0.33, abs=2e-3)
    # pi allows 0.1 m, the controller's own per-call cap (0.08 m) still refuses.
    out = run(f, "move_delta([0.09, 0, 0])", tier="low")
    assert out["status"] == "error" and "limits.max_move_m" in out["error"], out


def test_polymetis_joint_moves_are_bounded_per_call_and_checked_along_the_path(
    monkeypatch,
):
    f = polymetis(monkeypatch, ik=True)
    f._manifest_ready()
    low = f._rpc["code.api"]("low")["available"]
    assert {"solve_ik", "traj_plan", "move_to_joints", "move_along_trajectory"} <= set(
        low
    )
    q0 = np.asarray(f.controller.state()["arm_joint_position"])
    assert f.solve_ik([0.5, 0.0, 0.3], DOWN_WXYZ) == pytest.approx(list(q0 + 0.1))
    robot = f._robot
    # A joint turned beyond limits.max_joint_step_rad (0.3): refused, nothing streamed.
    with pytest.raises(ValueError, match="max_joint_step_rad"):
        f._rpc["env.move_to_joints"](joints=list(q0 + [0.4, 0, 0, 0, 0, 0, 0]))
    # The TCP would move 0.0625 m (> --max-move 0.05), or end below the floor: refused.
    with pytest.raises(ValueError, match="limit is 0.05 m per call"):
        f._rpc["env.move_to_joints"](joints=list(q0 + [0.25, 0, 0, 0, 0, 0, 0]))
    f._reach._client.tcp[2] = 0.16
    with pytest.raises(ValueError, match="outside the workspace"):
        f._rpc["env.move_to_joints"](joints=list(q0 - [0.15, 0, 0, 0, 0, 0, 0]))
    f._reach._client.tcp[2] = 0.3
    assert robot.joint_setpoints == 0
    out = f._rpc["env.move_to_joints"](joints=list(q0 + [0.15, 0, 0, 0, 0, 0, 0]))
    assert out["ok"] and robot.joint_setpoints > 0
    assert robot.controller == "cartesian", "never left in joint impedance"
    assert np.allclose(robot.q, q0 + [0.15, 0, 0, 0, 0, 0, 0])
    # Ten waypoints of 0.045 m each: 0.45 m > MAX_LEGS x --max-move, refused before moving.
    n = robot.joint_setpoints
    q = robot.q.copy()
    zigzag = [list(q + [0.18 * (k % 2), 0, 0, 0, 0, 0, 0]) for k in range(1, 11)]
    with pytest.raises(ValueError, match="at most 0.400 m per call"):
        f._rpc["env.move_along_trajectory"](trajectory=zigzag)
    assert robot.joint_setpoints == n
    out = f._rpc["env.move_along_trajectory"](trajectory=zigzag[:4])
    assert out["waypoints_done"] == 4 and robot.joint_setpoints > n
    # The RLinf backend has no joint command: its code.api never lists them.
    g, _ = rlinf()
    g._reach = f._reach
    assert "move_to_joints" not in g._rpc


def test_polymetis_stop_cancels_a_joint_stream(monkeypatch):
    f = polymetis(monkeypatch, ik=True)
    q0 = np.asarray(f.controller.state()["arm_joint_position"])
    f.request_stop()
    monkeypatch.setattr(f.controller, "_stop", lambda: True)
    out = f.controller.move_joints(q0 + [0.1, 0, 0, 0, 0, 0, 0])
    assert out.get("cancelled") and f._robot.controller == "cartesian"


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
        raise AssertionError("never a program's call")


def test_dual_checks_the_arm_it_moves_and_the_posture_reset_is_no_primitive():
    from pi_embodied_services.robots.dual_franka.env_server import DualFrankaEnvFacade

    w = FakeDualWorker()
    f = DualFrankaEnvFacade(
        w,
        code=True,
        limits={**LIMITS, "workspace_xy": [0.1, 1.15, -0.35, 0.35]},
    )
    f._manifest_ready()
    assert f.manifest["robot"] == "dual_franka"
    out = run(
        f, "move_delta('left', [0, 0.04, 0])\nset_gripper('right', True)\n", tier="low"
    )
    assert out["status"] == "ran", out
    assert out["frames"] == [], "pi picks the dual rig's video camera"
    out = run(f, "move_delta('left', [0, 0.04, 0])", tier="low")
    assert out["status"] == "error" and "outside the workspace" in out["error"], out
    assert run(f, "move_delta('right', [0, 0.04, 0])", tier="low")["status"] == "ran"
    # The tool's path is limited the same way.
    with pytest.raises(ValueError, match="left arm"):
        f._rpc["env.move_delta"]("left", delta_xyz=[0, 0.04, 0])
    out = run(f, "recover_joint_posture('stuck')", tier="low")
    assert out["status"] == "error" and "recover_joint_posture" in out["error"]
    assert run(f, "close_gripper('left')", tier="high")["status"] == "ran"
    assert [m[0] for m in w.moves] == ["move", "gripper", "move", "gripper"]
    assert not {"goto_pose", "get_object_pose"} & set(
        f._rpc["code.api"]("high")["available"]
    )
    assert "env.goto_pose" not in f._rpc, "the single arm's high tier only"


def test_the_self_check_passes_with_the_geometric_toolset():
    w = FakeWorker()
    f = FrankaEnvFacade(w, geometry=True, limits=LIMITS)
    f._manifest_ready()
    assert {"view_points", "mark_point", "grip_target", "grip_state"} <= set(
        f._rpc["code.api"]("low")["available"]
    ), (
        "the geometry primitives are the server's; move_grip stays pi's (it executes the plan)"
    )
