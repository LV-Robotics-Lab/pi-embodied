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

"""Franka Polymetis env server against test doubles (no NUC, no cameras, no arm)."""

from __future__ import annotations

import copy
import json
import math
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

import numpy as np
import pytest
import yaml

from pi_embodied_services.robots.franka_polymetis import control, env_server, nuc_server
from pi_embodied_services.robots.franka_polymetis.calibration import (
    easy_handeye_yaml,
    matrix_to_quat_xyzw,
)
from pi_embodied_services.robots.franka_polymetis.control import (
    quat_angle,
    quat_from_euler_xyz,
    quat_rotate,
    tool_tilt,
)
from pi_embodied_services.robots.franka_polymetis.env_server import (
    DEFAULT_CONFIG,
    METHODS,
    FrankaPolymetisFacade,
    letterbox_geometry,
    letterbox_intrinsics,
    limits_from_config,
    load_config,
    main,
)
from pi_embodied_services.robots.franka_polymetis.mock import (
    MOCK_ENV,
    MockPolymetisRobot,
    MockRGBD,
)
from pi_embodied_services.utils.rpc.http_rpc import HttpRpcClient
from pi_embodied_services.utils.rpc.main_thread_serve import MainThreadServeMixin

SERVICES = Path(__file__).resolve().parents[1]
DOWN = (1.0, 0.0, 0.0, 0.0)  # gripper pointing down: 180 deg about x

CFG = {
    "robot": {"nuc_ip": "127.0.0.1", "tcp_offset_m": [0.0, 0.0, 0.0]},
    "cameras": {
        "image_size": 256,
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
    "reset": {"begin_joints": [0.0, -0.5, 0.0, -2.6, 0.0, 2.07, 0.86], "lift_m": 0.05},
}


@pytest.fixture(autouse=True)
def _mock_env(monkeypatch):
    monkeypatch.setenv(MOCK_ENV, "1")


def cfg(**sections) -> dict:
    out = copy.deepcopy(CFG)
    for name, values in sections.items():
        out.setdefault(name, {}).update(values)
    return out


def facade(robot=None, config=None, sleep=lambda s: None) -> FrankaPolymetisFacade:
    robot = robot or MockPolymetisRobot((0.5, 0.0, 0.3, *DOWN))
    cams = {"wrist": MockRGBD("1"), "third_person": MockRGBD("2", depth_m=0.9)}
    return FrankaPolymetisFacade(config or cfg(), robot, cams, sleep=sleep)


def call(f: FrankaPolymetisFacade, method: str, *args, **kwargs):
    return f._serve_dispatch(method, args, kwargs)


class WallRobot(MockPolymetisRobot):
    """A wall at x = ``wall_x`` stops the measured pose; ``reflex`` makes contact
    terminate the controller (a libfranka collision reflex)."""

    def __init__(self, *args, wall_x: float, reflex: bool = False, **kwargs):
        super().__init__(*args, **kwargs)
        self.wall_x, self.reflex = wall_x, reflex

    def update_desired_ee_pose(self, pose):
        super().update_desired_ee_pose(pose)
        if self.pose[0] > self.wall_x:
            self.pose[0] = self.wall_x
            if self.reflex:
                self.lose_controller()


# -- RPC parity with the RLinf backend ----------------------------------------


def test_method_list_matches_the_rlinf_server():
    pytest.importorskip("omegaconf")
    from pi_embodied_services.robots.franka.env_server import FrankaEnvFacade

    assert METHODS == FrankaEnvFacade._METHODS
    f = facade()
    # Both servers also answer env.preview_reach (utils/reach.py; "unknown" without --ik).
    assert {m for m in f._rpc if m.startswith("env.")} == {
        f"env.{m}" for m in FrankaEnvFacade._METHODS
    } | {"env.preview_reach"}
    # Both backends serve the same primitive registry (without perception: the base set).
    from pi_embodied_services.robots.franka.primitives import (
        FRANKA_PRIMITIVES,
        franka_primitives,
    )

    assert "code.api" in f._rpc and FrankaEnvFacade._PRIMITIVES is FRANKA_PRIMITIVES
    assert franka_primitives(None) == FRANKA_PRIMITIVES
    assert [p["name"] for p in f._rpc["code.api"]("high")["primitives"]] == [
        p.name for p in FRANKA_PRIMITIVES if "high" in p.tiers
    ]


def test_capabilities_have_the_same_keys_on_both_backends():
    pytest.importorskip("omegaconf")
    from pi_embodied_services.robots.franka.env_server import rlinf_capabilities

    rlinf = rlinf_capabilities(
        {
            "main_image_key": "wrist_1",
            "override_cfg": {
                "camera_names": {"1": "wrist_1", "2": "third_person"},
                "ee_pose_limit_min": [0.15, -0.45, 0.03, 2.9, -0.1, -1.7],
                "ee_pose_limit_max": [1.15, 0.54, 0.27, 2.9, -0.1, 1.4],
                "enable_camera_depth": True,
            },
        },
        [0.02, 0.1, 1.0],
    )
    poly = facade().capabilities()
    assert set(rlinf) == set(poly)
    assert rlinf["has_vla"] and not poly["has_vla"]
    assert rlinf["cameras"] == {"main": "wrist_1", "extra_0": "third_person"}
    assert rlinf["z_floor_m"] == 0.03 and rlinf["workspace"]["max"][2] == 0.27


def test_results_carry_the_rlinf_keys_and_healthz_names_the_service():
    f = facade()
    meta = call(f, "env.get_env_meta")
    assert {"ok", "action_dim", "action_scale", "use_relative_frame"} <= set(meta)
    assert meta["capabilities"]["backend"] == "polymetis"
    move = call(f, "env.move_delta", [0.02, 0.0, 0.0])
    assert {
        "ok",
        "requested_delta_xyz_base",
        "start_tcp_pose",
        "final_tcp_pose",
        "final_error_m",
        "steps_used",
        "states",
    } <= set(move)
    rot = call(f, "env.rotate_delta", [0.0, 0.0, 0.1])
    assert {"ok", "requested_delta_rpy_base", "final_error_rad", "steps_used"} <= set(
        rot
    )
    grip = call(f, "env.set_gripper", open=True)
    assert {"ok", "target_gripper_open", "steps_used", "robot_state", "states"} <= set(
        grip
    )
    state = call(f, "env.get_robot_state")
    base = state["raw_base_state"]
    assert len(base["tcp_pose"]) == 7 and len(base["gripper_position"]) == 1
    assert isinstance(base["gripper_open"], bool)
    assert call(f, "healthz")["service"] == "franka-polymetis-env"
    with pytest.raises(ValueError, match="no VLA action space"):
        call(f, "env.chunk_step", np.zeros((4, 7), np.float32))


# -- motion and limits ----------------------------------------------------------


@pytest.mark.parametrize("smooth", [False, True])
def test_move_ramps_the_setpoint_in_servo_ticks_and_reaches_the_target(smooth):
    robot = MockPolymetisRobot((0.5, 0.0, 0.3, *DOWN))
    f = facade(robot, cfg(smooth={"enabled": smooth}))
    before = len(robot.setpoints)
    r = call(f, "env.move_delta", [0.02, -0.01, 0.0])
    assert r["ok"] and r["final_error_m"] < 1e-9
    # Linear: equal servo steps. Smooth: Show-Harness's 20 min-jerk waypoints.
    assert r["steps_used"] == (
        20 if smooth else math.ceil(math.hypot(0.02, 0.01) / 0.0025)
    )
    path = [(0.5, 0.0, 0.3)] + [p[:3] for p in robot.setpoints[before:]]
    steps = np.diff(path, axis=0)
    assert np.max(np.linalg.norm(steps, axis=1)) <= 0.0025 + 1e-9
    np.testing.assert_allclose(robot.pose[:3], [0.52, -0.01, 0.3], atol=1e-9)


def test_setpoint_accumulates_without_integrating_measurement_noise():
    robot = MockPolymetisRobot((0.5, 0.0, 0.3, *DOWN))
    f = facade(robot)
    call(f, "env.move_delta", [0.02, 0.0, 0.0])
    robot.pose[0] += 0.004  # measured pose sags/jitters (below the resync gap)
    call(f, "env.move_delta", [0.02, 0.0, 0.0])
    np.testing.assert_allclose(robot.setpoints[-1][:3], [0.54, 0.0, 0.3], atol=1e-9)


@pytest.mark.parametrize(
    "delta, match",
    [
        ([0.09, 0.0, 0.0], "limit is 0.08 m per call"),
        ([0.0, 0.0, -0.08], "outside the workspace"),  # z 0.12 < the 0.14 floor
        ([0.0, 0.08, 0.0], "outside the workspace"),  # y 0.38 > ymax 0.35
        ([float("nan"), 0.0, 0.0], "finite"),
    ],
)
def test_move_refuses_instead_of_clamping(delta, match):
    robot = MockPolymetisRobot((0.5, 0.30, 0.2, *DOWN))
    f = facade(robot)
    sent = len(robot.setpoints)
    with pytest.raises(ValueError, match=match):
        call(f, "env.move_delta", delta)
    assert len(robot.setpoints) == sent, "a refused move must command nothing"
    np.testing.assert_allclose(robot.pose[:3], [0.5, 0.30, 0.2])


def test_z_floor_refuses_descent_but_allows_moving_back_up():
    robot = MockPolymetisRobot((0.5, 0.0, 0.15, *DOWN))
    f = facade(robot)
    with pytest.raises(ValueError, match="z 0.14"):
        call(f, "env.move_delta", [0.0, 0.0, -0.02])
    robot.pose[2] = 0.12  # hand-guided below the floor: only moves back in run
    f.controller.sync()
    with pytest.raises(ValueError, match="outside the workspace"):
        call(f, "env.move_delta", [0.01, 0.0, 0.0])
    assert call(f, "env.move_delta", [0.0, 0.0, 0.01])["final_tcp_pose"][
        2
    ] == pytest.approx(0.13)


def test_rotation_limit_and_base_frame_semantics():
    robot = MockPolymetisRobot((0.5, 0.0, 0.3, *DOWN))
    f = facade(robot)
    with pytest.raises(ValueError, match="limit is 0.2 rad per call"):
        call(f, "env.rotate_delta", [0.0, 0.0, 0.25])
    r = call(f, "env.rotate_delta", [0.0, 0.0, 0.15])
    assert r["ok"]
    expected = np.array(
        [math.cos(0.075), math.sin(0.075), 0.0, 0.0]
    )  # Rz(0.15) * Rx(pi): yaw about base +z
    assert quat_angle(r["final_tcp_pose"][3:], expected) < 1e-9
    np.testing.assert_allclose(r["final_tcp_pose"][:3], [0.5, 0.0, 0.3], atol=1e-9)


def test_euler_matches_scipy_extrinsic_xyz():
    def rx(a):
        return np.array(
            [[1, 0, 0], [0, math.cos(a), -math.sin(a)], [0, math.sin(a), math.cos(a)]]
        )

    def ry(a):
        return np.array(
            [[math.cos(a), 0, math.sin(a)], [0, 1, 0], [-math.sin(a), 0, math.cos(a)]]
        )

    def rz(a):
        return np.array(
            [[math.cos(a), -math.sin(a), 0], [math.sin(a), math.cos(a), 0], [0, 0, 1]]
        )

    rpy = (0.3, -0.2, 0.7)
    q = quat_from_euler_xyz(rpy)
    m = np.stack([quat_rotate(q, e) for e in np.eye(3)], axis=1)
    np.testing.assert_allclose(m, rz(rpy[2]) @ ry(rpy[1]) @ rx(rpy[0]), atol=1e-12)


def test_blocked_descent_is_reported_and_stops_pressing():
    robot = MockPolymetisRobot((0.5, 0.0, 0.22, *DOWN), surface_z=0.20)
    f = facade(robot)
    r = call(f, "env.move_delta", [0.0, 0.0, -0.05])
    assert r["descent_blocked"] is True and r["ok"] is False
    assert r["descent_travelled_m"] == pytest.approx(0.02)
    assert robot.setpoints[-1][2] == pytest.approx(0.20)  # re-anchored, not 0.17
    free = MockPolymetisRobot((0.5, 0.0, 0.30, *DOWN))
    assert "descent_blocked" not in call(facade(free), "env.move_delta", [0, 0, -0.05])


def test_lost_controller_is_restarted_once_and_a_far_restart_aborts():
    robot = MockPolymetisRobot((0.5, 0.0, 0.3, *DOWN))
    f = facade(robot)
    starts = robot.starts
    robot.lose_controller()
    assert call(f, "env.move_delta", [0.02, 0.0, 0.0])["ok"]
    assert robot.starts == starts + 1
    # Mid-move a reflex stops the controller with the arm pushed 5 cm aside.
    command = robot.update_desired_ee_pose
    sent = len(robot.setpoints)

    def reflex(pose):
        if len(robot.setpoints) == sent + 3 and robot.controller == "cartesian":
            robot.lose_controller()
            robot.pose[1] += 0.05
        command(pose)

    robot.update_desired_ee_pose = reflex
    with pytest.raises(RuntimeError, match="motion aborted"):
        call(f, "env.move_delta", [0.0, 0.0, 0.02])
    assert len(robot.setpoints) == sent + 3, "nothing is sent after the far restart"
    np.testing.assert_allclose(f.controller.target_pos, robot.pose[:3])


# -- gripper --------------------------------------------------------------------


def test_empty_grasp_reopens():
    robot = MockPolymetisRobot((0.5, 0.0, 0.3, *DOWN))
    f = facade(robot)
    r = call(f, "env.set_gripper", open=False)
    assert r["grasp_empty"] is True and r["ok"] is False
    assert robot.gripper_commands[-2:] == [True, False]
    assert robot.width == pytest.approx(0.08)
    assert r["robot_state"]["raw_base_state"]["gripper_open"] is True


def test_real_grasp_holds():
    robot = MockPolymetisRobot((0.5, 0.0, 0.3, *DOWN), object_width=0.039)
    r = call(facade(robot), "env.set_gripper", open=False)
    assert r["ok"] is True and "grasp_empty" not in r
    assert r["robot_state"]["raw_base_state"]["gripper_open"] is False
    assert robot.gripper_commands[-1] is True


def test_gripper_state_is_measured_not_commanded():
    # A close the fingers never execute (fault, or stopped before they moved): the
    # command says closed, the fingers are still open.
    robot = MockPolymetisRobot((0.5, 0.0, 0.3, *DOWN))
    robot.jammed = True
    f = facade(robot)
    call(f, "env.set_gripper", open=False)
    base = call(f, "env.get_robot_state")["raw_base_state"]
    assert robot.gripper_commands[-1] is True
    assert base["gripper_commanded_open"] is False
    assert base["gripper_position"] == [pytest.approx(0.08)]
    assert base["gripper_open"] is True and base["gripper_closed"] is False
    assert base["gripper_grasped"] is False
    # A wide object stops the fingers above the width threshold: closed per the
    # gripper's grasp flag, and the width is where the fingers stopped.
    robot = MockPolymetisRobot((0.5, 0.0, 0.3, *DOWN), object_width=0.075)
    f = facade(robot)
    call(f, "env.set_gripper", open=False)
    base = call(f, "env.get_robot_state")["raw_base_state"]
    assert base["gripper_position"] == [pytest.approx(0.075)]
    assert base["gripper_grasped"] is True and base["gripper_moving"] is False
    assert base["gripper_open"] is False and base["gripper_closed"] is True
    call(f, "env.set_gripper", open=True)
    base = call(f, "env.get_robot_state")["raw_base_state"]
    assert base["gripper_open"] is True and base["gripper_grasped"] is False
    assert base["gripper_commanded_open"] is True


def test_jammed_gripper_is_flagged():
    # Fingers that ignore a close (still open, nothing grasped): jammed, not ok.
    robot = MockPolymetisRobot((0.5, 0.0, 0.3, *DOWN))
    robot.jammed = True
    f = facade(robot)
    r = call(f, "env.set_gripper", open=False)
    assert r["gripper_jammed"] is True and r["ok"] is False
    assert "gripper jammed" in r["note"]
    # Fingers stuck closed on nothing ignore an open.
    robot.jammed = False
    call(f, "env.set_gripper", open=False)  # empty close: reopened
    robot.width, robot.jammed = 0.0002, True
    r = call(f, "env.set_gripper", open=True)
    assert r["gripper_jammed"] is True and r["ok"] is False
    # Not jammed: an open gripper told to open, a grasped object told to close again,
    # and fingers that move.
    robot.jammed, robot.width = False, 0.08
    assert "gripper_jammed" not in call(f, "env.set_gripper", open=True)
    held = MockPolymetisRobot((0.5, 0.0, 0.3, *DOWN), object_width=0.075)
    g = facade(held)
    call(g, "env.set_gripper", open=False)
    held.jammed = True
    r = call(g, "env.set_gripper", open=False)
    assert "gripper_jammed" not in r and r["ok"] is True
    r = call(facade(), "env.set_gripper", open=False)
    assert "gripper_jammed" not in r and r["grasp_empty"] is True


# -- reset ---------------------------------------------------------------------


@pytest.mark.parametrize("method", ["joint_stream", "move_to_joint_positions"])
def test_reset_opens_lifts_homes_and_restarts_impedance(method):
    robot = MockPolymetisRobot((0.5, 0.0, 0.2, *DOWN), width=0.0002)
    f = facade(robot, cfg(reset={"method": method}))
    starts = robot.starts
    r = call(f, "env.reset")
    assert r["ok"] is True, r
    assert r["info"]["lifted_m"] == pytest.approx(0.05)
    assert robot.gripper_commands[0] is False  # opened first
    assert robot.starts == starts + 1 and robot.controller == "cartesian"
    np.testing.assert_allclose(robot.q, CFG["reset"]["begin_joints"])
    np.testing.assert_allclose(f.controller.target_pos, robot.home_pose[:3])
    if method == "joint_stream":
        assert robot.joint_setpoints > 10


def test_reset_needs_a_begin_pose():
    f = facade(config=cfg(reset={"begin_joints": None}))
    with pytest.raises(ValueError, match="begin_joints is not set"):
        call(f, "env.reset")


# -- stop -----------------------------------------------------------------------


def test_stop_halts_a_move_between_control_ticks():
    robot = MockPolymetisRobot((0.5, 0.0, 0.3, *DOWN))
    f = facade(
        robot,
        cfg(limits={"tick_s": 0.01, "servo_step_m": 0.0005, "servo_step_rad": 0.0025}),
        sleep=time.sleep,
    )
    box: dict = {}
    thread = threading.Thread(
        target=lambda: box.update(r=call(f, "env.move_delta", [0.05, 0.0, 0.0]))
    )
    thread.start()
    deadline = time.time() + 5
    while len(robot.setpoints) < 20:
        assert time.time() < deadline
        time.sleep(0.005)
    reply = call(f, "stop")
    assert reply["call_in_progress"] is True
    thread.join(timeout=5)
    r = box["r"]
    assert r["cancelled"] is True and r["ok"] is False
    assert 0 < r["steps_used"] < 100
    # The arm holds the last commanded setpoint, and nothing is sent after the stop.
    sent = len(robot.setpoints)
    time.sleep(0.05)
    assert len(robot.setpoints) == sent
    np.testing.assert_allclose(r["target_tcp_pose"][:3], robot.setpoints[-1][:3])
    # Calls after the stop run normally.
    assert call(f, "env.move_delta", [0.001, 0.0, 0.0])["ok"]


def test_stop_halts_the_joint_stream_reset():
    robot = MockPolymetisRobot((0.5, 0.0, 0.3, *DOWN))
    f = facade(
        robot,
        cfg(limits={"tick_s": 0.025}, reset={"begin_time_s": 5.0}),
        sleep=time.sleep,
    )
    box: dict = {}
    thread = threading.Thread(target=lambda: box.update(r=call(f, "env.reset")))
    thread.start()
    deadline = time.time() + 5
    while robot.joint_setpoints < 5:
        assert time.time() < deadline
        time.sleep(0.005)
    call(f, "stop")
    thread.join(timeout=5)
    assert box["r"]["cancelled"] is True and box["r"]["ok"] is False
    assert robot.joint_setpoints < 400 and robot.controller == "cartesian"


# -- smooth motion (Show-Harness plugins/smooth) -------------------------------


def controller(robot, stop=lambda: False, clock=lambda: 0.0, **smooth):
    """A controller on the test CFG with a fake clock and no real sleeps."""
    lim = limits_from_config(cfg(smooth=smooth))
    c = control.PolymetisController(robot, lim, stop, sleep=lambda s: None, clock=clock)
    c.start_impedance()
    return c


def xs(robot, start, x0=0.5):
    """Setpoint x per command since ``start``, from the pre-move x."""
    return np.array([x0] + [p[0] for p in robot.setpoints[start:]])


def test_min_jerk_profile_shape():
    n = 20
    fr = np.array(control.smooth_fractions(n))
    t = np.arange(1, n + 1) / n
    # Show-Harness SmoothPlugin.plan(0, 0, 0): the classic min-jerk polynomial.
    np.testing.assert_allclose(fr, 10 * t**3 - 15 * t**4 + 6 * t**5, atol=1e-12)
    steps = np.diff(np.concatenate([[0.0], fr]))
    assert fr[-1] == 1.0 and np.all(steps > 0)
    # Rest to rest: near-zero first and last advances, 1.875x the mean at mid-move.
    assert steps[0] < 0.03 / n and steps[-1] < 0.03 / n
    assert np.max(steps) == pytest.approx(1.875 / n, rel=0.02)
    np.testing.assert_allclose(steps, steps[::-1], atol=1e-12)
    dense = np.diff(np.concatenate([[0.0], control.smooth_fractions(10000)]))
    assert dense[0] * 10000 < 1e-6 and dense[-1] * 10000 < 1e-6  # zero end velocities
    # Cruise boundaries: a chained middle move is a constant-rate line; a first
    # (last) chained move leaves (arrives) at the cruise rate.
    np.testing.assert_allclose(control.smooth_fractions(n, 1.0, 1.0), t, atol=1e-12)
    first = np.diff(np.concatenate([[0.0], control.smooth_fractions(n, 0.0, 1.0)]))
    last = np.diff(np.concatenate([[0.0], control.smooth_fractions(n, 1.0, 0.0)]))
    assert first[0] < 0.03 / n and first[-1] == pytest.approx(1 / n, rel=0.05)
    assert last[0] == pytest.approx(1 / n, rel=0.05) and last[-1] < 0.03 / n
    # Never past the target, never backward, even for an aggressive cruise.
    wild = np.array(control.smooth_fractions(n, 1.5, 1.5))
    assert np.all(np.diff(wild) >= 0) and wild.max() == 1.0


@pytest.mark.parametrize("dt_s", [0.05, 0.02])
def test_smooth_ramp_never_exceeds_the_servo_step_cap(dt_s):
    cap = 0.0025 * min(1.0, dt_s / 0.05)  # 2.5 mm per 50 ms tick, scaled to dt_s
    cap_rad = 0.0125 * min(1.0, dt_s / 0.05)
    for delta in ([0.08, 0.0, 0.0], [0.04, 0.04, 0.05], [0.002, 0.0, 0.0]):
        robot = MockPolymetisRobot((0.5, 0.0, 0.3, *DOWN))
        c = controller(robot, dt_s=dt_s)
        start = len(robot.setpoints)
        assert c.move_delta(delta)["ok"]
        path = [(0.5, 0.0, 0.3)] + [p[:3] for p in robot.setpoints[start:]]
        assert np.max(np.linalg.norm(np.diff(path, axis=0), axis=1)) <= cap + 1e-12
    # 8 cm rest-to-rest peaks at 1.875x its mean speed: stretched to >= 3 s.
    fractions, delay = c.plan(0.08, 0.0)
    assert delay == dt_s and len(fractions) * delay >= 1.875 * 0.08 / 0.05 - 1e-9
    # 2 cm: Show-Harness's 20 waypoints unless the cap needs more (>= 0.75 s).
    duration = len(c.plan(0.02, 0.0)[0]) * dt_s
    assert max(20 * dt_s, 0.75) <= duration + 1e-9 <= max(20 * dt_s, 0.75) + dt_s
    # Rotations too.
    robot = MockPolymetisRobot((0.5, 0.0, 0.3, *DOWN))
    c = controller(robot, dt_s=dt_s)
    start = len(robot.setpoints)
    assert c.rotate_delta([0.0, 0.0, 0.2])["ok"]
    quats = [DOWN] + [p[3:] for p in robot.setpoints[start:]]
    assert max(quat_angle(a, b) for a, b in zip(quats, quats[1:])) <= cap_rad + 1e-9
    # A chain of maximal moves, joins included.
    robot = MockPolymetisRobot((0.4, 0.0, 0.3, *DOWN))
    c = controller(robot, dt_s=dt_s, blend=True)
    start = len(robot.setpoints)
    for i in range(3):
        assert c.move_delta([0.08, 0.0, 0.0], continuous=i < 2)["ok"]
    assert np.max(np.abs(np.diff(xs(robot, start, 0.4)))) <= cap + 1e-12


def test_chained_moves_flow_without_stopping():
    robot = MockPolymetisRobot((0.5, 0.0, 0.3, *DOWN))
    c = controller(robot, blend=True)
    start = len(robot.setpoints)
    r1 = c.move_delta([0.02, 0.0, 0.0], continuous=True)
    assert r1["ok"] and r1["flowing"] and not r1["chained"]
    assert len(robot.setpoints) - start == r1["steps_used"]  # no settle re-commands
    r2 = c.move_delta([0.02, 0.0, 0.0], continuous=True)
    assert r2["ok"] and r2["chained"] and r2["flowing"]
    r3 = c.move_delta([0.02, 0.0, 0.0])
    assert r3["ok"] and r3["chained"] and not r3["flowing"]
    v = np.diff(xs(robot, start))
    k1, k2 = r1["steps_used"], r1["steps_used"] + r2["steps_used"]
    cruise = 0.02 / 20  # one move length per move duration
    # Accelerate once, cruise through both joins, decelerate once, then settle.
    assert v[0] < 0.03 * cruise
    for j in (k1, k2):
        assert v[j - 1] == pytest.approx(cruise, rel=0.05)
        assert v[j] == pytest.approx(cruise, rel=0.05)
    np.testing.assert_allclose(v[k1:k2], cruise, rtol=1e-9)
    moving = v[: k2 + r3["steps_used"]]
    assert np.all(moving[1:-1] > 0.03 * cruise)  # never at rest in between
    assert np.all(v[k2 + r3["steps_used"] :] == 0) and len(v) - len(moving) == 4
    np.testing.assert_allclose(robot.setpoints[-1][:3], [0.56, 0.0, 0.3], atol=1e-12)
    # The same three moves rest to rest stop at every join.
    robot = MockPolymetisRobot((0.5, 0.0, 0.3, *DOWN))
    c = controller(robot)
    start = len(robot.setpoints)
    for _ in range(3):
        assert "chained" in c.move_delta([0.02, 0.0, 0.0])
    v = np.diff(xs(robot, start))
    assert np.sum(v == 0) == 3 * 4 and v[20] < 0.03 * cruise


def test_chain_breaks_on_turns_late_moves_rotations_and_the_gripper():
    now = [0.0]
    robot = MockPolymetisRobot((0.5, 0.0, 0.3, *DOWN))
    c = controller(robot, clock=lambda: now[0], blend=True)

    def settled_before(result, start):
        """The stream was brought to rest (4 settle re-commands) before the ramp."""
        first = robot.setpoints[start : start + 4]
        return all(np.allclose(p, first[0]) for p in first) and not result["chained"]

    c.move_delta([0.02, 0.0, 0.0], continuous=True)
    start = len(robot.setpoints)
    r = c.move_delta([0.0, 0.02, 0.0], continuous=True)  # a turn: from rest
    assert settled_before(r, start) and r["flowing"]
    now[0] += 5.0  # past chain_window_s: the arm has long stopped
    start = len(robot.setpoints)
    r = c.move_delta([0.0, 0.02, 0.0])
    assert not r["chained"] and robot.setpoints[start][1] < 0.0201  # no settle, no jump
    assert not np.allclose(robot.setpoints[start], robot.setpoints[start + 1])
    for action in (
        lambda: c.rotate_delta([0.0, 0.0, 0.1]),
        lambda: c.set_gripper(open=False),
    ):
        c.move_delta([0.02, 0.0, 0.0], continuous=True)
        start = len(robot.setpoints)
        gripper = len(robot.gripper_commands)
        action()
        first = robot.setpoints[start : start + 4]
        assert len(first) == 4 and all(np.allclose(p, first[0]) for p in first)
        assert len(robot.gripper_commands) <= gripper + 2
    # blend off: continuous is ignored (Show-Harness's Franka default).
    robot = MockPolymetisRobot((0.5, 0.0, 0.3, *DOWN))
    c = controller(robot, blend=False)
    assert c.move_delta([0.02, 0.0, 0.0], continuous=True)["flowing"] is False
    # smooth off: the linear ramp, no chaining keys.
    robot = MockPolymetisRobot((0.5, 0.0, 0.3, *DOWN))
    r = controller(robot, enabled=False).move_delta([0.02, 0.0, 0.0], continuous=True)
    assert r["ok"] and "flowing" not in r and r["steps_used"] == 8


def test_stop_halts_a_smooth_and_a_chained_ramp_between_waypoints():
    stop_at = [10**9]
    robot = MockPolymetisRobot((0.5, 0.0, 0.3, *DOWN))
    c = controller(robot, stop=lambda: len(robot.setpoints) >= stop_at[0], blend=True)
    start = len(robot.setpoints)
    stop_at[0] = start + 7
    r = c.move_delta([0.04, 0.0, 0.0])
    assert r["cancelled"] and not r["ok"] and r["steps_used"] == 7
    assert len(robot.setpoints) == start + 7  # nothing is sent after the stop
    np.testing.assert_allclose(c.target_pos, robot.setpoints[-1][:3])
    assert 0.5 < c.target_pos[0] < 0.51  # held mid-ramp
    # Mid-chain: the stream is dropped; the next move starts from rest.
    stop_at[0] = 10**9
    assert c.move_delta([0.02, 0.0, 0.0], continuous=True)["flowing"]
    stop_at[0] = len(robot.setpoints) + 5
    r = c.move_delta([0.02, 0.0, 0.0], continuous=True)
    assert r["cancelled"] and r["chained"] and not r["flowing"]
    stop_at[0] = 10**9
    r = c.move_delta([0.02, 0.0, 0.0])
    assert r["ok"] and not r["chained"]


def test_move_delta_continuous_over_rpc_and_smooth_meta():
    # Default (and example.yaml): min-jerk on, chaining off (Show-Harness's Franka).
    assert yaml.safe_load(DEFAULT_CONFIG.read_text())["smooth"]["blend"] is False
    f = facade()
    meta = call(f, "env.get_env_meta")["smooth"]
    assert meta["enabled"] and not meta["blend"] and not meta["chaining"]
    r = call(f, "env.move_delta", [0.02, 0.0, 0.0], continuous=True)
    assert r["ok"] and r["flowing"] is False
    f = facade(config=cfg(smooth={"blend": True}))
    meta = call(f, "env.get_env_meta")["smooth"]
    assert meta["enabled"] and meta["chaining"] and meta["duration_s"] == 1.0
    assert call(f, "env.move_delta", [0.02, 0.0, 0.0], continuous=True)["flowing"]
    assert call(f, "env.move_delta", [0.02, 0.0, 0.0])["chained"]


def test_a_recurring_reflex_fails_the_call_after_one_restart():
    robot = WallRobot((0.5, 0.0, 0.3, *DOWN), wall_x=0.51, reflex=True)
    f = facade(robot)
    with pytest.raises(RuntimeError, match="lost again in the same call"):
        call(f, "env.move_delta", [0.08, 0.0, 0.0])
    assert f.controller.restarts == 2  # one resume, then a restart to hold only
    planned = len(f.controller.plan(0.08, 0.0)[0])
    assert robot.controller == "cartesian" and len(robot.setpoints) < planned / 2
    np.testing.assert_allclose(f.controller.target_pos, robot.pose[:3])
    # A single loss resumes and reports the restart.
    robot = MockPolymetisRobot((0.5, 0.0, 0.3, *DOWN))
    f = facade(robot)
    robot.lose_controller()
    r = call(f, "env.move_delta", [0.02, 0.0, 0.0])
    assert r["ok"] and r["controller_restarts"] == 1


def test_contact_stops_the_ramp_and_reanchors_on_all_axes():
    robot = WallRobot((0.5, 0.0, 0.3, *DOWN), wall_x=0.52)
    f = facade(robot)
    r = call(f, "env.move_delta", [0.06, 0.02, 0.0])
    assert r["ok"] is False and r["blocked"] is True
    # Stopped well before the ramp ended.
    assert len(robot.setpoints) < 0.7 * len(
        f.controller.plan(math.hypot(0.06, 0.02), 0.0)[0]
    )
    np.testing.assert_allclose(f.controller.target_pos, robot.pose[:3], atol=1e-12)
    np.testing.assert_allclose(robot.setpoints[-1], robot.pose)
    for _ in range(3):  # repeated pushes never leave the setpoint inside the wall
        call(f, "env.move_delta", [0.02, 0.0, 0.0])
        assert abs(f.controller.target_pos[0] - robot.pose[0]) < 1e-12
    # Pushed 2.5 cm aside while idle: a pure-z move does not pull the arm back.
    robot.pose[1] += 0.025
    call(f, "env.move_delta", [0.0, 0.0, 0.01])
    assert robot.setpoints[-1][1] == pytest.approx(robot.pose[1])
    # Displaced by more than move_tolerance_m at the end of a call: re-anchored.
    command = robot.update_desired_ee_pose

    def sag(pose):
        command(pose)
        robot.pose[1] -= 0.012

    robot.update_desired_ee_pose = sag
    r = call(f, "env.move_delta", [-0.01, 0.0, 0.0])
    assert r["reanchored"] is True and r["ok"] is False
    np.testing.assert_allclose(f.controller.target_pos, robot.setpoints[-1][:3])


@pytest.mark.parametrize(
    "section, values, match",
    [
        ("limits", {"z_floor_m": float("nan")}, "finite"),
        ("limits", {"workspace_max": [0.75, float("inf"), 0.6]}, "finite"),
        ("limits", {"tick_s": 0.0}, "tick_s"),
        ("limits", {"tick_s": 0.01}, "m/s"),
        ("limits", {"max_tracking_error_m": 0.2}, "max_tracking_error_m"),
        ("limits", {"max_tilt_rad": 2.0}, "max_tilt_rad"),
        ("robot", {"tcp_offset_m": [float("nan"), 0.0, 0.0]}, "finite"),
        ("impedance", {"kx": [5000, 750, 750, 15, 15, 15]}, "gains"),
        ("impedance", {"kxd": [37, 37, 37, 2, 2, 9]}, "gains"),
        ("reset", {"begin_joints": [float("nan")] * 7}, "finite"),
        ("reset", {"begin_joints": [0, -0.5, 0, -2.6, 0, 0.0, 0.86]}, "joint limits"),
        ("reset", {"begin_time_s": 0.1}, "begin_time_s"),
        ("smooth", {"dt_s": 0.0}, "smooth_dt_s"),
        ("smooth", {"substeps": 2.5}, "integer"),
        ("smooth", {"enabled": "yes"}, "true or false"),
        ("smooth", {"cruise": 3.0}, "smooth_cruise"),
        ("smooth", {"window_s": 1.0}, "unknown smooth keys"),
    ],
)
def test_config_rejects_non_finite_and_out_of_range_values(section, values, match):
    with pytest.raises(ValueError, match=match):
        limits_from_config(cfg(**{section: values}))


def test_reset_is_speed_bounded_and_never_leaves_joint_impedance_running():
    robot = MockPolymetisRobot((0.5, 0.0, 0.3, *DOWN))
    f = facade(robot, cfg(reset={"begin_time_s": 0.5}))
    sent = []
    move = robot.move_to_joint_positions
    robot.move_to_joint_positions = lambda q, t: (sent.append(t), move(q, t))
    q0 = robot.q.copy()
    r = call(f, "env.reset")
    distance = np.max(np.abs(np.asarray(CFG["reset"]["begin_joints"]) - q0))
    assert r["info"]["duration_s"] == pytest.approx(1.875 * distance / 0.5)
    assert robot.joint_setpoints >= r["info"]["duration_s"] / 0.05
    # A NaN joint reading is refused and Cartesian impedance holds the arm.
    robot.q[:] = np.nan
    with pytest.raises(RuntimeError, match="invalid joint positions"):
        call(f, "env.reset")
    assert robot.controller == "cartesian"
    # An error mid-stream goes back to Cartesian impedance.
    robot.q = q0.copy()
    stream = robot.update_desired_joint_pos

    def fail(q):
        if robot.joint_setpoints >= 3:
            raise RuntimeError("NUC connection lost")
        stream(q)

    robot.joint_setpoints = 0
    robot.update_desired_joint_pos = fail
    with pytest.raises(RuntimeError, match="NUC connection lost"):
        call(f, "env.reset")
    assert robot.controller == "cartesian"
    np.testing.assert_allclose(f.controller.target_pos, robot.pose[:3])
    # move_to_joint_positions gets the speed-bounded duration too.
    f = facade(
        MockPolymetisRobot((0.5, 0.0, 0.3, *DOWN)),
        cfg(reset={"begin_time_s": 0.5, "method": "move_to_joint_positions"}),
    )
    f.controller.robot.move_to_joint_positions = lambda q, t: sent.append(t)
    call(f, "env.reset")
    assert sent[-1] >= 1.875 * distance / 0.5 - 1e-9


def test_workspace_is_checked_per_axis_and_on_the_begin_pose():
    robot = MockPolymetisRobot((0.74, 0.0, 0.09, *DOWN))  # 5 cm below the floor
    f = facade(robot)
    with pytest.raises(ValueError, match="outside the workspace"):
        call(f, "env.move_delta", [0.04, 0.0, 0.06])  # x would end 3 cm out
    assert call(f, "env.move_delta", [0.0, 0.0, 0.06])["ok"]
    robot = MockPolymetisRobot((0.5, 0.0, 0.3, *DOWN), home_pose=(0.9, 0.0, 0.4, *DOWN))
    r = call(facade(robot), "env.reset")
    assert r["ok"] is False and r["info"]["begin_pose_outside_workspace"] is True


def test_rotation_tilt_limit_wrap_and_flange_box_check():
    robot = MockPolymetisRobot((0.5, 0.0, 0.3, *DOWN))
    f = facade(robot)
    assert call(f, "env.rotate_delta", [0.0, 0.2, 0.0])["ok"]
    assert call(f, "env.rotate_delta", [0.0, 0.2, 0.0])["ok"]
    with pytest.raises(ValueError, match="max_tilt_rad"):
        call(f, "env.rotate_delta", [0.0, 0.2, 0.0])
    assert tool_tilt(robot.pose[3:]) == pytest.approx(0.4)
    assert call(f, "env.rotate_delta", [0.0, -0.2, 0.0])["ok"]  # back toward down
    with pytest.raises(ValueError, match="without wrap-around"):
        call(f, "env.rotate_delta", [0.0, 0.0, 2 * math.pi + 0.1])
    for _ in range(20):
        call(f, "env.rotate_delta", [0.0, 0.0, 0.2])
    assert np.linalg.norm(f.controller.target_quat) == pytest.approx(1.0, abs=1e-12)
    # With a 10 cm TCP offset, pitching swings the flange past xmax 0.75.
    edge = MockPolymetisRobot((0.74, 0.0, 0.4, *DOWN))
    f = facade(edge, cfg(robot={"tcp_offset_m": [0.0, 0.0, 0.1]}))
    with pytest.raises(ValueError, match="rotation's flange"):
        call(f, "env.rotate_delta", [0.0, 0.2, 0.0])
    assert call(f, "env.rotate_delta", [0.0, -0.2, 0.0])["ok"]


def _pose_matrix(pose7) -> np.ndarray:
    t = np.eye(4)
    t[:3, :3] = np.stack([quat_rotate(pose7[3:], e) for e in np.eye(3)], 1)
    t[:3, 3] = pose7[:3]
    return t


def test_rlinf_tcp_frame_needs_the_hand_yaw_for_wrist_calibration():
    """A wrist extrinsic calibrated on RLinf (``T_tcp_cam``, TCP = libfranka O_T_EE =
    flange * Trans(0, 0, 0.1034) * Rz(-45 deg)) lands the camera at the true pose only
    when the Polymetis TCP carries the same yaw; the offset alone is cm off."""
    yaw0 = quat_from_euler_xyz([0.0, 0.0, 0.3])
    flange = np.array([0.5, 0.05, 0.35, *control.quat_mul(yaw0, DOWN)])
    f_t_ee = np.eye(4)
    f_t_ee[:3, :3] = _pose_matrix(
        [0, 0, 0, *quat_from_euler_xyz([0, 0, -math.pi / 4])]
    )[:3, :3]
    f_t_ee[2, 3] = 0.1034
    rlinf_tcp = _pose_matrix(flange) @ f_t_ee  # what RLinf reports as tcp_pose
    t_tcp_cam = np.eye(4)  # D405 on the wrist: 9 cm off the tool axis, 4 cm up
    t_tcp_cam[:3, 3] = [0.09, 0.0, -0.04]
    truth = rlinf_tcp @ t_tcp_cam

    hand = {"tcp_offset_m": [0.0, 0.0, 0.1034], "tcp_yaw_deg": -45.0}
    robot = MockPolymetisRobot(flange)
    f = facade(robot, cfg(robot=hand))
    tcp = np.asarray(call(f, "env.get_robot_state")["raw_base_state"]["tcp_pose"])
    np.testing.assert_allclose(_pose_matrix(tcp), rlinf_tcp, atol=1e-9)
    np.testing.assert_allclose(_pose_matrix(tcp) @ t_tcp_cam, truth, atol=1e-9)
    pos, quat = control.flange_to_tcp(flange, hand["tcp_offset_m"], -45.0)
    np.testing.assert_allclose(_pose_matrix([*pos, *quat]), rlinf_tcp, atol=1e-9)

    # Actions round-trip: a translation keeps the flange orientation (no 45 deg
    # twist from the TCP yaw) and a base-z rotation turns the flange by the same.
    assert call(f, "env.move_delta", [0.0, 0.0, 0.02])["ok"]
    assert quat_angle(robot.pose[3:], flange[3:]) < 1e-9
    np.testing.assert_allclose(robot.pose[:3], flange[:3] + [0, 0, 0.02], atol=1e-9)
    assert call(f, "env.rotate_delta", [0.0, 0.0, 0.1])["ok"]
    expected = control.quat_mul(quat_from_euler_xyz([0, 0, 0.1]), flange[3:])
    assert quat_angle(robot.pose[3:], expected) < 1e-9

    # Translation-only (the old config): the TCP origin agrees, the camera does not.
    old = facade(
        MockPolymetisRobot(flange), cfg(robot={"tcp_offset_m": [0, 0, 0.1034]})
    )
    tcp = call(old, "env.get_robot_state")["raw_base_state"]["tcp_pose"]
    np.testing.assert_allclose(tcp[:3], rlinf_tcp[:3, 3], atol=1e-9)
    cam_err = np.linalg.norm((_pose_matrix(tcp) @ t_tcp_cam)[:3, 3] - truth[:3, 3])
    assert cam_err > 0.06  # 2 * 0.09 * sin(22.5 deg) = 6.9 cm
    with pytest.raises(ValueError, match="tcp_yaw_deg"):
        limits_from_config(cfg(robot={"tcp_yaw_deg": 270.0}))


def test_shutdown_and_parent_death_stop_a_running_motion(monkeypatch):
    for trigger in ("shutdown", "parent_death"):
        robot = MockPolymetisRobot((0.5, 0.0, 0.3, *DOWN))
        f = facade(
            robot,
            cfg(
                limits={
                    "tick_s": 0.01,
                    "servo_step_m": 0.0005,
                    "servo_step_rad": 0.0025,
                }
            ),
            sleep=time.sleep,
        )
        hooks: dict = {}
        monkeypatch.setattr(
            env_server, "watch_parent_death", lambda cb: hooks.update(cb=cb)
        )
        monkeypatch.setattr(MainThreadServeMixin, "serve", lambda self, **kw: None)
        f.serve(transport="http", host="127.0.0.1", port=0, parent_watch=True)
        box: dict = {}
        thread = threading.Thread(
            target=lambda: box.update(r=call(f, "env.move_delta", [0.05, 0.0, 0.0]))
        )
        thread.start()
        deadline = time.time() + 5
        while len(robot.setpoints) < 10:
            assert time.time() < deadline
            time.sleep(0.005)
        if trigger == "shutdown":
            call(f, "shutdown")
        else:
            hooks["cb"]()
        thread.join(timeout=5)
        assert box["r"]["cancelled"] is True, trigger
        assert f._shutdown_event.is_set()


class FakePolymetis:
    """Polymetis RobotInterface + GripperInterface for nuc_server guard tests."""

    def __init__(self):
        self.pos = np.array([0.5, 0.0, 0.3])
        self.quat = np.array([1.0, 0.0, 0.0, 0.0])
        self.q = np.array([0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785])
        self.metadata = type("M", (), {"max_width": 0.08})()
        self.calls: list = []

    def get_ee_pose(self):
        return self.pos.copy(), self.quat.copy()

    def get_joint_positions(self):
        return self.q.copy()

    def __getattr__(self, name):
        return lambda *a, **kw: self.calls.append((name, a, kw))


def test_nuc_server_validates_every_command():
    assert nuc_server.MAX_KX == control.MAX_KX and nuc_server.MAX_KXD == control.MAX_KXD
    assert nuc_server.JOINT_MIN == control.JOINT_MIN
    assert nuc_server.JOINT_MAX == control.JOINT_MAX
    fake = FakePolymetis()
    limits = nuc_server.NucLimits(
        workspace_min=(0.3, -0.35, 0.1), workspace_max=(0.75, 0.35, 0.6), z_floor_m=0.14
    )
    s = nuc_server.FrankaServer(fake, fake, 0.1, 20.0, limits, np.asarray)
    s.update_desired_ee_pose([0.505, 0.0, 0.3, 1.0, 0.0, 0.0, 0.0])
    assert fake.calls[-1][0] == "update_desired_ee_pose"
    bad = [
        ([float("nan"), 0.0, 0.3, 1.0, 0.0, 0.0, 0.0], "finite"),
        ([0.55, 0.0, 0.3, 1.0, 0.0, 0.0, 0.0], "jumps"),
        ([0.5, 0.0, 0.3, 0.0, 1.0, 0.0, 0.0], "jumps"),
        ([0.5, 0.0, 0.3, 2.0, 0.0, 0.0, 0.0], "unit length"),
    ]
    for pose, match in bad:
        with pytest.raises(ValueError, match=match):
            s.update_desired_ee_pose(pose)
    fake.pos = np.array([0.5, 0.0, 0.145])
    with pytest.raises(ValueError, match="outside the NUC workspace"):
        s.update_desired_ee_pose([0.5, 0.0, 0.138, 1.0, 0.0, 0.0, 0.0])
    s.update_desired_ee_pose([0.5, 0.0, 0.145, 1.0, 0.0, 0.0, 0.0])  # hold: accepted
    with pytest.raises(ValueError, match="gains"):
        s.start_cartesian_impedance([5000.0] * 3 + [15.0] * 3, [37.0] * 3 + [2.0] * 3)
    s.start_cartesian_impedance(list(control.DEFAULT_KX), list(control.DEFAULT_KXD))
    goal = [0.0, -0.5, 0.0, -2.6, 0.0, 2.07, 0.86]
    with pytest.raises(ValueError, match="time_to_go"):
        s.move_to_joint_positions(goal, 0.05)
    with pytest.raises(ValueError, match="joint limits"):
        s.move_to_joint_positions([0.0, -0.5, 0.0, -2.6, 0.0, 0.0, 0.86], 10.0)
    s.move_to_joint_positions(goal, 4.0)
    with pytest.raises(ValueError, match="jumps"):
        s.update_desired_joint_pos(fake.q + 0.2)
    s.update_desired_joint_pos(fake.q + 0.01)
    with pytest.raises(ValueError, match="width"):
        s.set_gripper_position(float("nan"))


def test_gripper_state_travels_from_polymetis_to_the_client():
    fake = FakePolymetis()
    fake.get_state = lambda: type(
        "GripperState", (), {"width": 0.041, "is_grasped": True, "is_moving": False}
    )()
    s = nuc_server.FrankaServer(
        fake, fake, 0.1, 20.0, nuc_server.NucLimits(), np.asarray
    )
    assert "get_gripper_state" in nuc_server.READ_ONLY
    served = s.get_gripper_state()
    assert served == {
        "width": pytest.approx(0.041),
        "is_grasped": True,
        "is_moving": False,
        "prev_command_successful": None,
    }

    from pi_embodied_services.robots.franka_polymetis.hardware import PolymetisRobot

    class NoSuchMethod(Exception):
        name = "NameError"  # what zerorpc raises for a method the server lacks

    class OldServer:
        def get_gripper_state(self):
            raise NoSuchMethod("get_gripper_state")

        def get_gripper_position(self):
            return 0.03

    client = PolymetisRobot.__new__(PolymetisRobot)
    client._gripper_state_rpc = True
    client.server = type("New", (), {"get_gripper_state": lambda self: served})()
    assert client.get_gripper_state()["is_grasped"] is True
    client.server = OldServer()
    old = {"width": pytest.approx(0.03), "is_grasped": None, "is_moving": None}
    assert client.get_gripper_state() == old
    assert client._gripper_state_rpc is False


# -- cameras and perception -------------------------------------------------------


def test_letterboxed_observation_matches_its_intrinsics():
    f = facade()
    obs = call(f, "env.get_observation")
    assert obs["main_images"].shape == (256, 256, 3)
    assert obs["extra_view_images"].shape == (1, 256, 256, 3)
    assert obs["main_depths"].shape == (256, 256)
    assert obs["extra_view_depths"].shape == (1, 256, 256)
    assert obs["main_depths"][0, 0] == 0 and obs["main_depths"][
        128, 128
    ] == pytest.approx(0.5)
    assert obs["extra_view_depths"][0, 128, 128] == pytest.approx(0.9)
    meta = call(f, "env.get_camera_meta")
    assert meta["observation_camera_map"] == {
        "main": "wrist",
        "extra_0": "third_person",
    }
    k = np.asarray(meta["cameras"]["wrist"]["intrinsic_K"])
    raw = meta["cameras"]["wrist"]["raw_color_intrinsics"]
    geo = letterbox_geometry(640, 480, 256)
    assert geo == {"size": 256, "resized": [256, 192], "offset_xy": [0, 32]}
    # A raw pixel and its letterboxed position see the same ray.
    for u, v in [(0.0, 0.0), (320.0, 240.0), (639.0, 100.0)]:
        u2 = (u + 0.5) * 0.4 - 0.5 + 0
        v2 = (v + 0.5) * 0.4 - 0.5 + 32
        ray = [(u - raw["ppx"]) / raw["fx"], (v - raw["ppy"]) / raw["fy"]]
        ray2 = [(u2 - k[0, 2]) / k[0, 0], (v2 - k[1, 2]) / k[1, 1]]
        np.testing.assert_allclose(ray, ray2, atol=1e-12)
    assert letterbox_intrinsics(raw, None)[0][2] == raw["ppx"]


def test_calibration_writes_an_easy_handeye_yaml(tmp_path):
    angle = 0.7
    axis = np.array([0.2, -0.5, 0.8]) / np.linalg.norm([0.2, -0.5, 0.8])
    k = np.array(
        [[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]]
    )
    r = np.eye(3) + math.sin(angle) * k + (1 - math.cos(angle)) * k @ k
    t = np.eye(4)
    t[:3, :3], t[:3, 3] = r, [0.1, -0.2, 0.5]
    doc = easy_handeye_yaml(t, eye_on_hand=True)
    tr = doc["transformation"]
    q = [tr["qx"], tr["qy"], tr["qz"], tr["qw"]]
    np.testing.assert_allclose(
        np.stack([quat_rotate(q, e) for e in np.eye(3)], 1), r, atol=1e-9
    )
    assert doc["parameters"]["eye_on_hand"] is True and tr["z"] == 0.5
    rot180 = np.diag([1.0, -1.0, -1.0])  # trace < 0 branch
    np.testing.assert_allclose(
        np.abs(matrix_to_quat_xyzw(rot180)), [1, 0, 0, 0], atol=1e-12
    )
    path = tmp_path / "wrist.yaml"
    path.write_text(yaml.safe_dump(doc))
    pytest.importorskip("omegaconf")
    from pi_embodied_services.robots.franka.runtime_config import load_easy_handeye_yaml

    assert load_easy_handeye_yaml(path)["transformation"]["x"] == 0.1


# -- config and mock gating -------------------------------------------------------


def test_example_config_refuses_until_calibrated():
    with pytest.raises(ValueError, match="z_floor_m is not set"):
        load_config(DEFAULT_CONFIG)
    example = yaml.safe_load(DEFAULT_CONFIG.read_text())
    assert set(example["perception"]["calibration"]) == {"external", "wrist"}


def test_unknown_limit_keys_are_refused(tmp_path):
    bad = cfg(limits={"max_step": 0.1})
    path = tmp_path / "bad.yaml"
    path.write_text(yaml.safe_dump(bad))
    with pytest.raises(ValueError, match="unknown limits keys"):
        load_config(path)


def test_mocks_refuse_outside_tests(monkeypatch, tmp_path):
    monkeypatch.delenv(MOCK_ENV)
    with pytest.raises(RuntimeError, match="tests only"):
        MockPolymetisRobot()
    with pytest.raises(RuntimeError, match="tests only"):
        MockRGBD()
    path = tmp_path / "c.yaml"
    path.write_text(yaml.safe_dump(cfg()))
    with pytest.raises(SystemExit):
        main(["--mock", "--robot-config", str(path)])


def test_mock_server_over_http_with_parent_watch(tmp_path):
    path = tmp_path / "c.yaml"
    path.write_text(yaml.safe_dump(cfg()))
    proc = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "pi_embodied_services.robots.franka_polymetis.env_server",
            "--mock",
            "--robot-config",
            str(path),
            "--port",
            "0",
            "--parent-watch",
        ],
        cwd=SERVICES,
        env={**os.environ, MOCK_ENV: "1", "PYTHONPATH": str(SERVICES)},
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    try:
        url = None
        deadline = time.time() + 30
        while url is None and time.time() < deadline:
            line = proc.stdout.readline()
            if not line:
                break
            if line.startswith("RPC server listening on "):
                url = line.split()[-1]
        assert url, "server did not start"
        client = HttpRpcClient(url)
        assert client.call("healthz", timeout_s=5)["service"] == "franka-polymetis-env"
        meta = client.call("env.get_env_meta", timeout_s=5)
        assert meta["capabilities"]["has_vla"] is False
        assert client.call("env.reset", timeout_s=10)["ok"] is True
        r = client.call("env.move_delta", ([0.0, 0.0, 0.02],), timeout_s=10)
        assert r["ok"] is True
        refused = None
        try:
            client.call("env.move_delta", ([0.2, 0.0, 0.0],), timeout_s=10)
        except Exception as exc:  # the RPC error envelope
            refused = str(exc)
        assert refused and "limit is 0.08 m per call" in refused
        obs = client.call("env.get_observation", timeout_s=10)
        assert obs["main_images"].shape == (256, 256, 3)
        json.dumps(client.call("env.get_camera_meta", timeout_s=5))
        proc.stdin.close()  # parent death: the server exits
        assert proc.wait(timeout=15) == 0
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.wait()


def test_code_api_lists_the_primitives_and_resolves_to_the_facade_methods():
    f = facade()
    high = call(f, "code.api", tier="high")
    assert [p["name"] for p in high["primitives"]] == [
        "get_robot_state",
        "preview_reach",
        "move_delta",
        "rotate_delta",
        "set_gripper",
    ]
    assert len(high["digest"]) == 64
    low = call(f, "code.api", tier="low")
    assert "get_observation" in [p["name"] for p in low["primitives"]]
    assert call(f, "code.api")["primitives"] == low["primitives"]
    method, kwargs = f.code_api.resolve("move_delta", {"delta_xyz": [0.0, 0.0, 0.01]})
    assert method == "env.move_delta" and method in f._rpc
    # A resolved call is the tool's call: the facade's own limits refuse an oversized move.
    with pytest.raises(ValueError, match="per\\s+call"):
        call(f, method, delta_xyz=[0.0, 0.0, 1.0])


def test_the_grasp_arguments_parse(tmp_path, capsys):
    """Finding: pi passes --graspnet/--anyplace/... to every franka backend; this argparse
    refused them, so the Polymetis server could not start with a grasp service."""
    path = tmp_path / "c.yaml"
    path.write_text(yaml.safe_dump(cfg()))
    argv = ["--print-config", "--robot-config", str(path)]
    grasp = ["--graspnet", "http://127.0.0.1:1", "--anyplace", "http://127.0.0.1:2"]
    grasp += ["--anygrasp", "http://127.0.0.1:3", "--graspgenx", "http://127.0.0.1:4"]
    grasp += ["--grasp-to-eef", '{"translation": [0, 0, 0.01]}']
    assert main(argv + grasp) == 0
    assert "nuc_ip" in capsys.readouterr().out


def test_the_planner_segments_object_text_with_the_perception_sam3(monkeypatch):
    """plan_grasp(object=<text>) works on Polymetis (and the RLinf server, which shares
    franka_grasp_planner): the planner uses --sam3's client. Ids survive an observation and a
    refused move, and expire once the arm moved."""
    from test_grasp import FakeSam3, FakeServer

    from pi_embodied_services.robots.franka import perception as franka_perception
    from pi_embodied_services.utils import grasp as G
    from pi_embodied_services.utils.perception import (
        FRANKA_CAMERAS,
        Perception,
        franka_intrinsics,
    )

    monkeypatch.setattr(
        franka_perception,
        "load_calibration_bundle",
        lambda: {"external": {"matrix": np.eye(4)}, "wrist": {"matrix": np.eye(4)}},
    )
    mask = np.zeros((256, 256), bool)
    mask[100:160, 100:160] = True
    sam3 = FakeSam3(mask)
    server = FakeServer(
        [
            G.make_candidate(
                score=0.9,
                rotation=G.ZX_NATIVE_TO_GRASPNET,
                center=[0.0, 0.0, 0.5],
                width=0.04,
                depth=0.0,
                source_model="fake",
            )
        ]
    )
    robot = MockPolymetisRobot((0.5, 0.0, 0.3, *DOWN))
    cams = {"wrist": MockRGBD("1"), "third_person": MockRGBD("2", depth_m=0.9)}
    holder: dict = {}
    perception = Perception(
        sam3=sam3,
        cameras=FRANKA_CAMERAS,
        intrinsics=lambda key: franka_intrinsics(holder["f"].get_camera_meta(), key),
    )
    f = FrankaPolymetisFacade(
        cfg(),
        robot,
        cams,
        sleep=lambda s: None,
        perception=perception,
        grasp={"graspnet": server},
    )
    holder["f"] = f
    assert {"env.plan_grasp", "env.plan_place", "env.claim_waypoints"} <= set(f._rpc)
    out = call(f, "env.plan_grasp", object="block", camera="third_person")
    assert out["candidate_count"] == 1 and out["mask_id"].startswith("d")
    assert server.calls[0][1]["mask"].sum() == 60 * 60
    gid = out["active"]
    call(f, "env.get_observation")
    with pytest.raises(ValueError):
        call(f, "env.move_delta", [0.2, 0.0, 0.0])  # refused, unmoved
    assert call(f, "env.resolve_grasp", gid)["id"] == gid
    call(f, "env.move_delta", [0.0, 0.0, 0.02])
    with pytest.raises(G.GraspError, match="stale"):
        call(f, "env.resolve_grasp", gid)
