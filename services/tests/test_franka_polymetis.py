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

from pi_embodied_services.robots.franka_polymetis.calibration import (
    easy_handeye_yaml,
    matrix_to_quat_xyzw,
)
from pi_embodied_services.robots.franka_polymetis.control import (
    quat_angle,
    quat_from_euler_xyz,
    quat_rotate,
)
from pi_embodied_services.robots.franka_polymetis.env_server import (
    DEFAULT_CONFIG,
    METHODS,
    FrankaPolymetisFacade,
    letterbox_geometry,
    letterbox_intrinsics,
    load_config,
    main,
)
from pi_embodied_services.robots.franka_polymetis.mock import (
    MOCK_ENV,
    MockPolymetisRobot,
    MockRGBD,
)
from pi_embodied_services.utils.rpc.http_rpc import HttpRpcClient

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
        "tick_s": 0.0,
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


# -- RPC parity with the RLinf backend ----------------------------------------


def test_method_list_matches_the_rlinf_server():
    pytest.importorskip("omegaconf")
    from pi_embodied_services.robots.franka.env_server import FrankaEnvFacade

    assert METHODS == FrankaEnvFacade._METHODS
    f = facade()
    assert {m for m in f._rpc} == {f"env.{m}" for m in FrankaEnvFacade._METHODS}


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


def test_move_ramps_the_setpoint_in_servo_ticks_and_reaches_the_target():
    robot = MockPolymetisRobot((0.5, 0.0, 0.3, *DOWN))
    f = facade(robot)
    before = len(robot.setpoints)
    r = call(f, "env.move_delta", [0.02, -0.01, 0.0])
    assert r["ok"] and r["final_error_m"] < 1e-9
    assert r["steps_used"] == math.ceil(math.hypot(0.02, 0.01) / 0.0025)
    steps = np.diff([p[:3] for p in robot.setpoints[before:]], axis=0)
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
        cfg(limits={"tick_s": 0.01, "servo_step_m": 0.0005}),
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
        cfg(limits={"tick_s": 0.01}, reset={"begin_time_s": 5.0}),
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
