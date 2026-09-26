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

"""UR5e env server and hand-eye calibration against test doubles (no arm, no camera)."""

from __future__ import annotations

import copy
import json
import math
import threading
import time
from pathlib import Path

import numpy as np
import pytest
import yaml
from scipy.spatial.transform import Rotation

from pi_embodied_services.components.cameras import undistort_pixels
from pi_embodied_services.components.cameras.mock import MOCK_ENV, MockCamera
from pi_embodied_services.robots.ur5e import calibrate
from pi_embodied_services.robots.ur5e.calibrate import (
    EYE_IN_HAND,
    EYE_TO_HAND,
    Sample,
    capture_sample,
    load_calibration_yaml,
    write_calibration_yaml,
)
from pi_embodied_services.robots.ur5e.control import (
    EMPTY_WIDTH_FRACTION,
    UR5eController,
    UR5eLimits,
    async_phase,
    pose7_of,
    pose_rotvec,
    tool_tilt,
)
from pi_embodied_services.robots.ur5e.env_server import (
    DEFAULT_CONFIG,
    METHODS,
    UR5eEnvFacade,
    build_mock,
    camera_devices,
    limits_from_config,
    load_config,
    main,
)
from pi_embodied_services.robots.ur5e.mock import DOWN, MockRobotiq, MockUrArm

ARM = "2023300001"
CFG = {
    "robot": {"ip": "127.0.0.1", "gripper": {"type": "robotiq"}},
    "cameras": {
        "devices": {
            "wrist": {
                "type": "realsense",
                "serial": "1",
                "main": True,
                "mount": "wrist",
            },
            "front": {
                "type": "webcam",
                "device": 0,
                "mount": "fixed",
                "intrinsics": {"fx": 500, "fy": 500, "ppx": 320, "ppy": 240},
            },
        }
    },
    "calibration": {
        "arm_id": ARM,
        "begin_joints": [1.571, -1.571, 1.571, -1.571, -1.571, 0.0],
    },
    "limits": {
        "z_floor_m": 0.14,
        "workspace_min": [0.20, -0.35, 0.10],
        "workspace_max": [0.75, 0.35, 0.60],
        "max_move_m": 0.08,
        "max_rotate_rad": 0.2,
        "poll_s": 0.0,
    },
    "gripper": {"poll_s": 0.0},
}


@pytest.fixture(autouse=True)
def _mock_env(monkeypatch):
    monkeypatch.setenv(MOCK_ENV, "1")


def cfg(**sections) -> dict:
    out = copy.deepcopy(CFG)
    for name, values in sections.items():
        out.setdefault(name, {}).update(values)
    return out


def cameras() -> dict:
    return {"wrist": MockCamera("1"), "front": MockCamera("cam0", depth_m=None)}


def facade(arm=None, gripper=None, config=None, cams=None, **kw) -> UR5eEnvFacade:
    arm = arm or MockUrArm((0.45, 0.0, 0.30, *DOWN))
    gripper = MockRobotiq() if gripper is None else gripper
    return UR5eEnvFacade(
        config or cfg(), arm, gripper, cams or cameras(), sleep=lambda s: None, **kw
    )


def call(f: UR5eEnvFacade, method: str, *args, **kwargs):
    return f._serve_dispatch(method, args, kwargs)


# -- the rpy / rotvec fix -----------------------------------------------------------


def test_rpy_is_converted_to_a_rotation_vector_not_passed_through():
    # OpenETA ur5e.py:177 sent rpy as a rotvec: for [pi/2, 0, pi/2] that is a
    # 2.22 rad turn about (1, 0, 1); the real rotation is 120 deg about (1, 1, 1)/sqrt3.
    rpy = [math.pi / 2, 0.0, math.pi / 2]
    rv = pose_rotvec(rpy=rpy)
    assert not np.allclose(rv, rpy)
    expect = Rotation.from_euler("xyz", rpy).as_rotvec()
    np.testing.assert_allclose(rv, expect)
    np.testing.assert_allclose(np.linalg.norm(rv), math.radians(120), atol=1e-9)
    # A pure roll or yaw is the same in both parametrizations, which is how the bug hid.
    np.testing.assert_allclose(
        pose_rotvec(rpy=[math.pi, 0, 0]), [math.pi, 0, 0], atol=1e-12
    )
    np.testing.assert_allclose(pose_rotvec(rotvec=[0.1, 0.2, 0.3]), [0.1, 0.2, 0.3])
    with pytest.raises(ValueError, match="exactly one"):
        pose_rotvec(rotvec=[0, 0, 0], rpy=[0, 0, 0])
    with pytest.raises(ValueError, match="exactly one"):
        pose_rotvec()


def test_move_pose_with_rpy_commands_the_converted_rotvec():
    arm = MockUrArm((0.45, 0.0, 0.30, *DOWN), step_rad=1.0)
    f = facade(arm)
    # A small tilt expressed as rpy: the moveL target must carry its rotvec.
    start = Rotation.from_rotvec(DOWN)
    target = Rotation.from_euler("xyz", [0.1, 0.05, 0.0]) * start
    r = call(f, "env.move_pose", [0.45, 0.0, 0.30], rpy=target.as_euler("xyz").tolist())
    assert r["ok"]
    np.testing.assert_allclose(arm.moves[-1][3:], target.as_rotvec(), atol=1e-9)
    # The same pose given as a rotvec is identical.
    r2 = call(f, "env.move_pose", [0.45, 0.0, 0.30], rotvec=target.as_rotvec().tolist())
    assert r2["ok"]
    np.testing.assert_allclose(arm.moves[-1][3:], target.as_rotvec(), atol=1e-9)


# -- RPC surface ----------------------------------------------------------------------


def test_methods_and_capabilities_follow_the_franka_layout():
    pytest.importorskip("omegaconf")
    from pi_embodied_services.robots.franka.env_server import rlinf_capabilities

    f = facade()
    assert {m for m in f._rpc} == {f"env.{m}" for m in METHODS} | {"code.api"}
    assert "env.chunk_step" not in f._rpc
    names = {p["name"] for p in call(f, "code.api")["primitives"]}
    assert names == {
        "get_robot_state",
        "get_observation",
        "get_camera_meta",
        "move_delta",
        "move_pose",
        "rotate_delta",
        "set_gripper",
    }
    rlinf = rlinf_capabilities(
        {
            "main_image_key": "wrist_1",
            "override_cfg": {
                "camera_names": {"1": "wrist_1"},
                "ee_pose_limit_min": [0.15, -0.45, 0.03, 2.9, -0.1, -1.7],
                "ee_pose_limit_max": [1.15, 0.54, 0.27, 2.9, -0.1, 1.4],
            },
        },
        [0.02, 0.1, 1.0],
    )
    caps = f.capabilities()
    assert set(rlinf) <= set(caps)
    assert caps["backend"] == "ur_rtde" and not caps["has_vla"]
    assert caps["cameras"] == {"main": "wrist", "extra_0": "front"}
    assert caps["camera_depth"] == {"wrist": True, "front": False}
    assert caps["arm_id"] == ARM
    assert call(f, "healthz")["service"] == "ur5e-env"
    meta = call(f, "env.get_env_meta")
    assert meta["robot"] == "ur5e" and meta["has_begin_pose"] and meta["arm_id"] == ARM
    assert meta["limits"]["speed_mps"] == 0.25 and meta["limits"]["accel_mps2"] == 0.5


def test_state_observation_and_camera_meta():
    f = facade()
    state = call(f, "env.get_robot_state")
    base = state["raw_base_state"]
    assert len(base["tcp_pose"]) == 7 and len(base["tcp_pose_rotvec"]) == 6
    assert len(base["joints"]) == 6 and base["gripper_open"] is True
    assert base["gripper_position"] == [pytest.approx(0.085)]
    assert base["tool_tilt_rad"] == pytest.approx(0.0)
    obs = call(f, "env.get_observation")
    assert set(obs["images"]) == {"wrist", "front"}
    assert set(obs["depths"]) == {"wrist"}, "the webcam has no depth"
    assert (
        obs["images"]["wrist"].shape == (480, 640, 3)
        and obs["images"]["wrist"].dtype == np.uint8
    )
    assert obs["depths"]["wrist"].dtype == np.float32
    meta = call(f, "env.get_camera_meta")
    assert meta["observation_camera_map"] == {"main": "wrist", "extra_0": "front"}
    wrist, front = meta["cameras"]["wrist"], meta["cameras"]["front"]
    assert (
        wrist["has_depth"]
        and wrist["mount"] == "wrist"
        and wrist["intrinsic_K"][0][0] == 600.0
    )
    assert not front["has_depth"] and front["mount"] == "fixed"
    assert wrist["extrinsic"] is None, "no calibration file configured"


# -- safety layer -------------------------------------------------------------------------


def test_limits_are_required_and_bounded():
    with pytest.raises(ValueError, match="z_floor_m is not set"):
        UR5eLimits().validate()
    with pytest.raises(ValueError, match="workspace_min and limits.workspace_max"):
        UR5eLimits(z_floor_m=0.1).validate()
    with pytest.raises(ValueError, match="inside the workspace z range"):
        UR5eLimits(
            z_floor_m=0.9, workspace_min=(0, -1, 0), workspace_max=(1, 1, 0.5)
        ).validate()
    with pytest.raises(ValueError, match="speed_mps must be in"):
        limits_from_config(cfg(limits={"speed_mps": 2.0}))
    with pytest.raises(ValueError, match="unknown limits keys"):
        limits_from_config(cfg(limits={"enable_collision_check": True}))
    with pytest.raises(ValueError, match="arm_id is not set"):
        load_config_dict(cfg(calibration={"arm_id": None}))
    lim = limits_from_config(cfg())
    assert lim.speed_mps == 0.25 and lim.accel_mps2 == 0.5 and lim.max_move_m == 0.08


def load_config_dict(c: dict, tmp: Path | None = None):
    path = (tmp or Path(__file__).parent / "_ur5e_tmp.yaml") if tmp else None
    import tempfile

    with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as fh:
        yaml.safe_dump(c, fh)
        path = Path(fh.name)
    try:
        return load_config(path)
    finally:
        path.unlink()


def test_example_config_is_valid_once_the_hardware_values_are_filled():
    raw = yaml.safe_load(DEFAULT_CONFIG.read_text())
    with pytest.raises(ValueError, match="z_floor_m is not set"):
        limits_from_config(raw)
    raw["limits"].update(CFG["limits"])
    raw["calibration"].update(CFG["calibration"])
    raw["cameras"]["devices"]["wrist"].pop("calibration")
    raw["cameras"]["devices"]["front"].pop("calibration")
    load_config_dict(raw)


@pytest.mark.parametrize(
    "delta, match",
    [
        ([0.09, 0.0, 0.0], "limit is 0.08 m per call"),
        ([0.0, 0.0, -0.08], "outside the workspace"),  # z 0.12 < the 0.14 floor
        ([0.0, 0.08, 0.0], "outside the workspace"),  # y 0.38 > ymax 0.35
        ([float("nan"), 0.0, 0.0], "finite"),
        ([0.01, 0.0], "3 finite numbers"),
    ],
)
def test_move_delta_refuses_before_commanding(delta, match):
    arm = MockUrArm((0.5, 0.30, 0.2, *DOWN))
    f = facade(arm)
    with pytest.raises(ValueError, match=match):
        call(f, "env.move_delta", delta)
    assert arm.moves == [] and f.controller.commands == 0, (
        "a refused move commands nothing"
    )
    np.testing.assert_allclose(arm.pose[:3], [0.5, 0.30, 0.2])


def test_move_pose_is_bounded_from_the_setpoint_and_by_the_box():
    arm = MockUrArm((0.45, 0.0, 0.20, *DOWN))
    f = facade(arm)
    with pytest.raises(ValueError, match="limit is 0.08 m per call"):
        call(f, "env.move_pose", [0.60, 0.0, 0.20])
    with pytest.raises(ValueError, match="limit is 0.2 rad per call"):
        call(f, "env.move_pose", [0.45, 0.0, 0.20], rpy=[0.3, 0.0, 0.0])
    with pytest.raises(ValueError, match="outside the workspace"):
        call(f, "env.move_pose", [0.45, 0.0, 0.13])  # 7 cm down, below the 0.14 floor
    assert arm.moves == []
    r = call(f, "env.move_pose", [0.50, 0.02, 0.20])
    assert r["ok"] and r["final_error_m"] < 1e-9
    np.testing.assert_allclose(arm.pose[:3], [0.50, 0.02, 0.20])


def test_z_floor_refuses_descent_but_allows_moving_back_up():
    arm = MockUrArm((0.5, 0.0, 0.15, *DOWN))
    f = facade(arm)
    with pytest.raises(ValueError, match="z 0.14"):
        call(f, "env.move_delta", [0.0, 0.0, -0.02])
    arm.pose[2] = 0.12  # hand-guided below the floor: only moves back in run
    with pytest.raises(ValueError, match="outside the workspace"):
        call(f, "env.move_delta", [0.01, 0.0, 0.0])
    assert call(f, "env.move_delta", [0.0, 0.0, 0.01])["final_tcp_pose"][
        2
    ] == pytest.approx(0.13)


def test_rotation_limit_tilt_limit_and_base_frame_semantics():
    arm = MockUrArm((0.5, 0.0, 0.3, *DOWN), step_rad=1.0)
    f = facade(arm)
    with pytest.raises(ValueError, match="limit is 0.2 rad per call"):
        call(f, "env.rotate_delta", [0.0, 0.0, 0.25])
    assert arm.moves == []
    r = call(f, "env.rotate_delta", [0.0, 0.0, 0.15])
    assert r["ok"] and r["requested_delta_rpy_base"] == [0.0, 0.0, 0.15]
    # Rz(0.15) * Rx(pi): a yaw about base +z keeps the tool pointing down.
    expect = Rotation.from_euler("z", 0.15) * Rotation.from_rotvec(DOWN)
    assert (
        Rotation.from_quat(r["final_tcp_pose"][3:]) * expect.inv()
    ).magnitude() < 1e-9
    np.testing.assert_allclose(r["final_tcp_pose"][:3], [0.5, 0.0, 0.3], atol=1e-9)
    assert tool_tilt(np.asarray(arm.pose[3:])) == pytest.approx(0.0, abs=1e-6)
    # Three 0.2 rad tilts would pass 0.5 rad from straight down: the third is refused.
    for _ in range(2):
        assert call(f, "env.rotate_delta", [0.2, 0.0, 0.0])["ok"]
    with pytest.raises(ValueError, match="tilts the tool"):
        call(f, "env.rotate_delta", [0.2, 0.0, 0.0])
    assert call(f, "env.rotate_delta", [-0.2, 0.0, 0.0])["ok"], (
        "tilting back is allowed"
    )


def test_moves_use_the_configured_speed_and_accumulate_on_the_setpoint():
    arm = MockUrArm((0.5, 0.0, 0.3, *DOWN))
    f = facade(arm)
    r = call(f, "env.move_delta", [0.02, -0.01, 0.0])
    assert (
        r["ok"]
        and r["steps_used"] > 0
        and r["requested_delta_xyz_base"] == [0.02, -0.01, 0.0]
    )
    assert (arm.speed, arm.accel) == (0.25, 0.5)
    assert {
        "start_tcp_pose",
        "target_tcp_pose",
        "final_tcp_pose",
        "final_error_m",
        "states",
    } <= set(r)
    arm.pose[0] += 0.004  # measurement sag below the resync gap does not accumulate
    call(f, "env.move_delta", [0.02, 0.0, 0.0])
    np.testing.assert_allclose(arm.moves[-1][:3], [0.54, -0.01, 0.3], atol=1e-9)
    assert call(f, "env.get_robot_state")["raw_base_state"]["setpoint_pose"][
        :3
    ] == pytest.approx([0.54, -0.01, 0.3])


def test_a_blocked_move_clears_the_setpoint_and_reports_not_ok():
    arm = MockUrArm((0.5, 0.0, 0.3, *DOWN), wall_x=0.52)
    f = facade(arm)
    r = call(f, "env.move_delta", [0.05, 0.0, 0.0])
    assert r["ok"] is False and r["final_error_m"] == pytest.approx(0.03)
    assert f.controller.target is None, "a missed target clears the setpoint"
    assert call(f, "env.get_robot_state")["raw_base_state"]["setpoint_pose"] is None
    # The next move starts from the measured pose, not the old target.
    r2 = call(f, "env.move_delta", [0.0, 0.0, 0.01])
    np.testing.assert_allclose(arm.moves[-1][:3], [0.52, 0.0, 0.31], atol=1e-9)
    assert r2["ok"]


def test_a_driver_error_stops_the_arm_and_clears_the_setpoint():
    arm = MockUrArm((0.5, 0.0, 0.3, *DOWN), fail_after=2)
    f = facade(arm)
    call(f, "env.move_delta", [0.0, 0.0, 0.0])  # settles the setpoint
    arm.polls = 0
    with pytest.raises(RuntimeError, match="motion aborted.*setpoint was cleared"):
        call(f, "env.move_delta", [0.05, 0.0, 0.0])
    assert arm.stops[-1] == "stopL" and f.controller.target is None


def test_timeout_stops_the_move():
    arm = MockUrArm((0.5, 0.0, 0.3, *DOWN), step_m=0.0001)
    clock = {"t": 0.0}

    def tick():
        clock["t"] += 0.5
        return clock["t"]

    c = UR5eController(
        arm,
        None,
        limits_from_config(cfg(limits={"move_timeout_s": 1.0})),
        sleep=lambda s: None,
        clock=tick,
    )
    r = c.move_delta([0.05, 0.0, 0.0])
    assert (
        r["timed_out"] and not r["ok"] and arm.stops == ["stopL"] and c.target is None
    )


def test_stop_halts_a_running_move_with_stopL_and_clears_the_setpoint():
    arm = MockUrArm((0.5, 0.0, 0.3, *DOWN), step_m=0.0005)
    f = facade(arm, config=cfg(limits={"poll_s": 0.005}))
    f.controller._sleep = time.sleep
    box: dict = {}
    thread = threading.Thread(
        target=lambda: box.update(r=call(f, "env.move_delta", [0.05, 0.0, 0.0]))
    )
    thread.start()
    deadline = time.time() + 5
    while arm.polls < 10:
        assert time.time() < deadline
        time.sleep(0.002)
    reply = call(f, "stop")
    assert reply["call_in_progress"] is True
    thread.join(timeout=5)
    r = box["r"]
    assert r["cancelled"] is True and r["ok"] is False
    assert arm.stops == ["stopL"], "stopL really stops the moveL"
    assert 0 < np.linalg.norm(arm.pose[:3] - [0.5, 0, 0.3]) < 0.05
    assert f.controller.target is None, "the setpoint is cleared after a stop"
    # Calls after the stop run normally, from the measured pose.
    r2 = call(f, "env.move_delta", [0.001, 0.0, 0.0])
    assert r2["ok"]
    np.testing.assert_allclose(
        r2["start_tcp_pose"][:3], r["final_tcp_pose"][:3], atol=1e-9
    )


def test_stop_halts_the_reset_with_stopJ():
    arm = MockUrArm((0.5, 0.0, 0.3, *DOWN), joints=(0.0,) * 6, step_joint_rad=0.001)
    f = facade(arm, config=cfg(limits={"poll_s": 0.005}))
    f.controller._sleep = time.sleep
    box: dict = {}
    thread = threading.Thread(target=lambda: box.update(r=call(f, "env.reset")))
    thread.start()
    deadline = time.time() + 5
    while not arm.joint_moves or arm.polls < 5:
        assert time.time() < deadline
        time.sleep(0.002)
    call(f, "stop")
    thread.join(timeout=5)
    r = box["r"]
    assert r["cancelled"] is True and r["ok"] is False and "stopJ" in arm.stops
    assert f.controller.target is None


def test_reset_opens_lifts_then_moves_to_begin_and_needs_begin_joints():
    arm = MockUrArm((0.5, 0.0, 0.3, *DOWN))
    grip = MockRobotiq(position=255)
    f = facade(arm, grip)
    r = call(f, "env.reset")
    assert r["ok"] and r["gripper"]["ok"] and r["lift"]["ok"] and r["move"]["ok"]
    assert grip.commands[0] == 0 and grip.pos == 0
    # Order: gripper, a straight-up moveL of reset_lift_m, then the moveJ.
    assert len(arm.moves) == 1 and len(arm.joint_moves) == 1
    np.testing.assert_allclose(arm.moves[0][:3], [0.5, 0.0, 0.35], atol=1e-9)
    assert r["info"]["lifted_m"] == pytest.approx(0.05)
    np.testing.assert_allclose(arm.q, CFG["calibration"]["begin_joints"], atol=1e-9)
    assert r["robot_state"]["raw_base_state"]["setpoint_pose"] is not None
    bare = facade(
        MockUrArm(), config=cfg(calibration={"arm_id": ARM, "begin_joints": None})
    )
    with pytest.raises(ValueError, match="begin_joints is not set"):
        call(bare, "env.reset")
    assert bare.controller.commands == 0


def test_reset_lift_is_clamped_to_the_box_ceiling_and_a_blocked_lift_stops_the_reset():
    # 2 cm below the ceiling: the lift is 2 cm, not 5.
    arm = MockUrArm((0.5, 0.0, 0.58, *DOWN))
    f = facade(arm)
    r = call(f, "env.reset")
    assert r["ok"] and r["info"]["lifted_m"] == pytest.approx(0.02)
    np.testing.assert_allclose(arm.moves[0][:3], [0.5, 0.0, 0.60], atol=1e-9)
    # At the ceiling there is no lift at all, only the joint move.
    arm = MockUrArm((0.5, 0.0, 0.60, *DOWN))
    f = facade(arm)
    r = call(f, "env.reset")
    assert r["ok"] and r["lift"] is None and "lifted_m" not in r["info"]
    assert arm.moves == [] and len(arm.joint_moves) == 1
    # A lift that does not arrive (a wall) fails the reset before any moveJ.
    arm = MockUrArm((0.5, 0.0, 0.3, *DOWN), protective_stop_after=2)
    f = facade(arm)
    r = call(f, "env.reset")
    assert r["ok"] is False and r["lift"]["ok"] is False and r["move"] is None
    assert r["lift"]["protective_stopped"] is True and arm.joint_moves == []


def test_reset_refuses_a_begin_pose_outside_the_workspace_before_moving():
    # Forward kinematics puts the begin pose's TCP at x = 0.9 (the box ends at 0.75):
    # refused before the gripper, the lift or the moveJ is commanded.
    arm = MockUrArm(
        (0.5, 0.0, 0.3, *DOWN), joints=(0.0,) * 6, home_pose=(0.9, 0.0, 0.4, *DOWN)
    )
    grip = MockRobotiq(position=255)
    f = facade(arm, grip)
    with pytest.raises(
        ValueError, match="outside the workspace.*nothing was commanded"
    ):
        call(f, "env.reset")
    assert arm.moves == [] and arm.joint_moves == [] and grip.commands == []
    np.testing.assert_allclose(arm.fk_queries[0], CFG["calibration"]["begin_joints"])
    # Joints the controller's safety configuration rejects are refused the same way.
    arm = MockUrArm((0.5, 0.0, 0.3, *DOWN), joints_within_limits=False)
    with pytest.raises(ValueError, match="safety limits.*nothing was commanded"):
        call(facade(arm), "env.reset")
    assert arm.moves == [] and arm.joint_moves == []
    # Forward kinematics said inside (the arm is already at the begin joints) but the
    # measured TCP ends outside: still reported, not trusted.
    arm = MockUrArm((0.5, 0.0, 0.3, *DOWN), home_pose=(0.9, 0.0, 0.4, *DOWN))
    r = call(facade(arm), "env.reset")
    assert r["move"]["ok"] and r["ok"] is False
    assert r["info"]["begin_pose_outside_workspace"] is True
    assert "outside the workspace" in r["info"]["note"]


def test_reset_refuses_a_joint_path_that_dips_below_the_floor():
    begin = np.asarray(CFG["calibration"]["begin_joints"])
    q0 = np.zeros(6)

    def fk(q):
        # TCP z sags 35 cm in the middle of the joint path (a swing through the table).
        s = float(np.clip(np.linalg.norm(q - q0) / np.linalg.norm(begin - q0), 0, 1))
        return (0.45, 0.0, 0.40 - 0.35 * math.sin(math.pi * s), *DOWN)

    arm = MockUrArm((0.45, 0.0, 0.40, *DOWN), joints=q0, fk=fk)
    f = facade(arm)
    r = call(f, "env.reset")
    assert r["ok"] is False and r["move"]["path_outside_workspace"] is True
    assert arm.joint_moves == [], "the moveJ was never commanded"
    assert r["info"]["begin_path_outside_workspace"] is True
    assert "leaves the workspace" in r["info"]["note"]
    # A path that stays in the box runs.
    arm = MockUrArm(
        (0.45, 0.0, 0.40, *DOWN),
        joints=q0,
        fk=lambda q: (0.45, 0.0, 0.40 - 0.01 * float(np.max(np.abs(q))), *DOWN),
    )
    r = call(facade(arm), "env.reset")
    assert r["ok"] and len(arm.joint_moves) == 1
    assert len(arm.fk_queries) >= f.controller.limits.reset_path_samples


def test_reset_releases_a_held_object_and_says_so():
    grip = MockRobotiq(object_pos=150)
    f = facade(gripper=grip)
    assert call(f, "env.set_gripper", open=False)["object_detected"]
    r = call(f, "env.reset")
    assert r["ok"] and grip.pos == 0 and r["info"]["released_object"] is True
    assert "released" in r["info"]["note"]
    # A closed but empty gripper is simply opened (nothing to report).
    f2 = facade(gripper=MockRobotiq(position=255, object_pos=None))
    r2 = call(f2, "env.reset")
    assert r2["ok"] and "released_object" not in r2["info"]


# -- protective stop and rejected commands -------------------------------------------


def test_motion_is_refused_while_the_robot_is_protective_stopped():
    arm = MockUrArm((0.5, 0.0, 0.3, *DOWN))
    f = facade(arm)
    call(f, "env.move_delta", [0.0, 0.0, 0.0])
    assert f.controller.target is not None
    arm.protective_stopped = True
    for method, kwargs in (
        ("env.move_delta", {"delta_xyz": [0.01, 0.0, 0.0]}),
        ("env.move_pose", {"xyz": [0.5, 0.0, 0.31]}),
        ("env.rotate_delta", {"delta_rpy": [0.0, 0.0, 0.1]}),
        ("env.reset", {}),
    ):
        with pytest.raises(
            RuntimeError, match="protective stopped.*nothing was commanded"
        ):
            call(f, method, **kwargs)
    assert len(arm.moves) == 1, "only the settling move before the stop was sent"
    assert arm.joint_moves == [] and f.controller.target is None
    state = call(f, "env.get_robot_state")["raw_base_state"]
    assert state["robot_status"]["protective_stopped"] is True
    arm.protective_stopped = False
    assert call(f, "env.move_delta", [0.01, 0.0, 0.0])["ok"]


def test_a_rejected_move_raises_and_clears_the_setpoint():
    arm = MockUrArm((0.5, 0.0, 0.3, *DOWN))
    f = facade(arm)
    call(f, "env.move_delta", [0.0, 0.0, 0.0])
    arm.accept_moves = False
    with pytest.raises(RuntimeError, match="moveL was rejected by the controller"):
        call(f, "env.move_delta", [0.01, 0.0, 0.0])
    assert f.controller.target is None and len(arm.moves) == 1
    with pytest.raises(RuntimeError, match="moveL was rejected"):
        call(f, "env.reset")  # the lift is the first command
    with pytest.raises(RuntimeError, match="moveJ was rejected by the controller"):
        f.controller.move_joints(CFG["calibration"]["begin_joints"])
    assert arm.joint_moves == []


def test_a_protective_stop_during_a_move_is_reported_in_the_result():
    arm = MockUrArm((0.5, 0.0, 0.3, *DOWN), protective_stop_after=2)
    f = facade(arm)
    r = call(f, "env.move_delta", [0.05, 0.0, 0.0])
    assert r["ok"] is False and r["protective_stopped"] is True
    assert "protective stopped" in r["note"] and "teach pendant" in r["note"]
    assert f.controller.target is None
    arm = MockUrArm((0.5, 0.0, 0.3, *DOWN), joints=(0.0,) * 6, protective_stop_after=2)
    f = facade(arm)
    r = call(f, "env.reset")
    assert r["ok"] is False
    stage = r["lift"] if r["move"] is None else r["move"]
    assert stage["protective_stopped"] is True


# -- gripper ------------------------------------------------------------------------


def test_gripper_open_close_width_and_grasp():
    grip = MockRobotiq(object_pos=150)  # an object stops the fingers at 150/255
    f = facade(gripper=grip)
    r = call(f, "env.set_gripper", open=False)
    assert r["ok"] and r["object_detected"] and not r.get("grasp_empty")
    assert r["gripper_width_m"] == pytest.approx(0.085 * (1 - 150 / 255))
    state = r["robot_state"]["raw_base_state"]
    assert state["gripper_open"] is False and state["gripper_grasped"] is True
    assert state["gripper_commanded_open"] is False
    r = call(f, "env.set_gripper", open=True)
    assert r["ok"] and r["gripper_width_m"] == pytest.approx(0.085)
    assert {"target_gripper_open", "steps_used", "robot_state", "states"} <= set(r)


def test_empty_grasp_reopens_and_jammed_fingers_are_reported():
    grip = MockRobotiq()  # nothing between the fingers: closes to 255 = 0 m
    f = facade(gripper=grip)
    r = call(f, "env.set_gripper", open=False)
    assert r["grasp_empty"] is True and r["ok"] is False and grip.pos == 0
    assert r["steps_used"] == 2 and "reopened" in r["note"]
    jam = MockRobotiq(jammed=True)
    g = facade(gripper=jam)
    r = call(g, "env.set_gripper", open=False)
    assert r["gripper_jammed"] is True and r["ok"] is False
    assert "jammed" in r["note"] and not r.get("grasp_empty")
    assert r["robot_state"]["raw_base_state"]["gripper_commanded_open"] is False
    # An idle open of an open gripper is not a jam.
    assert "gripper_jammed" not in call(
        facade(gripper=MockRobotiq()), "env.set_gripper", open=True
    )
    # Without a gripper the call is refused.
    none = facade(
        gripper=False and None,
        config=cfg(robot={"ip": "127.0.0.1", "gripper": {"type": "none"}}),
    )
    none.controller.gripper = None
    with pytest.raises(ValueError, match="no gripper configured"):
        call(none, "env.set_gripper", open=True)


def test_gripper_is_activated_once_before_the_first_command():
    grip = MockRobotiq(active=False)
    f = facade(gripper=grip)
    call(f, "env.set_gripper", open=True)
    call(f, "env.set_gripper", open=False)
    assert grip.activations == 1


def test_a_stale_object_status_is_not_taken_for_settled_fingers():
    # After GTO the OBJ register still shows the previous motion (3 = at position)
    # for a few reads; the old check settled at once and called the unmoved fingers
    # jammed. Grasping an object through the lag must report the grasp.
    grip = MockRobotiq(object_pos=150, stale_polls=3, moving_polls=2)
    f = facade(gripper=grip)
    r = call(f, "env.set_gripper", open=False)
    assert r["ok"] and r["object_detected"] and "gripper_jammed" not in r
    assert grip.pos == 150
    # Opening an already open gripper (POS == PRE) settles without waiting.
    grip = MockRobotiq(stale_polls=3)
    f = facade(gripper=grip)
    r = call(f, "env.set_gripper", open=True)
    assert r["ok"] and "gripper_jammed" not in r
    # Fingers that never report motion are still jammed, after the ack timeout.
    clock = {"t": 0.0}

    def tick():
        clock["t"] += 0.1
        return clock["t"]

    jam = MockRobotiq(jammed=True, stale_polls=100)
    c = UR5eController(
        MockUrArm(), jam, limits_from_config(cfg()), sleep=lambda s: None, clock=tick
    )
    r = c.set_gripper(open=False)
    assert r["gripper_jammed"] is True and r["ok"] is False
    assert 0.5 < clock["t"] < 2.0, "settled after gripper_ack_timeout_s, not at once"


def test_stop_during_a_gripper_command_stops_the_gripper():
    grip = MockRobotiq(object_pos=150, stale_polls=100)
    f = facade(gripper=grip)
    f.controller._stop = lambda: True
    r = call(f, "env.set_gripper", open=False)
    assert r["cancelled"] is True and r["ok"] is False and grip.stopped == 1


def test_setpoint_resyncs_on_orientation_drift_too():
    arm = MockUrArm((0.5, 0.0, 0.3, *DOWN), step_rad=1.0)
    f = facade(arm)
    call(f, "env.move_delta", [0.0, 0.0, 0.0])
    # The tool was turned 0.1 rad by hand (position unchanged): the next command
    # starts from the measured orientation, not the stale setpoint.
    turned = (Rotation.from_euler("z", 0.1) * Rotation.from_rotvec(DOWN)).as_rotvec()
    arm.pose[3:] = turned
    r = call(f, "env.move_delta", [0.01, 0.0, 0.0])
    assert r["ok"]
    np.testing.assert_allclose(arm.moves[-1][3:], turned, atol=1e-9)
    assert f.controller.limits.divergence_resync_rad == 0.05


# -- camera health and freshness -------------------------------------------------------


def test_a_dead_camera_blocks_motion_until_it_delivers_frames_again():
    front = MockCamera("cam0", depth_m=None, fail_reads=4)
    cams = {"wrist": MockCamera("1"), "front": front}
    arm = MockUrArm((0.5, 0.0, 0.3, *DOWN))
    f = facade(arm, cams=cams)
    with pytest.raises(RuntimeError, match="camera read failed: front"):
        call(f, "env.get_observation")
    assert "front" in f.camera_errors and "wrist" not in f.camera_errors
    with pytest.raises(RuntimeError, match="camera 'front' is not delivering frames"):
        call(f, "env.move_delta", [0.01, 0.0, 0.0])
    with pytest.raises(RuntimeError, match="not delivering frames"):
        call(f, "env.set_gripper", open=True)
    assert arm.moves == [] and f.controller.commands == 0
    # The camera came back (its reads succeed): motion is allowed again.
    front.fail_reads = 0
    assert call(f, "env.move_delta", [0.01, 0.0, 0.0])["ok"]
    assert f.camera_errors == {}


def test_observations_are_fresh_and_carry_the_frame_age():
    cams = {
        "wrist": MockCamera("1"),
        "front": MockCamera("cam0", depth_m=None, buffered=True),
    }
    f = facade(cams=cams)
    cams["front"].scene = 7
    obs = call(f, "env.get_observation")
    assert MockCamera.scene_of(obs["images"]["front"]) == 7, (
        "the queued pre-motion frame was drained"
    )
    assert set(obs["frame_age_s"]) == {"wrist", "front"}
    assert obs["frame_age_s"]["front"] < 0.5 and obs["max_frame_age_s"] == 0.5
    # A source that only hands out old frames is refused after one re-read.
    stale = {
        "wrist": MockCamera("1"),
        "front": MockCamera("cam0", depth_m=None, age_s=2),
    }
    g = facade(cams=stale)
    with pytest.raises(RuntimeError, match="front: frame is 2.0. s old"):
        call(g, "env.get_observation")
    assert stale["front"].reads == 4, "read_fresh twice, each discarding one frame"
    with pytest.raises(ValueError, match="max_frame_age_s must be positive"):
        facade(config=cfg(cameras={"max_frame_age_s": 0}))
    with pytest.raises(ValueError, match="unknown cameras keys"):
        facade(config=cfg(cameras={"max_frame_age": 1}))


def test_hand_eye_capture_waits_for_rest_and_takes_a_fresh_frame():
    arm = MockUrArm((0.5, 0.0, 0.3, *DOWN))
    cam = MockCamera("cam0", depth_m=None, buffered=True)
    clock = {"t": 0.0}

    def tick():
        clock["t"] += 0.05
        return clock["t"]

    # The arm was moved to a new pose (scene 1) since the camera's last read of
    # scene 0: a plain read would pair the new pose with the old image.
    cam.read()
    arm.pose[0] = 0.55
    cam.scene = 1
    assert MockCamera.scene_of(cam.read()) == 0
    pose, joints, frame = capture_sample(
        arm, cam, clock=tick, sleep=lambda s: None, settle_s=0.2
    )
    assert MockCamera.scene_of(frame) == 1 and pose[0] == pytest.approx(0.55)
    assert joints.shape == (6,)
    # A swaying arm is not sampled.
    arm.speeds = np.array([0.0, 0.01, 0.0, 0.0, 0.0, 0.0])
    with pytest.raises(RuntimeError, match="still moving"):
        capture_sample(arm, cam, clock=tick, sleep=lambda s: None, timeout_s=1.0)
    arm.speeds = np.zeros(6)
    # A frame that is too old is refused (real clock: the mock backdates against it).
    with pytest.raises(RuntimeError, match="s old"):
        capture_sample(
            arm,
            MockCamera("x", depth_m=None, age_s=3.0),
            sleep=lambda s: None,
            max_age_s=0.5,
            settle_s=0.0,
        )


# -- arm binding ---------------------------------------------------------------------


def test_the_config_is_bound_to_the_arm_serial():
    with pytest.raises(
        ValueError, match="reports serial 'other'.*belongs to another arm"
    ):
        facade(MockUrArm(serial="other"))
    with pytest.raises(ValueError, match="serial number could not be read"):
        facade(MockUrArm(serial=None))
    unbound = facade(
        MockUrArm(serial="whatever"),
        config=cfg(robot={"ip": "127.0.0.1", "identity": "none"}),
    )
    assert unbound.arm_id is None


def test_calibration_files_are_bound_to_the_arm_and_the_mount(tmp_path):
    result = {
        "mode": EYE_IN_HAND,
        "method": "PARK",
        "transform": "T_tcp_cam",
        "matrix": calibrate.transform(
            Rotation.from_euler("z", 0.3).as_matrix(), [0.05, 0.0, 0.1]
        ),
        "samples": {"total": 5, "used": 5, "rejected": 0},
        "rejected": [],
        "residuals": {"translation_mm": {"mean": 1.0, "median": 1.0, "max": 2.0}},
    }
    path = tmp_path / "wrist.yaml"
    write_calibration_yaml(
        path,
        result,
        arm_id=ARM,
        camera="wrist",
        camera_serial="1",
        target_path=str(path),
    )
    c = cfg()
    c["cameras"]["devices"]["wrist"]["calibration"] = str(path)
    f = facade(config=c)
    ext = call(f, "env.get_camera_meta")["cameras"]["wrist"]["extrinsic"]
    assert ext["frame"] == "tcp" and ext["arm_id"] == ARM
    np.testing.assert_allclose(ext["matrix"], result["matrix"])
    # Another arm's calibration is refused.
    write_calibration_yaml(
        path,
        result,
        arm_id="1999",
        camera="wrist",
        camera_serial="1",
        target_path=str(path),
    )
    with pytest.raises(ValueError, match="made on arm '1999', not this arm"):
        facade(config=c)
    # A wrist file on a fixed camera is refused, and so is another camera's serial.
    write_calibration_yaml(
        path,
        result,
        arm_id=ARM,
        camera="wrist",
        camera_serial="1",
        target_path=str(path),
    )
    c["cameras"]["devices"]["wrist"]["mount"] = "fixed"
    with pytest.raises(ValueError, match="eye-in-hand but the camera is mounted fixed"):
        facade(config=c)
    c["cameras"]["devices"]["wrist"]["mount"] = "wrist"
    write_calibration_yaml(
        path,
        result,
        arm_id=ARM,
        camera="wrist",
        camera_serial="9",
        target_path=str(path),
    )
    with pytest.raises(ValueError, match="made with camera '9'"):
        facade(config=c)
    with pytest.raises(ValueError, match="arm_id is required"):
        write_calibration_yaml(
            path,
            result,
            arm_id="",
            camera="wrist",
            camera_serial="1",
            target_path=str(path),
        )


# -- cameras from flags ---------------------------------------------------------------


def test_camera_flag_overrides_the_config_devices():
    main, extras, devices = camera_devices(
        cfg(), "front=webcam:0,side=rtsp://cam.local/live,wrist=realsense:1234"
    )
    assert main == "front" and extras == ["side", "wrist"]
    assert devices["side"] == {"type": "rtsp", "url": "rtsp://cam.local/live"}
    assert (
        devices["wrist"]["serial"] == "1234" and devices["wrist"]["mount"] == "wrist"
    ), "config keys carry over"
    assert devices["front"]["intrinsics"]["fx"] == 500
    with pytest.raises(ValueError, match="must be name=type:source"):
        camera_devices(cfg(), "front")
    with pytest.raises(ValueError, match="realsense:<serial>"):
        camera_devices(cfg(), "front=kinect:0")
    f = facade(
        cams={"wrist": MockCamera("1"), "front": MockCamera("f", depth_m=None)},
        camera_flag="front=webcam:0,wrist=realsense:1",
    )
    assert f.capabilities()["cameras"]["main"] == "front"


def test_mock_cli_serves_and_read_pose_is_read_only(tmp_path, monkeypatch):
    path = tmp_path / "ur5e.yaml"
    path.write_text(yaml.safe_dump(cfg()))
    arm, gripper, cams = build_mock(load_config(path))
    assert (
        gripper is not None
        and set(cams) == {"wrist", "front"}
        and not cams["front"].has_depth
    )
    assert arm.identity() == ARM
    monkeypatch.delenv(MOCK_ENV)
    with pytest.raises(SystemExit):
        main(["--mock", "--robot-config", str(path)])
    monkeypatch.setenv(MOCK_ENV, "1")
    assert main(["--print-config", "--robot-config", str(path)]) == 0


# -- hand-eye calibration on synthetic poses ---------------------------------------------


def rand_transform(rng, t_scale=0.3):
    return calibrate.transform(
        Rotation.random(random_state=rng).as_matrix(), rng.uniform(-t_scale, t_scale, 3)
    )


def synthetic(mode: str, n: int, rng, noise_m=0.0, noise_rad=0.0):
    """Ground-truth camera transform and n consistent samples; returns (T_cam, samples, T_board)."""
    T_cam = rand_transform(rng)  # T_base_cam (eye_to_hand) or T_tcp_cam (eye_in_hand)
    T_board = rand_transform(
        rng, 0.1
    )  # T_tcp_board (eye_to_hand) or T_base_board (eye_in_hand)
    samples = []
    for i in range(n):
        T_base_tcp = rand_transform(rng, 0.4)
        if mode == EYE_TO_HAND:
            T_cam_board = calibrate.invert(T_cam) @ T_base_tcp @ T_board
        else:
            T_cam_board = (
                calibrate.invert(T_cam) @ calibrate.invert(T_base_tcp) @ T_board
            )
        if noise_m or noise_rad:
            jitter = calibrate.transform(
                Rotation.from_rotvec(rng.normal(0, noise_rad, 3)).as_matrix(),
                rng.normal(0, noise_m, 3),
            )
            T_cam_board = T_cam_board @ jitter
        samples.append(Sample(f"{i:03d}", T_base_tcp, T_cam_board, reprojection_px=0.3))
    return T_cam, samples, T_board


@pytest.mark.parametrize("mode", [EYE_TO_HAND, EYE_IN_HAND])
@pytest.mark.parametrize("method", ["PARK", "TSAI", "DANIILIDIS"])
def test_hand_eye_recovers_the_camera_transform(mode, method):
    pytest.importorskip("cv2")
    rng = np.random.default_rng(7)
    T_cam, samples, T_board = synthetic(mode, 12, rng)
    result = calibrate.calibrate(samples, mode, method=method)
    np.testing.assert_allclose(result["matrix"], T_cam, atol=1e-6)
    np.testing.assert_allclose(result["board_matrix"], T_board, atol=1e-6)
    assert result["samples"] == {"total": 12, "used": 12, "rejected": 0}
    assert result["residuals"]["translation_mm"]["max"] < 1e-3
    assert result["transform"] == ("T_base_cam" if mode == EYE_TO_HAND else "T_tcp_cam")


@pytest.mark.parametrize("mode", [EYE_TO_HAND, EYE_IN_HAND])
def test_outliers_are_rejected_and_the_solve_recovers(mode):
    pytest.importorskip("cv2")
    rng = np.random.default_rng(3)
    T_cam, samples, _ = synthetic(mode, 15, rng, noise_m=0.0005, noise_rad=0.001)
    # Two corrupt observations: a mis-detected board 12 cm off, and a bad reprojection.
    samples[4].T_cam_board = samples[4].T_cam_board @ calibrate.transform(
        np.eye(3), [0.12, 0.0, 0.0]
    )
    samples[9].reprojection_px = 9.0
    result = calibrate.calibrate(samples, mode)
    rejected = {r["id"] for r in result["rejected"]}
    assert rejected == {"004", "009"}, result["rejected"]
    assert any("reprojection 9.00 px" in r["reason"] for r in result["rejected"])
    assert result["samples"]["used"] == 13
    assert np.linalg.norm(result["matrix"][:3, 3] - T_cam[:3, 3]) < 0.005
    assert calibrate.rotation_deg(result["matrix"][:3, :3], T_cam[:3, :3]) < 0.5
    assert result["residuals"]["translation_mm"]["max"] < 5


def test_too_few_samples_are_refused():
    pytest.importorskip("cv2")
    rng = np.random.default_rng(1)
    _, samples, _ = synthetic(EYE_TO_HAND, 2, rng)
    with pytest.raises(ValueError, match="need >= 3"):
        calibrate.calibrate(samples, EYE_TO_HAND)
    with pytest.raises(ValueError, match="mode must be one of"):
        calibrate.solve_hand_eye(samples, "sideways")


def test_solve_writes_a_new_file_and_apply_replaces_only_with_yes(tmp_path, capsys):
    pytest.importorskip("cv2")
    rng = np.random.default_rng(11)
    T_cam, samples, _ = synthetic(EYE_TO_HAND, 10, rng)
    sdir = tmp_path / "samples"
    sdir.mkdir()
    (sdir / "intrinsics.json").write_text(
        json.dumps({"fx": 600, "fy": 600, "ppx": 320, "ppy": 240})
    )
    (sdir / "camera.json").write_text(
        json.dumps({"camera": "front", "serial": "cam-7"})
    )
    for s in samples:
        pose7 = pose7_of(
            np.concatenate(
                [
                    s.T_base_tcp[:3, 3],
                    Rotation.from_matrix(s.T_base_tcp[:3, :3]).as_rotvec(),
                ]
            )
        )
        (sdir / f"{s.id}.json").write_text(
            json.dumps(
                {
                    "tcp_pose": pose7,
                    "T_cam_board": s.T_cam_board.tolist(),
                    "reprojection_px": 0.3,
                }
            )
        )
    target = tmp_path / "cal" / "front.yaml"
    c = cfg()
    c["cameras"]["devices"]["front"]["calibration"] = str(target)
    config = tmp_path / "ur5e.yaml"
    config.write_text(yaml.safe_dump(c))
    assert (
        calibrate.main(
            [
                "solve",
                "--robot-config",
                str(config),
                "--camera",
                "front",
                "--samples",
                str(sdir),
            ]
        )
        == 0
    )
    out = capsys.readouterr().out
    assert (
        "T_base_cam (eye_to_hand, PARK) from 10 of 10 samples" in out
        and "translation_mm" in out
    )
    new = tmp_path / "cal" / "front.new.yaml"
    assert new.exists() and not target.exists(), (
        "solve never touches the live calibration"
    )
    loaded = load_calibration_yaml(new)
    assert (
        loaded["arm_id"] == ARM
        and loaded["camera"] == "front"
        and loaded["camera_serial"] == "cam-7"
    )
    assert loaded["eye_on_hand"] is False
    np.testing.assert_allclose(loaded["matrix"], T_cam, atol=1e-6)
    # Dry run: nothing replaced.
    assert (
        calibrate.main(["apply", "--robot-config", str(config), "--camera", "front"])
        == 2
    )
    assert new.exists() and not target.exists()
    # Another arm's config refuses.
    c["calibration"]["arm_id"] = "1999"
    config.write_text(yaml.safe_dump(c))
    with pytest.raises(SystemExit, match="bound to arm"):
        calibrate.main(
            ["apply", "--robot-config", str(config), "--camera", "front", "--yes"]
        )
    c["calibration"]["arm_id"] = ARM
    config.write_text(yaml.safe_dump(c))
    assert (
        calibrate.main(
            ["apply", "--robot-config", str(config), "--camera", "front", "--yes"]
        )
        == 0
    )
    assert target.exists() and not new.exists()
    # The env server now serves the extrinsic; a second apply backs the old file up.
    f = facade(
        config=c,
        cams={"wrist": MockCamera("1"), "front": MockCamera("cam-7", depth_m=None)},
    )
    assert (
        call(f, "env.get_camera_meta")["cameras"]["front"]["extrinsic"]["frame"]
        == "base"
    )
    assert (
        calibrate.main(
            [
                "solve",
                "--robot-config",
                str(config),
                "--camera",
                "front",
                "--samples",
                str(sdir),
            ]
        )
        == 0
    )
    assert (
        calibrate.main(
            ["apply", "--robot-config", str(config), "--camera", "front", "--yes"]
        )
        == 0
    )
    assert "change from the current calibration" in capsys.readouterr().out
    assert len(list((tmp_path / "cal").glob("front.yaml.bak-*"))) == 1


def test_solve_needs_an_arm_id(tmp_path):
    pytest.importorskip("cv2")
    c = cfg(calibration={"arm_id": None, "begin_joints": None})
    c["robot"]["identity"] = "none"
    c["cameras"]["devices"]["front"]["calibration"] = str(tmp_path / "front.yaml")
    config = tmp_path / "ur5e.yaml"
    config.write_text(yaml.safe_dump(c))
    with pytest.raises(SystemExit, match="--arm-id is required"):
        calibrate.main(
            [
                "solve",
                "--robot-config",
                str(config),
                "--camera",
                "front",
                "--samples",
                str(tmp_path),
            ]
        )


def test_board_detection_round_trips_a_rendered_checkerboard():
    cv2 = pytest.importorskip("cv2")
    board = calibrate.Board(8, 5, 0.03)  # even x odd: no 180-degree ambiguity
    K = np.array([[800.0, 0, 320], [0, 800.0, 240], [0, 0, 1]])
    R = Rotation.from_euler("xyz", [0.25, -0.2, 0.1]).as_matrix()
    t = np.array([-0.08, -0.05, 0.6])
    # Render the board's squares by projecting their outer corners.
    img = np.full((480, 640), 255, np.uint8)
    for r in range(-1, board.rows):
        for c in range(-1, board.cols):
            if (r + c) % 2:
                continue
            corners = (
                np.array([[c, r], [c + 1, r], [c + 1, r + 1], [c, r + 1]], float)
                * board.square_m
            )
            pts = np.c_[corners, np.zeros(4)] @ R.T + t
            uv = (K @ pts.T).T
            uv = uv[:, :2] / uv[:, 2:]
            cv2.fillConvexPoly(img, np.round(uv).astype(np.int32), 0)
    intr = {"fx": 800.0, "fy": 800.0, "ppx": 320.0, "ppy": 240.0}
    T, reproj = calibrate.detect_board(img, intr, board)
    assert reproj < 1.0
    # The board frame's origin corner is a convention of the detector; the recovered
    # corners must coincide with the rendered ones (as a set) and lie in the same plane.
    obj = board.object_points()
    got = obj @ T[:3, :3].T + T[:3, 3]
    true = obj @ R.T + t
    gaps = np.linalg.norm(got[:, None, :] - true[None, :, :], axis=-1)
    assert gaps.min(axis=1).max() < 0.003 and gaps.min(axis=0).max() < 0.003
    normal = calibrate.rotation_deg(T[:3, :3], R)
    assert normal < 0.5 or abs(normal - 180) < 0.5
    with pytest.raises(ValueError, match="checkerboard_not_found"):
        calibrate.detect_board(np.zeros((480, 640), np.uint8), intr, board)
    # The same board seen through an inverse-Brown-Conrady lens (RealSense colour
    # streams) solves to the same pose once the corners are undistorted: distort the
    # rendered image's corner set analytically instead of re-rendering.
    coeffs = [0.05, -0.02, 0.001, -0.001, 0.0]
    inv = {**intr, "distortion_model": "inverse_brown_conrady", "coeffs": coeffs}
    und = undistort_pixels(np.array([[100.0, 80.0], [500.0, 400.0]]), inv)
    assert not np.allclose(und, [[100.0, 80.0], [500.0, 400.0]])


# -- async motion start, interruption, control script ----------------------------------


def _ticking_clock(step: float = 0.1):
    clock = {"t": 0.0}

    def tick():
        clock["t"] += step
        return clock["t"]

    return clock, tick


def test_async_phase_waits_for_the_new_operation_id():
    # getAsyncOperationProgressEx: id changes when the move thread starts.
    assert async_phase((4, False), (4, False), False) == "pending"
    assert async_phase((4, False), (5, True), False) == "running"
    assert async_phase((4, False), (5, False), False) == "done"  # a short op
    assert async_phase((127, False), (0, True), False) == "running"  # id wraps
    # Legacy getAsyncOperationProgress (no id): done only after it was seen running.
    assert async_phase((None, False), (None, False), False) == "pending"
    assert async_phase((None, False), (None, True), False) == "running"
    assert async_phase((None, False), (None, False), True) == "done"


def test_a_stale_async_register_does_not_end_the_move_early():
    # ur_rtde returns from moveL(async) before the script's move thread ran; the
    # register still shows the previous (finished) operation. The pre-fix poll
    # (progress >= 0) read that as "done" while the arm was moving.
    probe = MockUrArm((0.5, 0.0, 0.3, *DOWN), stale_polls=3)
    before = probe.async_status()
    probe.move_l((0.55, 0.0, 0.3, *DOWN), 0.25, 0.5)
    assert probe.busy() is False and probe.running, (
        "the stale read the old code trusted"
    )
    assert async_phase(before, probe.async_status(), False) == "pending"

    arm = MockUrArm((0.5, 0.0, 0.3, *DOWN), stale_polls=3)
    f = facade(arm)
    r = call(f, "env.move_delta", [0.05, 0.0, 0.0])
    assert r["ok"] and r["final_error_m"] < 1e-9 and not arm.running
    assert f.controller.target is not None
    # A stop requested while the register is still stale reaches the moving arm.
    arm = MockUrArm((0.5, 0.0, 0.3, *DOWN), stale_polls=3, step_m=0.001)
    f = facade(arm)
    call(f, "env.move_delta", [0.0, 0.0, 0.0])
    polls = arm.polls
    f.controller._stop = lambda: arm.polls >= polls + 2
    r = call(f, "env.move_delta", [0.05, 0.0, 0.0])
    assert r["cancelled"] is True and arm.stops == ["stopL"] and not arm.running
    assert f.controller.target is None


def test_a_move_that_never_starts_is_stopped_and_reported():
    clock, tick = _ticking_clock()
    arm = MockUrArm((0.5, 0.0, 0.3, *DOWN), stale_polls=10**6, step_m=0.0)
    c = UR5eController(
        arm, None, limits_from_config(cfg()), sleep=lambda s: None, clock=tick
    )
    r = c.move_delta([0.01, 0.0, 0.0])
    assert r["ok"] is False and r["not_started"] is True and "timed_out" not in r
    assert arm.stops == ["stopL"] and c.target is None
    assert clock["t"] < 2.0, "gave up after start_timeout_s, not move_timeout_s"


def test_a_protective_stop_mid_move_ends_the_wait_at_once():
    # The script is paused: the async register keeps reading "running". The pre-fix
    # loop spun until move_timeout_s and reported timed_out.
    clock, tick = _ticking_clock()
    arm = MockUrArm((0.5, 0.0, 0.3, *DOWN), protective_stop_after=3)
    c = UR5eController(
        arm, None, limits_from_config(cfg()), sleep=lambda s: None, clock=tick
    )
    r = c.move_delta([0.05, 0.0, 0.0])
    assert r["ok"] is False and r["interrupted"] == "safety_stop"
    assert r["protective_stopped"] is True and "timed_out" not in r
    assert "protective stopped" in r["note"] and clock["t"] < 2.0
    assert arm.running, "the register still read running: the flag did not end it"


def test_a_stopped_control_script_is_reported_and_reuploaded_on_the_next_command():
    clock, tick = _ticking_clock()
    # An unreachable target: the controller's IK fails and the script halts
    # (moveL(async) had already returned True).
    arm = MockUrArm((0.5, 0.0, 0.3, *DOWN), script_stop_after=2)
    c = UR5eController(
        arm, None, limits_from_config(cfg()), sleep=lambda s: None, clock=tick
    )
    r = c.move_delta([0.05, 0.0, 0.0])
    assert r["ok"] is False and r["interrupted"] == "control_script_stopped"
    assert "timed_out" not in r and "unreachable" in r["note"] and clock["t"] < 2.0
    assert c.target is None
    # The next command re-uploads the script instead of needing a server restart.
    arm.script_stop_after = None
    r = c.move_delta([0.01, 0.0, 0.0])
    assert r["ok"] and r["control_script_reuploaded"] is True and arm.reuploads == 1
    # When the script cannot come back (remote control off), the command is refused.
    arm.program_running, arm.reupload_fails = False, True
    moves = len(arm.moves)
    with pytest.raises(RuntimeError, match="could not be re-uploaded.*nothing was"):
        c.move_delta([0.01, 0.0, 0.0])
    assert len(arm.moves) == moves


def test_other_safety_modes_refuse_motion_too():
    arm = MockUrArm((0.5, 0.0, 0.3, *DOWN))
    f = facade(arm)

    def status(robot_mode, safety_mode):
        return lambda: {
            "robot_mode": robot_mode,
            "safety_mode": safety_mode,
            "protective_stopped": False,
            "emergency_stopped": False,
            "program_running": True,
        }

    arm.status = status(7, 5)  # a safeguard stop, no protective flag
    with pytest.raises(RuntimeError, match=r"safety mode 5 \(safeguard stop\)"):
        call(f, "env.move_delta", [0.01, 0.0, 0.0])
    arm.status = status(5, 1)  # powered but idle, brakes engaged
    with pytest.raises(RuntimeError, match=r"robot mode 5 \(idle"):
        call(f, "env.move_delta", [0.01, 0.0, 0.0])
    assert arm.moves == []


# -- gripper settle and empty-grasp threshold ------------------------------------------


def test_gripper_waits_for_the_position_to_settle_when_obj_lags():
    # POS moves while OBJ still shows the previous motion's 3: the pre-fix check
    # settled mid-motion and reported a half-closed width without the grasp.
    grip = MockRobotiq(object_pos=150, moving_polls=6, obj_lag=True)
    f = facade(gripper=grip)
    r = call(f, "env.set_gripper", open=False)
    assert r["ok"] and r["object_detected"] and grip._pending is None
    assert r["gripper_width_m"] == pytest.approx(0.085 * (1 - 150 / 255))
    # An OBJ that disagrees with the command (contact-while-closing during an open)
    # is not taken as settled either.
    grip = MockRobotiq(position=150, object_pos=150, moving_polls=4, obj_lag=True)
    grip.obj = 2
    f = facade(gripper=grip)
    r = call(f, "env.set_gripper", open=True)
    assert r["ok"] and grip.pos == 0 and r["gripper_width_m"] == pytest.approx(0.085)


def test_empty_grasp_threshold_follows_the_2f85_closed_width():
    # A real 2F-85 closed on nothing stops at POS ~228 (9 mm), above the old 5 mm.
    assert limits_from_config(cfg()).empty_width_m == pytest.approx(0.011, abs=2e-4)
    assert EMPTY_WIDTH_FRACTION * 0.085 > 0.085 * (1 - 227 / 255)
    grip = MockRobotiq(closed_pos=228)
    f = facade(gripper=grip)
    r = call(f, "env.set_gripper", open=False)
    assert r["grasp_empty"] is True and r["ok"] is False and grip.pos == 0
    # A thin object (6 mm) still grasps: contact was reported.
    thin = MockRobotiq(object_pos=int(255 * (1 - 0.006 / 0.085)))
    r = call(facade(gripper=thin), "env.set_gripper", open=False)
    assert r["ok"] and r["object_detected"] and not r.get("grasp_empty")
    # Scaled to the stroke (2F-140), explicit values win, null disables.
    big = limits_from_config(cfg(gripper={"poll_s": 0.0, "stroke_m": 0.14}))
    assert big.empty_width_m == pytest.approx(round(0.13 * 0.14, 4))
    explicit = limits_from_config(cfg(gripper={"empty_width_m": 0.02}))
    assert explicit.empty_width_m == 0.02
    off = limits_from_config(cfg(gripper={"empty_width_m": None}))
    assert off.empty_width_m is None
    with pytest.raises(ValueError, match="empty_width_m must be in"):
        limits_from_config(cfg(gripper={"empty_width_m": 0.08}))


def test_old_sample_dirs_with_the_prefixed_distortion_model_load(tmp_path):
    sdir = tmp_path / "samples"
    sdir.mkdir()
    (sdir / "intrinsics.json").write_text(
        json.dumps(
            {
                "width": 640,
                "height": 480,
                "fx": 615.0,
                "fy": 615.0,
                "ppx": 320.0,
                "ppy": 240.0,
                "distortion_model": "distortion.inverse_brown_conrady",
                "coeffs": [0.1, -0.2, 0.001, 0.001, 0.05],
            }
        )
    )
    intr = calibrate.load_intrinsics(sdir / "intrinsics.json")
    assert intr["distortion_model"] == "inverse_brown_conrady"
    assert intr["coeffs"][0] == 0.1
