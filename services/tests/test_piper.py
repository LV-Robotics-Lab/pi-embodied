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

"""Piper step math and safety limits against a mocked ROS transport (no ROS, no arm)."""

from __future__ import annotations

import threading
import time

import numpy as np
import pytest

from pi_embodied_services.robots.piper.controller import PiperController, PiperLimits
from pi_embodied_services.robots.piper.env_server import (
    DEFAULT_CONFIG,
    PiperEnvFacade,
    limits_from_config,
    load_config,
    resize_with_pad,
)
from pi_embodied_services.robots.piper.kinematics import PiperKinematics

#: Show-Harness's left-rig begin pose (clear of joint limits and the j5=0 singularity).
HOME = np.array([-0.44299, 0.85654, -0.9814, 0.02017, 0.92179, 0.0])


class FakeArm:
    """The ``ros_io.PiperRosArm`` surface, synchronous and FK-consistent.

    ``drop`` makes it ignore commands (a node not in mode 1); ``pose_offset`` makes its
    pose feedback disagree with FK (an unknown DH model); ``object_width`` is where a
    closing gripper stops (None = nothing between the fingers).
    """

    def __init__(self, drop=False, pose_offset=0.0, object_width=None, width=0.07):
        self.kin = PiperKinematics(0x01)
        self.q = HOME.copy()
        self.pose = self.kin.fk_pose7(self.q)
        self.width = width
        self.drop = drop
        self.pose_offset = pose_offset
        self.object_width = object_width
        self.streamed = 0
        self.poses: list[np.ndarray] = []
        self.noted = None
        self.closed = False

    def get_ee_pose(self):
        p = self.pose.copy()
        p[0] += self.pose_offset
        return p

    def get_joint_positions(self):
        return self.q.copy()

    def get_gripper_width(self):
        return self.width

    def command_pose(self, pose7):
        self.poses.append(np.asarray(pose7, float))
        if not self.drop:
            self.pose = np.asarray(pose7, float).copy()

    def stream_joints(self, q):
        self.streamed += 1
        if not self.drop:
            self.q = np.asarray(q, float)[:6].copy()
            self.pose = self.kin.fk_pose7(self.q)

    def set_gripper_width(self, w):
        if self.drop:
            return
        if w <= 0.0 and self.object_width is not None:
            self.width = self.object_width
        else:
            self.width = float(w)

    def note_commanded_pose(self, pose7):
        self.noted = pose7

    def close(self):
        self.closed = True


def limits(**kw) -> PiperLimits:
    base = dict(
        z_floor_m=0.0,
        speed_mps=2.0,
        yaw_speed_radps=10.0,
        settle_steps=1,
        settle_dt_s=0.0,
        gripper_settle_s=0.0,
        reset_time_s=0.1,
    )
    base.update(kw)
    return PiperLimits(**base)


def controller(arm=None, stop=lambda: False, **kw) -> PiperController:
    c = PiperController(arm or FakeArm(), limits(**kw), stop, sleep=lambda s: None)
    c.sync()
    return c


def test_step_refuses_oversized_translation_and_yaw():
    c = controller(max_step_m=0.05, max_yaw_rad=0.2)
    before = c.robot.pose.copy()
    with pytest.raises(ValueError, match="limit is 0.05 m per call"):
        c.step([0.04, 0.04, 0.0])
    with pytest.raises(ValueError, match="exceeds the limit"):
        c.step(yaw=0.25)
    with pytest.raises(ValueError, match="finite"):
        c.step([float("nan"), 0, 0])
    with pytest.raises(ValueError, match="gripper"):
        c.step(gripper="half")
    assert c.robot.streamed == 0
    np.testing.assert_allclose(c.robot.pose, before)


def test_z_floor_is_required_unless_disabled():
    with pytest.raises(ValueError, match="z_floor_m is not calibrated"):
        PiperLimits().validate()
    PiperLimits(enable_z_floor=False).validate()
    with pytest.raises(ValueError, match="both"):
        PiperLimits(z_floor_m=0.1, workspace_min=[0, 0, 0]).validate()


def test_joint_stream_reaches_a_base_frame_step():
    c = controller()
    assert c.backend == "joint_stream"
    start = c.robot.pose[:3].copy()
    out = c.step([0.02, 0.0, 0.0])
    assert out["ok"] and not out["notes"], out["notes"]
    np.testing.assert_allclose(c.robot.pose[:3], start + [0.02, 0, 0], atol=1e-3)
    assert c.robot.streamed > 2, "moved by streaming joint waypoints"
    assert out["moved_m"] == pytest.approx(0.02, abs=1e-3)


def test_heading_frame_rotates_the_delta_by_the_gripper_heading():
    c = controller()
    heading = c.heading_yaw()
    assert heading is not None
    out = c.step([0.02, 0.0, 0.0], frame="heading")
    expect = [0.02 * np.cos(heading), 0.02 * np.sin(heading), 0.0]
    np.testing.assert_allclose(out["delta_xyz_base"], expect, atol=1e-9)
    np.testing.assert_allclose(
        np.subtract(out["post_pose"][:3], out["pre_pose"][:3]), expect, atol=1e-3
    )


def test_z_floor_blocks_descent():
    arm = FakeArm()
    floor = float(arm.pose[2]) - 0.01
    c = controller(arm, z_floor_m=floor)
    out = c.step([0.0, 0.0, -0.03])
    assert any(n.startswith("z-floor: blocked 0.0200") for n in out["notes"])
    assert c.robot.pose[2] == pytest.approx(floor, abs=1e-3)
    assert c.target_pose[2] == pytest.approx(floor)


def test_workspace_box_clamps_the_setpoint():
    arm = FakeArm()
    x, y, z = arm.pose[:3]
    c = controller(
        arm, workspace_min=[x - 1, y - 1, z - 1], workspace_max=[x + 0.01, y + 1, z + 1]
    )
    out = c.step([0.03, 0.0, 0.0])
    assert "workspace: clamped x to the box" in out["notes"]
    assert c.robot.pose[0] == pytest.approx(x + 0.01, abs=1e-3)


def test_fk_mismatch_falls_back_to_endpose():
    arm = FakeArm(pose_offset=0.05)
    c = PiperController(arm, limits(), sleep=lambda s: None)
    notes = c.sync()
    assert c.backend == "endpose"
    assert "joint_stream disabled" in notes[0]
    c.step([0.0, 0.0, 0.02])
    assert arm.poses and arm.streamed == 0


def test_dropped_commands_resync_the_setpoint():
    arm = FakeArm(drop=True)
    c = controller(arm, motion_backend="endpose", divergence_resync_m=0.01)
    start = c.target_pose.copy()
    c.step([0.0, 0.0, 0.02])
    out = c.step([0.0, 0.0, 0.02])
    assert not out["ok"]
    assert any(n.startswith("divergence") for n in out["notes"])
    np.testing.assert_allclose(c.target_pose[:3], start[:3], atol=1e-9)


def test_stop_halts_the_joint_stream_between_waypoints():
    arm = FakeArm()
    c = controller(arm, stop=lambda: arm.streamed >= 3, speed_mps=0.05)
    start = arm.pose[:3].copy()
    out = c.step([0.04, 0.0, 0.0])
    assert out["cancelled"] and not out["ok"] and "stopped" in out["notes"]
    moved = float(np.linalg.norm(arm.pose[:3] - start))
    assert 0 < moved < 0.01, moved
    # The setpoint restarts from where the arm is, not where it was headed.
    np.testing.assert_allclose(c.target_pose[:3], arm.pose[:3], atol=1e-9)


def test_gripper_hold_empty_grasp_and_dropped_command():
    holding = controller(FakeArm(object_width=0.03))
    out = holding.step(gripper="close")
    assert out["ok"] and out["gripper_closed"] and out["gripper_width_m"] == 0.03

    empty = controller(FakeArm())
    out = empty.step(gripper="close")
    assert any(n.startswith("empty grasp") for n in out["notes"])
    assert out["gripper_closed"] is False and out["gripper_width_m"] == 0.07

    left = controller(FakeArm()).step(gripper="close", reopen_empty=False)
    assert left["gripper_closed"] and left["gripper_width_m"] == 0.0
    assert any(n.endswith("(left closed)") for n in left["notes"])

    dropped = controller(FakeArm(drop=True))
    out = dropped.step(gripper="close")
    assert not out["ok"] and any("likely dropped" in n for n in out["notes"])

    already = controller(FakeArm())
    assert already.step(gripper="open")["notes"] == ["gripper already open"]


def test_move_to_joints_checks_limits_and_resyncs():
    arm = FakeArm()
    c = controller(arm)
    with pytest.raises(ValueError, match="joint limits"):
        c.move_to_joints([0, -1, 0, 0, 0, 0])
    goal = HOME + [0.05, 0.02, -0.02, 0.0, 0.03, 0.0]
    out = c.move_to_joints(goal)
    assert out["ok"], out
    np.testing.assert_allclose(arm.q, goal, atol=1e-9)
    np.testing.assert_allclose(c.target_pose, arm.pose, atol=1e-9)


def test_example_config_fails_closed_until_calibrated(tmp_path):
    with pytest.raises(ValueError, match="z_floor_m is not calibrated"):
        load_config(DEFAULT_CONFIG)
    text = DEFAULT_CONFIG.read_text().replace("z_floor_m: null", "z_floor_m: 0.19")
    path = tmp_path / "piper.yaml"
    path.write_text(text)
    cfg = load_config(path)
    lim = limits_from_config(cfg)
    assert lim.z_floor_m == 0.19 and lim.max_step_m == 0.05
    assert lim.open_width_m == 0.07 and lim.gripper_settle_s == 1.5
    assert lim.ori_flex_rad == pytest.approx(np.radians(15))
    assert "banana_plate" in cfg["tasks"]


class FakeCamera:
    def read(self):
        return np.full((48, 64, 3), 7, dtype=np.uint8)

    def close(self):
        pass


def facade(arm, tmp_path, **cfg_limits):
    cfg = {
        "arm": "left",
        "cameras": {"front": "/f", "wrist": "/w", "image_size": 32},
        "calibration": {"z_floor_m": 0.0, "begin_joints": HOME.tolist()},
        "limits": {"speed_mps": 0.05, "reset_time_s": 0.1, **cfg_limits},
        "gripper": {"settle_s": 0.0},
        "motion": {"settle_steps": 1, "settle_dt_s": 0.0},
    }
    return PiperEnvFacade(cfg, arm, {"front": FakeCamera(), "wrist": FakeCamera()})


def test_facade_step_observation_and_lock_free_stop(tmp_path):
    arm = FakeArm()
    f = facade(arm, tmp_path)
    call = f._serve_dispatch
    assert call("env.get_env_meta", (), {})["limits"]["max_step_m"] == 0.05
    obs = call("env.get_observation", (), {})
    assert obs["images"]["front"].shape == (32, 32, 3)
    assert obs["robot_state"]["motion_backend"] == "joint_stream"
    with pytest.raises(ValueError, match="per call"):
        call("env.step", (), {"delta_xyz": [0.1, 0, 0]})

    # A 4 cm move at 5 cm/s takes ~0.8 s; `stop` bypasses the call lock and halts it.
    result = {}
    worker = threading.Thread(
        target=lambda: result.update(call("env.step", (), {"delta_xyz": [0.04, 0, 0]}))
    )
    worker.start()
    time.sleep(0.2)
    assert call("stop", (), {})["call_in_progress"] is True
    worker.join(5)
    assert result["cancelled"] is True
    assert result["moved_m"] < 0.035

    reset = call("env.reset", (), {})
    assert reset["ok"], reset
    np.testing.assert_allclose(arm.q, HOME, atol=1e-9)
    f.close()
    assert arm.closed


def test_resize_with_pad_letterboxes():
    img = np.ones((48, 64, 3), dtype=np.uint8)
    out = resize_with_pad(img, 32)
    assert out.shape == (32, 32, 3)
    assert out[0].sum() == 0 and out[16].sum() == 32 * 3
