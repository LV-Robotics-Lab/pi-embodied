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
import yaml
from scipy.spatial.transform import Rotation as Rot

from pi_embodied_services.robots.piper.controller import PiperController, PiperLimits
from pi_embodied_services.robots.piper.env_server import (
    DEFAULT_CONFIG,
    PiperEnvFacade,
    arm_config,
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

    def __init__(
        self,
        drop=False,
        pose_offset=0.0,
        object_width=None,
        width=0.07,
        ident="usb-A",
        drop_pose=False,
    ):
        self.ident = ident
        self.kin = PiperKinematics(0x01)
        self.q = HOME.copy()
        self.pose = self.kin.fk_pose7(self.q)
        self.width = width
        self.drop = drop
        #: Ignore MOVE P only (a reach fallback that does not move the arm).
        self.drop_pose = drop_pose
        self.rot_log: list[np.ndarray] = []
        self.pose_offset = pose_offset
        self.object_width = object_width
        self.streamed = 0
        self.stream_log: list[np.ndarray] = []
        self.fail_at_stream: int | None = None
        self.stale = False
        self.poses: list[np.ndarray] = []
        self.noted = None
        self.closed = False

    def _check_fresh(self):
        if self.stale:
            raise RuntimeError("feedback is stale")

    def get_ee_pose(self):
        self._check_fresh()
        p = self.pose.copy()
        p[0] += self.pose_offset
        return p

    def get_joint_positions(self):
        self._check_fresh()
        return self.q.copy()

    def get_gripper_width(self):
        return self.width

    def command_pose(self, pose7):
        self.poses.append(np.asarray(pose7, float))
        if not (self.drop or self.drop_pose):
            self.pose = np.asarray(pose7, float).copy()

    def stream_joints(self, q):
        if self.fail_at_stream is not None and self.streamed >= self.fail_at_stream:
            raise RuntimeError("feedback on /puppet/joint_left is 0.80s old")
        self.streamed += 1
        pose = self.kin.fk_pose7(np.asarray(q, float)[:6])
        self.stream_log.append(pose[:3])
        self.rot_log.append(pose[3:])
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

    def identity(self):
        return self.ident

    def close(self):
        self.closed = True


def limits(**kw) -> PiperLimits:
    base = dict(
        z_floor_m=0.0,
        speed_mps=0.3,
        yaw_speed_radps=2.0,
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
    # 15 of the 50 min-jerk waypoints of a 1 s move: ~16 % of the way.
    c = controller(arm, stop=lambda: arm.streamed >= 15, speed_mps=0.05)
    start = arm.pose[:3].copy()
    out = c.step([0.04, 0.0, 0.0])
    assert out["cancelled"] and not out["ok"] and "stopped" in out["notes"]
    moved = float(np.linalg.norm(arm.pose[:3] - start))
    assert 0 < moved < 0.01, moved
    # The setpoint restarts from where the arm is, not where it was headed.
    np.testing.assert_allclose(c.target_pose[:3], arm.pose[:3], atol=1e-9)


def assert_next_step_starts_from_measured(c, arm):
    """The next 1 cm step streams only near the measured pose, never the old target."""
    arm.fail_at_stream = None
    start = arm.pose[:3].copy()
    arm.stream_log.clear()
    out = c.step([0.0, 0.0, 0.01])
    assert out["ok"], out["notes"]
    far = max(float(np.linalg.norm(p - start)) for p in arm.stream_log)
    assert far < 0.011, far
    np.testing.assert_allclose(arm.pose[:3], start + [0, 0, 0.01], atol=1e-3)


def test_failure_mid_step_invalidates_the_setpoint():
    arm = FakeArm()
    c = controller(arm, speed_mps=0.05)
    old_target = arm.pose[:3] + [0.04, 0.0, 0.0]
    arm.fail_at_stream = 5
    with pytest.raises(RuntimeError, match="old"):
        c.step([0.04, 0.0, 0.0])
    assert c._target_pos is None and c._q_cmd is None and arm.noted is None
    assert np.linalg.norm(arm.pose[:3] - old_target) > 0.03
    assert_next_step_starts_from_measured(c, arm)


def test_failure_mid_reset_invalidates_the_setpoint():
    arm = FakeArm()
    c = controller(arm, reset_time_s=1.0)
    home = arm.pose[:3].copy()
    goal = HOME + [0.3, 0.2, -0.2, 0.0, 0.2, 0.0]
    arm.fail_at_stream = 20
    with pytest.raises(RuntimeError, match="old"):
        c.move_to_joints(goal)
    assert np.linalg.norm(arm.pose[:3] - home) > 0.02, "stopped away from the old pose"
    assert_next_step_starts_from_measured(c, arm)


def test_no_motion_after_a_failure_while_feedback_is_stale():
    arm = FakeArm()
    c = controller(arm)
    arm.fail_at_stream = 2
    with pytest.raises(RuntimeError):
        c.step([0.02, 0.0, 0.0])
    arm.fail_at_stream, arm.stale = None, True
    streamed = arm.streamed
    with pytest.raises(RuntimeError, match="stale"):
        c.step([0.02, 0.0, 0.0])
    assert arm.streamed == streamed and not arm.poses
    assert c._target_pos is None


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
    with pytest.raises(ValueError, match="calibration.arm_id is not set"):
        load_config(path)
    path.write_text(text.replace("arm_id: null", "arm_id: usb-A"))
    cfg = load_config(path)
    lim = limits_from_config(cfg)
    assert lim.z_floor_m == 0.19 and lim.max_step_m == 0.05
    assert lim.open_width_m == 0.07 and lim.gripper_settle_s == 1.5
    assert lim.ori_flex_rad == pytest.approx(np.radians(15))
    assert "banana_plate" in cfg["tasks"]
    assert lim.smooth and lim.smooth_substeps * lim.smooth_dt_s == pytest.approx(1.0)
    assert lim.max_total_yaw_rad == pytest.approx(np.radians(150), abs=1e-3)


class FakeCamera:
    def read(self):
        return np.full((48, 64, 3), 7, dtype=np.uint8)

    def close(self):
        pass


def facade(arm, tmp_path, **cfg_limits):
    cfg = {
        "arm": "left",
        "cameras": {"front": "/f", "wrist": "/w", "image_size": 32},
        "calibration": {
            "z_floor_m": 0.0,
            "begin_joints": HOME.tolist(),
            "arm_id": "usb-A",
        },
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

    # A smooth 4 cm move takes 1.0 s; `stop` bypasses the call lock and halts it.
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


# -- dual arm (the Show-Harness Cobot Magic rig) ------------------------------------

DUAL_CONFIG = DEFAULT_CONFIG.with_name("dual_example.yaml")


def dual_facade(left, right, per_arm=None):
    """Both arms behind one facade; ``per_arm[side]`` overrides that arm's block."""
    per_arm = per_arm or {}
    arms = {
        side: {
            "cameras": {"wrist": f"/camera_{side[0]}/color/image_raw"},
            "calibration": {
                "z_floor_m": 0.0,
                "begin_joints": HOME.tolist(),
                "arm_id": f"usb-{side}",
            },
            **per_arm.get(side, {}),
        }
        for side in ("left", "right")
    }
    cfg = {
        "cameras": {"front": "/camera_f/color/image_raw", "image_size": 32},
        "limits": {"speed_mps": 0.3, "reset_time_s": 0.1},
        "gripper": {"settle_s": 0.0},
        "motion": {"settle_steps": 1, "settle_dt_s": 0.0, "units_frame": "base"},
        "smooth": {"dt_s": 0.001},
        "arms": arms,
    }
    cams = {n: FakeCamera() for n in ("front", "wrist_left", "wrist_right")}
    left.ident, right.ident = "usb-left", "usb-right"
    return PiperEnvFacade(cfg, {"left": left, "right": right}, cams)


def test_dual_example_config_fails_closed_per_arm(tmp_path):
    with pytest.raises(ValueError, match="left arm: z_floor_m is not calibrated"):
        load_config(DUAL_CONFIG)
    text = DUAL_CONFIG.read_text()
    once = text.replace("z_floor_m: null", "z_floor_m: 0.19", 1)
    path = tmp_path / "dual.yaml"
    path.write_text(once)
    with pytest.raises(ValueError, match="left arm: calibration.arm_id is not set"):
        load_config(path)
    once = once.replace("arm_id: null", "arm_id: usb-left", 1)
    path.write_text(once)
    with pytest.raises(ValueError, match="right arm: z_floor_m is not calibrated"):
        load_config(path)
    both = once.replace("z_floor_m: null", "z_floor_m: 0.21", 1)
    path.write_text(both)
    with pytest.raises(ValueError, match="right arm: calibration.arm_id is not set"):
        load_config(path)
    path.write_text(both.replace("arm_id: null", "arm_id: usb-right", 1))
    cfg = load_config(path)
    left, right = arm_config(cfg, "left"), arm_config(cfg, "right")
    assert left["arm"] == "left" and right["arm"] == "right"
    assert limits_from_config(left).z_floor_m == 0.19
    assert limits_from_config(right).z_floor_m == 0.21
    # Shared keys stay, per-arm keys override section by section.
    assert left["cameras"] == {
        **cfg["cameras"],
        "wrist": "/camera_l/color/image_raw",
    }
    assert right["cameras"]["wrist"] == "/camera_r/color/image_raw"
    assert limits_from_config(right).max_step_m == 0.05
    assert "banana_handover" in cfg["tasks"]
    with pytest.raises(ValueError, match="unknown keys"):
        arm_config({"arms": {"left": {"z_floor_m": 0.1}, "right": {}}}, "left")
    with pytest.raises(ValueError, match="needs a `left` and a `right`"):
        arm_config({"arms": {"left": {}}}, "left")


def test_dual_facade_meta_observation_and_per_arm_state():
    f = dual_facade(FakeArm(), FakeArm())
    call = f._serve_dispatch
    meta = call("env.get_env_meta", (), {})
    assert meta["arms"] == ["left", "right"] and meta["arm"] == "dual"
    assert meta["cameras"] == ["front", "wrist_left", "wrist_right"]
    assert meta["has_begin_pose"] and meta["arm_limits"]["right"]["z_floor_m"] == 0.0
    obs = call("env.get_observation", (), {})
    assert set(obs["images"]) == {"front", "wrist_left", "wrist_right"}
    assert set(obs["robot_state"]["arms"]) == {"left", "right"}
    state = call("env.get_robot_state", (), {"arm": "right"})
    assert state["arm"] == "right" and state["halted"] is None
    topics = call("env.get_camera_meta", (), {})["cameras"]
    assert topics["wrist_right"]["topic"] == "/camera_r/color/image_raw"
    with pytest.raises(ValueError, match="pass arm="):
        call("env.step", (), {"delta_xyz": [0.01, 0, 0]})


def test_dual_step_moves_only_the_named_arm_within_its_own_limits():
    left, right = FakeArm(), FakeArm()
    floor = float(right.pose[2]) - 0.01
    cal = {"z_floor_m": floor, "begin_joints": HOME.tolist(), "arm_id": "usb-right"}
    f = dual_facade(left, right, {"right": {"calibration": cal}})
    call = f._serve_dispatch
    l0 = left.pose.copy()
    out = call("env.step", (), {"delta_xyz": [0.0, 0.0, -0.03], "arm": "right"})
    assert out["arm"] == "right"
    assert any(n.startswith("z-floor: blocked 0.0200") for n in out["notes"])
    assert right.pose[2] == pytest.approx(floor, abs=1e-3)
    np.testing.assert_allclose(left.pose, l0)  # the left arm stays still
    # The left arm's own floor (0.0) lets the same descent through.
    out = call("env.step", (), {"delta_xyz": [0.0, 0.0, -0.03], "arm": "left"})
    assert not out["notes"] and left.pose[2] == pytest.approx(l0[2] - 0.03, abs=1e-3)
    assert right.pose[2] == pytest.approx(floor, abs=1e-3)
    with pytest.raises(ValueError, match="per call"):
        call("env.step", (), {"delta_xyz": [0.1, 0, 0], "arm": "left"})


def test_dual_per_arm_workspace_box():
    left, right = FakeArm(), FakeArm()
    x = float(left.pose[0])
    box = {"workspace_min": [x - 1, -1, -1], "workspace_max": [x + 0.005, 1, 1]}
    cal = {"z_floor_m": 0.0, "begin_joints": HOME.tolist(), "arm_id": "usb-left"}
    f = dual_facade(left, right, {"left": {"calibration": cal, "limits": box}})
    out = f.step([0.03, 0, 0], arm="left")
    assert "workspace: clamped x to the box" in out["notes"]
    out = f.step([0.03, 0, 0], arm="right")
    assert not out["notes"], "the right arm has no box"


def test_dual_fault_halts_one_arm_until_its_reset():
    left, right = FakeArm(drop=True), FakeArm()
    f = dual_facade(left, right)
    for c in f._controllers.values():
        c.limits.divergence_resync_m = 0.01
    out = f.step([0.0, 0.0, 0.02], arm="left")
    assert out["halted"] and any(n.startswith("divergence") for n in out["notes"])
    with pytest.raises(RuntimeError, match="the left arm is halted .*divergence"):
        f.step([0.0, 0.0, 0.01], arm="left")
    # The right arm keeps working.
    assert f.step([0.0, 0.0, 0.01], arm="right")["ok"]
    assert f.get_env_meta()["halted"].keys() == {"left"}
    # A stale-feedback error halts the right arm too; reset clears one arm at a time.
    right.fail_at_stream = right.streamed
    with pytest.raises(RuntimeError, match="old"):
        f.step([0.0, 0.0, 0.01], arm="right")
    with pytest.raises(RuntimeError, match="the right arm is halted .*error"):
        f.step([0.0, 0.0, 0.01], arm="right")
    right.fail_at_stream = None
    assert f.reset(arm="right")["ok"]
    assert f.step([0.0, 0.0, 0.01], arm="right")["ok"]
    assert "left" in f.get_env_meta()["halted"]


def test_dual_halt_arm_and_reset_both():
    left, right = FakeArm(), FakeArm()
    f = dual_facade(left, right)
    f.step([0.02, 0.0, 0.0], arm="left")
    assert f.halt_arm(arm="left", reason="stage done")["halted"] == "halted: stage done"
    with pytest.raises(RuntimeError, match="halted: stage done"):
        f.step([0.01, 0.0, 0.0], arm="left")
    out = f.reset(both=True)
    assert out["ok"] and set(out["arms"]) == {"left", "right"}
    np.testing.assert_allclose(left.q, HOME, atol=1e-9)
    np.testing.assert_allclose(right.q, HOME, atol=1e-9)
    assert not f.get_env_meta()["halted"]
    f.close()
    assert left.closed and right.closed


def test_single_arm_facade_keeps_its_flat_protocol(tmp_path):
    arm = FakeArm(drop=True)
    f = facade(arm, tmp_path, divergence_resync_m=0.01)
    f._controllers["left"].limits.smooth_dt_s = 0.001
    meta = f.get_env_meta()
    assert meta["arm"] == "left" and meta["arms"] == []
    assert "arm" not in f.get_robot_state()
    f.step([0.0, 0.0, 0.02])
    f.step([0.0, 0.0, 0.02])
    # One arm: a divergence is reported, not latched (the agent is told to finish).
    f.step([0.0, 0.0, 0.01], arm="left")
    with pytest.raises(ValueError, match="not driven here"):
        f.step([0.0, 0.0, 0.01], arm="right")


# -- smooth joint stream (Show-Harness plugins/smooth) ------------------------------


def test_smooth_move_follows_the_min_jerk_profile_at_the_stream_rate():
    arm = FakeArm()
    c = controller(arm)
    assert c.limits.smooth
    fr, dt = c.plan(0.02, 0.0)
    # 1.0 s per move (20 x 0.05 s), resampled at joint_stream_hz = 50.
    assert len(fr) == 50 and dt == pytest.approx(0.02)
    assert fr[-1] == 1.0 and all(b >= a for a, b in zip(fr, fr[1:]))
    t = 0.5
    assert fr[24] == pytest.approx(10 * t**3 - 15 * t**4 + 6 * t**5)
    # A long move is stretched until its peak stays within smooth_max_speed_mps.
    fr, dt = c.plan(0.2, 0.0)
    peak = max(np.diff([0.0, *fr])) * 0.2 / dt
    assert peak <= c.limits.smooth_max_speed_mps + 1e-3 and len(fr) * dt > 1.0
    start = arm.pose[:3].copy()
    out = c.step([0.0, 0.02, 0.0])
    assert out["ok"] and not out["chained"] and not out["flowing"]
    np.testing.assert_allclose(arm.pose[:3], start + [0, 0.02, 0], atol=1e-3)
    # Min-jerk: the first streamed waypoints barely move, the middle ones the most.
    gaps = np.linalg.norm(np.diff(np.array([start, *arm.stream_log]), axis=0), axis=1)
    assert gaps[0] < 0.1 * gaps.max() and np.argmax(gaps) in range(15, 35)


def test_smooth_off_is_the_constant_rate_stream():
    c = controller(smooth=False, speed_mps=0.05)
    fr, dt = c.plan(0.02, 0.0)
    assert len(fr) * dt == pytest.approx(0.4) and np.allclose(np.diff(fr), fr[0])


def test_continuous_translations_chain_at_cruise_speed():
    now = [0.0]
    arm = FakeArm()
    c = PiperController(arm, limits(), sleep=lambda s: None, clock=lambda: now[0])
    c.sync()
    first = c.step([0.02, 0.0, 0.0], continuous=True)
    assert first["flowing"] and not first["chained"]
    settles = arm.streamed
    second = c.step([0.02, 0.002, 0.0], continuous=True)  # cos > 0.9: same direction
    assert second["chained"] and second["flowing"]
    # Chained: no settle re-commands at the join, and the first waypoint is at cruise.
    assert arm.streamed - settles == 50
    third = c.step([0.02, 0.0, 0.0])
    assert third["chained"] and not third["flowing"]
    # A turn in direction ends the stream first (settles), then starts at rest.
    c.step([0.02, 0.0, 0.0], continuous=True)
    turn = c.step([0.0, 0.02, 0.0], continuous=True)
    assert not turn["chained"] and turn["flowing"]
    # The window passes: the next move starts at rest.
    now[0] += 5.0
    late = c.step([0.0, 0.02, 0.0])
    assert not late["chained"]
    # A yaw or the gripper never chains, and brings a flowing stream to rest.
    c.step([0.02, 0.0, 0.0], continuous=True)
    grip = c.step(gripper="close", continuous=True)
    assert not grip["flowing"]
    c.step([0.02, 0.0, 0.0], continuous=True)
    yawed = c.step([0.01, 0.0, 0.0], yaw=0.05, continuous=True)
    assert not yawed["chained"] and not yawed["flowing"]


def test_stop_ends_a_chained_stream():
    arm = FakeArm()
    stop = [False]
    c = controller(arm, stop=lambda: stop[0])
    c.step([0.02, 0.0, 0.0], continuous=True)
    stop[0] = True
    out = c.step([0.02, 0.0, 0.0], continuous=True)
    assert out["cancelled"] and not out["flowing"]
    stop[0] = False
    assert not c.step([0.02, 0.0, 0.0])["chained"]


def test_facade_passes_continuous_and_reports_smooth():
    f = dual_facade(FakeArm(), FakeArm())
    assert f.get_env_meta()["smooth"]["enabled"] is True
    assert f.step([0.02, 0, 0], arm="left", continuous=True)["flowing"]
    assert f.step([0.02, 0, 0], arm="left")["chained"]
    with pytest.raises(ValueError, match="unknown smooth keys"):
        limits_from_config({"smooth": {"speed": 1}, "calibration": {"z_floor_m": 0}})
    with pytest.raises(ValueError, match="smooth_cruise"):
        limits(smooth_cruise=3.0).validate()


# -- audit: yaw budget, reset path, calibration binding ----------------------------


def test_accumulated_yaw_is_capped_from_the_reset_heading():
    arm = FakeArm()
    c = controller(arm, max_yaw_rad=0.2, max_total_yaw_rad=np.radians(30))
    assert c.yaw_from_reset() == pytest.approx(0.0)
    c.step(yaw=0.2)
    c.step(yaw=0.2)
    assert c.yaw_from_reset() == pytest.approx(0.4, abs=1e-3)
    streamed = arm.streamed
    with pytest.raises(
        ValueError, match="would turn the gripper 34 deg .* limit is 30 deg"
    ):
        c.step(yaw=0.2)
    assert arm.streamed == streamed, "nothing was commanded"
    # Turning back is always allowed, and a reset restarts the budget.
    c.step(yaw=-0.2)
    assert c.state()["yaw_from_reset_rad"] == pytest.approx(0.2, abs=1e-3)
    c.step(yaw=0.2)
    c.move_to_joints(HOME)
    assert c.yaw_from_reset() == pytest.approx(0.0, abs=1e-6)
    c.step(yaw=0.2)
    with pytest.raises(ValueError, match="max_total_yaw_rad"):
        PiperLimits(z_floor_m=0.0, max_total_yaw_rad=4.0).validate()


def test_reset_path_respects_the_z_floor_and_the_box():
    arm = FakeArm()
    kin = arm.kin
    low = HOME + [0.0, 0.35, 0.0, 0.0, 0.0, 0.0]
    z_low = float(kin.fk_pose7(low)[2])
    z_home = float(arm.pose[2])
    assert z_low < z_home - 0.02, (z_low, z_home)
    # The goal itself is below the floor: refused before any motion.
    c = controller(arm, z_floor_m=z_low + 0.01)
    with pytest.raises(ValueError, match="at the goal .* outside the Z floor"):
        c.move_to_joints(low)
    assert arm.streamed == 0
    np.testing.assert_allclose(arm.q, HOME)
    # A box that excludes the goal in x/y refuses too; the begin pose inside is fine.
    x, y, z = arm.pose[:3]
    boxed = controller(
        FakeArm(),
        workspace_min=[x - 0.01, y - 0.01, 0.0],
        workspace_max=[x + 0.01, y + 0.01, 1.0],
    )
    far = HOME + [0.3, 0.0, 0.0, 0.0, 0.0, 0.0]
    with pytest.raises(ValueError, match="outside the Z floor / workspace box"):
        boxed.move_to_joints(far)
    assert boxed.move_to_joints(HOME)["ok"]
    # A start below the floor may climb out (the violation never deepens).
    start_low = FakeArm()
    start_low.q = low.copy()
    start_low.pose = kin.fk_pose7(low)
    climb = controller(start_low, z_floor_m=z_low + 0.01)
    assert climb.move_to_joints(HOME)["ok"]


def test_calibration_is_bound_to_the_arm_it_was_captured_on(tmp_path):
    arm = FakeArm(ident="usb-B")
    with pytest.raises(
        ValueError, match="reports identity 'usb-B', but calibration.arm_id is 'usb-A'"
    ):
        facade(arm, tmp_path)
    assert arm.streamed == 0 and not arm.poses
    # Two arms: swapped adapters (or configs) are refused.
    left, right = FakeArm(), FakeArm()
    cfg_right = {
        "calibration": {
            "z_floor_m": 0.0,
            "begin_joints": HOME.tolist(),
            "arm_id": "usb-left",
        }
    }
    with pytest.raises(
        ValueError, match="the right arm reports identity 'usb-right', but arms.right"
    ):
        dual_facade(left, right, {"right": cfg_right})
    # ros.identity: none skips the binding (and reports no id).
    cfg = {
        "arm": "left",
        "ros": {"identity": "none"},
        "cameras": {"front": "/f"},
        "calibration": {"z_floor_m": 0.0},
    }
    f = PiperEnvFacade(cfg, FakeArm(ident="whatever"), {"front": FakeCamera()})
    assert f.get_env_meta()["arm_ids"] == {"left": None}
    ok = facade(FakeArm(), tmp_path)
    assert ok.get_env_meta()["arm_ids"] == {"left": "usb-A"}


def test_identity_sources(tmp_path):
    from pi_embodied_services.robots.piper import ros_io

    usb = tmp_path / "devices" / "usb1" / "1-2"
    (usb / "1-2:1.0").mkdir(parents=True)
    (usb / "serial").write_text("0039004A5553501020313332\n")
    net = tmp_path / "net" / "can_left"
    net.mkdir(parents=True)
    (net / "device").symlink_to(usb / "1-2:1.0")
    assert (
        ros_io.can_usb_serial("can_left", tmp_path / "net")
        == "0039004A5553501020313332"
    )
    with pytest.raises(RuntimeError, match="no CAN interface can_right"):
        ros_io.can_usb_serial("can_right", tmp_path / "net")
    params = {"/piper_left/serial": 1234}
    got = ros_io.arm_identity(
        "left", "param:/piper_left/serial", lambda k, d: params.get(k, d)
    )
    assert got == "1234"
    with pytest.raises(RuntimeError, match="is not set"):
        ros_io.arm_identity("left", "param:/nope", lambda k, d: d)
    assert ros_io.arm_identity("left", "none") is None
    with pytest.raises(ValueError, match="ros.identity must be"):
        ros_io.arm_identity("left", "serial")


# -- review fixes: stale setpoints, per-arm locks, reset scope, config, wrist ramp --


def fail_ik_once(c):
    """Make the first IK call fail (the arm at its reach limit), the rest real."""
    real = c._kin.ik_bounded
    calls = [0]

    def ik(*a, **kw):
        calls[0] += 1
        return (None, 0.0) if calls[0] == 1 else real(*a, **kw)

    c._kin.ik_bounded = ik


def test_reach_fallback_shortfall_fails_and_the_next_step_starts_from_measured():
    # HIGH 1: IK fails -> one MOVE P that does not arrive (5 cm short). The old code kept
    # the setpoint at the target, so the next +z 1 cm step's first waypoint jumped ~5 cm.
    arm = FakeArm(drop_pose=True)
    c = controller(arm)
    fail_ik_once(c)
    start = arm.pose[:3].copy()
    out = c.step([0.0, 0.0, 0.05])
    assert not out["ok"], out["notes"]
    assert any("50.0 mm short of its target" in n for n in out["notes"]), out["notes"]
    np.testing.assert_allclose(arm.pose[:3], start, atol=1e-9)
    np.testing.assert_allclose(c.target_pose[:3], arm.pose[:3], atol=1e-9)
    # A later gripper command re-sends the measured pose, never the unreached one.
    np.testing.assert_allclose(arm.noted[:3], start, atol=1e-9)
    assert_next_step_starts_from_measured(c, arm)
    # A MOVE P that arrives is fine.
    ok = controller(FakeArm())
    fail_ik_once(ok)
    good = ok.step([0.0, 0.0, 0.02])
    assert good["ok"], good["notes"]


def test_dual_reach_fallback_shortfall_halts_that_arm():
    left, right = FakeArm(drop_pose=True), FakeArm()
    f = dual_facade(left, right)
    fail_ik_once(f._controllers["left"])
    out = f.step([0.0, 0.0, 0.05], arm="left")
    assert out["halted"] and not out["ok"]
    with pytest.raises(
        RuntimeError, match="the left arm is halted .*short of its target"
    ):
        f.step([0.0, 0.0, 0.01], arm="left")
    assert f.step([0.0, 0.0, 0.01], arm="right")["ok"]


def test_a_waypoint_jump_is_refused_before_it_is_streamed():
    arm = FakeArm()
    c = controller(arm)
    real = c._kin.ik_bounded
    calls = [0]

    def ik(*a, **kw):
        calls[0] += 1
        sol, dev = real(*a, **kw)
        return (sol + [0.0, 0.1, 0.0, 0.0, 0.0, 0.0] if calls[0] == 3 else sol), dev

    c._kin.ik_bounded = ik
    with pytest.raises(RuntimeError, match="waypoint 3/50 would move the gripper"):
        c.step([0.02, 0.0, 0.0])
    assert arm.streamed == 2, "nothing past the previous waypoint was commanded"
    assert c._target_pos is None
    assert_next_step_starts_from_measured(c, arm)


def test_every_move_starts_from_the_measured_pose_despite_an_fk_offset():
    # Pose feedback 4 mm off the vendored FK (within the 1 cm joint_stream gate): the
    # first waypoint starts at the measured joints, and the arm moves the commanded 2 cm.
    arm = FakeArm(pose_offset=0.004)
    c = controller(arm)
    assert c.backend == "joint_stream"
    fk0 = arm.kin.fk_pose7(arm.q)[:3]
    start = arm.get_ee_pose()[:3]
    out = c.step([0.0, 0.02, 0.0])
    assert out["ok"], out["notes"]
    assert np.linalg.norm(arm.stream_log[0] - fk0) < 1e-3
    np.testing.assert_allclose(arm.get_ee_pose()[:3], start + [0, 0.02, 0], atol=1e-3)


def max_tick_turn(arm, q0):
    rots = [arm.kin.fk_pose7(q0)[3:], *arm.rot_log]
    return max(
        (Rot.from_quat(b) * Rot.from_quat(a).inv()).magnitude()
        for a, b in zip(rots, rots[1:])
    )


def test_a_clipped_yaw_never_snaps_the_wrist():
    # MEDIUM: HOME reaches ~0.53 rad of yaw; the second 0.5 rad turn is clipped at the
    # reach boundary. The old setpoint kept the full 1.0 rad, and the next move's first
    # waypoint snapped the wrist toward it in one tick.
    arm = FakeArm()
    c = controller(arm, max_yaw_rad=0.5, max_total_yaw_rad=np.radians(120))
    c.step(yaw=0.5)
    out = c.step(yaw=0.5)
    assert any(n.startswith("reach clamp") for n in out["notes"]), out["notes"]
    measured = Rot.from_quat(arm.pose[3:]).as_euler("xyz")[2]
    assert np.remainder(c._target_euler[2] - measured + np.pi, 2 * np.pi) - np.pi == (
        pytest.approx(0.0, abs=1e-3)
    )
    q0 = arm.q.copy()
    arm.rot_log.clear()
    c.step([0.0, 0.0, 0.01])
    assert max_tick_turn(arm, q0) < np.radians(2), np.degrees(max_tick_turn(arm, q0))


def test_argument_refusals_do_not_halt_an_arm_but_faults_do():
    f = dual_facade(FakeArm(), FakeArm())
    for c in f._controllers.values():
        c.limits.max_total_yaw_rad = 0.3
    with pytest.raises(ValueError, match="per call"):
        f.step([0.1, 0, 0], arm="left")
    f.step(yaw=0.2, arm="left")
    with pytest.raises(ValueError, match="max_total_yaw_rad"):
        f.step(yaw=0.2, arm="left")
    with pytest.raises(ValueError, match="gripper"):
        f.step(gripper="half", arm="left")
    assert not f.get_env_meta()["halted"]
    assert f.step([0.0, 0.0, 0.01], arm="left")["ok"]


def test_a_failed_or_cancelled_reset_keeps_the_arm_halted():
    left, right = FakeArm(), FakeArm()
    f = dual_facade(left, right)
    f.halt_arm(arm="left", reason="fault")
    # Cancelled mid-reset (stop): still halted, and the arm does not move.
    c = f._controllers["left"]
    streamed = left.streamed
    c.stop_requested = lambda: left.streamed >= streamed + 2
    out = f.reset(arm="left")
    assert out["cancelled"] and not out["ok"]
    assert f.get_env_meta()["halted"]["left"] == "reset cancelled"
    c.stop_requested = lambda: False
    pose = left.pose.copy()
    with pytest.raises(RuntimeError, match="the left arm is halted"):
        f.step([0.0, 0.0, 0.01], arm="left")
    np.testing.assert_allclose(left.pose, pose)
    # Failed (stale feedback mid-reset): still halted.
    left.fail_at_stream = left.streamed + 3
    with pytest.raises(RuntimeError, match="old"):
        f.reset(arm="left")
    assert f.get_env_meta()["halted"]["left"].startswith("reset failed")
    # Did not arrive (commands dropped): still halted.
    left.fail_at_stream, left.drop = None, True
    left.q = HOME + [0.2, 0.0, 0.0, 0.0, 0.0, 0.0]
    left.pose = left.kin.fk_pose7(left.q)
    assert not f.reset(arm="left")["ok"]
    assert f.get_env_meta()["halted"]["left"].startswith("reset failed")
    with pytest.raises(RuntimeError, match="the left arm is halted"):
        f.step([0.0, 0.0, 0.01], arm="left")
    # Only a successful reset clears it.
    left.drop = False
    assert f.reset(arm="left")["ok"]
    assert f.step([0.0, 0.0, 0.01], arm="left")["ok"]


def test_dual_reset_names_its_arm_and_never_drops_a_held_object():
    left, right = FakeArm(object_width=0.03), FakeArm()
    f = dual_facade(left, right)
    assert f.step(gripper="close", arm="left")["gripper_width_m"] == 0.03
    states = f.get_robot_state()["arms"]
    assert states["left"]["holding_object"] and not states["right"]["holding_object"]
    f.step([0.0, 0.0, 0.01], arm="right")
    streamed = left.streamed
    with pytest.raises(ValueError, match="reset names its arm"):
        f.reset()
    # Resetting the right arm leaves the left hand's object where it is.
    assert f.reset(arm="right")["ok"]
    assert left.width == 0.03 and left.streamed == streamed
    # The left gripper holds something: refused before any motion without release.
    with pytest.raises(ValueError, match="left \\(0.030 m\\) arm holds an object"):
        f.reset(arm="left")
    with pytest.raises(ValueError, match="holds an object"):
        f.reset(both=True)
    assert left.width == 0.03 and left.streamed == streamed
    assert not f.get_env_meta()["halted"], "a refused reset halts nothing"
    out = f.reset(both=True, release=True)
    assert out["ok"] and left.width == 0.07
    # One arm: the same guard.
    one = facade(FakeArm(object_width=0.03), None)
    one.step(gripper="close")
    with pytest.raises(ValueError, match="holds an object"):
        one.reset()
    assert one.reset(release=True)["ok"]


@pytest.mark.parametrize(
    "key, value, match",
    [
        ("z_floor_m", ".nan", "z_floor_m must be a finite"),
        ("z_floor_m", ".inf", "z_floor_m must be a finite"),
        ("max_yaw_rad", "3.0", "max_yaw_rad"),
        ("max_yaw_rad", "0.6", "must stay below pi"),
        ("max_total_yaw_rad", ".nan", "max_total_yaw_rad"),
        ("max_step_m", ".nan", "max_step_m"),
        ("max_step_m", "0.5", "max_step_m"),
        ("speed_mps", "5.0", "speed_mps"),
        ("divergence_resync_m", ".nan", "divergence_resync_m"),
        ("divergence_resync_m", "5.0", "divergence_resync_m"),
        ("enable_z_floor", "'no'", "enable_z_floor"),
        ("workspace_min", "[0, .nan, 0]", "finite"),
        ("reset_time_s", "0.0", "reset_time_s"),
        ("ori_flex_deg", "90", "ori_flex_rad"),
        ("joint_stream_hz", ".nan", "joint_stream_hz"),
    ],
)
def test_unsafe_config_values_refuse_to_start(tmp_path, key, value, match):
    text = DEFAULT_CONFIG.read_text().replace("z_floor_m: null", "z_floor_m: 0.19")
    text = text.replace("arm_id: null", "arm_id: usb-A")
    path = tmp_path / "piper.yaml"
    path.write_text(text)
    load_config(path)
    cfg = yaml.safe_load(text)
    section = {"z_floor_m": "calibration"}.get(key, "limits")
    if key in cfg["motion"]:
        section = "motion"
    cfg[section][key] = yaml.safe_load(value)
    if key == "workspace_min":
        cfg["limits"]["workspace_max"] = [1, 1, 1]
    path.write_text(yaml.safe_dump(cfg))
    with pytest.raises(ValueError, match=match):
        load_config(path)
