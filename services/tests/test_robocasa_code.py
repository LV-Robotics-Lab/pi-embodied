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

"""Code mode on the RoboCasa server (``code.run`` over its registry), without a simulator: the
facade's ``step`` runs on a fake robosuite env that reports robot and object observations."""

from __future__ import annotations

import numpy as np
import pytest

from pi_embodied_services.robots.robocasa.env_server import (
    ARM_M_PER_STEP,
    BASE_M_PER_STEP,
    CALIBRATE_BASE_STEPS,
    CODE_MAX_FRAMES,
    GRIPPER_STEPS,
    PROBE_STEPS,
    RoboCasaEnvFacade,
)
from pi_embodied_services.utils.rpc import RpcFacade


class FakeKitchen:
    """A PandaOmron in an empty kitchen: the arm action moves the eef along the base's axes
    (ARM_M_PER_STEP per unit), the base drives along its heading and turns (BASE_M_PER_STEP,
    0.5 rad/s), the gripper closes on +1."""

    action_dim = 12

    def __init__(self):
        self.eef = np.array([0.5, 0.0, 1.0])
        self.base = np.array([0.0, 0.0, 0.0])
        self.yaw = 0.3
        self.finger = 0.04
        self.actions: list[np.ndarray] = []

    def _R(self):
        c, s = np.cos(self.yaw), np.sin(self.yaw)
        return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])

    def obs(self):
        q = np.array([0, 0, np.sin(self.yaw / 2), np.cos(self.yaw / 2)])  # xyzw
        return {
            "robot0_eef_pos": self.eef.copy(),
            "robot0_eef_quat": np.array([0, 0, 0, 1.0]),
            "robot0_gripper_qpos": np.array([self.finger, -self.finger]),
            "robot0_base_pos": self.base.copy(),
            "robot0_base_quat": q,
            "robot0_base_to_eef_pos": self._R().T @ (self.eef - self.base),
            "robot0_base_to_eef_quat": np.array([0, 0, 0, 1.0]),
            "robot0_agentview_left_image": np.zeros((4, 4, 3), np.uint8),
            "obj_pos": np.array([0.6, 0.1, 0.9]),
            "obj_to_robot0_eef_pos": np.array([0.1, 0.1, -0.1]),
            "object-state": np.zeros(14),
        }

    def reset(self):
        return self.obs()

    def step(self, a):
        a = np.asarray(a, dtype=np.float64)
        self.actions.append(a)
        self.eef = self.eef + self._R() @ (np.clip(a[:3], -1, 1) * ARM_M_PER_STEP)
        self.finger = float(
            np.clip(self.finger - 0.004 * np.clip(a[6], -1, 1), 0.0, 0.04)
        )
        if a[11] > 0:
            move = self._R() @ np.array([a[7], -a[8], 0.0]) * BASE_M_PER_STEP
            self.base = self.base + move
            self.eef = self.eef + move
            turn = float(np.clip(a[9], -1, 1)) * 0.5 / 20
            c, s = np.cos(turn), np.sin(turn)
            rel = self.eef - self.base
            self.eef = self.base + np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]]) @ rel
            self.yaw += turn
        return self.obs(), 0.0, False, {"obj_dist": 0.12}

    def _check_success(self):
        return self.eef[2] > 1.05


def facade() -> RoboCasaEnvFacade:
    f = object.__new__(RoboCasaEnvFacade)
    RpcFacade.__init__(f)
    f.env = FakeKitchen()
    f._steps, f._run_start, f._run_obs, f._run_frames = 0, 0, None, []
    f._obs = f.env.obs()
    f._pos_jac = f._fwd_offset = None
    f._home_rel = np.asarray(f._obs["robot0_base_to_eef_pos"])
    f._motion_frames, f._policy_frames, f._motion_steps = [], [], 0
    f._recording, f._sam3_url, f._sam3 = False, "", None
    f.renders = []

    def render(camera_name, height, width, depth):
        f.renders.append((camera_name, height, width, depth))
        img = np.zeros((height, width, 3), np.uint8)
        img[0] = 9  # bottom-up: the first row is the image's bottom
        return img

    f.render_camera = render
    RoboCasaEnvFacade._register_rpc(f)
    f._rpc["env.render_camera"] = render
    return f


def test_the_server_serves_code_run_with_a_token_and_exclusivity():
    f = facade()
    assert RoboCasaEnvFacade.REQUIRE_TOKEN and f._rpc_token
    assert "code.run" in f._rpc and "code.helpers" in f._rpc
    with pytest.raises(PermissionError):
        f._serve_dispatch("env.check_success", (), {})


def test_a_step_answers_the_robot_obs_only_and_the_run_reports_its_effect():
    f = facade()
    out = f._rpc["code.run"](
        "for _ in range(3):\n"
        "    r = step([0, 0, 1, 0, 0, 0, -1, 0, 0, 0, 0, -1])\n"
        "RESULT = {'keys': sorted(r), 'obs': sorted(r['obs'])}\n",
        timeout_s=30,
        tier="raw",
    )
    assert out["status"] == "ran", out
    robot = sorted(
        k for k in f.env.obs() if k.startswith("robot0_") and "image" not in k
    )
    assert out["result"] == {
        "keys": ["done", "obs", "reward"],
        "obs": robot,
    }, "no object observations, no camera images, no info"
    assert out["steps"] == 3 and out["success"] is True
    assert sorted(out["obs"]) == robot
    assert np.allclose(out["obs"]["robot0_eef_pos"], [0.5, 0.0, 1.15])
    assert len(out["frames"]) == 3 and out["frames"][0][-1, 0, 0] == 9, "top-down"
    assert out["calls"][0]["move_m"] == pytest.approx(ARM_M_PER_STEP)


def test_the_move_cap_counts_the_arm_the_base_and_the_turn():
    f = facade()
    est = f._code_move_m(
        "env.step", {"flat_action": [3, 0, 0, 0, 0, 0, 1, 1, 0, 1, 0, 1]}
    )
    assert est == pytest.approx(ARM_M_PER_STEP + 2 * BASE_M_PER_STEP)
    out = f._rpc["code.run"](
        "for _ in range(10):\n    step([1, 0, 0, 0, 0, 0, -1, 0, 0, 0, 0, -1])\n",
        timeout_s=30,
        tier="raw",
        max_move_m=0.12,
    )
    assert out["status"] == "error" and out["limit"] == "max_move_m", out
    assert len(f.env.actions) == 2


def test_oversized_renders_are_refused_before_they_run():
    f = facade()
    out = f._rpc["code.run"](
        "render_camera('robot0_eye_in_hand', 4096, 4096, False)\n",
        timeout_s=30,
        tier="low",
    )
    assert out["status"] == "error" and "1024" in out["error"], out
    assert f.renders == []


def test_the_run_video_is_bounded_and_a_run_without_steps_has_no_obs():
    f = facade()
    f._begin_run()
    for _ in range(CODE_MAX_FRAMES + 5):
        f._keep_frame(np.zeros((1, 1, 3), np.uint8))
    assert len(f._run_frames) <= CODE_MAX_FRAMES
    out = f._rpc["code.run"]("RESULT = check_success()\n", timeout_s=30, tier="low")
    assert out["status"] == "ran" and out["steps"] == 0 and out["obs"] is None


def test_the_low_tier_shows_examples_and_s4_drops_them():
    f = facade()
    low = f._rpc["code.api"]("low")
    s4 = f._rpc["code.api"]("low-noexamples")
    assert s4["available"] == low["available"] and s4["digest"] != low["digest"]
    docs = {p["name"]: p["doc"] for p in f._code.api("low")}
    assert "Example:" in docs["move_to"] and "Example:" in docs["render_camera"]
    assert all("Example:" not in p["doc"] for p in f._code.api("low-noexamples"))


def test_progress_is_privileged_and_the_high_and_low_tiers_move_the_robot():
    """Audit #10: the task's progress (object distances) and the contact read are privileged; the
    high tier has CaP-X's motion functions, the low tier the servo, base and gripper."""
    f = facade()
    tier = {
        t: set(f._rpc["code.api"](t)["available"])
        for t in ("high", "low", "raw", "privileged")
    }
    for t in ("high", "low", "raw"):
        assert (
            not {"get_task_progress", "grasp_contact", "ground_truth_poses"} & tier[t]
        ), t
    assert {"goto_pose", "home_pose", "open_gripper", "close_gripper"} <= tier["high"]
    assert "get_object_pose" not in tier["high"], "needs --sam3"
    assert {"move_to", "move_delta", "rotate_pitch", "set_gripper", "release"} <= tier[
        "low"
    ]
    assert {"navigate_to", "move_base", "scripted_grasp", "get_state"} <= tier["low"]
    assert tier["raw"] == {"raw_obs", "step"}
    assert {
        "get_task_progress",
        "grasp_contact",
        "ground_truth_poses",
        "goto_pose",
    } <= tier["privileged"]
    from pi_embodied_services.components import manifest

    m = manifest.load_manifest("robocasa")
    names = {p.name for p in manifest.code_primitives(m, lambda c: c == "sam3")}
    assert "get_object_pose" in names, "with --sam3"


def test_the_server_checks_itself_against_its_manifest():
    """A facade method the manifest does not declare (nor lists as internal) stops the server,
    and so does a declared one that is missing."""
    from pi_embodied_services.components.manifest import ManifestError

    f = facade()
    f._rpc["env.secret_teleport"] = lambda: None
    with pytest.raises(ManifestError, match="env.secret_teleport"):
        f._manifest_ready()
    g = facade()
    del g._rpc["env.navigate_to"]
    with pytest.raises(ManifestError, match="navigate_to"):
        g._manifest_ready()
    facade()._manifest_ready()


def test_move_to_calibrates_the_arm_then_servos_on_the_server():
    f = facade()
    target = [0.6, 0.1, 1.02]
    r = f._rpc["env.move_to"](target, gripper="open")
    assert r["ok"] and r["final_dist"] < 0.012, r
    # Three probe axes of PROBE_STEPS each, then the servo steps.
    assert r["env_steps"] == 3 * PROBE_STEPS + r["steps"]
    assert len(r["frames"]) == r["env_steps"] and r["frames"][0][-1, 0, 0] == 9, (
        "top-down"
    )
    assert np.allclose(r["obs"]["robot0_eef_pos"], f.env.eef)
    assert "policy_frames" not in r
    assert all(a[6] == -1 for a in f.env.actions)
    # Calibrated once: the next move servos at once; a base turn drops the calibration.
    n = len(f.env.actions)
    r = f._rpc["env.move_delta"]([0.0, 0.0, -0.05])
    assert r["ok"] and r["env_steps"] == r["steps"] and len(f.env.actions) > n
    f._rpc["env.move_base"](turn=1.0, steps=4)
    assert f._pos_jac is None


def test_hold_keeps_the_finger_width_and_set_gripper_drives_it():
    f = facade()
    f._rpc["env.set_gripper"](1.0, steps=5)
    width = f.env.finger
    f._rpc["env.move_to"]([0.5, 0.05, 1.0])
    assert abs(f.env.finger - width) < 0.004
    r = f._rpc["env.release"](steps=10)
    assert r["ok"] and f.env.finger == pytest.approx(0.04)


def test_navigate_to_measures_the_heading_and_drives_there():
    f = facade()
    r = f._rpc["env.navigate_to"]([1.0, 1.0], tol=0.1)
    assert r["ok"] and r["final_dist"] < 0.1, r
    assert f._fwd_offset is not None and f._pos_jac is None
    assert r["env_steps"] == CALIBRATE_BASE_STEPS + r["steps"]


def test_a_stop_cancels_a_motion_between_steps():
    f = facade()
    calls = {"n": 0}

    def stop():
        calls["n"] += 1
        return calls["n"] > 4

    f.stop_requested = stop
    r = f._rpc["env.move_base"](forward=1.0, steps=10)
    assert r["cancelled"] and r["env_steps"] == 4 and len(f.env.actions) == 4


def test_the_high_tier_goes_home_and_to_a_position():
    f = facade()
    home = f.env.eef.copy()
    f._rpc["env.move_delta"]([0.05, 0.0, -0.05])
    r = f._rpc["env.home_pose"]()
    assert r["ok"] and np.linalg.norm(f.env.eef - home) < 0.012
    r = f._rpc["env.goto_pose"]([0.55, -0.05, 0.98], z_approach=0.05)
    assert r["ok"] and np.linalg.norm(f.env.eef - [0.55, -0.05, 0.98]) < 0.012
    r = f._rpc["env.close_gripper"]()
    assert r["ok"] and r["env_steps"] == GRIPPER_STEPS


def test_the_flywheel_records_while_recording():
    f = facade()
    f._rpc["env.set_recording"](True)
    r = f._rpc["env.set_gripper"](-1.0, steps=2)
    assert len(r["policy_frames"]) == 2
    pf = r["policy_frames"][0]
    assert pf["action"][6] == -1 and pf["success"] is False
    assert sorted(pf["video"]) == [
        "video.robot0_agentview_left",
        "video.robot0_agentview_right",
        "video.robot0_eye_in_hand",
    ]
    assert "state.end_effector_position_relative" in pf["state"]


def test_a_program_moves_through_the_motion_methods_under_the_move_cap():
    f = facade()
    out = f._rpc["code.run"](
        "r = move_to([0.5, 0.0, 1.1])\nRESULT = sorted(r)\n", timeout_s=30, tier="low"
    )
    assert out["status"] == "ran", out
    assert out["result"] == ["eef", "final_dist", "gripper_qpos", "ok", "steps"]
    assert out["success"] is True and out["obs"] is not None
    assert len(out["frames"]) == out["steps"] and out["steps"] > 0
    assert out["calls"][0]["move_m"] == pytest.approx(0.1)
    out = f._rpc["code.run"](
        "navigate_to([3.0, 0.0])\n", timeout_s=30, tier="low", max_move_m=1.0
    )
    assert out["status"] == "error" and out["limit"] == "max_move_m", out
    out = f._rpc["code.run"]("goto_pose([0.5, 0.0, 1.0])\n", timeout_s=30, tier="high")
    assert out["status"] == "ran", out
