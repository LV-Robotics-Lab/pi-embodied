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
    CODE_MAX_FRAMES,
    RoboCasaEnvFacade,
)
from pi_embodied_services.utils.rpc import RpcFacade


class FakeKitchen:
    action_dim = 12

    def __init__(self):
        self.eef = np.array([0.5, 0.0, 1.0])
        self.actions: list[np.ndarray] = []

    def step(self, a):
        self.actions.append(a)
        self.eef = self.eef + np.clip(a[:3], -1, 1) * ARM_M_PER_STEP
        obs = {
            "robot0_eef_pos": self.eef.copy(),
            "robot0_gripper_qpos": np.array([0.04, -0.04]),
            "robot0_agentview_left_image": np.zeros((4, 4, 3), np.uint8),
            "obj_pos": np.array([0.6, 0.1, 0.9]),
            "obj_to_robot0_eef_pos": np.array([0.1, 0.1, -0.1]),
            "object-state": np.zeros(14),
        }
        return obs, 0.0, False, {"obj_dist": 0.12}

    def _check_success(self):
        return self.eef[2] > 1.05


def facade() -> RoboCasaEnvFacade:
    f = object.__new__(RoboCasaEnvFacade)
    RpcFacade.__init__(f)
    f.env = FakeKitchen()
    f._steps, f._run_start, f._run_obs, f._run_frames = 0, 0, None, []
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
        tier="low",
    )
    assert out["status"] == "ran", out
    assert out["result"] == {
        "keys": ["done", "obs", "reward"],
        "obs": ["robot0_eef_pos", "robot0_gripper_qpos"],
    }, "no object observations, no camera images, no info"
    assert out["steps"] == 3 and out["success"] is True
    assert sorted(out["obs"]) == ["robot0_eef_pos", "robot0_gripper_qpos"]
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
        tier="low",
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
    assert out["status"] == "error" and "at most 1024" in out["error"], out
    assert f.renders == []


def test_the_run_video_is_bounded_and_a_run_without_steps_has_no_obs():
    f = facade()
    f._begin_run()
    for _ in range(CODE_MAX_FRAMES + 5):
        f._keep_frame(np.zeros((1, 1, 3), np.uint8))
    assert len(f._run_frames) <= CODE_MAX_FRAMES
    out = f._rpc["code.run"]("RESULT = check_success()\n", timeout_s=30)
    assert out["status"] == "ran" and out["steps"] == 0 and out["obs"] is None


def test_the_low_tier_shows_examples_and_s4_drops_them():
    f = facade()
    low = {p["name"]: p for p in f._rpc["code.api"]("low")["primitives"]}
    assert "example" in low["step"] and "example" in low["render_camera"]
    s4 = f._rpc["code.api"]("low-noexamples")["primitives"]
    assert all("example" not in p for p in s4)
