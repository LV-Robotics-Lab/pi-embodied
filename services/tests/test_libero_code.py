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

"""The LIBERO facade's code-mode primitives (`code.api` / `code.run`) over a mock
LiberoEnv, no simulator: tiers, the servo primitives, back-projection, stop handling and
the run's bookkeeping for pi."""

from __future__ import annotations

import numpy as np
import pytest

from pi_embodied_services.robots.libero.env_server import (
    CODE_RES,
    LiberoEnvFacade,
)


class ArmSim:
    """A LiberoEnv stand-in: an OSC-like arm (action[:3] * 0.05 m per step, action[5] * 0.1 rad
    yaw, action[6] drives the fingers), a flat table 0.5 m below a camera looking straight down
    with a wrist camera, and success once the arm rises above z = 0.5 with the gripper closed."""

    def __init__(self):
        self.pos = np.array([0.0, 0.0, 0.2])
        self.yaw = 0.0
        self.width = 0.08
        self.steps = 0
        self.succeeded = False
        self.workers = [self]

    def _quat(self):
        return np.array([0.0, 0.0, np.sin(self.yaw / 2), np.cos(self.yaw / 2)])

    @property
    def current_raw_obs(self):
        half = self.width / 2
        return [
            {
                "robot0_eef_pos": self.pos.copy(),
                "robot0_eef_quat": self._quat(),
                "robot0_gripper_qpos": np.array([half, -half]),
            }
        ]

    def _obs(self):
        return {
            "main_images": np.full((1, 4, 4, 3), self.steps % 256, dtype=np.uint8),
            "wrist_images": np.zeros((1, 4, 4, 3), dtype=np.uint8),
            "states": np.zeros((1, 8), dtype=np.float32),
        }

    def _info(self):
        return {"episode": {"success_once": np.array([self.succeeded])}}

    def reset(self):
        self.__init__()
        return self._obs(), self._info()

    def step(self, action):
        a = np.asarray(action, dtype=np.float64).reshape(7)
        self.steps += 1
        self.pos = self.pos + a[:3] * 0.05
        self.pos[2] = max(self.pos[2], 0.0)  # the table
        self.yaw += a[5] * 0.1
        if a[6] > 0:
            self.width = max(0.02, self.width - 0.02)  # closes on a 2 cm object
        elif a[6] < 0:
            self.width = min(0.08, self.width + 0.02)
        if self.pos[2] > 0.5 and self.width < 0.05:
            self.succeeded = True
        zeros = np.zeros(1, dtype=bool)
        return self._obs(), np.zeros(1), zeros, zeros, self._info()

    def render_camera(self, camera_name, height, width, depth):
        rgb = np.zeros((height, width, 3), dtype=np.uint8)
        rgb[0, 0] = (
            255  # a marker at the raw image's first row: upright it is the last row
        )
        d = np.full((height, width), 0.5, dtype=np.float32)
        return (rgb, d) if depth else rgb

    def get_camera_meta(self, camera_name, height, width):
        f = height / 2
        return {
            "intrinsic_K": [[f, 0, width / 2], [0, f, height / 2], [0, 0, 1]],
            # Camera 0.7 m above the table looking down: cam +z is world -z, cam +x is world +x.
            "extrinsic_cam2world": [
                [1, 0, 0, 0],
                [0, -1, 0, 0],
                [0, 0, -1, 0.7],
                [0, 0, 0, 1],
            ],
        }

    def env_call(self, name, target):
        return {"cube": {"pos": [0.1, 0.0, 0.02], "quat_xyzw": [0, 0, 0, 1]}}

    task_descriptions = ["lift the cube"]

    @property
    def env(self):
        return self


def facade(sam3=None) -> LiberoEnvFacade:
    f = LiberoEnvFacade(ArmSim(), meta={}, sam3=sam3)
    f.reset()
    return f


def names(api):
    """Primitive names of a `code.api` reply, a registry tier list or a runner list."""
    if isinstance(api, dict):
        api = api["primitives"]
    return [p["name"] if isinstance(p, dict) else p.name for p in api]


RAW_HIGH = ["get_task_language", "render_camera", "get_camera_meta"]
RAW_LOW = RAW_HIGH + ["raw_obs", "step", "chunk_step"]


CODE_HIGH = [
    "get_state",
    "get_observation",
    "back_project",
    "move_to",
    "rotate_wrist",
    "set_gripper",
]
CODE_LOW = ["get_state", "get_observation", "move_delta", "rotate_delta", "set_gripper"]


def test_the_registry_tiers_list_the_code_primitives_and_privileged_adds_ground_truth():
    f = facade()
    high = f._rpc["code.api"]("high")
    low = f._rpc["code.api"]("low")
    priv = f._rpc["code.api"]("privileged")
    # Other servers' primitives (the raw surface, reach previews) sit beside these in the registry.
    assert set(CODE_HIGH) <= set(names(high)) and set(CODE_LOW) <= set(names(low))
    assert set(names(low)) & {"back_project", "move_to", "rotate_wrist"} == set()
    assert set(names(high)) & {"move_delta", "rotate_delta"} == set()
    assert "ground_truth_poses" not in names(high) + names(low)
    assert set(names(priv)) == set(names(high)) | {"ground_truth_poses"}
    assert isinstance(high["digest"], str) and high["tier"] == "high"
    move_to = next(p for p in high["primitives"] if p["name"] == "move_to")
    assert move_to["method"] == "env.move_to" and move_to["mutating"] is True
    assert list(move_to["params"]) == ["xyz", "gripper", "tol", "max_steps"]
    # The runner renders the same declaration as a signature and a doc for the program.
    runner = {p.name: p for p in f._code.primitives("high")}
    assert set(runner) == set(names(high))
    assert runner["move_to"].describe()["signature"] == (
        "(xyz: vec3, gripper: number = None, tol: number = None, max_steps: integer = None)"
    )
    assert "Args:" in runner["move_to"].describe()["doc"]
    assert "Moves the robot." in runner["move_to"].describe()["doc"]
    assert "ground_truth_poses" in [p.name for p in f._code.primitives("privileged")]
    helpers = names(f._rpc["code.helpers"]())
    assert helpers[:2] == ["rotation_matrix_to_quaternion", "decompose_transform"]


def test_segment_needs_a_sam3_server():
    assert "segment" not in names(facade()._rpc["code.api"]("high"))
    assert "env.segment" not in facade()._rpc
    with_sam3 = facade(sam3="http://127.0.0.1:1")
    assert "segment" in names(with_sam3._rpc["code.api"]("high"))
    assert "segment" not in names(with_sam3._rpc["code.api"]("low"))
    assert "env.segment" in with_sam3._rpc


def test_move_to_servos_with_the_tools_step_rule_and_keeps_the_gripper_command():
    f = facade()
    out = f.move_to([0.1, 0.05, 0.2])
    assert out["final_dist_m"] < 0.012
    assert out["eef_pos"] == pytest.approx([0.1, 0.05, 0.2], abs=0.012)
    # 2.5 cm per step at most: 0.1 m takes at least 4 steps.
    assert 4 <= out["steps_used"] <= 6
    assert out["gripper_width"] == pytest.approx(0.08)
    f.set_gripper(True)
    assert f.get_state()["gripper_cmd"] == 1
    f.move_to([0.1, 0.05, 0.3])
    assert f.get_state()["gripper_cmd"] == 1, "a move keeps the last gripper command"
    opened = f.move_to([0.1, 0.05, 0.2], gripper=-1)
    assert opened["gripper_width"] == pytest.approx(0.08)
    assert f.get_state()["gripper_cmd"] == -1
    with pytest.raises(ValueError):
        f.move_to([0, 0, 0.2], gripper=0.5)


def test_move_delta_and_rotate_delta_are_bounded_and_report_travel():
    f = facade()
    out = f.move_delta([0, 0, -0.05])
    assert out["moved_m"] == pytest.approx(0.05, abs=0.005)
    with pytest.raises(ValueError, match="0.10 m"):
        f.move_delta([0.2, 0, 0])
    f.move_to([0, 0, 0.05])
    blocked = f.move_delta([0, 0, -0.1])  # the table is at z = 0
    assert blocked["moved_m"] == pytest.approx(0.05, abs=0.005)
    turn = f.rotate_delta(0.5)
    assert turn["yaw"] == pytest.approx(0.5, abs=0.02)
    with pytest.raises(ValueError):
        f.rotate_delta(2.0)
    assert f.rotate_wrist(target_yaw=0.0)["yaw"] == pytest.approx(0.0, abs=0.02)
    with pytest.raises(ValueError):
        f.rotate_wrist()


def test_set_gripper_stops_when_the_fingers_stop():
    f = facade()
    out = f.set_gripper(True)
    assert out["gripper_width"] == pytest.approx(0.02)
    assert out["steps_used"] < 12, "the fingers stopped on the object"
    assert f.set_gripper(False)["gripper_width"] == pytest.approx(0.08)


def test_observation_is_upright_metric_and_back_projects_with_its_calibration():
    f = facade()
    obs = f.get_observation()
    assert set(obs) >= {"agentview", "wrist", "eef_pos", "gripper_width", "terminated"}
    view = obs["agentview"]
    assert (
        view["rgb"].shape == (CODE_RES, CODE_RES, 3) and view["rgb"].dtype == np.uint8
    )
    assert (
        view["depth"].shape == (CODE_RES, CODE_RES)
        and view["depth"].dtype == np.float32
    )
    assert view["rgb"][-1, 0, 0] == 255, "rows are flipped: LIBERO renders upside down"
    # No depth_near/far in the meta: the depth is already metric (0.5 m to the table).
    assert float(view["depth"][10, 10]) == pytest.approx(0.5)
    # The principal pixel looks straight down from 0.7 m onto the table at z = 0.2.
    centre = f.back_project(CODE_RES // 2, CODE_RES // 2)["world_xyz"]
    assert centre == pytest.approx([0.0, 0.0, 0.2], abs=1e-3)
    # One pixel right of centre is +x in the camera, so +x in the world (f = 256 px, z = 0.5).
    right = f.back_project(CODE_RES // 2, CODE_RES // 2 + 128)["world_xyz"]
    assert right[0] == pytest.approx(0.25, abs=1e-3)
    with pytest.raises(ValueError, match="out of bounds"):
        f.back_project(CODE_RES, 0)
    with pytest.raises(ValueError, match="camera"):
        f.back_project(0, 0, camera="overhead")


def test_a_run_reports_steps_success_the_latest_obs_and_frames():
    f = facade()
    out = f._rpc["code.run"](
        "g = set_gripper(True)\n"
        "r = move_to([0, 0, 0.6])\n"
        "RESULT = [get_state()['terminated'], r['steps_used'] + g['steps_used']]\n",
        timeout_s=30,
    )
    assert out["status"] == "ran", out
    assert out["result"][0] is True
    assert out["terminated"] is True and out["truncated"] is False
    assert out["success_step"] is not None and out["success_step"] <= out["steps"]
    assert out["steps"] == out["result"][1], "the gripper's steps and the move's"
    assert out["obs"]["main_images"].shape == (4, 4, 3)
    assert len(out["frames"]) == 2, "one agentview frame per motion primitive"
    assert [c["name"] for c in out["calls"]] == ["set_gripper", "move_to", "get_state"]
    assert out["calls"][1]["move_m"] == pytest.approx(0.4)


def test_a_finished_episode_stops_every_motion_primitive():
    f = facade()
    f.set_gripper(True)
    f.move_to([0, 0, 0.6])
    assert f.get_state()["terminated"]
    assert f.move_to([0.3, 0, 0.6])["steps_used"] == 0
    assert f.set_gripper(False)["steps_used"] == 0
    f.reset()
    assert not f.get_state()["terminated"] and f.get_state()["gripper_cmd"] == -1


def test_a_stop_ends_a_servo_loop_between_steps():
    f = facade()
    f._active_generation = f._stop_generation  # as inside a running call
    f.request_stop()
    out = f.move_to([0.5, 0, 0.2])
    assert out["steps_used"] == 0 and out["cancelled"] is True


def test_move_budget_is_estimated_from_the_target_and_refuses():
    f = facade()
    out = f._rpc["code.run"](
        "log = []\n"
        "for t in ([0.05, 0, 0.2], [0.10, 0, 0.2], [0.30, 0, 0.2]):\n"
        "    try:\n"
        "        move_to(t); log.append('ok')\n"
        "    except Exception as e:\n"
        "        log.append(type(e).__name__)\n"
        "RESULT = log\n",
        timeout_s=30,
        max_move_m=0.12,
    )
    assert out["result"] == ["ok", "ok", "CodeLimitError"]
    assert out["limit"] == "max_move_m"
    assert out["steps"] > 0


def test_ground_truth_primitive_only_with_privileged():
    f = facade()
    out = f._rpc["code.run"]("RESULT = ground_truth_poses()", timeout_s=30)
    assert "NameError" in out["error"]
    out = f._rpc["code.run"](
        "RESULT = ground_truth_poses(['cube'])", timeout_s=30, tier="privileged"
    )
    assert out["result"]["poses"]["cube"]["pos"] == [0.1, 0.0, 0.02]


def test_calls_go_through_the_registry_resolve():
    f = facade()
    out = f._rpc["code.run"](
        "log = []\n"
        "calls = (lambda: move_to([0, 0, 0.3], sideways=1), lambda: move_to(),"
        " lambda: move_to([0, 0, 0.3], 1, tol=0.01, max_steps=5, extra=2))\n"
        "for call in calls:\n"
        "    try:\n"
        "        call(); log.append('ok')\n"
        "    except Exception as e:\n"
        "        log.append(str(e))\n"
        "RESULT = log\n",
        timeout_s=30,
    )
    assert "unknown parameter(s) sideways" in out["result"][0]
    assert "missing parameter(s) xyz" in out["result"][1]
    assert "unknown parameter(s) extra" in out["result"][2]
    assert out["steps"] == 0, "a refused call never reaches the env"
    # Positional arguments fill the declared parameters in order; a raw low-tier step is capped too.
    out = f._rpc["code.run"](
        "r = move_delta([0, 0, 0.05], -1)\n"
        "step([0.2, 0, 0, 0, 0, 0, -1])\n"
        "RESULT = r['moved_m']",
        timeout_s=30,
        tier="low",
    )
    assert out["result"] == pytest.approx(0.05, abs=0.005)
    assert out["calls"][1]["move_m"] == pytest.approx(0.01)
    assert len(out["frames"]) == 2, "raw steps are mutating too"
