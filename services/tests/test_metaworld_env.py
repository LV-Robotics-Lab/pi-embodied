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

"""Metaworld env server helpers with a mock simulator: the task table, success handling,
the workspace refusal and the ground-truth names."""

from __future__ import annotations

import math
from types import SimpleNamespace

import numpy as np
import pytest

from pi_embodied_services.robots.metaworld import env_server as mw


def test_task_table_is_mt50_with_the_ml45_split():
    assert len(mw.TASKS) == 50
    assert len(set(mw.TASKS)) == 50
    assert all(t.endswith("-v3") for t in mw.TASKS)
    assert len(mw.ML45_TEST) == 5 and len(mw.ML45_TRAIN) == 45
    assert set(mw.ML45_TRAIN) | set(mw.ML45_TEST) == set(mw.TASKS)
    assert mw.instruction("reach-v3").startswith("Move the gripper")
    with pytest.raises(ValueError, match="unknown Metaworld task 'reach-v9'"):
        mw.instruction("reach-v9")


def test_task_table_matches_the_installed_metaworld():
    env_dict = pytest.importorskip("metaworld.env_dict")
    assert list(env_dict.ALL_V3_ENVIRONMENTS) == mw.TASKS
    assert list(env_dict.ML45_V3["test"]) == mw.ML45_TEST


class _Env:
    """A Metaworld-like env: the hand follows the action at 1 cm per unit, the gripper pads
    close over 5 steps, success when the TCP is within 2 cm of ``goal``."""

    def __init__(self, goal, low=(-0.2, 0.5, 0.05), high=(0.2, 0.75, 0.3)):
        self.goal = np.asarray(goal, dtype=np.float64)
        self.mocap_low = np.asarray(low, dtype=np.float64)
        self.mocap_high = np.asarray(high, dtype=np.float64)
        self.hand = np.array([0.0, 0.6, 0.2])
        self.width = 1.0
        self.data = SimpleNamespace(mocap_pos=np.array([self.hand + [0, 0, 0.045]]))
        self.model = None
        self._target_pos = self.goal
        self.seeds: list[int] = []
        self.resets = 0

    @property
    def tcp_center(self):
        return self.hand.copy()

    def get_body_com(self, name):
        side = -1 if name == "leftpad" else 1
        return self.hand + [side * self.width * 0.047, 0, 0]

    def seed(self, s):
        self.seeds.append(s)

    def reset(self):
        self.resets += 1
        self.hand = np.array([0.0, 0.6, 0.2])
        self.width = 1.0
        return self._obs(), {}

    def _obs(self):
        obs = np.zeros(39)
        obs[:3] = self.hand
        obs[3] = self.width
        return obs

    def step(self, a):
        a = np.clip(np.asarray(a, dtype=np.float64), -1, 1)
        self.hand = np.clip(self.hand + a[:3] * 0.01, self.mocap_low, self.mocap_high)
        self.data.mocap_pos = np.array([self.hand + [0, 0, 0.045]])
        self.width = float(np.clip(self.width - 0.2 * a[3], 0.0, 1.0))
        d = float(np.linalg.norm(self.hand - self.goal))
        info = {"success": float(d < 0.015), "obj_to_target": d, "grasp_success": 0.0}
        return self._obs(), 1.0 - d, False, False, info

    def close(self):
        pass


def _facade(goal=(0.1, 0.7, 0.2)):
    f = object.__new__(mw.MetaworldEnvFacade)
    mw.BaseEnvFacade.__init__(f)
    f._env = _Env(goal)
    f._seed = 7
    f._view_size = 4
    f._renderers = {}
    f._obs = np.zeros(39)
    f._info = {}
    f._success_once = False
    f._gripper_effort = mw.OPEN
    f._closed = False
    f._box = None
    f._meta = {"task": "reach-v3", "seed": 7}
    frame = np.zeros((4, 4, 3), dtype=np.uint8)
    f._render = lambda camera, h, w, depth=False: frame
    return f


def test_reset_seeds_the_env_and_the_global_rngs_and_opens_the_gripper():
    f = _facade()
    obs, info = f.reset()
    assert f._env.seeds == [7]
    a = np.random.random()
    f.reset(seed=7)
    assert np.random.random() == a, "the global numpy RNG is restored by the seed"
    assert f._env.seeds == [7, 7]
    assert obs["tcp_pos"].tolist() == pytest.approx([0.0, 0.6, 0.2])
    assert obs["gripper_width"] == pytest.approx(0.094)
    assert f._gripper_effort == mw.OPEN
    assert f.get_env_meta()["workspace"]["max"] == pytest.approx([0.2, 0.75, 0.255])


def test_success_is_the_envs_flag_and_latches_as_success_once():
    f = _facade(goal=(0.0, 0.6, 0.15))
    f.reset()
    r = f.move_delta([0, 0, -0.04])
    assert r["ok"] is True
    assert r["info"]["success"] is True and r["info"]["success_once"] is True
    assert r["steps_used"] == 4, "the servo stops at success"
    assert f.state()["success"] is True
    # The env's flag follows the state; the latch stays.
    f._env.goal = np.array([1.0, 1.0, 1.0])
    f.move_delta([0, 0, 0.05])
    assert f.state()["success"] is False and f.state()["success_once"] is True
    # step / chunk_step report success as terminated; chunk_step stops there.
    f.reset()
    f._env.goal = np.array([0.0, 0.6, 0.15])
    _obs, _rew, term, _trunc, info = f.step([0, 0, -1, -1])
    assert term is False and info["success"] is False
    frames, rews, terms, _truncs, info = f.chunk_step(
        [[0, 0, -1, -1]] * 10, return_all_frames=True
    )
    assert terms.tolist() == [False, False, True]
    assert len(frames) == 3 and len(rews) == 3
    assert info["success"] is True


def test_move_delta_refuses_long_moves_and_targets_outside_the_workspace_box():
    f = _facade()
    f.reset()
    with pytest.raises(ValueError, match="limit is 0.2 m per call"):
        f.move_delta([0.3, 0, 0])
    with pytest.raises(ValueError, match="outside the workspace box"):
        f.move_delta([0, 0.19, 0])
    assert f._env.tcp_center.tolist() == pytest.approx([0.0, 0.6, 0.2]), (
        "nothing commanded"
    )
    with pytest.raises(ValueError, match="gripper must be"):
        f.move_delta([0, 0, 0], "shut")
    # The fingers settle first (GRIPPER_STEPS), then 5 steps of 1 cm, then one settle step
    # (the mock hand tracks the mocap exactly).
    r = f.move_delta([0.05, 0, 0], "close")
    assert r["gripper"] == "close" and r["steps_used"] == mw.GRIPPER_STEPS + 5 + 1
    assert r["final_tcp_pos"] == pytest.approx([0.05, 0.6, 0.2])
    assert r["gripper_width"] == pytest.approx(0.0)
    assert len(r["frames"]) == r["steps_used"]
    # The gripper command persists across moves; set_gripper changes it in place.
    r = f.move_delta([0, 0.02, 0])
    assert r["gripper"] == "close" and math.isclose(r["final_error_m"], 0, abs_tol=1e-6)
    r = f.set_gripper(open=True)
    assert r["target_gripper_open"] is True and r["gripper"] == "open"
    assert r["steps_used"] == mw.GRIPPER_STEPS, "no settle: the arm never moved"
    assert r["moved_m"] == pytest.approx([0, 0, 0])


def test_stop_cancels_a_move_between_control_steps():
    f = _facade()
    f.reset()
    # As in a served call: the stop arrives after the call was received.
    f._active_generation = f._stop_generation
    f.request_stop()
    r = f.move_delta([0.05, 0, 0])
    assert r["cancelled"] is True and r["steps_used"] == 1
    assert f._env.tcp_center.tolist() == pytest.approx([0.0, 0.6, 0.2])


class _Body:
    def __init__(self, i, name):
        self.id, self.name = i, name


def test_ground_truth_names_are_the_non_robot_bodies_plus_the_goal():
    names = [
        "world",
        "right_arm_base_link",
        "right_l0",
        "hand",
        "rightpad",
        "mocap",
        "obj",
        "peg",
        "tableTop",
    ]
    parent = [0, 0, 1, 2, 3, 0, 0, 0, 0]

    def body(key):
        if isinstance(key, int):
            return _Body(key, names[key])
        return _Body(names.index(key), key)

    model = SimpleNamespace(nbody=len(names), body=body, body_parentid=parent)
    assert mw.object_bodies(model) == ["obj", "peg", "tableTop"]


def test_code_api_is_the_registry_over_the_facades_rpc_methods():
    """`code.api` comes from the shared registry: the high tier is the closed-loop set, the
    low tier adds the raw surface, `privileged` adds the simulator's ground truth; every
    primitive's method is a registered RPC of the facade."""
    f = _facade()
    names = lambda tier: [p["name"] for p in f.code_api.describe(tier)["primitives"]]  # noqa: E731
    assert names("high") == ["get_task_language", "state", "move_delta", "set_gripper"]
    assert names("low") == [
        "get_task_language",
        "state",
        "move_delta",
        "set_gripper",
        "render_camera",
        "get_camera_meta",
        "step",
        "chunk_step",
    ]
    assert names("privileged")[-1] == "ground_truth_poses"
    for p in f.code_api.describe(None)["primitives"]:
        assert p["method"] in f._rpc
    assert f.code_api.resolve("move_delta", {"delta_xyz": [0, 0, 0.01]}) == (
        "env.move_delta",
        {"delta_xyz": [0, 0, 0.01]},
    )
    with pytest.raises(ValueError, match="not a primitive"):
        f.code_api.resolve("teleport", {})


def test_camera_meta_is_opencv_from_mujocos_fovy_and_frame():
    model = SimpleNamespace(
        camera=lambda name: _Body(0, name), cam_fovy=np.array([90.0])
    )
    data = SimpleNamespace(
        cam_xmat=np.array([np.eye(3).reshape(-1)]), cam_xpos=np.array([[1.0, 2.0, 3.0]])
    )
    m = mw.camera_meta(model, data, "corner4", 256, 256)
    assert m["intrinsic_K"][0][0] == pytest.approx(128.0)
    assert m["intrinsic_K"][0][2] == 128 and m["intrinsic_K"][1][2] == 128
    r = np.asarray(m["extrinsic_cam2world"])
    assert r[:3, 3].tolist() == [1.0, 2.0, 3.0]
    # MuJoCo looks along -z with y up; OpenCV along +z with y down.
    assert r[:3, :3].tolist() == np.diag([1.0, -1.0, -1.0]).tolist()
    # A view turned 180 degrees is the camera rolled half a turn about its optical axis.
    m = mw.camera_meta(model, data, "corner4", 256, 256, rotated=True)
    assert (
        np.asarray(m["extrinsic_cam2world"])[:3, :3].tolist()
        == np.diag([-1.0, 1.0, -1.0]).tolist()
    )
