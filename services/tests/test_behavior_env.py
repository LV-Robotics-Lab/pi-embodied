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

"""The BEHAVIOR env server against a mock OmniGibson: success is the BDDL task's, q_score is
BEHAVIOR's partial credit, the "picked" judgement is a reference field apart from success,
move_hand keeps the planner's obstacles, the task/scene lookup and the code API tiers."""

from __future__ import annotations

import math
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from pi_embodied_services.robots.behavior import env_server, sim, tasks


class Obj:
    def __init__(self, name: str, pos, exists: bool = True):
        self.name = name
        self.pos = np.asarray(pos, dtype=np.float64)
        self.exists = exists
        self.is_system = False

    def get_position_orientation(self):
        return self.pos, np.array([0.0, 0.0, 0.0, 1.0])


class PrimitiveError(Exception):
    pass


class FakeSim:
    """An R1Pro in a BDDL task with two goal predicates. Every primitive yields ``n`` actions;
    each ``env.step`` moves the base/hand toward the pending goal and advances the scripted
    ``goals`` (which predicates are satisfied after step k) and ``success``."""

    def __init__(self, n: int = 3):
        self.n = n
        self.steps = 0
        self.goals: dict[int, list[bool]] = {}
        self.success_at: int | None = None
        #: With success_at: the BDDL goal holds only for steps in [success_at, success_until).
        self.success_until: int | None = None
        self.stop_after: int | None = None
        self.satisfied = [False, False]
        self.radio = Obj("radio_89", [1.0, 0.5, 0.45])
        self.table = Obj("coffee_table_1", [1.0, 0.5, 0.0])
        self.gone = Obj("bowl_2", [0, 0, 0], exists=False)
        self.held = {"left": None, "right": None}
        self.base = np.array([0.0, 0.0, 0.0])
        self.yaw = 0.0
        self.hand = {
            "left": np.array([0.3, 0.2, 0.9]),
            "right": np.array([0.3, -0.2, 0.9]),
        }
        self.fingers = {"left": 0.05, "right": 0.05}
        self.calls: list[tuple] = []
        H = W = 4
        rgb = np.zeros((H, W, 4), np.uint8)
        depth = np.ones((H, W), np.float32)
        self.frame = {
            f"robot_r:{link}:Camera:0": {"rgb": rgb, "depth_linear": depth}
            for link in sim.CAMERAS.values()
        }
        self.frame["proprio"] = np.zeros(8, np.float32)
        robot = self.robot = SimpleNamespace(
            name="robot_r",
            sensors={
                f"robot_r:{link}:Camera:0": SimpleNamespace(
                    intrinsic_matrix=np.array(
                        [[2.0, 0, 2.0], [0, 2.0, 2.0], [0, 0, 1]]
                    ),
                    image_width=W,
                    image_height=H,
                    get_position_orientation=lambda: (
                        np.array([0.1, 0.0, 1.2]),
                        np.array([0.0, 0.0, 0.0, 1.0]),
                    ),
                )
                for link in sim.CAMERAS.values()
            },
            _ag_obj_in_hand=self.held,
            gripper_control_idx={
                "left": np.array([24, 25]),
                "right": np.array([26, 27]),
            },
            grasping_mode="sticky",
            joints={f"j{i}": None for i in range(28)},
            get_position_orientation=lambda: (self.base, self._quat()),
            get_eef_pose=lambda arm: (
                self.hand[arm],
                np.array([0.0, 0.7071, 0.0, 0.7071]),
            ),
            get_joint_positions=self._joints,
        )
        self.task = SimpleNamespace(
            object_scope={
                "radio_receiver.n.01_1": self.radio,
                "coffee_table.n.01_1": self.table,
                "bowl.n.01_1": self.gone,
            },
            get_goal_option_satisfaction=lambda idx: [list(self.satisfied)],
        )
        self.env = SimpleNamespace(
            robots=[robot],
            task=self.task,
            reset=self._reset,
            step=self._step,
            close=lambda: None,
        )
        ctrl = self.ctrl = SimpleNamespace(arm="left")
        ctrl._settle_robot = lambda: self._gen("settle", None)
        ctrl._navigate_to_pose = lambda pose: self._gen("navigate", pose)
        ctrl._move_hand = lambda pose, **kw: self._gen("move_hand", (pose, kw))
        ctrl._execute_release = lambda: self._gen("release", None)
        ctrl._execute_grasp = lambda: self._gen("grasp", None)
        self.handle = sim.Handle(
            og=None, env=self.env, controller=ctrl, error=PrimitiveError
        )

    def _quat(self):
        return np.array([0.0, 0.0, np.sin(self.yaw / 2), np.cos(self.yaw / 2)])

    def _joints(self):
        q = np.zeros(28)
        q[24] = q[25] = self.fingers["left"] / 2
        q[26] = q[27] = self.fingers["right"] / 2
        return q

    def _reset(self):
        self.steps = 0
        self.satisfied = [False, False]
        return {"robot_r": self.frame}, {}

    def _gen(self, kind, arg):
        self.calls.append((kind, arg))
        if kind == "navigate" and arg[0] > 100:
            raise PrimitiveError("no base plan: the goal is in collision")
        for i in range(self.n):
            yield (kind, arg, i)

    def _step(self, action):
        self.steps += 1
        kind, arg = action[0], action[1]
        arm = self.ctrl.arm
        if kind == "navigate":
            self.base = np.array([arg[0], arg[1], 0.0])
            self.yaw = arg[2]
        elif kind == "move_hand":
            self.hand[arm] = np.asarray(arg[0][0], dtype=np.float64)
            if self.held[arm] is not None:
                self.radio.pos = self.hand[arm].copy()
        elif kind == "release":
            self.fingers[arm] = 0.05
            self.held[arm] = None
        elif kind == "grasp":
            near = np.linalg.norm(self.hand[arm] - self.radio.pos) < 0.15
            self.fingers[arm] = 0.02 if near else 0.0
            self.held[arm] = self.radio if near else None
        if self.steps in self.goals:
            self.satisfied = list(self.goals[self.steps])
        solved = self.success_at is not None and self.steps >= self.success_at
        if self.success_until is not None and self.steps >= self.success_until:
            solved = False
        return (
            {"robot_r": self.frame},
            0.0,
            solved,
            False,
            {"done": {"success": solved}},
        )

    def facade(self):
        f = env_server.BehaviorEnvFacade(
            handle=self.handle,
            meta={
                "task": "turning_on_radio",
                "instruction": tasks.LANGUAGE["turning_on_radio"],
            },
        )
        f.stop_requested = lambda: (
            self.stop_after is not None and self.steps >= self.stop_after
        )
        return f


def test_success_is_bddl_and_q_score_is_the_partial_credit():
    fake = FakeSim()
    f = fake.facade()
    obs, info = f.reset()
    assert info["instruction"] == tasks.LANGUAGE["turning_on_radio"]
    assert obs["success"] is False and obs["q_score"] == 0.0
    assert obs["goals"] == {"satisfied": 0, "total": 2}
    assert obs["head"].shape == (4, 4, 3) and obs["head_depth"].dtype == np.float32
    assert set(obs) >= {
        "left_wrist",
        "right_wrist",
        "left_wrist_depth",
        "right_wrist_depth",
    }
    # One of two goal predicates newly satisfied: half credit, not success.
    fake.goals = {fake.steps + 2: [True, False]}
    r = f.navigate_to_pose(1.0, 0.0, 0.5)
    assert r["ok"] is True and r["primitive"] == "navigate_to_pose" and r["steps"] == 3
    assert r["success"] is False and r["q_score"] == 0.5
    assert r["goals"] == {"satisfied": 1, "total": 2}
    assert r["reached_pos"] == [1.0, 0.0, 0.0] and r["reached_yaw"] == 0.5
    assert r["distance_left_m"] == 0.0 and list(r["base_pos"]) == [1.0, 0.0, 0.0]
    # BDDL says success: q_score is 1 whatever the predicate count says.
    fake.success_at = fake.steps + 1
    r = f.open_gripper("left")
    assert r["success"] is True and r["q_score"] == 1.0 and r["terminated"] is True
    assert (
        sim.q_score(
            False, [[True, True], [False, True]], [[True, False], [False, False]]
        )
        == 0.5
    )


def test_picked_is_a_reference_field_apart_from_success():
    fake = FakeSim()
    f = fake.facade()
    obs, _ = f.reset()
    assert obs["privileged"] == {
        "in_hand": {"left": None, "right": None},
        "picked": False,
    }
    # A grasp at the radio: open, pregrasp above, close (sticky), approach, settle, lift.
    r = f.grasp_object("left", fake.radio.pos.tolist(), None, pregrasp_offset_m=0.1)
    assert r["ok"] is True and r["grasping_mode"] == "sticky"
    assert [c[0] for c in fake.calls[-6:]] == [
        "release",
        "move_hand",
        "grasp",
        "move_hand",
        "settle",
        "move_hand",
    ]
    assert fake.calls[-3][1][1] == {"stop_on_ag": True}, (
        "the approach stops on the grasp, no ignored obstacles"
    )
    assert r["privileged"]["in_hand"]["left"] == "radio_89"
    # The radio rose with the lift: CaP-X's judgement holds, but BDDL success does not follow from it.
    assert r["privileged"]["picked"] is True
    assert r["success"] is False and r["q_score"] == 0.0
    assert r["gripper_width"] == 0.02
    # The sim-only facts stay in `privileged`; the plain state has the gripper width only.
    assert "in_hand" not in r and "picked" not in r
    assert r["eef"]["left"]["gripper_width"] == 0.02


def test_move_hand_never_ignores_obstacles_and_refuses_out_of_reach():
    fake = FakeSim()
    f = fake.facade()
    f.reset()
    r = f.move_hand("right", [0.5, -0.3, 1.0], [0, 0, 0, 1])
    assert r["ok"] is True and r["arm"] == "right"
    kind, (pose, kw) = fake.calls[-1]
    assert kind == "move_hand" and kw == {}, (
        "no ignore_all_obstacles / skip_obstacle_update"
    )
    assert np.allclose(np.asarray(pose[0]), [0.5, -0.3, 1.0])
    assert r["eef_pos"] == [0.5, -0.3, 1.0] and r["distance_left_m"] == 0.0
    with pytest.raises(ValueError, match="navigate first"):
        f.move_hand("left", [3.0, 0.0, 1.0])
    with pytest.raises(ValueError, match="arm must be"):
        f.move_hand("head", [0.5, 0.0, 1.0])
    with pytest.raises(ValueError, match="the limit is 5.0 m"):
        f.navigate_to_pose(9.0, 0.0, 0.0)


def test_a_failed_plan_and_a_stop_are_reported_not_raised():
    fake = FakeSim()
    f = fake.facade()
    f.reset()
    # The planner fails (an ActionPrimitiveError): the call answers with the error, the episode goes on.
    fake.base = np.array([199.0, 0.0, 0.0])
    r = f.navigate_to_pose(200.0, 0.0, 0.0)
    assert (
        r["ok"] is False and "no base plan" in r["error"] and r["phase"] == "navigate"
    )
    fake.stop_after = fake.steps + 2
    r = f.move_hand("left", [199.4, 0.2, 0.9])
    assert r["ok"] is False and r["cancelled"] is True and r["steps"] == 2


def test_camera_meta_ground_truth_and_reads():
    fake = FakeSim()
    f = fake.facade()
    f.reset()
    meta = f.get_camera_meta("head")
    assert meta["convention"] == "opengl" and meta["width"] == 4
    assert np.allclose(meta["extrinsic_cam2world"][:3, 3], [0.1, 0.0, 1.2])
    rgb, depth = f.render_camera("left_wrist", depth=True)
    assert rgb.shape == (4, 4, 3) and depth.shape == (4, 4)
    with pytest.raises(ValueError, match="camera_name"):
        f.render_camera("agentview")
    gt = f.ground_truth_poses()
    assert gt["frame"] == "world" and sorted(gt["poses"]) == [
        "coffee_table.n.01_1",
        "radio_receiver.n.01_1",
    ]
    assert gt["poses"]["radio_receiver.n.01_1"]["pos"] == [1.0, 0.5, 0.45]
    with pytest.raises(ValueError, match="unknown objects"):
        f.ground_truth_poses(["bowl.n.01_1"])
    p = f.get_robot_position()
    assert (
        p["pos"] == [0.0, 0.0, 0.0]
        and p["yaw"] == 0.0
        and set(p["eef"]) == {"left", "right"}
    )
    raw = f.raw_obs()
    assert raw["proprio"].shape == (8,) and len(raw["joint_names"]) == 28
    assert "task" not in raw, "the BDDL low-dim state (object poses) is privileged"
    assert f.get_task_language() == tasks.LANGUAGE["turning_on_radio"]


def test_code_api_is_the_manifest_every_motion_in_the_low_tier():
    f = FakeSim().facade()
    f._manifest_ready()
    api = f._rpc["code.api"]
    assert api("high")["available"] == []
    low = api("low")["available"]
    assert low[:8] == [
        "get_task_language",
        "get_robot_position",
        "navigate_to_pose",
        "move_hand",
        "move_hand_delta",
        "grasp_object",
        "open_gripper",
        "close_gripper",
    ]
    assert {"back_project", "state", "move_to_joints", "move_along_trajectory"} <= set(
        low
    )
    # segment needs --sam3, point --molmo; neither is set here.
    assert "segment" not in low and "point" not in low
    assert api("raw")["available"] == ["raw_obs", "step", "chunk_step"]
    assert "ground_truth_poses" in api("privileged")["available"]
    assert "ground_truth_poses" not in api()["available"]
    # Every declared method is a registered RPC method, and a call resolves to it.
    method, _kwargs = f.code_api.resolve(
        "move_hand", {"arm": "left", "xyz": [0.5, 0.2, 0.9]}, "low"
    )
    assert method == "env.move_hand" and method in f._rpc
    with pytest.raises(ValueError, match="unknown parameter"):
        f.code_api.resolve(
            "move_hand",
            {"arm": "left", "xyz": [0, 0, 0], "ignore_all_obstacles": True},
            "low",
        )


def test_the_server_checks_itself_against_its_manifest():
    """A facade method the manifest does not declare (nor lists as internal) stops the server,
    and so does a declared one it lacks."""
    from pi_embodied_services.components.manifest import ManifestError

    f = FakeSim().facade()
    f._rpc["env.secret_teleport"] = lambda: None
    with pytest.raises(ManifestError, match="env.secret_teleport"):
        f._manifest_ready()
    g = FakeSim().facade()
    del g._rpc["env.back_project"]
    with pytest.raises(ManifestError, match="back_project"):
        g._manifest_ready()
    h = FakeSim().facade()
    h._molmo_url = "http://molmo"
    h._manifest_ready()
    assert "point" in h._rpc["code.api"]("low")["available"]


def test_task_manifest_and_scene_lookup(tmp_path):
    assert len(tasks.TASKS) == 50 and len(set(tasks.TASK_NAMES)) == 50
    assert tasks.TASK_NAMES[:2] == ("turning_on_radio", "picking_up_trash"), (
        "CaP-X's two tasks first"
    )
    assert tasks.TASK_INDEX["turning_on_radio"] == 0
    scenes = tmp_path / "og_dataset" / "scenes"
    inst = scenes / "Rs_int" / "json" / "Rs_int_task_turning_on_radio_instances"
    inst.mkdir(parents=True)
    for i in (0, 1, 3):
        (inst / f"Rs_int_task_turning_on_radio_0_{i}_template.json").write_text("{}")
    (inst / "Rs_int_task_turning_on_radio_1_0_template.json").write_text(
        "{}"
    )  # another definition
    other = (
        scenes
        / "house_single_floor"
        / "json"
        / "house_single_floor_task_picking_up_trash_instances"
    )
    other.mkdir(parents=True)
    (other / "house_single_floor_task_picking_up_trash_0_0_template.json").write_text(
        "{}"
    )
    assert sim.find_task_scene(tmp_path, "turning_on_radio") == ("Rs_int", [0, 1, 3])
    assert sim.find_task_scene(tmp_path, "picking_up_trash") == (
        "house_single_floor",
        [0],
    )
    with pytest.raises(FileNotFoundError, match="make_pizza"):
        sim.find_task_scene(tmp_path, "make_pizza")
    cfg = sim.task_config(
        activity="picking_up_trash",
        scene_model="house_single_floor",
        instance_id=0,
        image_size=480,
        grasping_mode="sticky",
        max_steps=20000,
    )
    assert (
        cfg["task"]["type"] == "BehaviorTask"
        and cfg["task"]["activity_instance_id"] == 0
    )
    assert cfg["robots"][0]["obs_modalities"] == ["rgb", "depth_linear", "proprio"]
    assert (
        cfg["robots"][0]["sensor_config"]["VisionSensor"]["sensor_kwargs"][
            "image_height"
        ]
        == 480
    )
    assert cfg["scene"]["scene_model"] == "house_single_floor"
    with pytest.raises(ValueError, match="grasping_mode"):
        sim.task_config(
            activity="x",
            scene_model="s",
            instance_id=0,
            image_size=64,
            grasping_mode="magnet",
            max_steps=1,
        )


def test_success_and_goal_fallbacks_for_older_omnigibson():
    # OmniGibson 3.7: no get_goal_option_satisfaction; goal_status of the predicate termination.
    task = SimpleNamespace(
        _termination_conditions={
            "predicate": SimpleNamespace(
                goal_status={"satisfied": [1], "unsatisfied": [0, 2]}
            )
        },
        success=np.array([False]),
    )
    assert sim.goal_satisfaction(task) == [[False, True, False]]
    assert sim.success(None, task) is False
    assert sim.success({"done": {"success": True}}, task) is True
    task.success = np.array([True])
    assert sim.success({}, task) is True
    assert sim.quat_yaw([0, 0, np.sin(0.25), np.cos(0.25)]) == pytest.approx(0.5)


def test_success_is_latched_at_its_first_step_until_reset():
    """A goal that holds mid-primitive and is undone by its end (an object set down and knocked
    off) still counts, as on the other simulators (success_once), until the next reset."""
    fake = FakeSim()
    f = fake.facade()
    f.reset()
    # Steps 1..3 of the primitive: the goal holds at step 2 only.
    fake.success_at, fake.success_until = fake.steps + 2, fake.steps + 3
    r = f.navigate_to_pose(1.0, 0.0, 0.0)
    assert r["success"] is True and r["q_score"] == 1.0
    assert f.state()["success"] is True, "latched: the last step's info says False"
    # The goal never holds again: the latch stays.
    fake.success_at = None
    r = f.open_gripper("left")
    assert r["success"] is True and r["q_score"] == 1.0
    _obs, _rew, _term, _trunc, info = f.step(np.zeros(4))
    assert info["success"] is True
    # A reset starts afresh.
    obs, _ = f.reset()
    assert obs["success"] is False and obs["q_score"] == 0.0
    assert f.state()["success"] is False


def test_move_hand_delta_is_a_bounded_base_frame_step_with_the_gripper_first():
    """The units' primitive: the delta is in the base frame (turned by the base yaw into the
    world), the yaw turns the hand about world +z, a gripper command runs first, and a step
    beyond the per-call limits is refused before anything moves."""
    fake = FakeSim()
    f = fake.facade()
    f.reset()
    fake.yaw = math.pi / 2  # the base faces world +y: its +x (ahead) is world +y
    before = len(fake.calls)
    r = f.move_hand_delta("left", [0.05, 0.0, -0.02], yaw=0.1, gripper="close")
    kinds = [k for k, _ in fake.calls[before:]]
    assert kinds == ["grasp", "move_hand"], kinds
    (pos, quat), _kw = fake.calls[-1][1]
    assert np.allclose(np.asarray(pos), [0.3, 0.25, 0.88], atol=1e-6)
    turned = (
        Rotation.from_quat(np.asarray(quat))
        * Rotation.from_quat([0.0, 0.7071, 0.0, 0.7071]).inv()
    )
    assert np.allclose(turned.as_rotvec(), [0, 0, 0.1], atol=1e-4)
    assert r["arm"] == "left" and "distance_left_m" in r
    before = len(fake.calls)
    f.move_hand_delta("right", [0, 0, 0])
    assert [k for k, _ in fake.calls[before:]] == ["settle"], "a zero step holds"
    with pytest.raises(ValueError, match="the limit is 0.1 m"):
        f.move_hand_delta("left", [0.2, 0, 0])
    with pytest.raises(ValueError, match="the limit is 0.3 rad"):
        f.move_hand_delta("left", [0, 0, 0], yaw=0.5)
    with pytest.raises(ValueError, match="gripper must be"):
        f.move_hand_delta("left", [0, 0, 0.01], gripper="half")


@pytest.mark.parametrize("fork", [False, True])
def test_joint_target_action_converts_the_base_as_each_omnigibson_does(
    monkeypatch, fork
):
    import sys
    import types

    curobo = types.ModuleType("omnigibson.action_primitives.curobo")
    # CaP-X's 3.7 fork: the base joints go through its world-frame helper first.
    if fork:
        curobo.holonomic_base_command_in_world_frame = lambda robot, q: ("world", q)
    pkg = types.ModuleType("omnigibson.action_primitives")
    pkg.curobo = curobo
    monkeypatch.setitem(sys.modules, "omnigibson", types.ModuleType("omnigibson"))
    monkeypatch.setitem(sys.modules, "omnigibson.action_primitives", pkg)
    monkeypatch.setitem(sys.modules, "omnigibson.action_primitives.curobo", curobo)
    monkeypatch.setattr(sim, "tensor", lambda q: tuple(q))
    robot = SimpleNamespace(q_to_action=lambda q: ("action", q))
    got = sim.joint_target_action(robot, [0.1, 0.2])
    # OmniGibson 3.9 has no such helper: q_to_action takes the joint positions as they are.
    assert got == (
        ("action", ("world", (0.1, 0.2))) if fork else ("action", (0.1, 0.2))
    )


def test_primitives_arm_is_settable_on_both_omnigibson_versions():
    class Og39:  # OmniGibson 3.9: a read-only property, the robot's first arm
        robot = SimpleNamespace(default_arm="left")

        @property
        def arm(self):
            return self.robot.default_arm

    class CapxFork:  # CaP-X's 3.7 fork: a plain attribute
        arm = "left"

    p = sim.with_settable_arm(Og39)()
    assert p.arm == "left"
    p.arm = "right"
    assert p.arm == "right" and isinstance(p, Og39)
    assert sim.with_settable_arm(CapxFork) is CapxFork


@pytest.mark.parametrize("major", [0, 5, 6])
def test_set_render_gpu_leaves_kit_the_visible_gpu_on_isaac_sim_6(monkeypatch, major):
    monkeypatch.setattr(sim, "isaacsim_major", lambda: major)
    env = {"CUDA_VISIBLE_DEVICES": "1", "OMNIGIBSON_GPU_ID": "0"}
    sim.set_render_gpu(env)
    if major >= 6:
        # An explicit active_gpu indexes RTX's Vulkan devices, the hidden GPU 0 among them.
        assert env == {"CUDA_VISIBLE_DEVICES": "1"}
    else:
        assert env == {"CUDA_VISIBLE_DEVICES": "1", "OMNIGIBSON_GPU_ID": "0"}


def test_omnigibson_39_instances_are_overlays_found_in_every_split(tmp_path):
    root = tmp_path / sim.CHALLENGE_INSTANCES
    scene, act = "house_double_floor_lower", "turning_on_radio"
    for split, ids in (("scenes", [0, 1]), ("scene_test/public", [301])):
        inst = root / split / scene / "json" / f"{scene}_task_{act}_instances"
        inst.mkdir(parents=True)
        for i in ids:
            (inst / f"{scene}_task_{act}_0_{i}_template-tro_state.json").write_text(
                "{}"
            )
        # the full template of instance 0 and another task's overlay are not instances of it
        (inst.parent / f"{scene}_task_{act}_0_0_template.json").write_text("{}")
    assert sim.find_task_scene(tmp_path, act) == (scene, [0, 1, 301])
    assert sim.instance_split(tmp_path, scene, act, 1) == "train"
    assert sim.instance_split(tmp_path, scene, act, 301) == "public_test"
    with pytest.raises(FileNotFoundError):
        sim.instance_split(tmp_path, scene, act, 7)
    # CaP-X's og_dataset has a full template per instance: loaded directly, no overlay.
    full = (
        tmp_path
        / "og_dataset"
        / "scenes"
        / scene
        / "json"
        / f"{scene}_task_{act}_instances"
    )
    full.mkdir(parents=True)
    (full / f"{scene}_task_{act}_0_5_template.json").write_text("{}")
    assert sim.find_task_scene(tmp_path, act) == (scene, [5])
    assert sim.instance_split(tmp_path, scene, act, 5) is None


def test_launch_sizes_curobo_for_a_house_and_a_shared_gpu(monkeypatch):
    """The primitives are built with the base joint limit that reaches a house's pre-sampled
    poses (OmniGibson's +-5 m fails IK at y = 6.04), no headless viewer camera, and the caller's
    cuRobo batch (its warmup memory grows with it)."""
    import sys
    import types

    built = {}

    class Primitives:
        def __init__(self, env, robot, **kw):
            built.update(kw, env=env, robot=robot)

    class Environment:
        def __init__(self, configs):
            self.configs = configs
            self.robots = ["r1pro"]
            self.task = "task"

    og = types.ModuleType("omnigibson")
    og.Environment = Environment
    pkg = types.ModuleType("omnigibson.action_primitives")
    curobo = types.ModuleType("omnigibson.action_primitives.curobo")
    curobo.m = SimpleNamespace(HOLONOMIC_BASE_PRISMATIC_JOINT_LIMIT=5.0)
    pkg.curobo = curobo
    base = types.ModuleType("omnigibson.action_primitives.action_primitive_set_base")
    base.ActionPrimitiveError = RuntimeError
    starter = types.ModuleType(
        "omnigibson.action_primitives.starter_semantic_action_primitives"
    )
    starter.StarterSemanticActionPrimitives = Primitives
    macros = types.ModuleType("omnigibson.macros")
    macros.gm = SimpleNamespace(RENDER_VIEWER_CAMERA=True)
    for name, mod in {
        "omnigibson": og,
        "omnigibson.action_primitives": pkg,
        "omnigibson.action_primitives.curobo": curobo,
        "omnigibson.action_primitives.action_primitive_set_base": base,
        "omnigibson.action_primitives.starter_semantic_action_primitives": starter,
        "omnigibson.macros": macros,
    }.items():
        monkeypatch.setitem(sys.modules, name, mod)

    h = sim.launch({"scene": {}}, curobo_batch_size=1)
    assert h.env.configs == {"scene": {}} and built["robot"] == "r1pro"
    assert built["curobo_batch_size"] == 1 and built["enable_head_tracking"] is False
    assert (
        curobo.m.HOLONOMIC_BASE_PRISMATIC_JOINT_LIMIT == sim.BASE_JOINT_LIMIT_M == 30.0
    )
    assert macros.gm.RENDER_VIEWER_CAMERA is False
    assert (
        macros.gm.USE_GPU_DYNAMICS is False and macros.gm.ENABLE_OBJECT_STATES is True
    )
