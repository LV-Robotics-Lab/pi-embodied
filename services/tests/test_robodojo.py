"""RoboDojo glue that runs without Isaac Sim: the config assembly of RoboDojo's main.py, the task
inventory, the policy-client stub, the geometry helpers (waypoints, back-projection), and the env
server's primitives on a fake EvalEnv with RoboDojo's action-dict contract."""

import sys
from types import SimpleNamespace

import numpy as np
import pytest
import yaml

from pi_embodied_services.robots.robodojo import env_server, sim
from pi_embodied_services.robots.robodojo.env_server import RobodojoEnvFacade

# -- config assembly -----------------------------------------------------------------


def write(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(data if isinstance(data, str) else yaml.safe_dump(data))


@pytest.fixture
def root(tmp_path):
    """A RoboDojo checkout's config tree, shaped like @726e9aa's."""
    cfg = tmp_path / "env_cfg"
    write(
        cfg / "arx_x5.yml",
        {
            "config_name": "arx_x5",
            "config": {
                "sim": "sim_config",
                "scene": "default",
                "robot": "dual_x5",
                "camera": "camera_config",
            },
            "observation": {"collect_freq": 25, "vision": {"depth": False}},
        },
    )
    write(
        cfg / "sim" / "sim_config.yml",
        {"dt": 0.004, "render_interval": 10, "scene": {"num_envs": 10}},
    )
    scene = {
        "Table": {"default": "t"},
        "Ground": {"materials": {"default": "g"}},
        "Background": {},
    }
    write(cfg / "scene" / "default.yml", scene)
    write(cfg / "scene" / "conveyor.yml", {**scene, "Conveyor": True})
    write(
        cfg / "camera" / "camera_config.yml",
        {
            "annotator": {
                "common": {"enabled": True, "rgb_capture": {"type": "rgb"}},
                "cam_head": {"enabled": True, "rgb_capture": {"type": "rgb"}},
                "cam_off": {"enabled": False},
            }
        },
    )
    robots = {"robots": [{"robot_name": "x5"}, {"robot_name": "x5"}]}
    write(cfg / "robot" / "dual_x5.yml", robots)
    write(
        cfg / "robot" / "dual_x5_and_franka_competition.yml",
        {"robots": [*robots["robots"], {"robot_name": "franka"}]},
    )
    task = tmp_path / "task" / "RoboDojo"
    write(task / "task_registry.py", "")
    write(
        task / "config" / "_task.yml",
        {
            "common": {
                "data_source": "datagen",
                "scene_config": "default",
                "render_interval": 10,
                "eval_nums": 50,
            },
            "tasks": {
                "stack_bowls": {"data_source": "teleop", "eval_nums": 25},
                "make_kong": {"robot_config": "dual_x5_and_franka_competition"},
                "fill_egg_holder": {
                    "render_interval": 5,
                    "robot_self_collision": False,
                },
                "pick_from_conveyor_by_image": {"scene_config": "conveyor"},
            },
        },
    )
    for name in (
        "stack_bowls",
        "make_kong",
        "fill_egg_holder",
        "pick_from_conveyor_by_image",
        "push_T_random",
    ):
        write(
            task / "config" / f"{name}.yml",
            {"Rigid": [{"select_mode": {"label": ["a"]}}]},
        )
        write(task / "tasks" / f"{name}.py", "")
    write(task / "tasks" / "orphan.py", "")  # no config: not a task
    layouts = tmp_path / "Assets" / "Eval_Layout" / "RoboDojo" / "arx_x5" / "0"
    for i in (0, 1, 2, 10):
        write(layouts / f"stack_bowls_{i}.json", "{}")
    write(layouts / "stack_bowls_random_0.json", "{}")
    return tmp_path


def test_task_inventory_skips_the_shared_config_and_orphans(root):
    assert sim.task_names(root) == [
        "fill_egg_holder",
        "make_kong",
        "pick_from_conveyor_by_image",
        "push_T_random",
        "stack_bowls",
    ]
    assert sim.layout_count(root, "stack_bowls") == 4  # not stack_bowls_random_*
    assert sim.layout_count(root, "stack_bowls", eval_seed=1) == 0


def test_dimensions_cover_42_base_tasks_and_random_variants_inherit():
    base = [t for tasks in sim.DIMENSIONS.values() for t in tasks]
    assert len(base) == len(set(base)) == 42
    assert {k: len(v) for k, v in sim.DIMENSIONS.items()} == {
        "generalization": 12,
        "memory": 6,
        "precision": 8,
        "long-horizon": 8,
        "open": 8,
    }
    assert sim.dimension("push_T_random") == sim.dimension("push_T") == "generalization"
    assert sim.dimension("nope") is None


def test_build_config_follows_main_py(root):
    cfg, n = sim.build_config(root, "stack_bowls")
    assert n == 25
    assert cfg["sim"]["physx"]["enable_stabilization"] is True  # teleop data
    assert cfg["sim"]["scene"]["num_envs"] == 1 and cfg["sim"]["seed"] == [0]
    assert cfg["camera"]["default_frequency"] == 25
    assert (
        cfg["scene"]["Table"]["random"] is False
        and cfg["scene"]["Ground"]["materials"]["random"] is False
    )
    e = cfg["eval_cfg"]
    assert (e["task_name"], e["num_envs"], e["eval_num"], e["seed"]) == (
        "stack_bowls",
        1,
        25,
        0,
    )
    # Depth for back-projection: every enabled camera also renders distance_to_image_plane.
    ann = cfg["camera"]["annotator"]
    assert (
        ann["cam_head"]["distance_to_image_plane_capture"]["type"]
        == "distance_to_image_plane"
    )
    assert (
        "distance_to_image_plane_capture" in ann["common"]
        and "distance_to_image_plane_capture" not in ann["cam_off"]
    )
    assert e["observation"]["vision"]["depth"] is True
    plain, _ = sim.build_config(root, "stack_bowls", depth=False)
    assert (
        "distance_to_image_plane_capture"
        not in plain["camera"]["annotator"]["cam_head"]
    )


def test_build_config_per_task_overrides(root):
    kong, n = sim.build_config(root, "make_kong")
    assert n == 50 and len(kong["robot"]["robots"]) == 3  # the Franka support arm
    assert "physx" not in kong["sim"]  # datagen: no stabilization
    egg, _ = sim.build_config(root, "fill_egg_holder")
    assert egg["sim"]["render_interval"] == 5
    assert all(r["enabled_self_collisions"] is False for r in egg["robot"]["robots"])
    conv, _ = sim.build_config(root, "pick_from_conveyor_by_image")
    assert conv["scene"]["Conveyor"] is True


def test_model_client_stub_stands_in_for_xpolicylab(monkeypatch):
    for name in ("client_server", "client_server.ws", "client_server.ws.model_client"):
        monkeypatch.delitem(sys.modules, name, raising=False)
    sim.install_model_client_stub()
    from client_server.ws.model_client import WsModelClient

    c = WsModelClient(url="ws://x", evaluation_id="e")
    c.call(func_name="reset")
    assert c.calls == ["reset"]


# -- geometry ----------------------------------------------------------------------


def test_waypoints_bound_translation_and_rotation_per_step():
    q0 = np.array([1.0, 0, 0, 0])
    q1 = sim.yaw_quat(q0, 0.4)
    path = sim.waypoints([0, 0, 0], [0.1, 0, 0], q0, q1, step_m=0.01, step_rad=0.05)
    assert len(path) == 10  # 0.1 m at 1 cm; 0.4 rad at 0.05 would be 8
    assert np.allclose(path[-1][:3], [0.1, 0, 0]) and np.allclose(path[-1][3:], q1)
    prev = np.r_[0.0, 0, 0, q0]
    for p in path:
        assert np.linalg.norm(p[:3] - prev[:3]) <= 0.01 + 1e-9
        assert sim.quat_angle(p[3:], prev[3:]) <= 0.05 + 1e-9
        prev = p
    turn = sim.waypoints(
        [0, 0, 0], [0, 0, 0], q0, sim.yaw_quat(q0, 0.3), step_m=0.01, step_rad=0.05
    )
    assert len(turn) == 6


def test_yaw_quat_is_counter_clockwise_about_world_z():
    q = sim.yaw_quat([1.0, 0, 0, 0], np.pi / 2)
    w, x, y, z = q
    R = np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )
    assert np.allclose(R @ [1, 0, 0], [0, 1, 0])
    assert env_server._yaw_between([1.0, 0, 0, 0], q) == pytest.approx(np.pi / 2)


def test_intrinsics_honour_a_non_square_aperture():
    K = sim.intrinsics(640, 480, 10.0, 22.212, 14.266)
    assert K[0, 0] == pytest.approx(640 * 10 / 22.212) and K[1, 1] == pytest.approx(
        480 * 10 / 14.266
    )
    assert (K[0, 2], K[1, 2]) == (320.0, 240.0)
    square = sim.intrinsics(640, 480, 10.0, 20.0, None)
    assert square[0, 0] == square[1, 1]


def test_back_project_inverts_projection_through_an_opengl_camera():
    # A USD camera 1 m above the table plane z = 0, looking straight down (-z of the camera = -z of the world).
    gl = np.eye(4)
    gl[:3, 3] = [0.2, -0.1, 1.0]
    T = sim.gl_to_cv(gl)
    K = sim.intrinsics(64, 48, 10.0, 20.0, None)
    depth = np.full((48, 64), 1.0, dtype=np.float32)
    depth[0, 0] = np.inf  # no hit
    p = np.array([0.25, -0.12, 0.0])
    c = np.linalg.inv(T) @ np.r_[p, 1.0]
    px = [K[0, 0] * c[0] / c[2] + K[0, 2], K[1, 1] * c[1] / c[2] + K[1, 2]]
    xyz = sim.back_project(depth, K, T, [px, [0, 0], [100, 5]], radius=0)
    assert np.allclose(xyz[0], p, atol=1e-5)
    assert xyz[1] is None and xyz[2] is None
    # Image rows grow downward: a point further +y in the world (camera up is +y) is higher in the image.
    up = sim.back_project(depth, K, T, [[32, 10]], radius=0)[0]
    assert up[1] > -0.1


# -- the facade on a fake EvalEnv ---------------------------------------------------------


class FakeRobot:
    def __init__(self, arm):
        self.arm_name = f"{arm}_arm"
        self.gripper_name = f"{arm}_ee"
        self.type = "target"
        self.gripper_scale = [0.0, 0.044]
        self.gripper_move = {"sign": 1, "mimic": [0, 1, 0]}
        self.gripper_bias = 0.145


class FakeRobotManager:
    """Joint 0..2 are the end-effector xyz (an identity 'arm'); IK fails beyond 0.5 m of reach."""

    BASE = {
        "left": np.array([-0.3, -0.45, 0.765]),
        "right": np.array([0.3, -0.45, 0.765]),
    }

    def __init__(self, env):
        self.env = env
        self.robot_list = [FakeRobot("left"), FakeRobot("right")]
        self.q = {
            r.arm_name: np.array([0.0, 0.3, 0.2, 0.0, 0.0, 0.0])
            for r in self.robot_list
        }
        self.grip = {r.arm_name: 0.044 for r in self.robot_list}
        self.control_manager = SimpleNamespace(prev_control=[{}])

    def _side(self, robot):
        return robot.arm_name.split("_")[0]

    def get_joint(self, robot, env_idx_list=None):
        return {0: self.q[robot.arm_name].copy()}

    def get_real_endpose(self, robot, env_idx_list=None, is_relative=True):
        q = self.q[robot.arm_name]
        return {
            0: np.r_[
                self.BASE[self._side(robot)] + q[:3], sim.yaw_quat([1.0, 0, 0, 0], q[3])
            ]
        }

    def get_end_effector_real_val(self, robot, env_idx_list=None):
        return {0: np.array([self.grip[robot.arm_name], self.grip[robot.arm_name]])}

    def solve_ik(self, target_pose, env_idx, robot):
        rel = np.asarray(target_pose[:3]) - self.BASE[self._side(robot)]
        if np.linalg.norm(rel) > 0.5:
            return {"status": "Fail"}
        w, x, y, z = target_pose[3:]
        yaw = np.arctan2(2 * (x * y + z * w), 1 - 2 * (y * y + z * z))
        return {"status": "Success", "joint_value": np.r_[rel, yaw, 0.0, 0.0]}


class FakeEvalEnv:
    """RoboDojo's EvalEnv contract as the facade uses it (src/eval_client/eval_env.py)."""

    def __init__(self, step_lim=200):
        self.robot_manager = FakeRobotManager(self)
        self.step_lim = step_lim
        self.take_action_cnt = [0]
        self.end_flag = [False]
        self.success = [True]
        self.actions: list[dict] = []
        self.resets: list = []
        self.closed = 0
        self.solve_when = None  # a predicate on the fake arms: success
        self.obs_manager = SimpleNamespace(
            instruction=["Stack the three bowls together."]
        )
        self.reward_manager = SimpleNamespace(get_score=lambda: [15.0])
        self.run_reward_calls = 0

    def reset(self, seed):
        self.resets.append(list(seed))
        self.take_action_cnt = [0]
        self.end_flag = [False]
        self.success = [True]
        self.robot_manager.q = {
            r.arm_name: np.array([0.0, 0.3, 0.2, 0.0, 0.0, 0.0])
            for r in self.robot_manager.robot_list
        }

    def close(self):
        self.closed += 1

    def run_reward(self):
        self.run_reward_calls += 1

    def get_score(self):
        pass

    def validate_action_dict(self, action):
        allowed = {
            f"{a}_{k}"
            for a in ("left", "right")
            for k in ("arm_joint_state", "ee_joint_state", "ee_pose")
        }
        bad = set(action) - allowed
        if bad:
            raise ValueError(f"Unexpected state keys: {sorted(bad)}")

    def take_action(self, action):
        self.validate_action_dict(action)
        if self.end_flag[0] or self.take_action_cnt[0] >= self.step_lim:
            return
        self.actions.append(action)
        self.take_action_cnt[0] += 1
        rm = self.robot_manager
        for a in ("left", "right"):
            if f"{a}_arm_joint_state" in action:
                rm.q[f"{a}_arm"] = np.asarray(
                    action[f"{a}_arm_joint_state"], dtype=float
                )
            if f"{a}_ee_joint_state" in action:
                rm.grip[f"{a}_arm"] = float(action[f"{a}_ee_joint_state"][0]) * 0.044
        # is_episode_end
        if self.solve_when and self.solve_when(rm):
            self.end_flag[0], self.success[0] = True, True
        elif self.take_action_cnt[0] >= self.step_lim:
            self.end_flag[0], self.success[0] = True, False

    def get_obs(self):
        img = np.zeros((48, 64, 4), dtype=np.uint8)
        img[..., 0] = self.take_action_cnt[0] % 256
        vision = {
            c: {"color": img[..., :3], "depth": np.ones((48, 64), np.float32)}
            for c in sim.CAMERAS
        }
        return {
            "vision": vision,
            "state": {},
            "instruction": self.obs_manager.instruction[0],
        }


@pytest.fixture
def facade(monkeypatch):
    monkeypatch.setattr(
        sim, "unstable_error", lambda: type("UnStableError", (Exception,), {})
    )
    env = FakeEvalEnv()
    f = RobodojoEnvFacade(
        app=SimpleNamespace(close=lambda: None),
        env=env,
        meta={"task": "stack_bowls", "seed": 0, "layouts": 25, "eval_seed": 0},
    )
    f.stop_requested = lambda: False
    f.reset()
    return f


def test_reset_loads_the_layout_and_registers_the_checks(facade):
    env = facade._env
    assert env.resets == [[0]] and env.run_reward_calls == 1 and env.closed == 0
    obs, info = facade.reset(seed=3)
    # A second reset relaunches the simulation first, as RoboDojo's main.py does between batches.
    assert env.resets[-1] == [3] and env.closed == 1 and info["seed"] == 3
    assert set(obs) >= {
        "head",
        "left_wrist",
        "right_wrist",
        "arms",
        "success",
        "score",
        "env_steps",
    }
    assert obs["head"].shape == (48, 64, 3) and obs["head"].dtype == np.uint8
    assert info["instruction"] == "Stack the three bowls together."
    with pytest.raises(ValueError, match="layout id"):
        facade.reset(seed=25)


def test_move_delta_moves_one_arm_and_holds_the_other(facade):
    env = facade._env
    right_before = env.robot_manager.q["right_arm"].copy()
    r = facade.move_delta("left", [0.0, 0.05, 0.03], return_frames=True)
    assert r["executed"] == r["waypoints"] == 6  # 5.8 cm at 1 cm per control step
    assert (
        np.allclose(r["moved_m"], [0.0, 0.05, 0.03], atol=1e-4)
        and r["final_error_m"] < 1e-4
    )
    assert np.allclose(env.robot_manager.q["right_arm"], right_before)
    # Every control step is a joint-mode RoboDojo action with both arms and both grippers.
    a = env.actions[-1]
    assert set(a) == {
        "left_arm_joint_state",
        "right_arm_joint_state",
        "left_ee_joint_state",
        "right_ee_joint_state",
    }
    assert a["left_ee_joint_state"] == [1.0]  # open after reset
    assert r["env_steps"] == 6 and "stopped" not in r
    assert len(r["frames"]) == 1  # one head frame every FRAME_EVERY control steps


def test_move_delta_with_a_gripper_command_holds_it_first(facade):
    r = facade.move_delta("right", [0.0, 0.0, -0.02], gripper=0.0)
    assert r["control_steps"] == env_server.GRIPPER_STEPS + 2
    assert facade._env.actions[0]["right_ee_joint_state"] == [0.0]
    assert r["arms"]["right"]["gripper"] == pytest.approx(0.0)


def test_move_to_stops_at_an_unreachable_waypoint(facade):
    r = facade.move_to(
        "left", [-0.3, 0.05, 1.215]
    )  # 0.67 m from the base: the tail is out of reach
    assert r["stopped"] == "ik_failed" and 0 < r["executed"] < r["waypoints"]
    with pytest.raises(ValueError, match="limit"):
        facade.move_to("left", [0.5, 0.5, 0.5])


def test_an_ik_branch_flip_is_refused(facade, monkeypatch):
    rm = facade._env.robot_manager
    real = rm.solve_ik
    monkeypatch.setattr(
        rm,
        "solve_ik",
        lambda **k: {
            **real(**k),
            "joint_value": real(**k)["joint_value"] + [0, 0, 0, 0, 1.2, 0],
        },
    )
    r = facade.move_delta("left", [0.0, 0.0, 0.02])
    assert r["stopped"].startswith("ik_jump") and r["executed"] == 0


def test_rotate_delta_turns_about_the_vertical_and_clips(facade):
    r = facade.rotate_delta("right", 0.3)
    assert r["commanded_yaw"] == 0.3 and r["yaw"] == pytest.approx(0.3, abs=1e-3)
    assert np.allclose(r["moved_m"], 0.0, atol=1e-6)
    r = facade.rotate_delta("right", 5.0)
    assert r["clipped"] is True and r["commanded_yaw"] == env_server.MAX_ROTATE_RAD


def test_go_home_returns_both_arms_to_their_reset_joints(facade):
    facade.move_delta("left", [0.1, 0.0, 0.1])
    facade.move_delta("right", [-0.05, 0.0, 0.05])
    facade.go_home()
    for arm in ("left", "right"):
        assert np.allclose(facade._env.robot_manager.q[f"{arm}_arm"], facade._home[arm])


def test_success_and_the_step_limit_end_the_episode(facade):
    env = facade._env
    env.solve_when = lambda rm: rm.q["left_arm"][2] > 0.25
    r = facade.move_delta("left", [0.0, 0.0, 0.1])
    assert r["success"] and r["ended"] and r["score"] == 1.0
    assert r["executed"] == 5  # stopped once RoboDojo ended the episode
    assert facade.move_delta("left", [0.0, 0.0, 0.01])["error"] == "the episode is over"
    facade.reset()
    env.solve_when, env.step_lim = None, 4
    r = facade.move_delta("right", [0.0, 0.1, 0.0])
    assert r["truncated"] and not r["success"] and r["score"] == pytest.approx(0.15)
    assert r["env_steps"] == 4


def test_native_actions_pass_through_validation_and_update_the_hold(facade):
    env = facade._env
    q = [0.05, 0.3, 0.25, 0.0, 0.0, 0.0]
    act = {"left_arm_joint_state": q, "left_ee_joint_state": [0.2]}
    obs, score, term, trunc, info = facade.step(act)
    assert (
        env.actions[-1] is act
        and np.allclose(facade._q["left"], q)
        and facade._grip["left"] == 0.2
    )
    with pytest.raises(ValueError, match="Unexpected"):
        facade.step({"arm_joint_state": q})
    obs, term, trunc, info = facade.chunk_step([act, act, act], return_all_frames=True)
    assert info["executed"] == 3 and len(obs["frames"]) == 3


def test_get_obs_drops_depth_unless_asked(facade):
    assert "depth" not in facade.get_obs()["vision"]["cam_head"]
    assert "depth" in facade.get_obs(depth=True)["vision"]["cam_head"]


def test_flywheel_recording_returns_one_frame_per_control_step(facade):
    first = facade.set_recording(True)
    assert first["state"].shape == (14,) and first["head"].shape == (48, 64, 3)
    r = facade.move_delta("left", [0.0, 0.0, 0.03])
    frames = r["policy_frames"]
    assert len(frames) == 3 and all(f["action"].shape == (14,) for f in frames)
    assert frames[-1]["action"][2] == pytest.approx(
        0.23
    )  # left joint 2 = z after the move
    assert facade.state().get("policy_frames") is None
    assert facade.set_recording(False) is None
    assert "policy_frames" not in facade.move_delta("left", [0.0, 0.0, 0.01])


def test_an_unstable_layout_is_reported_and_refuses_motion(monkeypatch):
    err = type("UnStableError", (Exception,), {})
    monkeypatch.setattr(sim, "unstable_error", lambda: err)
    env = FakeEvalEnv()

    def unstable(seed):
        raise err("All scene Unstable Error!")

    env.reset = unstable
    f = RobodojoEnvFacade(
        app=SimpleNamespace(close=lambda: None),
        env=env,
        meta={"task": "t", "seed": 0, "layouts": 5},
    )
    f.stop_requested = lambda: False
    _, info = f.reset()
    assert "unstable" in info["error"]
    assert "unstable" in f.move_delta("left", [0, 0, 0.01])["error"]
    assert env.run_reward_calls == 0


def test_the_tcp_is_gripper_bias_along_the_fingers(facade):
    arm = facade.state()["arms"]["left"]
    # At reset the fake hand has no yaw: the fingers point along +x.
    assert np.allclose(arm["tcp_pos"], arm["eef_pos"] + [0.145, 0, 0], atol=1e-6)
    facade.rotate_delta("left", 1.0)  # clipped to 0.8 rad
    arm = facade.state()["arms"]["left"]
    yaw = env_server.MAX_ROTATE_RAD
    assert np.allclose(
        arm["tcp_pos"] - arm["eef_pos"],
        [0.145 * np.cos(yaw), 0.145 * np.sin(yaw), 0],
        atol=1e-5,
    )


def test_a_motion_stopped_by_contact_is_reported_blocked_and_holds_where_it_is(
    facade, monkeypatch
):
    env = facade._env
    real = env.take_action

    def table(action):  # the hand cannot go below joint z = 0.15 (fingers on the table)
        a = dict(action)
        if "left_arm_joint_state" in a:
            q = np.asarray(a["left_arm_joint_state"], dtype=float).copy()
            q[2] = max(q[2], 0.15)
            a["left_arm_joint_state"] = q.tolist()
        real(a)

    monkeypatch.setattr(env, "take_action", table)
    r = facade.move_delta("left", [0.0, 0.0, -0.1])
    assert r["stopped"] == "blocked" and "table" in r["hint"]
    assert r["executed"] == r["waypoints"] == 10 and r[
        "final_error_m"
    ] == pytest.approx(0.05, abs=1e-6)
    # The held target is where the arm is, not the unreached goal.
    assert facade._q["left"][2] == pytest.approx(0.15)
    free = facade.move_delta("left", [0.0, 0.0, 0.02])
    assert "stopped" not in free
