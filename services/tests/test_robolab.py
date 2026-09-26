"""RoboLab glue that runs without Isaac Sim: quaternion order across Isaac Lab 2.x/3.x, the
orientation hold, tilt, the yaw primitive on a mock hand, RoboLab's freeze-based success, and the
subtask read-out."""

import sys
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest

from pi_embodied_services.robots.robolab import env_server, sim


@pytest.mark.parametrize("xyzw", [False, True])
def test_lab_quat_and_ee_quat_follow_the_installed_isaaclab_order(monkeypatch, xyzw):
    monkeypatch.setattr(sim, "isaaclab_xyzw", lambda: xyzw)
    wxyz = (0.9, 0.1, 0.2, 0.3)
    lab = sim.lab_quat(wxyz)
    assert lab == ((0.1, 0.2, 0.3, 0.9) if xyzw else wxyz)
    # rl_ee_quat reads the hand body's quaternion in Isaac Lab's order and always returns wxyz.
    robot = SimpleNamespace(
        data=SimpleNamespace(
            body_names=["panda_link0", "panda_hand"],
            body_quat_w=np.array([[[1.0, 0, 0, 0], list(lab)]]),
        )
    )
    env = SimpleNamespace(scene={"robot": robot})
    assert np.allclose(sim.rl_ee_quat(env), wxyz)


def test_to_np_unwraps_a_proxy_array():
    class Proxy:  # Isaac Lab 3 ProxyArray: explicit .torch
        torch = np.arange(3.0)

    assert np.array_equal(sim.to_np(Proxy()), np.arange(3.0))


def test_tilt_and_orientation_hold():
    down = np.array([0.0, 1.0, 0.0, 0.0])  # 180 deg about x: +Z points down
    assert sim.ee_tilt_deg(down) == pytest.approx(0.0, abs=1e-3)
    assert np.allclose(sim.hold_orientation_rotvec(down, down), 0.0)
    a = 0.1  # 0.1 rad off about world x
    off = np.array([np.cos((np.pi + a) / 2), np.sin((np.pi + a) / 2), 0.0, 0.0])
    rv = sim.hold_orientation_rotvec(down, off)
    assert np.allclose(rv, [-a, 0.0, 0.0], atol=1e-9)
    big = np.array([np.cos((np.pi + 1.0) / 2), np.sin((np.pi + 1.0) / 2), 0.0, 0.0])
    assert np.linalg.norm(sim.hold_orientation_rotvec(down, big)) == pytest.approx(
        sim.ORIENT_HOLD_MAX_RAD
    )


def quat_mul(q, r):
    qw, qx, qy, qz = q
    rw, rx, ry, rz = r
    return np.array(
        [
            qw * rw - qx * rx - qy * ry - qz * rz,
            qw * rx + qx * rw + qy * rz - qz * ry,
            qw * ry - qx * rz + qy * rw + qz * rx,
            qw * rz + qx * ry - qy * rx + qz * rw,
        ]
    )


def quat_from_rotvec(v):
    angle = float(np.linalg.norm(v))
    if angle < 1e-12:
        return np.array([1.0, 0.0, 0.0, 0.0])
    axis = np.asarray(v) / angle
    return np.array([np.cos(angle / 2), *(axis * np.sin(angle / 2))])


def x_axis(q):
    w, x, y, z = q
    return np.array([1 - 2 * (y * y + z * z), 2 * (x * y + z * w), 2 * (x * z - y * w)])


DOWN = np.array([0.0, 1.0, 0.0, 0.0])  # 180 deg about x: the approach axis points down


def test_yaw_quat_turns_about_world_z_counter_clockwise_and_keeps_the_tilt():
    q = sim.yaw_quat(DOWN, 0.3)
    assert sim.yaw_between(DOWN, q) == pytest.approx(0.3, abs=1e-9)
    assert sim.yaw_between(q, DOWN) == pytest.approx(-0.3, abs=1e-9)
    assert sim.ee_tilt_deg(q) == pytest.approx(sim.ee_tilt_deg(DOWN), abs=1e-6)
    # +yaw is right-handed about +z: the hand's x axis swings from +x toward +y (counter-clockwise
    # seen from above), and the hold's correction from the old heading is that same yaw.
    before, after = x_axis(DOWN), x_axis(q)
    assert np.cross(before, after)[2] > 0
    assert np.allclose(
        sim.hold_orientation_rotvec(q, DOWN, max_rad=np.inf), [0, 0, 0.3]
    )
    tilted = sim.yaw_quat(quat_mul(quat_from_rotvec([0.1, 0.0, 0.0]), DOWN), -0.2)
    assert sim.ee_tilt_deg(tilted) == pytest.approx(np.degrees(0.1), abs=1e-6)


class Scene:
    """``env.scene``: subscriptable for the robot, with the env origins."""

    def __init__(self, robot):
        self.robot = robot
        self.env_origins = np.zeros((1, 3))

    def __getitem__(self, name):
        assert name == "robot"
        return self.robot


class FakeIsaac:
    """A Panda hand under RoboLab's saturated relative IK: every control step achieves ``achieved``
    of the commanded (scaled) translation and axis-angle, in the world frame like Isaac Lab's
    ``apply_delta_pose``. ``ending`` scripts (terminated, truncated, freeze) per step index."""

    IK_SCALE = 0.5

    def __init__(self, monkeypatch, achieved=0.3, ending=None):
        self.pos = np.array([0.4, 0.0, 0.3])
        self.quat = DOWN.copy()
        self.n = 0
        self.ending = ending or {}
        self.achieved = achieved
        data = SimpleNamespace(
            body_names=["panda_link0", "panda_hand"],
            joint_names=["panda_joint1", "panda_finger_joint1", "panda_finger_joint2"],
            joint_pos=np.array([[0.0, 0.04, 0.04]]),
        )
        self.env = SimpleNamespace(
            scene=Scene(SimpleNamespace(data=data)),
            device="cpu",
            _frozen_envs=np.array([False]),
            _env_results={},
            _has_stepped=False,
        )
        self.obs = {
            "image_obs": {
                "front_cam": np.zeros((1, 4, 6, 3), np.uint8),
                "wrist_cam": np.zeros((1, 4, 4, 3), np.uint8),
            }
        }
        self.sync()
        monkeypatch.setattr(sim, "isaaclab_xyzw", lambda: False)
        monkeypatch.setattr(sim, "step", self.step)
        monkeypatch.setattr(sim, "reset", self.reset)
        self.facade = env_server.RobolabEnvFacade(
            app=None,
            handle=SimpleNamespace(
                env=self.env, ik_scale=self.IK_SCALE, instruction="x"
            ),
            meta={
                "agentview_camera": "front_cam",
                "wrist_camera": "wrist_cam",
                "settle_steps": 2,
                "subtask": False,
            },
        )

    def sync(self):
        data = self.env.scene.robot.data
        data.body_pos_w = np.array([[[0.0, 0.0, 0.0], self.pos]])
        data.body_quat_w = np.array([[[1.0, 0.0, 0.0, 0.0], self.quat]])

    def reset(self, env):
        assert env is self.env
        self.env._frozen_envs[:] = False
        self.env._env_results.clear()
        self.env._has_stepped = False
        return self.obs

    def step(self, env, action):
        assert env is self.env
        a = np.asarray(action, dtype=float)
        assert a.shape == (7,)
        self.env._has_stepped = True
        self.n += 1
        if not self.env._frozen_envs[0]:
            self.pos = self.pos + a[:3] * self.IK_SCALE * self.achieved
            self.quat = quat_mul(
                quat_from_rotvec(a[3:6] * self.IK_SCALE * self.achieved), self.quat
            )
            self.sync()
        term, trunc, freeze = self.ending.get(self.n, (False, False, False))
        if freeze:
            self.env._frozen_envs[0] = True
            self.env._env_results[0] = bool(term)
        return self.obs, term, trunc


def test_rotate_delta_turns_the_hand_by_the_yaw_and_holds_position_and_tilt(
    monkeypatch,
):
    fake = FakeIsaac(monkeypatch)
    f = fake.facade
    obs, _ = f.reset()
    assert obs["yaw_deg"] == 0.0
    r = f.rotate_delta(0.3)
    assert r["commanded_yaw"] == 0.3 and "clipped" not in r
    assert r["yaw"] == pytest.approx(0.3, abs=env_server.YAW_TOL_RAD)
    assert 1 <= r["decisions"] <= env_server.MAX_ROTATE_DECISIONS
    assert r["control_steps"] == r["decisions"] * env_server.STEPS_PER_DECISION
    assert np.allclose(r["moved_m"], 0.0)
    assert r["tilt_deg"] == pytest.approx(0.0, abs=0.01)
    assert r["yaw_deg"] == pytest.approx(np.degrees(0.3), abs=1)
    assert np.cross(x_axis(DOWN), x_axis(fake.quat))[2] > 0, (
        "counter-clockwise from above"
    )
    # Beyond the per-call limit the yaw is clipped and the clip reported; the sign is honoured.
    r = f.rotate_delta(-0.6, return_frames=True)
    assert (
        r["clipped"] is True
        and r["commanded_yaw"] == -0.3
        and r["requested_yaw"] == -0.6
    )
    assert r["yaw"] == pytest.approx(-0.3, abs=env_server.YAW_TOL_RAD)
    assert r["yaw_deg"] == pytest.approx(0.0, abs=1)
    assert len(r["frames"]) == r["decisions"]
    # A turn within tolerance costs no step.
    assert f.rotate_delta(0.0)["control_steps"] == 0


def test_move_delta_keeps_the_heading_the_turn_left(monkeypatch):
    fake = FakeIsaac(monkeypatch)
    f = fake.facade
    f.reset()
    # Untouched by the yaw work: with the reset heading held, a move commands no rotation.
    r = f.move_delta([0.02, 0.0, 0.0])
    assert r["yaw_deg"] == 0.0 and r["decisions"] == 1 and r["moved_m"][0] > 0
    f.rotate_delta(0.3)
    before = sim.rl_ee_quat(fake.env)
    r = f.move_delta([0.0, 0.02, 0.0])
    assert r["yaw_deg"] == pytest.approx(np.degrees(0.3), abs=1)
    assert sim.yaw_between(before, sim.rl_ee_quat(fake.env)) == pytest.approx(
        0, abs=0.01
    )
    # Once the episode is over neither primitive moves.
    fake.ending = {fake.n + 1: (True, False, True)}
    f.move_delta([0.02, 0.0, 0.0])
    assert f.state()["success"] is True
    r = f.rotate_delta(0.3)
    assert r["error"] == "the episode is over" and r["yaw"] == 0.0


def test_a_termination_robolab_resets_away_is_not_a_success(monkeypatch):
    # Isaac may report `terminated` in an episode's first two steps (a physics artifact);
    # RobolabEnv then resets the env instead of freezing it: no result, the episode goes on.
    fake = FakeIsaac(
        monkeypatch, ending={1: (True, False, False), 2: (True, False, False)}
    )
    f = fake.facade
    obs, _ = f.reset()  # settle_steps 2: both artifact steps happen here
    assert obs["success"] is False and obs["terminated"] is False
    assert obs["env_steps"] == 2
    r = f.move_delta([0.02, 0.0, 0.0])
    assert r["decisions"] == 1 and r["success"] is False
    # The real ending freezes the env and stores the predicate.
    fake.ending = {fake.n + 3: (True, False, True)}
    r = f.move_delta([0.04, 0.0, 0.0])
    assert r["success"] is True and r["terminated"] is True
    assert r["control_steps"] == 3 and r["decisions"] == 1
    # A stock env without RoboLab's bookkeeping ends on any termination.
    plain = SimpleNamespace()
    assert sim.rl_ended(plain, True, False) is True
    assert sim.rl_success(plain) is False


def test_a_fresh_episode_reports_no_subtask_progress(monkeypatch):
    class Term:
        subtask_state_machines = [object()]
        infos = [{"status": 0, "completed": 2, "total": 3, "score": 0.6, "info": "x"}]

    mod = ModuleType("robolab.core.events.subtask_recorder")
    mod.SubtaskCompletionRecorderTerm = Term
    for name in ("robolab", "robolab.core", "robolab.core.events"):
        monkeypatch.setitem(sys.modules, name, ModuleType(name))
    monkeypatch.setitem(sys.modules, mod.__name__, mod)
    manager = SimpleNamespace(get_term=lambda cls: Term() if cls is Term else None)
    env = SimpleNamespace(recorder_manager=manager, _has_stepped=True)
    assert sim.rl_subtask(env) == {
        "completed": 2,
        "total": 3,
        "score": 0.6,
        "info": "x",
    }
    # Right after reset the term's infos still hold the last episode's values.
    env._has_stepped = False
    assert sim.rl_subtask(env) == {"completed": 0, "total": 3, "score": 0.0, "info": ""}


def test_renderer_overrides_are_refused_before_isaac_starts_on_isaac_lab_3(monkeypatch):
    monkeypatch.setattr(sim, "isaaclab_major", lambda: 3)
    monkeypatch.setattr(
        sim, "launch_isaac", lambda **_: pytest.fail("Kit must not start")
    )
    monkeypatch.setattr(sys, "argv", ["env_server", "--renderer", "pathtracing"])
    with pytest.raises(SystemExit, match="only the realtime renderer"):
        env_server.main()
    monkeypatch.setattr(sys, "argv", ["env_server", "--rendering-type", "quality"])
    with pytest.raises(SystemExit, match="only the realtime renderer"):
        env_server.main()
