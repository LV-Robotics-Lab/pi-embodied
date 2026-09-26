"""Collision-free motion under --ik: the collision primitives (utils/collision.py), ``ik.check``
and ``robot`` obstacles (components/ik_server.py), the env-server planner (utils/motion.py),
and its wiring into the LIBERO, Franka and dual-Franka env servers. Fakes only: a point
"robot" whose first three joints are its TCP, no simulator, GPU or arm."""

from __future__ import annotations

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from pi_embodied_services.components.ik_server import CuroboBackend, IkFacade
from pi_embodied_services.utils import collision, motion
from pi_embodied_services.utils.rpc import RpcError

UP = [0.0, 0.0, 0.0, 1.0]
POINT_RADIUS = 0.02


# ---------------------------------------------------------------------------
# utils/collision.py


def test_signed_distances_of_every_primitive():
    box = collision.parse_obstacle(
        {"type": "box", "position": [1, 0, 0], "extent": [0.2, 0.2, 0.2]}
    )
    d = collision.point_distance(box, [[1.3, 0, 0], [1.0, 0, 0], [1.3, 0.4, 0.1]])
    assert d[0] == pytest.approx(0.2)
    assert d[1] == pytest.approx(-0.1)  # inside: minus the depth to the nearest face
    assert d[2] == pytest.approx(np.hypot(0.2, 0.3))
    turned = collision.parse_obstacle(
        {
            "type": "box",
            "position": [0, 0, 0],
            "extent": [1.0, 0.1, 0.1],
            "quat_xyzw": Rotation.from_euler("z", 90, degrees=True).as_quat(),
        }
    )
    assert collision.point_distance(turned, [[0, 0.45, 0]])[0] < 0  # long axis is y now
    sphere = collision.parse_obstacle(
        {"type": "sphere", "center": [0, 0, 1], "radius": 0.5}
    )
    assert collision.point_distance(sphere, [[0, 0, 0]])[0] == pytest.approx(0.5)
    capsule = collision.parse_obstacle(
        {"type": "capsule", "position": [0, 0, 0], "radius": 0.1, "height": 1.0}
    )
    assert collision.point_distance(
        capsule, [[0.3, 0, 0.4], [0, 0, 0.8]]
    ) == pytest.approx([0.2, 0.2])
    floor = collision.parse_obstacle(
        {"type": "halfspace", "point": [0, 0, 0.1], "normal": [0, 0, 2]}
    )
    assert collision.point_distance(floor, [[5, 5, 0.4]])[0] == pytest.approx(0.3)
    gap, which = collision.sphere_clearance(
        [box, sphere], [[0, 0, 0.3, 0.1], [0.7, 0, 0, 0.05], [9, 9, 9, 0.0]]
    )
    assert which == 1 and gap == pytest.approx(0.1)  # zero-radius spheres are padding


def test_obstacles_move_between_frames():
    matrix = np.eye(4)
    matrix[:3, :3] = Rotation.from_euler("z", 90, degrees=True).as_matrix()
    matrix[:3, 3] = [1, 0, 0]
    box = collision.parse_obstacle(
        {"type": "box", "name": "b", "position": [1, 0, 0], "extent": [0.4, 0.1, 0.1]}
    )
    moved = collision.transform_obstacle(box, matrix)
    assert moved["position"] == pytest.approx([1, 1, 0])
    assert (
        collision.point_distance(moved, [[1, 1.19, 0]])[0] < 0
    )  # long axis turned to y
    plane = collision.transform_obstacle(
        collision.parse_obstacle(
            {"type": "halfspace", "point": [0, 0, 0], "normal": [1, 0, 0]}
        ),
        matrix,
    )
    assert plane["normal"] == pytest.approx([0, 1, 0])
    wire = collision.to_wire(moved)
    assert wire["name"] == "b" and isinstance(wire["position"], list)


# ---------------------------------------------------------------------------
# ik.check and robot obstacles (components/ik_server.py)


class PointBackend:
    """An ik backend whose robot is a sphere at q[:3]; another arm is two spheres."""

    name = "point"

    def __init__(self):
        self.obstacles: list = []

    def check(self, robot, path, obstacles):
        self.obstacles = obstacles
        spheres = [[*np.asarray(q, float)[:3], POINT_RADIUS] for q in path]
        out = [collision.sphere_clearance(obstacles, [s]) for s in spheres]
        return np.array([c for c, _ in out]), np.array(
            [-1 if w is None else w for _, w in out]
        )

    def robot_spheres(self, robot, q):
        return np.array([[0, 0, 0.1, 0.1], [0, 0, 0.5, 0.05], [0, 0, 0, 0.0]])

    def plan(self, robot, start_q, **kwargs):
        self.obstacles = kwargs["obstacles"]
        return {"ok": True, "path": [start_q], "tcp_path": [[0, 0, 0, 0, 0, 0, 1]]}


def test_check_reports_clearance_and_the_nearest_obstacle():
    facade = IkFacade(PointBackend())
    wall = {
        "type": "box",
        "name": "wall",
        "position": [0.5, 0, 0],
        "extent": [0.1, 1, 1],
    }
    out = facade._dispatch(
        "ik.check",
        (),
        {"robot": "panda", "path": [[0.2, 0, 0], [0.44, 0, 0]], "obstacles": [wall]},
    )
    assert out["collision_free"] is False and out["worst_index"] == 1
    assert out["nearest"] == "wall" and out["min_clearance_m"] == pytest.approx(-0.01)
    assert out["clearances"][0] == pytest.approx(0.23)
    clear = facade.check("panda", q=[0.2, 0, 0], obstacles=[wall], margin=0.1)
    assert clear["collision_free"] is True and clear["checked"] == 1
    assert facade.check("panda", q=[0, 0, 0])["min_clearance_m"] is None
    with pytest.raises(ValueError, match="exactly one of q or path"):
        facade.check("panda", q=[0, 0, 0], path=[[0, 0, 0]])


def test_robot_obstacles_become_the_other_arms_spheres_in_the_planning_base():
    backend = PointBackend()
    facade = IkFacade(backend)
    other = {
        "type": "robot",
        "name": "left_arm",
        "robot": "panda",
        "q": [0.0] * 7,
        "base_pose": {"pos": [0.0, 0.8, 0.0], "quat_xyzw": [0, 0, 1, 0]},  # facing us
    }
    facade.plan("panda", [0.0] * 7, goal_q=[0.0] * 7, obstacles=[other])
    spheres = backend.obstacles
    assert [s["name"] for s in spheres] == [
        "left_arm/0",
        "left_arm/1",
    ]  # radius 0 dropped
    assert spheres[0]["center"] == pytest.approx([0.0, 0.8, 0.1])
    out = facade.check("panda", q=[0.0, 0.75, 0.1], obstacles=[other])
    assert out["collision_free"] is False and out["nearest"] == "left_arm/0"

    class NoSpheres:
        name = "pyroki"

    with pytest.raises(ValueError, match="use --backend curobo"):
        IkFacade(NoSpheres()).check("panda", q=[0] * 7, obstacles=[other])


class _T:
    def __init__(self, v):
        self.v = np.asarray(v, dtype=float)

    def detach(self):
        return self

    def cpu(self):
        return self

    def numpy(self):
        return self.v


class _Kin:
    """franka.yml-like kinematics: 9 dof; the hand at q[:3]; two spheres per configuration."""

    retract_config = _T([0.0] * 9)

    @staticmethod
    def get_dof():
        return 9

    def get_state(self, rows):
        rows = np.asarray(rows, dtype=float)

        class S:
            ee_position = _T(rows[:, :3])
            ee_quaternion = _T(np.tile([0.0, 1.0, 0.0, 0.0], (len(rows), 1)))
            link_spheres_tensor = _T(
                np.stack(
                    [[[*r[:3], 0.05], [r[0], r[1], r[2] + 0.3, 0.08]] for r in rows]
                )
            )

        return S()


class _TA:
    @staticmethod
    def to_device(v):
        return np.asarray(v, dtype=float)


def test_curobo_check_uses_its_collision_spheres_and_reports_tcp_poses(monkeypatch):
    import sys
    import types

    monkeypatch.setitem(sys.modules, "curobo", types.ModuleType("curobo"))

    class Solver:
        kinematics = _Kin()
        tensor_args = _TA()

    backend = CuroboBackend(
        solver_factory=lambda m: Solver(), planner_factory=lambda m: None
    )
    q = [0.4, 0.0, 0.2, 0, 0, 0, 0]
    tcp = backend.tcp_poses("panda", [q])
    assert tcp[0][:3] == pytest.approx([0.4, 0.0, 0.2 - 0.1034])  # hand pointing down
    assert tcp[0][3:] == pytest.approx([1, 0, 0, 0])
    shelf = collision.parse_obstacle(
        {
            "type": "box",
            "name": "shelf",
            "position": [0.4, 0, 0.55],
            "extent": [0.3, 0.3, 0.02],
        }
    )
    clear, which = backend.check("panda", [q, [0.4, 0, 0.0, 0, 0, 0, 0]], [shelf])
    assert clear[0] == pytest.approx(0.55 - 0.01 - 0.5 - 0.08)  # the upper sphere
    assert clear[1] > clear[0] and list(which) == [0, 0]
    spheres = backend.robot_spheres("panda", q)
    assert spheres.shape == (2, 4)


# ---------------------------------------------------------------------------
# utils/motion.py against a fake ik service (the point robot: q[:3] is the TCP)


class PointIk:
    """ik.plan: straight to the goal, or over the top when the straight line hits
    an obstacle (z = 0.24), else refused; ik.check: the point robot's clearance."""

    def __init__(self, fail: Exception | None = None):
        self.calls: list[tuple[str, dict]] = []
        self.fail = fail

    @staticmethod
    def _clear(obstacles, pts):
        parsed = [collision.parse_obstacle(o) for o in obstacles]
        if not parsed:
            return np.inf
        return collision.sphere_clearance(
            parsed, [[*p, POINT_RADIUS] for p in np.asarray(pts)]
        )[0]

    def call(self, method, args=(), kwargs=None, *, timeout_s=None):
        self.calls.append((method, kwargs))
        if self.fail is not None:
            raise self.fail
        if method == "ik.check":
            path = np.asarray(kwargs["path"], dtype=float)
            clear = [self._clear(kwargs["obstacles"], [q[:3]]) for q in path]
            worst = int(np.argmin(clear))
            return {
                "collision_free": bool(min(clear) > kwargs.get("margin", 0.0)),
                "min_clearance_m": None if np.isinf(min(clear)) else float(min(clear)),
                "worst_index": worst,
                "nearest": kwargs["obstacles"][0]["name"]
                if kwargs["obstacles"]
                else None,
            }
        assert method == "ik.plan"
        start = np.asarray(kwargs["start_q"], dtype=float)
        goal = np.asarray(kwargs["goal_pose"]["pos"], dtype=float)
        quat = kwargs["goal_pose"]["quat_xyzw"]
        obstacles = kwargs["obstacles"]
        for via in ([], [[start[0], start[1], 0.24], [goal[0], goal[1], 0.24]]):
            corners = [start[:3], *map(np.asarray, via), goal]
            pts = np.concatenate(
                [np.linspace(a, b, 20) for a, b in zip(corners[:-1], corners[1:])]
            )
            if self._clear(obstacles, pts) > 0:
                path = [[*p, 0, 0, 0, 0] for p in pts]
                return {
                    "ok": True,
                    "path": path,
                    "tcp_path": [[*p, *quat] for p in pts],
                    "backend": "point",
                }
        return {"ok": False, "error": "cuRobo found no collision-free path (IK_FAIL)"}


WALL = {
    "type": "box",
    "name": "wall",
    "position": [0.3, 0, 0.1],
    "extent": [0.05, 0.6, 0.2],
}
BOWL = {
    "type": "box",
    "name": "bowl",
    "position": [0.5, 0, 0.03],
    "extent": [0.1, 0.1, 0.06],
}


def test_plan_goes_around_obstacles_and_leaves_out_the_target():
    ik = PointIk()
    planner = motion.MotionPlanner("http://ik", "panda_libero", client=ik)
    start = [0.1, 0.0, 0.1]
    out = planner.plan([*start, 0, 0, 0, 0], start, [0.5, 0.0, 0.06], UP, [WALL, BOWL])
    assert out["status"] == "planned", out["message"]
    assert out["left_out"] == ["bowl"]  # the goal is at the bowl: it is the target
    assert out["obstacles"] == 1
    tops = [w[2] for w in out["waypoints"]]
    assert max(tops) == pytest.approx(0.24, abs=1e-6)  # over the wall
    assert out["waypoints"][-1][:3] == pytest.approx([0.5, 0.0, 0.06])
    assert 2 <= len(out["waypoints"]) <= motion.MAX_SEGMENTS
    assert len(out["q_path"]) == len(out["waypoints"])
    sent = ik.calls[0][1]
    assert [o["name"] for o in sent["obstacles"]] == ["wall"]


def test_plan_refuses_blocked_and_detouring_paths_and_is_unknown_without_a_service():
    roof = {
        "type": "box",
        "name": "roof",
        "position": [0.3, 0, 0.3],
        "extent": [0.05, 2, 0.8],
    }
    planner = motion.MotionPlanner("http://ik", "panda_libero", client=PointIk())
    blocked = planner.plan(
        [0.1, 0, 0.1, 0, 0, 0, 0], [0.1, 0, 0.1], [0.5, 0, 0.1], UP, [roof]
    )
    assert (
        blocked["status"] == "blocked"
        and "no collision-free path" in blocked["message"]
    )
    with pytest.raises(ValueError, match="move_to refused: no collision-free path"):
        motion.require_planned(blocked, "move_to")
    tight = motion.MotionPlanner(
        "http://ik", "panda_libero", client=PointIk(), max_detour_m=0.1
    )
    detour = tight.plan(
        [0.1, 0, 0.1, 0, 0, 0, 0], [0.1, 0, 0.1], [0.5, 0, 0.1], UP, [WALL]
    )
    assert detour["status"] == "blocked" and "detours" in detour["message"]
    down = motion.MotionPlanner(
        "http://ik", "panda_libero", client=PointIk(fail=RpcError("ik.plan", "down"))
    )
    unknown = down.plan([0.1, 0, 0.1, 0, 0, 0, 0], [0.1, 0, 0.1], [0.5, 0, 0.1], UP, [])
    assert unknown["status"] == "unknown"
    motion.require_planned(unknown, "move_to")  # not a refusal: the move runs unplanned
    assert down.check([[0] * 7], [WALL])["status"] == "unknown"


def test_plan_and_check_convert_the_scene_into_the_base_frame():
    ik = PointIk()
    planner = motion.MotionPlanner("http://ik", "panda_libero", client=ik)
    base = {"pos": [-0.5, 0.0, 0.9], "quat_xyzw": UP}
    wall_world = dict(WALL, position=[-0.2, 0.0, 1.0])
    out = planner.plan(
        [0.4, 0, 0.3, 0, 0, 0, 0],
        [-0.1, 0.0, 1.2],
        [0.0, 0.0, 0.95],
        UP,
        [wall_world],
        base_pose=base,
    )
    sent = ik.calls[0][1]
    assert sent["goal_pose"]["pos"] == pytest.approx([0.5, 0.0, 0.05])
    assert sent["obstacles"][0]["position"] == pytest.approx([0.3, 0.0, 0.1])
    assert out["waypoints"][-1][:3] == pytest.approx([0.0, 0.0, 0.95])  # back in world
    verdict = planner.check([[0.29, 0.0, 0.1]], [wall_world], base_pose=base)
    assert verdict["status"] == "contact" and verdict["nearest"] == "wall"
    assert "predicted contact with wall" in verdict["message"]


def test_waypoints_are_spaced_and_capped():
    line = np.array([[0.01 * i, 0, 0, 0, 0, 0, 1] for i in range(101)], dtype=float)
    idx = motion.select_waypoints(line, max_segments=5, min_segment_m=0.02)
    assert idx[-1] == 100 and len(idx) <= 5
    assert motion.select_waypoints(line[:3], min_segment_m=0.05) == [2]


def test_static_world_reads_the_env_file(tmp_path, monkeypatch):
    monkeypatch.delenv(motion.WORLD_ENV, raising=False)
    assert motion.static_world() == []
    path = tmp_path / "cell.json"
    path.write_text(
        '{"obstacles": [{"type": "halfspace", "name": "table", '
        '"point": [0, 0, 0], "normal": [0, 0, 1]}]}'
    )
    monkeypatch.setenv(motion.WORLD_ENV, str(path))
    assert motion.static_world()[0]["name"] == "table"


def test_follow_waypoints_checks_before_each_segment_and_stops_on_contact():
    pose = np.array([0.0, 0.0, 0.0, 0, 0, 0, 1])
    moves: list = []

    def move(d):
        pose[:3] += d
        moves.append(("move", d))
        return {"ok": True}

    def rotate(r):
        pose[3:] = (
            Rotation.from_euler("xyz", r) * Rotation.from_quat(pose[3:])
        ).as_quat()
        moves.append(("rotate", r))
        return {"ok": True}

    turned = Rotation.from_euler("z", 0.5).as_quat()
    wps = [[0.1, 0, 0, 0, 0, 0, 1], [0.1, 0.1, 0, *turned], [0.2, 0.1, 0, *turned]]
    checks: list[int] = []

    def check(i):
        checks.append(i)
        return {"status": "contact" if i == 2 else "clear", "message": "boom"}

    out = motion.follow_waypoints(
        wps, tcp_pose=lambda: pose, move=move, rotate=rotate, check=check
    )
    assert out["stopped"] == "contact" and out["segments"] == 2 and checks == [0, 1, 2]
    assert [m[0] for m in moves] == ["move", "move", "rotate"]
    assert pose[:3] == pytest.approx([0.1, 0.1, 0.0])
    assert Rotation.from_quat(pose[3:]).as_euler("xyz")[2] == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# LIBERO: move_to plans through the scene and checks before each servo segment


class PlanningArmSim:
    """An OSC-like LIBERO stand-in with the base at the world origin, joints whose first three
    are the TCP (the point robot), and a scene answering ``collision_world``."""

    def __init__(self, scene):
        # 0.3 m in xy from the targets below: move_to refuses a longer move.
        self.pos = np.array([0.2, 0.0, 0.1])
        self.scene = scene
        self.workers = [self]
        self.steps = 0

    @property
    def env(self):
        return self

    @property
    def current_raw_obs(self):
        return [
            {
                "robot0_eef_pos": self.pos.copy(),
                "robot0_eef_quat": np.array(UP),
                "robot0_joint_pos": np.array([*self.pos, 0, 0, 0, 0]),
                "robot0_gripper_qpos": np.array([0.04, -0.04]),
            }
        ]

    def env_call(self, name, target):
        if name == "robot_base_pose":
            return {"pos": [0.0, 0.0, 0.0], "quat_xyzw": UP}
        assert name == "collision_world"
        return {"obstacles": self.scene}

    def step(self, action):
        a = np.asarray(action, dtype=np.float64).reshape(-1, 7)[0]
        self.steps += 1
        self.pos = self.pos + a[:3] * 0.05
        zeros = np.zeros(1, dtype=bool)
        obs = {"main_images": np.zeros((1, 2, 2, 3), dtype=np.uint8)}
        return (
            obs,
            np.zeros(1),
            zeros,
            zeros,
            {"episode": {"success_once": np.array([False])}},
        )


def libero(scene, ik=None):
    from pi_embodied_services.robots.libero.env_server import LiberoEnvFacade

    ik = ik or PointIk()
    f = LiberoEnvFacade(
        PlanningArmSim(scene),
        meta={},
        ik_motion=motion.MotionPlanner("http://ik", "panda_libero", client=ik),
    )
    return f, ik


def test_libero_move_to_follows_a_planned_path_around_the_wall():
    f, ik = libero([WALL, BOWL])
    out = f.move_to([0.5, 0.0, 0.06], max_steps=120)
    assert out["final_dist_m"] < 0.012, out
    assert out["planned"]["segments"] >= 2 and "stopped" not in out
    # The arm went over the wall rather than through it.
    checks = [k for m, k in ik.calls if m == "ik.check"]
    assert len(checks) == out["planned"]["segments"]
    assert all([o["name"] for o in c["obstacles"]] == ["wall"] for c in checks)
    assert f._env.steps == out["steps_used"]
    # The RPCs exist only with --ik; the plan never carries obstacle geometry.
    plan = f._dispatch("env.plan_motion", ([0.1, 0.0, 0.2],), {})
    assert plan["status"] == "planned" and "q_path" not in plan
    assert isinstance(plan["left_out"], int)


def test_libero_move_to_refuses_without_a_collision_free_path_and_stops_on_contact():
    roof = {
        "type": "box",
        "name": "roof",
        "position": [0.3, 0, 0.3],
        "extent": [0.05, 2, 0.8],
    }
    f, _ = libero([roof])
    with pytest.raises(ValueError, match="move_to refused: no collision-free path"):
        f.move_to([0.5, 0.0, 0.1])
    assert f._env.steps == 0

    class ClosingIk(PointIk):
        """The wall moves into the way once the arm has started (a second check says contact)."""

        def call(self, method, args=(), kwargs=None, *, timeout_s=None):
            out = super().call(method, args, kwargs, timeout_s=timeout_s)
            if (
                method == "ik.check"
                and sum(m == "ik.check" for m, _ in self.calls) >= 2
            ):
                out = dict(
                    out, collision_free=False, min_clearance_m=-0.01, nearest="wall/3"
                )
            return out

    f, _ = libero([WALL], ClosingIk())
    out = f.move_to([0.5, 0.0, 0.06], max_steps=120)
    assert out["stopped"] == "contact" and "wall" in out["contact"]
    assert out["final_dist_m"] > 0.1


def test_libero_without_ik_has_no_planning_rpcs():
    from pi_embodied_services.robots.libero.env_server import LiberoEnvFacade

    f = LiberoEnvFacade(PlanningArmSim([WALL]), meta={})
    assert "env.plan_motion" not in f._rpc and "env.check_motion" not in f._rpc
    out = f.move_to([0.5, 0.0, 0.1])  # straight through: no world without --ik
    assert out["final_dist_m"] < 0.012


# ---------------------------------------------------------------------------
# Franka and dual Franka: planned move_delta, arm-arm checks


class PointFranka:
    """A single-Franka backend: the TCP moves by move_delta; joints are [tcp, 0...]."""

    def __init__(self):
        self.tcp = np.array([0.1, 0.0, 0.1, *UP])
        self.moves: list = []
        self.max_z = 0.0

    def get_robot_state(self):
        return {
            "raw_base_state": {
                "tcp_pose": self.tcp.copy(),
                "arm_joint_position": np.array([*self.tcp[:3], 0, 0, 0, 0]),
            }
        }

    def move_delta(self, delta_xyz):
        self.moves.append(("move", list(delta_xyz)))
        self.tcp[:3] += delta_xyz
        self.max_z = max(self.max_z, float(self.tcp[2]))
        return {"ok": True, "final_tcp_pose": self.tcp.tolist()}

    def rotate_delta(self, delta_rpy):
        self.moves.append(("rotate", list(delta_rpy)))
        return {"ok": True}

    def __getattr__(self, name):
        return lambda *a, **k: {}


def test_franka_move_delta_follows_a_plan_and_refuses_a_blocked_one(
    tmp_path, monkeypatch
):
    pytest.importorskip("omegaconf")
    from pi_embodied_services.robots.franka.env_server import FrankaEnvFacade

    cell = tmp_path / "cell.json"
    cell.write_text(
        '[{"type": "box", "name": "wall", "position": [0.3, 0, 0.1], '
        '"extent": [0.05, 0.6, 0.2]}]'
    )
    monkeypatch.setenv(motion.WORLD_ENV, str(cell))
    backend = PointFranka()
    ik = PointIk()
    f = FrankaEnvFacade(
        backend, ik_motion=motion.MotionPlanner("http://ik", "panda", client=ik)
    )
    out = f._dispatch("env.move_delta", (), {"delta_xyz": [0.4, 0.0, 0.0]})
    assert (
        out["ok"] is True
        and out["planned"]["segments_done"] == out["planned"]["segments"]
    )
    assert backend.tcp[:3] == pytest.approx([0.5, 0.0, 0.1])
    assert backend.max_z > 0.2  # climbed over the wall
    roof = '[{"type": "box", "name": "roof", "position": [0.3, 0, 0.3], "extent": [0.05, 2, 0.8]}]'
    cell.write_text(roof)
    blocked = FrankaEnvFacade(
        backend, ik_motion=motion.MotionPlanner("http://ik", "panda", client=PointIk())
    )
    n = len(backend.moves)
    with pytest.raises(ValueError, match="env.move_delta refused"):
        blocked._dispatch("env.move_delta", ([-0.4, 0.0, 0.0],), {})
    assert len(backend.moves) == n


class PointDualFranka:
    """Two point arms: right base at the world origin, left base 0.6 m along +y turned to face
    it (z 180 deg); poses in the rig frame (right_base) and each arm's own."""

    LEFT_BASE = np.array([0.6, 0.6, 0.0])

    def __init__(self):
        self.tcp = {
            "right": np.array([0.3, 0.1, 0.2]),
            "left": np.array([0.3, 0.5, 0.2]),
        }
        self.moves: list = []

    def _local(self, arm, p):
        if arm == "right":
            return p.copy()
        rel = p - self.LEFT_BASE
        return np.array([-rel[0], -rel[1], rel[2]])

    def get_robot_state(self):
        flip = Rotation.from_euler("z", 180, degrees=True)
        out = {}
        for arm in ("left", "right"):
            local = self._local(arm, self.tcp[arm])
            world_q = UP
            raw_q = (
                UP
                if arm == "right"
                else (flip.inv() * Rotation.from_quat(UP)).as_quat()
            )
            out[f"{arm}_arm"] = {
                "tcp_pose": np.array([*self.tcp[arm], *world_q]),
                "raw_tcp_pose": np.array([*local, *raw_q]),
                "arm_joint_position": np.array([*local, 0, 0, 0, 0]),
            }
        return out

    def move_delta(self, arm, delta_xyz):
        self.moves.append((arm, list(delta_xyz)))
        self.tcp[arm] = self.tcp[arm] + np.asarray(delta_xyz)
        return {"ok": True}

    def rotate_delta(self, arm, delta_rpy):
        return {"ok": True}

    def __getattr__(self, name):
        return lambda *a, **k: {}


class ArmSpheresIk(PointIk):
    """PointIk whose ``robot`` obstacles are the other point arm's TCP (a 5 cm sphere) and
    base column, as the ik facade would expand them."""

    def call(self, method, args=(), kwargs=None, *, timeout_s=None):
        kwargs = dict(kwargs)
        expanded = []
        for o in kwargs.get("obstacles", []):
            if o.get("type") != "robot":
                expanded.append(o)
                continue
            base = o["base_pose"]
            m = np.eye(4)
            m[:3, :3] = Rotation.from_quat(base["quat_xyzw"]).as_matrix()
            m[:3, 3] = base["pos"]
            tip = m[:3, :3] @ np.asarray(o["q"][:3]) + m[:3, 3]
            expanded.append(
                {
                    "type": "sphere",
                    "name": f"{o['name']}/tcp",
                    "center": tip.tolist(),
                    "radius": 0.05,
                }
            )
        kwargs["obstacles"] = expanded
        self.raw = kwargs
        return super().call(method, args, kwargs, timeout_s=timeout_s)


def dual(ik):
    pytest.importorskip("omegaconf")
    from pi_embodied_services.robots.dual_franka.env_server import DualFrankaEnvFacade

    backend = PointDualFranka()
    return backend, DualFrankaEnvFacade(
        backend, ik_motion=motion.MotionPlanner("http://ik", "panda", client=ik)
    )


def test_dual_franka_places_the_other_arm_in_the_moving_arms_base_frame(monkeypatch):
    monkeypatch.delenv(motion.WORLD_ENV, raising=False)
    ik = ArmSpheresIk()
    backend, f = dual(ik)
    frame = f._arm_frame("left")
    assert frame["base_pose"]["pos"] == pytest.approx([0.6, 0.6, 0.0])
    other = frame["robots"][0]
    assert other["name"] == "right_arm"
    # The right arm's base seen from the left base: 0.6 m ahead and 0.6 m to its right... turned.
    assert other["base_pose"]["pos"] == pytest.approx([0.6, 0.6, 0.0])
    out = f._dispatch("env.move_delta", ("right", [0.0, 0.1, 0.0]), {})
    assert out["ok"] is True and backend.tcp["right"] == pytest.approx([0.3, 0.2, 0.2])


def test_dual_franka_refuses_a_move_into_the_other_arm_and_stops_when_it_comes_close(
    monkeypatch,
):
    monkeypatch.delenv(motion.WORLD_ENV, raising=False)

    class NoDetourIk(ArmSpheresIk):
        """Plans only straight lines (a narrow cell)."""

        def call(self, method, args=(), kwargs=None, *, timeout_s=None):
            out = super().call(method, args, kwargs, timeout_s=timeout_s)
            if method == "ik.plan" and out.get("ok"):
                heights = [p[2] for p in out["path"]]
                if max(heights) > max(heights[0], heights[-1]) + 1e-6:
                    return {
                        "ok": False,
                        "error": "cuRobo found no collision-free path (TRAJOPT_FAIL)",
                    }
            return out

    backend, f = dual(NoDetourIk())
    # Right TCP (0.3, 0.1) toward the left TCP (0.3, 0.5): 0.35 m would end inside it.
    with pytest.raises(
        ValueError, match="env.move_delta refused: no collision-free path"
    ):
        f._dispatch(
            "env.move_delta", (), {"arm": "right", "delta_xyz": [0.0, 0.35, 0.0]}
        )
    assert backend.moves == []
    # A short move is planned; the left arm then swings into its way before segment 2.
    ik = ArmSpheresIk()
    backend, f = dual(ik)
    original = backend.move_delta

    def move_and_intrude(arm, delta_xyz):
        out = original(arm, delta_xyz)
        backend.tcp["left"] = backend.tcp["right"] + np.array([0.0, 0.03, 0.0])
        return out

    backend.move_delta = move_and_intrude
    out = f._dispatch("env.move_delta", ("right", [0.0, 0.15, 0.0]), {})
    assert out["stopped"] == "contact" and "left_arm" in out["contact"]
    assert (
        out["ok"] is False
        and out["planned"]["segments_done"] < out["planned"]["segments"]
    )
