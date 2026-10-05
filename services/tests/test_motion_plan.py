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
    fk = IkFacade(backend)._dispatch("ik.fk", (), {"robot": "panda", "q": q})
    assert fk["pos"] == pytest.approx(tcp[0][:3]) and fk["quat_xyzw"] == pytest.approx(
        [1, 0, 0, 0]
    )


# ---------------------------------------------------------------------------
# utils/motion.py against a fake ik service (the point robot: q[:3] is the TCP)


class PointIk:
    """ik.plan: straight to the goal, or over the top when the straight line hits
    an obstacle (z = 0.24), else refused; ik.check: the point robot's clearance."""

    def __init__(self, fail: Exception | None = None):
        self.calls: list[tuple[str, dict]] = []
        self.fail = fail

    @staticmethod
    def _nearest(obstacles, p):
        if not obstacles:
            return None
        parsed = [collision.parse_obstacle(o) for o in obstacles]
        _, j = collision.sphere_clearance(parsed, [[*p, POINT_RADIUS]])
        return obstacles[j]["name"]

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
        if method == "ik.solve":
            pos = kwargs["target_pose"]["pos"]
            ok = float(np.linalg.norm(pos)) < 0.9
            return {
                "ok": ok,
                "q": [*pos, 0, 0, 0, 0],
                "error": None if ok else "unreachable",
            }
        if method == "ik.check":
            path = np.asarray(
                kwargs["path"] if "path" in kwargs else [kwargs["q"]], dtype=float
            )
            clear = [self._clear(kwargs["obstacles"], [q[:3]]) for q in path]
            worst = int(np.argmin(clear))
            return {
                "collision_free": bool(min(clear) > kwargs.get("margin", 0.0)),
                "min_clearance_m": None if np.isinf(min(clear)) else float(min(clear)),
                "worst_index": worst,
                "nearest": self._nearest(kwargs["obstacles"], path[worst][:3]),
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
    args = ([0.1, 0, 0.1, 0, 0, 0, 0], [0.1, 0, 0.1], [0.5, 0, 0.1], UP, [])
    # A planner that answers with an error refuses the move, flag or not, and an unchecked
    # segment stops it (the dual rig's robot obstacle on PyRoKi was let through this way).
    errored = RpcError("ik.plan", "ValueError: robot obstacles need collision spheres")
    for allow in (False, True):
        bad = motion.MotionPlanner(
            "http://ik", "panda", client=PointIk(fail=errored), allow_unplanned=allow
        )
        out = bad.plan(*args)
        assert out["status"] == "blocked" and "collision spheres" in out["message"]
        with pytest.raises(ValueError, match="move_to refused: path not planned"):
            motion.require_planned(out, "move_to")
        assert bad.check([[0] * 7], [WALL])["status"] == "contact"
    # An unreachable service refuses by default ...
    down_exc = RpcError("ik.plan", "HTTP request failed: refused")
    down_exc.__cause__ = ConnectionRefusedError("refused")
    down = motion.MotionPlanner(
        "http://ik", "panda_libero", client=PointIk(fail=down_exc)
    )
    assert down.plan(*args)["status"] == "blocked"
    assert down.check([[0] * 7], [WALL])["status"] == "contact"
    # ... and runs unplanned only with --ik-allow-unplanned.
    lenient = motion.MotionPlanner(
        "http://ik", "panda_libero", client=PointIk(fail=down_exc), allow_unplanned=True
    )
    unknown = lenient.plan(*args)
    assert unknown["status"] == "unknown"
    motion.require_planned(unknown, "move_to")  # not a refusal: the move runs unplanned
    assert lenient.check([[0] * 7], [WALL])["status"] == "unknown"


def test_the_unplanned_flag_is_off_by_default():
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--ik")
    motion.add_unplanned_argument(p)
    assert motion.planner_from_args(p.parse_args([]), "panda") is None
    assert (
        motion.planner_from_args(
            p.parse_args(["--ik", "http://ik:1"]), "panda"
        ).allow_unplanned
        is False
    )
    on = p.parse_args(["--ik", "http://ik:1", "--ik-allow-unplanned"])
    assert motion.planner_from_args(on, "panda").allow_unplanned is True


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


def test_follow_waypoints_stops_after_a_failed_leg():
    """Audit a2c880c #3: a leg that failed (refused, or blocked short of its waypoint) was
    followed by the rest of the path, planned from a pose the arm never reached."""
    pose = np.array([0.0, 0.0, 0.0, 0, 0, 0, 1])
    moves: list = []

    def move(d):
        moves.append(d)
        if len(moves) == 2:  # blocked halfway along the second leg
            pose[:3] += np.asarray(d) / 2
            return {"ok": True}
        pose[:3] += d
        return {"ok": True}

    def rotate(r):
        return {"ok": True}

    wps = [
        [0.1, 0, 0, 0, 0, 0, 1],
        [0.1, 0.1, 0, 0, 0, 0, 1],
        [0.2, 0.1, 0, 0, 0, 0, 1],
    ]
    out = motion.follow_waypoints(
        wps, tcp_pose=lambda: pose, move=move, rotate=rotate, check=lambda i: {}
    )
    assert out["stopped"] == "stalled" and out["segments"] == 1 and len(moves) == 2
    assert out["stalled"] == {"waypoint": 1, "move": "translate", "short_m": 0.05}
    pose[:] = [0.0, 0.0, 0.0, 0, 0, 0, 1]
    refused = motion.follow_waypoints(
        wps,
        tcp_pose=lambda: pose,
        move=lambda d: {"ok": False, "error": "limit"},
        rotate=rotate,
        check=lambda i: {},
    )
    assert refused["stopped"] == "stalled" and refused["segments"] == 0


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
    # pi's --max-move wide enough for one 0.4 m call (the planner splits it into segments).
    f = FrankaEnvFacade(
        backend,
        ik_motion=motion.MotionPlanner("http://ik", "panda", client=ik),
        limits={"max_move_m": 1.0},
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
        backend,
        ik_motion=motion.MotionPlanner("http://ik", "panda", client=PointIk()),
        limits={"max_move_m": 1.0},
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
        backend,
        ik_motion=motion.MotionPlanner("http://ik", "panda", client=ik),
        limits={"max_move_m": 1.0},  # pi's --max-move out of the planner's way
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


def test_the_dual_rig_refuses_to_start_with_ik_on_a_backend_without_arm_spheres(
    monkeypatch, capsys
):
    pytest.importorskip("omegaconf")
    import sys

    from pi_embodied_services.robots.dual_franka.env_server import DualFrankaEnvFacade
    from pi_embodied_services.robots.franka import env_server as franka

    def run(backend_or_exc):
        def fake(url, **kw):
            if isinstance(backend_or_exc, Exception):
                raise backend_or_exc
            return backend_or_exc

        monkeypatch.setattr(motion, "ik_backend", fake)
        monkeypatch.setattr(
            sys, "argv", ["x", "--task-description", "t", "--ik", "http://ik:1"]
        )

        def no_config(*a, **k):
            raise RuntimeError("started past the ik check")

        return franka.main(
            create_worker_class=lambda: None,
            load_runtime_config=no_config,
            facade_class=DualFrankaEnvFacade,
        )

    for answer, why in (
        ("pyroki", "needs the ik server's --backend curobo"),
        (OSError("refused"), "cannot read the ik backend"),
    ):
        with pytest.raises(SystemExit):
            run(answer)
        assert why in capsys.readouterr().err
    with pytest.raises(RuntimeError, match="started past the ik check"):
        run("curobo")


def test_curobo_keeps_the_base_spheres_and_excludes_what_the_base_already_sits_in(
    monkeypatch,
):
    import sys
    import types

    monkeypatch.setitem(sys.modules, "curobo", types.ModuleType("curobo"))

    class Config:
        """Sphere 0 on panda_link0 (the base, 5 cm), sphere 1 on the hand."""

        link_name_to_idx_map = {"panda_link0": 0, "panda_hand": 1}

        @staticmethod
        def get_sphere_index_from_link_name(name):
            return _T([0] if name == "panda_link0" else [1])

    class Kin(_Kin):
        kinematics_config = Config()

        def get_state(self, rows):
            rows = np.asarray(rows, dtype=float)

            class S:
                link_spheres_tensor = _T(
                    np.stack([[[0, 0, 0.05, 0.05], [*r[:3], 0.04]] for r in rows])
                )

            return S()

    class Solver:
        kinematics = Kin()
        tensor_args = _TA()

    backend = CuroboBackend(
        solver_factory=lambda m: Solver(), planner_factory=lambda m: None
    )
    table = {
        "type": "box",
        "name": "table/7",
        "position": [0.3, 0, -0.1],
        "extent": [
            1.0,
            1.0,
            0.25,
        ],  # top at z = 0.025: the base sphere sits 7.5 cm in it
    }
    shelf = {
        "type": "box",
        "name": "shelf",
        "position": [0.4, 0, 0.3],
        "extent": [0.2, 0.2, 0.02],
    }
    kept, excluded = backend.base_excluded(
        "panda",
        [0.4, 0, 0.3, 0, 0, 0, 0],
        [collision.parse_obstacle(o) for o in (table, shelf)],
    )
    # Audit 92245e3 CUROBO-1: the table is not dropped from the arm's world; it is carved
    # around the base (four pieces keeping its name), so only the (table, base) pair is excused.
    assert excluded == ["table/7"]
    assert [o["name"] for o in kept] == ["table/7"] * 4 + ["shelf"]
    base_sphere = [[0, 0, 0.05, 0.05]]
    assert collision.sphere_clearance(kept[:4], base_sphere)[0] >= 0.01 - 1e-9
    assert min(collision.point_distance(o, [[0.4, 0, 0.0]])[0] for o in kept[:4]) < 0
    out = IkFacade(backend).check(
        "panda", q=[0.4, 0, 0.25, 0, 0, 0, 0], obstacles=[table, shelf]
    )
    assert out["excluded_by_base"] == ["table/7"]
    assert (
        out["collision_free"] is False and out["nearest"] == "shelf"
    )  # the hand still counts
    # A path whose hand dips into the table away from the base is refused, by the table.
    out = IkFacade(backend).check(
        "panda",
        path=[[0.4, 0, 0.3, 0, 0, 0, 0], [0.4, 0, 0.0, 0, 0, 0, 0]],
        obstacles=[table, shelf],
    )
    assert out["collision_free"] is False and out["worst_index"] == 1
    assert out["nearest"] == "table/7" and out["excluded_by_base"] == ["table/7"]
    # Robots without static links (ur5e) exclude nothing.
    assert backend.base_excluded("ur5e", [0] * 6, kept) == (kept, [])


def test_carve_column_cuts_only_the_base_out_of_a_box():
    """Audit 92245e3 CUROBO-1: the column the base's spheres reach is cut out of the box
    (whatever its orientation); the rest keeps the box's name and surfaces."""
    from scipy.spatial.transform import Rotation

    from pi_embodied_services.components.ik_server import carve_column

    for quat in (
        [0, 0, 0, 1],
        Rotation.from_euler("z", 45, degrees=True).as_quat(),
        Rotation.from_euler("x", 20, degrees=True).as_quat(),
    ):
        box = collision.parse_obstacle(
            {
                "type": "box",
                "name": "table",
                "position": [0.3, 0, -0.1],
                "extent": [1.0, 1.0, 0.25],
                "quat_xyzw": list(quat),
            }
        )
        pieces = carve_column(box, 0.06, 0.0, 0.1)
        assert 1 <= len(pieces) <= 4 and all(p["name"] == "table" for p in pieces)
        # The base's sphere is a margin clear of every piece ...
        assert (
            collision.sphere_clearance(pieces, [[0, 0, 0.05, 0.05]])[0] >= 0.01 - 1e-9
        )
        # ... while the table 10 cm and 40 cm out of the axis is still solid.
        for point in ([0.12, 0, -0.1], [0.4, 0, -0.1]):
            assert min(collision.point_distance(p, [point])[0] for p in pieces) < 0
    # A column that misses the box leaves it whole.
    far = collision.parse_obstacle(
        {
            "type": "box",
            "name": "shelf",
            "position": [1, 0, 0],
            "extent": [0.2, 0.2, 0.2],
        }
    )
    assert carve_column(far, 0.06, 0.0, 0.1) == [far]


def _fake_curobo(monkeypatch):
    """The cuRobo modules ``CuroboBackend.plan`` imports, as doubles."""
    import sys
    import types

    class WorldConfig:
        def __init__(self, d):
            self.d = d

        @classmethod
        def from_dict(cls, d):
            return cls(d)

    class JointState:
        def __init__(self, position):
            self.position = position

        @classmethod
        def from_position(cls, position):
            return cls(position)

    class MotionGenPlanConfig:
        def __init__(self, **kw):
            self.kw = kw

    for name, attrs in (
        ("curobo", {}),
        ("curobo.geom", {}),
        ("curobo.geom.types", {"WorldConfig": WorldConfig}),
        ("curobo.types", {}),
        ("curobo.types.state", {"JointState": JointState}),
        ("curobo.wrap", {}),
        ("curobo.wrap.reacher", {}),
        (
            "curobo.wrap.reacher.motion_gen",
            {"MotionGenPlanConfig": MotionGenPlanConfig},
        ),
    ):
        module = types.ModuleType(name)
        module.__dict__.update(attrs)
        monkeypatch.setitem(sys.modules, name, module)


def test_curobo_plans_against_the_table_carved_around_the_base(monkeypatch):
    """Audit 92245e3 CUROBO-1: ``plan`` gave cuRobo a world without the table the base stands
    in, so a trajectory sweeping the forearm through the table came back collision_free. The
    planner's world now holds the table's pieces around the base (and the check the same)."""
    _fake_curobo(monkeypatch)

    class Config:
        link_name_to_idx_map = {"panda_link0": 0, "panda_hand": 1}

        @staticmethod
        def get_sphere_index_from_link_name(name):
            return _T([0] if name == "panda_link0" else [1])

    class Kin(_Kin):
        kinematics_config = Config()

        def get_state(self, rows):
            rows = np.asarray(rows, dtype=float)

            class S:
                link_spheres_tensor = _T(
                    np.stack([[[0, 0, 0.05, 0.05], [*r[:3], 0.04]] for r in rows])
                )

            return S()

    class Solver:
        kinematics = Kin()
        tensor_args = _TA()

    class Planner:
        kinematics = Kin()
        tensor_args = _TA()

        class world_coll_checker:
            @staticmethod
            def clear_cache():
                pass

        def __init__(self):
            self.worlds = []

        def update_world(self, world):
            self.worlds.append(world.d)

        def plan_single_js(self, start, goal, cfg):
            class Result:
                success = _T([False])
                status = "MotionGenStatus.TRAJOPT_FAIL"

            return Result()

    planner = Planner()
    backend = CuroboBackend(
        solver_factory=lambda m: Solver(), planner_factory=lambda m: planner
    )
    table = {
        "type": "box",
        "name": "table/7",
        "position": [0.3, 0, -0.1],
        "extent": [1.0, 1.0, 0.25],
    }
    shelf = {
        "type": "box",
        "name": "shelf",
        "position": [0.4, 0, 0.3],
        "extent": [0.2, 0.2, 0.02],
    }
    out = backend.plan(
        "panda",
        [0.4, 0, 0.3, 0, 0, 0, 0],
        goal_q=[0.4, 0, 0.0, 0, 0, 0, 0],
        obstacles=[table, shelf],
    )
    assert out["ok"] is False and out["status"] == "MotionGenStatus.TRAJOPT_FAIL"
    assert out["excluded_by_base"] == ["table/7"] and out["obstacles"] == 5
    world = planner.worlds[-1]["cuboid"]
    assert sorted(world) == ["shelf", "table/7", "table/7#1", "table/7#2", "table/7#3"]
    pieces = [
        collision.parse_obstacle(
            {
                "type": "box",
                "position": c["pose"][:3],
                "extent": c["dims"],
                "quat_xyzw": [*c["pose"][4:], c["pose"][3]],
            }
        )
        for key, c in world.items()
        if key.startswith("table")
    ]
    # cuRobo sees the table everywhere but in the base's column.
    assert collision.sphere_clearance(pieces, [[0, 0, 0.05, 0.05]])[0] >= 0.01 - 1e-9
    assert min(collision.point_distance(p, [[0.4, 0, 0.0]])[0] for p in pieces) < 0


def test_the_plan_reports_what_the_base_excluded():
    class Ik(PointIk):
        def call(self, method, args=(), kwargs=None, *, timeout_s=None):
            out = super().call(method, args, kwargs, timeout_s=timeout_s)
            return (
                {**out, "excluded_by_base": ["table/7"]} if method == "ik.plan" else out
            )

    planner = motion.MotionPlanner("http://ik", "panda_libero", client=Ik())
    out = planner.plan([0.1, 0, 0.1, 0, 0, 0, 0], [0.1, 0, 0.1], [0.3, 0, 0.1], UP, [])
    assert out["status"] == "planned" and out["excluded_by_base"] == ["table/7"]


def test_a_blocked_plan_says_why():
    roof = {
        "type": "box",
        "name": "roof",
        "position": [0.3, 0, 0.3],
        "extent": [0.05, 2, 0.8],
    }
    bowl = {
        "type": "box",
        "name": "akita_black_bowl_2_main",
        "position": [0.5, 0.065, 0.1],
        "extent": [0.1, 0.1, 0.05],
    }
    planner = motion.MotionPlanner("http://ik", "panda_libero", client=PointIk())
    start = [0.1, 0, 0.1]
    # The path is blocked, the goal itself is clear.
    out = planner.plan([*start, 0, 0, 0, 0], start, [0.5, 0, 0.1], UP, [roof])
    assert "goal is reachable and clear" in out["message"]
    # The goal touches a neighbouring object's box outside the left-out radius (the point
    # robot is smaller than a hand, so the radius shrinks with it): named.
    tight = motion.MotionPlanner(
        "http://ik", "panda_libero", client=PointIk(), contact_radius=0.01
    )
    out = tight.plan([*start, 0, 0, 0, 0], start, [0.5, 0, 0.1], UP, [roof, bowl])
    assert (
        "the goal configuration found touches akita_black_bowl_2_main's bounding box"
        in out["message"]
    )
    # Out of reach without any obstacle.
    out = planner.plan([*start, 0, 0, 0, 0], start, [1.2, 0, 0.1], UP, [roof])
    assert "out of reach" in out["message"]


def test_a_blocked_plan_is_explained_by_the_planners_status():
    """Audit 92245e3 CUROBO-2: _why_blocked diagnosed every failed plan the same way (IK at
    the goal, then a check there), never reading cuRobo's status: an arm refused for standing
    in collision was told "the goal is reachable and clear; the path between is blocked" and
    kept re-aiming from the same spot, and a trajopt failure after cuRobo's own IK had found
    a clear goal could name a neighbour the one solution we checked happened to touch."""

    class StatusIk(PointIk):
        def __init__(self, status):
            super().__init__()
            self.status = status

        def call(self, method, args=(), kwargs=None, *, timeout_s=None):
            if method == "ik.plan":
                self.calls.append((method, kwargs))
                return {
                    "ok": False,
                    "error": f"cuRobo found no collision-free path ({self.status})",
                    "status": self.status,
                    "backend": "curobo",
                }
            return super().call(method, args, kwargs, timeout_s=timeout_s)

    def plan(status, start, obstacles):
        ik = StatusIk(status)
        planner = motion.MotionPlanner(
            "http://ik", "panda_libero", client=ik, contact_radius=0.01
        )
        out = planner.plan([*start, 0, 0, 0, 0], start, [0.5, 0, 0.1], UP, obstacles)
        assert out["status"] == "blocked"
        return out["message"], [m for m, _ in ik.calls]

    # The point robot (2 cm) overlaps the wall 1.5 cm from its face: the start is in collision.
    touching = [0.26, 0, 0.1]
    msg, calls = plan(
        "MotionGenStatus.INVALID_START_STATE_WORLD_COLLISION", touching, [WALL]
    )
    assert "the arm is already touching wall's bounding box where it is" in msg
    assert "back off before planning" in msg and "path between" not in msg
    assert calls == ["ik.plan", "ik.check"], (
        "the start is checked, the goal not re-solved"
    )
    msg, _ = plan(
        "MotionGenStatus.INVALID_START_STATE_SELF_COLLISION", [0.1, 0, 0.1], [WALL]
    )
    assert "not a valid start" in msg and "SELF_COLLISION" in msg
    # Trajectory optimisation failed after cuRobo's IK found the goal: the goal is fine.
    msg, calls = plan("MotionGenStatus.TRAJOPT_FAIL", [0.1, 0, 0.1], [WALL])
    assert "the goal is reachable and clear; the path between is blocked" in msg
    assert calls == ["ik.plan"], "no single IK solution is checked for a path failure"
    # IK failed: the goal is diagnosed; a clear single solution means cuRobo's seeds all hit.
    msg, calls = plan("MotionGenStatus.IK_FAIL", [0.1, 0, 0.1], [WALL])
    assert "the planner found no collision-free configuration for it" in msg
    assert calls == ["ik.plan", "ik.solve", "ik.check"]
    beside = {
        "type": "box",
        "name": "cup",
        "position": [0.5, 0.065, 0.1],
        "extent": [0.1, 0.1, 0.05],
    }
    msg, _ = plan("MotionGenStatus.IK_FAIL", [0.1, 0, 0.1], [WALL, beside])
    assert "the goal configuration found touches cup's bounding box" in msg
