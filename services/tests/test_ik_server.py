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

"""The ik service: request/response over HTTP with a fake backend, the pose and obstacle
parsing, the cuRobo backend against doubles, and (when PyRoKi is installed) real Panda solves."""

from __future__ import annotations

import threading
import time

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from pi_embodied_services.components import ik_server
from pi_embodied_services.components.ik_server import (
    ROBOTS,
    CuroboBackend,
    IkFacade,
    interpolate_poses,
    link_to_tcp,
    parse_obstacle,
    parse_pose,
    pose_dict,
    tcp_to_link,
)
from pi_embodied_services.utils.rpc import RpcError
from pi_embodied_services.utils.rpc.http_rpc import HttpRpcClient

DOWN = [1.0, 0.0, 0.0, 0.0]  # gripper pointing down: 180 deg about x (xyzw)


class FakeBackend:
    """Reachable inside a 0.8 m ball around the base; records the calls it gets."""

    name = "fake"

    def __init__(self) -> None:
        self.calls: list[tuple] = []

    def robots(self):
        return {
            "panda": {"arm_joints": list(ROBOTS["panda"].arm_joints), "loaded": True}
        }

    def solve(self, robot, target_pose, seed_q=None):
        self.calls.append(("solve", robot, target_pose, seed_q))
        if robot not in ROBOTS:
            raise ValueError(f"unknown robot {robot!r}")
        pos, _quat = parse_pose(target_pose)
        ok = float(np.linalg.norm(pos)) < 0.8
        return {
            "robot": robot,
            "q": [0.1] * 7 if ok else None,
            "ok": ok,
            "error": None if ok else "unreachable: too far",
            "position_err": 0.0 if ok else 0.3,
            "orientation_err": 0.0,
            "solve_ms": 0.5,
        }

    def plan(
        self, robot, start_q, goal_pose=None, goal_q=None, obstacles=None, waypoints=20
    ):
        self.calls.append(
            ("plan", robot, start_q, goal_pose, goal_q, obstacles, waypoints)
        )
        return {
            "robot": robot,
            "path": [list(start_q)] * int(waypoints),
            "ok": True,
            "error": None,
        }


def _serve(facade):
    bound: dict = {}
    original = facade._bind_and_announce

    def capture(*args, **kwargs):
        server = original(*args, **kwargs)
        bound["port"] = server.server_address[1]
        return server

    facade._bind_and_announce = capture
    thread = threading.Thread(
        target=facade.serve,
        kwargs={"transport": "http", "host": "127.0.0.1", "port": 0},
        daemon=True,
    )
    thread.start()
    deadline = time.time() + 5
    while "port" not in bound:
        assert time.time() < deadline, "server did not bind"
        time.sleep(0.01)
    client = HttpRpcClient(f"http://127.0.0.1:{bound['port']}")

    def stop():
        client.call("shutdown", timeout_s=10)
        thread.join(timeout=10)

    return client, stop


@pytest.fixture
def served():
    backend = FakeBackend()
    client, stop = _serve(IkFacade(backend))
    yield backend, client
    stop()


def test_healthz_names_the_service_and_robots_lists_the_backend(served):
    _backend, client = served
    assert client.call("healthz")["service"] == "ik"
    robots = client.call("ik.robots")
    assert robots["backend"] == "fake"
    assert robots["robots"]["panda"]["arm_joints"] == list(ROBOTS["panda"].arm_joints)


def test_solve_round_trips_both_pose_shapes_and_the_seed(served):
    backend, client = served
    near = client.call(
        "ik.solve",
        kwargs={
            "robot": "panda",
            "target_pose": {"pos": [0.4, 0.0, 0.3], "quat_xyzw": DOWN},
            "seed_q": [0.0] * 7,
        },
    )
    assert near["ok"] is True and len(near["q"]) == 7 and near["error"] is None
    far = client.call("ik.solve", args=("panda", [2.0, 0.0, 0.3, *DOWN]))
    assert far["ok"] is False and far["q"] is None
    assert "unreachable" in far["error"] and far["position_err"] == 0.3
    assert backend.calls[0][3] == [0.0] * 7 and backend.calls[1][3] is None


def test_errors_travel_as_rpc_errors(served):
    _backend, client = served
    with pytest.raises(RpcError, match="unknown robot"):
        client.call("ik.solve", args=("r2d2", [0.4, 0.0, 0.3, *DOWN]))
    with pytest.raises(RpcError, match="robot must be"):
        client.call("ik.solve", args=("", [0.4, 0.0, 0.3, *DOWN]))
    with pytest.raises(RpcError, match="obstacles must be a list"):
        client.call(
            "ik.plan",
            kwargs={
                "robot": "panda",
                "start_q": [0.0] * 7,
                "goal_q": [0.1] * 7,
                "obstacles": {},
            },
        )


def test_plan_forwards_goal_obstacles_and_waypoints(served):
    backend, client = served
    box = {"type": "box", "position": [0.5, 0, 0.1], "extent": [0.1, 0.1, 0.2]}
    out = client.call(
        "ik.plan",
        kwargs={
            "robot": "panda",
            "start_q": [0.0] * 7,
            "goal_pose": {"pos": [0.4, 0.1, 0.2], "quat_xyzw": DOWN},
            "obstacles": [box],
            "waypoints": 5,
        },
    )
    assert out["ok"] is True and len(out["path"]) == 5
    call = backend.calls[-1]
    assert call[0] == "plan" and call[3] == {"pos": [0.4, 0.1, 0.2], "quat_xyzw": DOWN}
    assert call[5] == [box] and call[6] == 5


# ---------------------------------------------------------------------------
# pure helpers


def test_parse_pose_accepts_dict_and_flat_and_normalises():
    pos, quat = parse_pose({"pos": [1, 2, 3], "quat_xyzw": [0, 0, 0, 2]})
    assert pos.tolist() == [1, 2, 3] and quat.tolist() == [0, 0, 0, 1]
    pos, quat = parse_pose([1, 2, 3, 0, 0, 0, 1])
    assert pos.tolist() == [1, 2, 3] and quat.tolist() == [0, 0, 0, 1]
    for bad in ([1, 2, 3], {"pos": [1, 2, 3]}, [1, 2, 3, 0, 0, 0, 0], [np.nan] * 7):
        with pytest.raises(ValueError):
            parse_pose(bad)


def test_tcp_offset_is_applied_along_the_tool_axis():
    model = ROBOTS["panda"]
    pos, quat = np.array([0.5, 0.0, 0.2]), np.array(DOWN)
    link_pos, _ = tcp_to_link(model, pos, quat)
    # Pointing down, the hand's z axis is world -z: the hand sits 0.1034 m above the TCP.
    assert np.allclose(link_pos, [0.5, 0.0, 0.3034])
    back, _ = link_to_tcp(model, link_pos, quat)
    assert np.allclose(back, pos)


def test_pose_interpolation_is_linear_in_position_and_slerps_orientation():
    start, goal = np.array([0.0, 0.0, 0.0]), np.array([0.2, 0.0, 0.0])
    q0 = Rotation.identity().as_quat()
    q1 = Rotation.from_euler("z", np.pi / 2).as_quat()
    path = interpolate_poses(start, q0, goal, q1, 5)
    assert len(path) == 5
    assert np.allclose(path[2][0], [0.1, 0.0, 0.0])
    assert np.isclose(Rotation.from_quat(path[2][1]).magnitude(), np.pi / 4)
    with pytest.raises(ValueError):
        interpolate_poses(start, q0, goal, q1, 1)


def test_obstacles_are_validated():
    box = parse_obstacle({"type": "box", "position": [0, 0, 0], "extent": [1, 1, 1]})
    assert box["quat_xyzw"].tolist() == [0, 0, 0, 1]
    assert (
        parse_obstacle({"type": "sphere", "center": [0, 0, 1], "radius": 0.1})["radius"]
        == 0.1
    )
    assert parse_obstacle(
        {"type": "halfspace", "point": [0, 0, 0], "normal": [0, 0, 1]}
    )
    for bad in (
        {"type": "cylinder"},
        {"type": "box", "position": [0, 0, 0]},
        {"type": "box", "position": [0, 0, 0], "extent": [0, 1, 1]},
        "box",
    ):
        with pytest.raises(ValueError):
            parse_obstacle(bad)


# ---------------------------------------------------------------------------
# cuRobo backend against doubles (the real one runs on the GPU box)


class _Tensor:
    def __init__(self, value):
        self._v = np.asarray(value, dtype=np.float64)

    def detach(self):
        return self

    def cpu(self):
        return self

    def numpy(self):
        return self._v


class _TensorArgs:
    @staticmethod
    def to_device(value):
        return np.asarray(value, dtype=np.float64)


class _Kinematics:
    """A 9-joint cspace like franka.yml (7 arm joints + 2 fingers at 0.04)."""

    retract_config = _Tensor([0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785, 0.04, 0.04])

    @staticmethod
    def get_dof():
        return 9


class _Solver:
    """cuRobo IKSolver double: reachable within 0.8 m; the seed is echoed in the solution."""

    tensor_args = _TensorArgs()
    kinematics = _Kinematics()

    def __init__(self):
        self.goals: list = []

    def solve_single(self, goal, seed_config=None):
        self.goals.append((goal, seed_config))
        ok = float(np.linalg.norm(goal.position[0])) < 0.8

        class R:
            success = _Tensor([ok])
            solution = _Tensor([[0.2] * 9])
            position_error = _Tensor([0.0 if ok else 0.4])
            rotation_error = _Tensor([0.0])

        return R()


class _Pose:
    def __init__(self, position, quaternion):
        self.position, self.quaternion = position, quaternion


@pytest.fixture
def curobo(monkeypatch):
    import sys
    import types

    math_mod = types.ModuleType("curobo.types.math")
    math_mod.Pose = _Pose
    state_mod = types.ModuleType("curobo.types.state")

    class JointState:
        @staticmethod
        def from_position(p):
            return ("js", np.asarray(p))

    state_mod.JointState = JointState
    for name, mod in (
        ("curobo", types.ModuleType("curobo")),
        ("curobo.types", types.ModuleType("curobo.types")),
        ("curobo.types.math", math_mod),
        ("curobo.types.state", state_mod),
    ):
        monkeypatch.setitem(sys.modules, name, mod)
    solver = _Solver()
    backend = CuroboBackend(
        solver_factory=lambda model: solver, planner_factory=lambda m: None
    )
    return backend, solver


def test_curobo_solve_converts_the_tcp_to_the_hand_and_reports_success(curobo):
    backend, solver = curobo
    out = backend.solve(
        "panda", {"pos": [0.5, 0.0, 0.2], "quat_xyzw": DOWN}, seed_q=[0.0] * 7
    )
    assert out["ok"] is True and out["q"] == [0.2] * 7 and out["self_collision_checked"]
    goal, seed = solver.goals[0]
    assert np.allclose(goal.position[0], [0.5, 0.0, 0.3034])  # panda_hand, 0.1034 m up
    assert np.allclose(goal.quaternion[0], [0.0, 1.0, 0.0, 0.0])  # wxyz of DOWN
    # A [1, 1, 9] seed: the 7 arm joints, then the fingers from the retract config.
    assert seed.shape == (1, 1, 9) and seed[0, 0, :7].tolist() == [0.0] * 7
    assert seed[0, 0, 7:].tolist() == [0.04, 0.04]
    far = backend.solve("panda", [2.0, 0.0, 0.2, *DOWN])
    assert far["ok"] is False and "cuRobo misses by 400.0 mm" in far["error"]


def test_curobo_refuses_robots_without_a_config(curobo):
    backend, _ = curobo
    with pytest.raises(ValueError, match="no cuRobo config"):
        backend.solve("piper", [0.3, 0.0, 0.2, *DOWN])
    assert backend.robots()["piper"]["supported"] is False
    assert backend.robots()["ur5e"]["curobo_config"] == "ur5e.yml"


def test_curobo_world_dict_maps_every_obstacle_type():
    world = CuroboBackend._world_dict(
        [
            parse_obstacle(
                {
                    "type": "box",
                    "name": "table",
                    "position": [0.5, 0, -0.05],
                    "extent": [1, 1, 0.1],
                }
            ),
            parse_obstacle({"type": "sphere", "center": [0, 0, 1], "radius": 0.1}),
            parse_obstacle(
                {
                    "type": "capsule",
                    "position": [0, 1, 0],
                    "radius": 0.05,
                    "height": 0.4,
                }
            ),
            parse_obstacle(
                {"type": "halfspace", "point": [0, 0, 0], "normal": [0, 0, 1]}
            ),
        ]
    )
    assert world["cuboid"]["table"]["dims"] == [1, 1, 0.1]
    assert world["cuboid"]["table"]["pose"] == [0.5, 0, -0.05, 1, 0, 0, 0]
    assert world["sphere"]["sphere_1"]["radius"] == 0.1
    assert world["capsule"]["capsule_2"]["base"] == [0, 0, -0.2]
    slab = world["cuboid"]["halfspace_3"]
    assert slab["pose"][2] < 0  # the slab lies below the plane
    # No obstacles: cuRobo still needs one primitive, far away.
    assert CuroboBackend._world_dict([]) == ik_server.CUROBO_EMPTY_WORLD


# ---------------------------------------------------------------------------
# PyRoKi on the Panda (CPU; skipped where the `ik` extra is not installed)

pyroki = pytest.importorskip("pyroki")


@pytest.fixture(scope="module")
def panda():
    import os

    os.environ.setdefault("JAX_PLATFORMS", "cpu")
    backend = ik_server.PyrokiBackend()
    backend.fk("panda", ROBOTS["panda"].home_q)  # downloads and compiles once
    return backend


@pytest.mark.timeout(300)
def test_pyroki_solves_a_known_reachable_pose_and_refuses_an_unreachable_one(panda):
    home = list(ROBOTS["panda"].home_q)
    pos, quat = panda.fk("panda", home)
    # The Franka TCP at the home pose: about 0.31 m ahead, 0.49 m up, pointing down.
    assert np.allclose(pos, [0.307, 0.0, 0.487], atol=2e-3)
    assert np.isclose(abs(Rotation.from_quat(quat).apply([0, 0, 1])[2]), 1.0, atol=1e-3)

    started = time.perf_counter()
    near = panda.solve("panda", pose_dict(pos + [0.1, 0.1, -0.1], quat), seed_q=home)
    elapsed = time.perf_counter() - started
    assert near["ok"] is True, near
    assert near["position_err"] < 0.005 and near["orientation_err"] < 0.05
    check_pos, _ = panda.fk("panda", near["q"])
    assert np.allclose(check_pos, pos + [0.1, 0.1, -0.1], atol=0.005)
    assert elapsed < 30.0  # includes the first compile

    warm = time.perf_counter()
    again = panda.solve("panda", pose_dict(pos + [0.05, -0.1, 0.0], quat), seed_q=home)
    assert again["ok"] and time.perf_counter() - warm < 2.0

    far = panda.solve("panda", pose_dict(pos + [1.5, 0.0, 0.0], quat), seed_q=home)
    assert far["ok"] is False and far["q"] is not None
    assert "unreachable" in far["error"] and far["position_err"] > 0.5
    assert far["seeds_tried"] == 1 + 1 + ik_server.RANDOM_SEEDS


@pytest.mark.timeout(300)
def test_pyroki_plans_a_short_path_and_reports_clearance(panda):
    home = list(ROBOTS["panda"].home_q)
    pos, quat = panda.fk("panda", home)
    plan = panda.plan(
        "panda", home, goal_pose=pose_dict(pos + [0.1, 0.0, -0.1], quat), waypoints=8
    )
    assert (
        plan["ok"] is True and len(plan["path"]) == 8 and plan["collision_free"] is None
    )
    assert plan["max_joint_step_rad"] < ik_server.MAX_JOINT_JUMP_RAD
    end_pos, _ = panda.fk("panda", plan["path"][-1])
    assert np.allclose(end_pos, pos + [0.1, 0.0, -0.1], atol=0.005)
    joint = panda.plan("panda", home, goal_q=[v + 0.1 for v in home], waypoints=4)
    assert joint["ok"] and np.allclose(joint["path"][-1], [v + 0.1 for v in home])
    clear = panda.plan(
        "panda",
        home,
        goal_pose=pose_dict(pos + [0.1, 0.0, -0.1], quat),
        obstacles=[{"type": "sphere", "center": [2.0, 2.0, 2.0], "radius": 0.1}],
        waypoints=4,
    )
    assert clear["ok"] is True and clear["collision_free"] is True
    assert clear["min_clearance_m"] > 1.0
    with pytest.raises(ValueError, match="exactly one"):
        panda.plan("panda", home)
