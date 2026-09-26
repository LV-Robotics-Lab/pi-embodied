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
#
# The PyRoKi solve is CaP-X's capx/serving/launch_pyroki_server.py (pose + limit
# costs, linear-interpolation planning seeded from the previous waypoint) on the
# pi-embodied RPC framework; the cuRobo backend follows CaP-X's
# launch_curobo_server.py and OpenETA's sim/mcp_server/collision.py.

"""RPC server solving inverse kinematics and short joint paths for named robots.

Run manually with::

    PYTHONPATH=/path/to/pi/services python -m pi_embodied_services.components.ik_server \
        --backend pyroki --port 18400

Runs under the ``ik`` extra's own interpreter (PyRoKi + JAX on the CPU; nothing
else of the services is imported besides numpy/scipy), which is why the env servers
reach it over RPC (``--ik <url>``) instead of importing it: their simulator venvs pin
their own numpy/torch and must not gain JAX.

Methods (``ik.*``): ``solve`` (one pose -> one joint vector), ``plan`` (a joint path
from a start configuration to a goal pose or configuration, refusing paths that hit
the given obstacles) and ``robots`` (the robot models this process knows). Poses are
TCP poses in the robot's base frame, as ``{"pos": [x, y, z], "quat_xyzw": [...]}`` or
a flat ``[x, y, z, qx, qy, qz, qw]`` list. The service is an internal dependency of
the env servers (``env.preview_reach``, the motion primitives' reach check, a
Robosuite ``move_to``); it is not an agent tool.

Backends: ``pyroki`` (default, MIT, CPU) solves IK as a least-squares problem and
plans by interpolating the pose and solving each waypoint from the previous one;
with obstacles it adds PyRoKi's world-collision cost and verifies the path against
them, so it avoids obstacles only locally (no graph search). ``curobo`` (optional,
GPU) uses cuRobo's IKSolver (self-collision aware) and MotionGen (collision-free
trajectory optimisation with graph seeds) for the robots cuRobo ships a config for.
"""

from __future__ import annotations

import argparse
import logging
import os
import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from scipy.spatial.transform import Rotation, Slerp

from pi_embodied_services.utils.logging import get_logger
from pi_embodied_services.utils.rpc import RpcFacade

logger = get_logger("ik_server")

DEFAULT_PORT = 18400
#: Position / orientation error at which a solution counts as reaching the target
#: (cuRobo's defaults in CaP-X: 5 mm, 0.05 rad).
DEFAULT_POS_TOL_M = 0.005
DEFAULT_ORI_TOL_RAD = 0.05
#: Random restarts after the caller's seed and the rest pose failed.
RANDOM_SEEDS = 6
#: A joint moving more than this between two path waypoints is a branch flip, not a path.
MAX_JOINT_JUMP_RAD = 0.5
#: Distance (m) the planners keep between the robot and an obstacle.
COLLISION_MARGIN_M = 0.01
#: cuRobo's primitive collision checker refuses an empty world ("Primitive Collision has no
#: obstacles", from the IK solver's and the planner's rollouts), so both are built with, and a
#: plan without obstacles gets, this 1 cm cube 100 m away.
CUROBO_EMPTY_WORLD = {
    "cuboid": {
        "_none": {"dims": [0.01, 0.01, 0.01], "pose": [100.0, 100.0, 100.0, 1, 0, 0, 0]}
    }
}


# ---------------------------------------------------------------------------
# Robot models
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RobotModel:
    """A robot the service can solve: its URDF, the arm joints a caller's ``q`` lists,
    the link the tool hangs off and where the TCP sits in that link's frame."""

    name: str
    #: ``robot_descriptions`` module name (the URDF, downloaded on first use).
    description: str
    ee_link: str
    arm_joints: tuple[str, ...]
    #: TCP position in the ``ee_link`` frame (metres); poses in and out are TCP poses.
    tool_offset_xyz: tuple[float, float, float] = (0.0, 0.0, 0.0)
    #: Other actuated joints (gripper fingers), held at these values.
    fixed_joints: dict[str, float] = field(default_factory=dict)
    #: Joint vector used as the rest pose and as the fallback IK seed.
    home_q: tuple[float, ...] = ()
    #: cuRobo's robot config file (``curobo/content/configs/robot``), if it ships one.
    curobo_config: str | None = None
    note: str = ""


_PANDA_JOINTS = tuple(f"panda_joint{i}" for i in range(1, 8))
_PANDA_HOME = (0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785)

ROBOTS: dict[str, RobotModel] = {
    "panda": RobotModel(
        name="panda",
        description="panda_description",
        ee_link="panda_hand",
        arm_joints=_PANDA_JOINTS,
        tool_offset_xyz=(0.0, 0.0, 0.1034),
        fixed_joints={"panda_finger_joint1": 0.04},
        home_q=_PANDA_HOME,
        curobo_config="franka.yml",
        note="Franka Panda / FR3 with the Franka Hand; the TCP is libfranka's default "
        "O_T_EE (flange + 0.1034 m), the tcp_pose the franka env servers report.",
    ),
    "panda_libero": RobotModel(
        name="panda_libero",
        description="panda_description",
        ee_link="panda_hand",
        arm_joints=_PANDA_JOINTS,
        tool_offset_xyz=(0.0, 0.0, 0.097),
        fixed_joints={"panda_finger_joint1": 0.04},
        home_q=_PANDA_HOME,
        curobo_config="franka.yml",
        note="robosuite's Panda (LIBERO, Robosuite): robot0_eef_pos is the gripper's "
        "grip_site, 0.097 m below panda_hand, and robot0_eef_quat is the hand's "
        "orientation. Poses are in the robot0_base frame.",
    ),
    "ur5e": RobotModel(
        name="ur5e",
        description="ur5e_description",
        ee_link="tool0",
        arm_joints=(
            "shoulder_pan_joint",
            "shoulder_lift_joint",
            "elbow_joint",
            "wrist_1_joint",
            "wrist_2_joint",
            "wrist_3_joint",
        ),
        home_q=(0.0, -1.571, 1.571, -1.571, -1.571, 0.0),
        curobo_config="ur5e.yml",
        note="UR5e without a tool: the TCP is tool0 (the flange).",
    ),
    "piper": RobotModel(
        name="piper",
        description="piper_description",
        ee_link="gripper_base",
        arm_joints=tuple(f"joint{i}" for i in range(1, 7)),
        fixed_joints={"joint7": 0.0, "joint8": 0.0},
        home_q=(0.0, 1.0, -1.0, 0.0, 1.0, 0.0),
        note="AgileX Piper: the TCP is the gripper_base link (the gripper mount); "
        "no cuRobo config ships for it.",
    ),
}


# ---------------------------------------------------------------------------
# Poses
# ---------------------------------------------------------------------------


def parse_pose(pose: Any, name: str = "pose") -> tuple[np.ndarray, np.ndarray]:
    """``{"pos", "quat_xyzw"}`` or a flat xyz+xyzw list -> (pos[3], unit quat xyzw[4])."""
    if isinstance(pose, dict):
        if "pos" not in pose or "quat_xyzw" not in pose:
            raise ValueError(
                f"{name} must have 'pos' and 'quat_xyzw', got {sorted(pose)}"
            )
        pos = np.asarray(pose["pos"], dtype=np.float64).reshape(-1)
        quat = np.asarray(pose["quat_xyzw"], dtype=np.float64).reshape(-1)
    else:
        flat = np.asarray(pose, dtype=np.float64).reshape(-1)
        if flat.shape != (7,):
            raise ValueError(
                f"{name} must be {{pos, quat_xyzw}} or [x, y, z, qx, qy, qz, qw], "
                f"got shape {flat.shape}"
            )
        pos, quat = flat[:3], flat[3:]
    if pos.shape != (3,) or quat.shape != (4,):
        raise ValueError(f"{name} needs 3 position and 4 quaternion values")
    if not (np.isfinite(pos).all() and np.isfinite(quat).all()):
        raise ValueError(f"{name} must be finite")
    norm = float(np.linalg.norm(quat))
    if norm < 1e-9:
        raise ValueError(f"{name} quaternion must be nonzero")
    return pos, quat / norm


def pose_dict(pos: np.ndarray, quat_xyzw: np.ndarray) -> dict[str, list[float]]:
    return {
        "pos": [float(v) for v in np.asarray(pos).reshape(3)],
        "quat_xyzw": [float(v) for v in np.asarray(quat_xyzw).reshape(4)],
    }


def orientation_error(quat_a_xyzw: np.ndarray, quat_b_xyzw: np.ndarray) -> float:
    """Angle (rad) between two orientations."""
    return float(
        (
            Rotation.from_quat(quat_a_xyzw) * Rotation.from_quat(quat_b_xyzw).inv()
        ).magnitude()
    )


def tcp_to_link(
    model: RobotModel, pos: np.ndarray, quat_xyzw: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """The ``ee_link`` pose whose TCP (``tool_offset_xyz`` along the link) is at the pose."""
    rot = Rotation.from_quat(quat_xyzw)
    return pos - rot.apply(np.asarray(model.tool_offset_xyz)), quat_xyzw


def link_to_tcp(
    model: RobotModel, pos: np.ndarray, quat_xyzw: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    rot = Rotation.from_quat(quat_xyzw)
    return pos + rot.apply(np.asarray(model.tool_offset_xyz)), quat_xyzw


def interpolate_poses(
    start_pos: np.ndarray,
    start_quat: np.ndarray,
    goal_pos: np.ndarray,
    goal_quat: np.ndarray,
    n: int,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """``n`` poses from start to goal: linear position, slerp orientation."""
    if n < 2:
        raise ValueError("a path needs at least 2 waypoints")
    times = np.linspace(0.0, 1.0, n)
    positions = start_pos[None] + (goal_pos - start_pos)[None] * times[:, None]
    rots = Rotation.from_quat(np.stack([start_quat, goal_quat]))
    quats = Slerp([0.0, 1.0], rots)(times).as_quat()
    return [(positions[i], quats[i]) for i in range(n)]


def _q_vector(q: Any, model: RobotModel, name: str) -> np.ndarray:
    vec = np.asarray(q, dtype=np.float64).reshape(-1)
    n = len(model.arm_joints)
    if vec.shape != (n,) or not np.isfinite(vec).all():
        raise ValueError(f"{name} must list {n} finite joint angles for {model.name}")
    return vec


def _joint_jumps(path: np.ndarray) -> float:
    return float(np.abs(np.diff(path, axis=0)).max()) if len(path) > 1 else 0.0


# ---------------------------------------------------------------------------
# Obstacles
# ---------------------------------------------------------------------------


def parse_obstacle(obstacle: Any) -> dict[str, Any]:
    """Validate one obstacle: ``box`` (``position``, ``extent``[, ``quat_xyzw``]),
    ``sphere`` (``center``, ``radius``), ``capsule`` (``position``, ``radius``,
    ``height``[, ``quat_xyzw``]) or ``halfspace`` (``point``, ``normal``), metres in
    the robot's base frame."""
    if not isinstance(obstacle, dict) or "type" not in obstacle:
        raise ValueError("each obstacle is a dict with a 'type'")
    kind = str(obstacle["type"])
    out: dict[str, Any] = {"type": kind, "name": str(obstacle.get("name", ""))}

    def vec(key: str, n: int) -> np.ndarray:
        if key not in obstacle:
            raise ValueError(f"{kind} obstacle needs '{key}'")
        v = np.asarray(obstacle[key], dtype=np.float64).reshape(-1)
        if v.shape != (n,) or not np.isfinite(v).all():
            raise ValueError(f"{kind} obstacle '{key}' must be {n} finite numbers")
        return v

    if kind == "box":
        out["position"], out["extent"] = vec("position", 3), vec("extent", 3)
        if (out["extent"] <= 0).any():
            raise ValueError("box extent must be positive")
        out["quat_xyzw"] = (
            vec("quat_xyzw", 4)
            if "quat_xyzw" in obstacle
            else np.array([0.0, 0.0, 0.0, 1.0])
        )
    elif kind == "sphere":
        out["center"], out["radius"] = vec("center", 3), float(obstacle["radius"])
    elif kind == "capsule":
        out["position"], out["radius"] = vec("position", 3), float(obstacle["radius"])
        out["height"] = float(obstacle["height"])
        out["quat_xyzw"] = (
            vec("quat_xyzw", 4)
            if "quat_xyzw" in obstacle
            else np.array([0.0, 0.0, 0.0, 1.0])
        )
    elif kind == "halfspace":
        out["point"], out["normal"] = vec("point", 3), vec("normal", 3)
    else:
        raise ValueError(
            f"unknown obstacle type {kind!r}; box, sphere, capsule or halfspace"
        )
    return out


# ---------------------------------------------------------------------------
# Results
# ---------------------------------------------------------------------------


def solve_result(
    *,
    robot: str,
    q: np.ndarray | None,
    ok: bool,
    error: str | None,
    position_err: float | None,
    orientation_err: float | None,
    started: float,
    **extra: Any,
) -> dict[str, Any]:
    out = {
        "robot": robot,
        "q": None if q is None else [float(v) for v in q],
        "ok": bool(ok),
        "error": error,
        "position_err": None if position_err is None else float(position_err),
        "orientation_err": None if orientation_err is None else float(orientation_err),
        "solve_ms": round((time.perf_counter() - started) * 1000.0, 3),
    }
    out.update(extra)
    return out


def plan_result(
    *,
    robot: str,
    path: np.ndarray | None,
    ok: bool,
    error: str | None,
    started: float,
    **extra: Any,
) -> dict[str, Any]:
    out = {
        "robot": robot,
        "path": None if path is None else np.asarray(path, dtype=float).tolist(),
        "ok": bool(ok),
        "error": error,
        "plan_ms": round((time.perf_counter() - started) * 1000.0, 3),
    }
    out.update(extra)
    return out


# ---------------------------------------------------------------------------
# PyRoKi backend
# ---------------------------------------------------------------------------


class PyrokiBackend:
    """IK and interpolated paths with PyRoKi (JAX, CPU).

    One robot model is a ``pyroki.Robot`` from the ``robot_descriptions`` URDF; the
    first solve per robot compiles the JAX program (about a second), later solves
    take well under a millisecond. Gripper joints are pinned to ``fixed_joints``.
    """

    name = "pyroki"

    def __init__(
        self,
        *,
        pos_tol: float = DEFAULT_POS_TOL_M,
        ori_tol: float = DEFAULT_ORI_TOL_RAD,
        random_seeds: int = RANDOM_SEEDS,
    ) -> None:
        try:
            import jax
            import pyroki  # noqa: F401
        except ImportError as exc:
            raise RuntimeError(
                "PyRoKi is missing; install the services' `ik` extra "
                '(`uv pip install -e "services[ik]"`)'
            ) from exc
        self._jax = jax
        self.pos_tol = float(pos_tol)
        self.ori_tol = float(ori_tol)
        self.random_seeds = int(random_seeds)
        self._loaded: dict[str, dict[str, Any]] = {}
        self._rng = np.random.default_rng(0)

    # -- models -----------------------------------------------------------

    def loaded(self, name: str) -> bool:
        return name in self._loaded

    def _load(self, name: str) -> dict[str, Any]:
        if name in self._loaded:
            return self._loaded[name]
        model = ROBOTS.get(name)
        if model is None:
            raise ValueError(f"unknown robot {name!r}; known: {sorted(ROBOTS)}")
        import jax.numpy as jnp
        import pyroki as pk
        from robot_descriptions.loaders.yourdfpy import load_robot_description

        started = time.perf_counter()
        urdf = load_robot_description(model.description)
        robot = pk.Robot.from_urdf(urdf)
        names = list(robot.joints.actuated_names)
        missing = [
            j for j in (*model.arm_joints, *model.fixed_joints) if j not in names
        ]
        if missing:
            raise RuntimeError(
                f"{model.description} lacks joints {missing}; actuated: {names}"
            )
        arm_idx = np.array([names.index(j) for j in model.arm_joints])
        pin_mask = np.zeros(len(names))
        pin_vals = np.zeros(len(names))
        for joint, value in model.fixed_joints.items():
            pin_mask[names.index(joint)] = 1.0
            pin_vals[names.index(joint)] = value
        if model.ee_link not in robot.links.names:
            raise RuntimeError(f"{model.description} has no link {model.ee_link!r}")
        lower = np.asarray(robot.joints.lower_limits)[arm_idx]
        upper = np.asarray(robot.joints.upper_limits)[arm_idx]
        entry = {
            "model": model,
            "robot": robot,
            "urdf": urdf,
            "coll": None,
            "link_index": robot.links.names.index(model.ee_link),
            "arm_idx": arm_idx,
            "pin_mask": jnp.asarray(pin_mask),
            "pin_vals": jnp.asarray(pin_vals),
            "lower": lower,
            "upper": upper,
            "solve": self._build_solver(),
            "solve_coll": None,
        }
        self._loaded[name] = entry
        logger.info(
            "loaded %s (%s) in %.1fs: %d arm joints, ee %s",
            name,
            model.description,
            time.perf_counter() - started,
            len(model.arm_joints),
            model.ee_link,
        )
        return entry

    def _collision(self, entry: dict[str, Any]) -> Any:
        """The robot's collision model (built on first use; needs the URDF's meshes)."""
        if entry["coll"] is None:
            import pyroki as pk

            entry["coll"] = pk.collision.RobotCollision.from_urdf(entry["urdf"])
            entry["solve_coll"] = self._build_solver(collision=True)
        return entry["coll"]

    @staticmethod
    def _build_solver(*, collision: bool = False):
        import jax_dataclasses as jdc
        import jaxlie
        import jaxls
        import pyroki as pk

        def factors(robot, jv, wxyz, pos, link_index, seed, pin_mask, pin_vals, rest_w):
            def pin(vals, var):
                return (vals[var] - pin_vals) * pin_mask * 100.0

            return [
                pk.costs.pose_cost_analytic_jac(
                    robot,
                    jv,
                    jaxlie.SE3.from_rotation_and_translation(jaxlie.SO3(wxyz), pos),
                    link_index,
                    pos_weight=50.0,
                    ori_weight=10.0,
                ),
                pk.costs.limit_cost(robot, jv, weight=100.0),
                pk.costs.rest_cost(jv, seed, weight=rest_w),
                jaxls.Cost(pin, (jv,)),
            ]

        def solve(problem_factors, jv, seed):
            return (
                jaxls.LeastSquaresProblem(problem_factors, [jv])
                .analyze()
                .solve(
                    verbose=False,
                    linear_solver="dense_cholesky",
                    trust_region=jaxls.TrustRegionConfig(lambda_initial=1.0),
                    initial_vals=jaxls.VarValues.make([jv.with_value(seed)]),
                )[jv]
            )

        if not collision:

            @jdc.jit
            def solve_ik(
                robot, link_index, wxyz, pos, seed, pin_mask, pin_vals, rest_w
            ):
                jv = robot.joint_var_cls(0)
                return solve(
                    factors(
                        robot,
                        jv,
                        wxyz,
                        pos,
                        link_index,
                        seed,
                        pin_mask,
                        pin_vals,
                        rest_w,
                    ),
                    jv,
                    seed,
                )

            return solve_ik

        @jdc.jit
        def solve_ik_coll(
            robot, coll, geoms, link_index, wxyz, pos, seed, pin_mask, pin_vals, rest_w
        ):
            jv = robot.joint_var_cls(0)
            problem = factors(
                robot, jv, wxyz, pos, link_index, seed, pin_mask, pin_vals, rest_w
            )
            problem.append(
                pk.costs.self_collision_cost(robot, coll, jv, margin=0.0, weight=5.0)
            )
            for geom in geoms:
                problem.append(
                    pk.costs.world_collision_cost(
                        robot, coll, jv, geom, margin=COLLISION_MARGIN_M, weight=20.0
                    )
                )
            return solve(problem, jv, seed)

        return solve_ik_coll

    @staticmethod
    def _geoms(obstacles: list[dict[str, Any]]) -> list[Any]:
        import pyroki as pk

        geoms = []
        for obs in obstacles:
            kind = obs["type"]
            if kind == "box":
                geoms.append(
                    pk.collision.Box.from_extent(
                        extent=obs["extent"],
                        position=obs["position"],
                        wxyz=np.asarray(obs["quat_xyzw"])[[3, 0, 1, 2]],
                    )
                )
            elif kind == "sphere":
                geoms.append(
                    pk.collision.Sphere.from_center_and_radius(
                        obs["center"], np.array([obs["radius"]])
                    )
                )
            elif kind == "capsule":
                geoms.append(
                    pk.collision.Capsule.from_radius_height(
                        position=obs["position"],
                        radius=np.array([obs["radius"]]),
                        height=np.array([obs["height"]]),
                        wxyz=np.asarray(obs["quat_xyzw"])[[3, 0, 1, 2]],
                    )
                )
            else:
                geoms.append(
                    pk.collision.HalfSpace.from_point_and_normal(
                        obs["point"], obs["normal"]
                    )
                )
        return geoms

    # -- kinematics -------------------------------------------------------

    def _full_q(self, entry: dict[str, Any], arm_q: np.ndarray) -> np.ndarray:
        full = np.asarray(entry["pin_vals"], dtype=np.float64).copy()
        full[entry["arm_idx"]] = arm_q
        return full

    def fk(self, name: str, q: Any) -> tuple[np.ndarray, np.ndarray]:
        """TCP pose (pos, quat_xyzw) of arm joints ``q``."""
        import jax.numpy as jnp

        entry = self._load(name)
        model: RobotModel = entry["model"]
        full = self._full_q(entry, _q_vector(q, model, "q"))
        poses = np.asarray(entry["robot"].forward_kinematics(jnp.asarray(full)))
        wxyz_xyz = poses[entry["link_index"]]
        return link_to_tcp(model, wxyz_xyz[4:], wxyz_xyz[[1, 2, 3, 0]])

    def _within_limits(self, entry: dict[str, Any], q: np.ndarray) -> bool:
        return bool(
            (q >= entry["lower"] - 1e-3).all() and (q <= entry["upper"] + 1e-3).all()
        )

    def _solve_once(
        self,
        entry: dict[str, Any],
        pos: np.ndarray,
        quat_xyzw: np.ndarray,
        seed_arm: np.ndarray,
        *,
        rest_w: float,
        geoms: list[Any] | None = None,
    ) -> tuple[np.ndarray, float, float]:
        """One solve from ``seed_arm``: (arm q, position error, orientation error)."""
        import jax.numpy as jnp

        model: RobotModel = entry["model"]
        link_pos, link_quat = tcp_to_link(model, pos, quat_xyzw)
        wxyz = jnp.asarray(link_quat[[3, 0, 1, 2]])
        seed = jnp.asarray(self._full_q(entry, seed_arm))
        if geoms is None:
            full = entry["solve"](
                entry["robot"],
                jnp.array(entry["link_index"]),
                wxyz,
                jnp.asarray(link_pos),
                seed,
                entry["pin_mask"],
                entry["pin_vals"],
                jnp.asarray(rest_w),
            )
        else:
            full = entry["solve_coll"](
                entry["robot"],
                entry["coll"],
                tuple(geoms),
                jnp.array(entry["link_index"]),
                wxyz,
                jnp.asarray(link_pos),
                seed,
                entry["pin_mask"],
                entry["pin_vals"],
                jnp.asarray(rest_w),
            )
        full = np.asarray(full, dtype=np.float64)
        arm_q = full[entry["arm_idx"]]
        fk_pos, fk_quat = self.fk(model.name, arm_q)
        return (
            arm_q,
            float(np.linalg.norm(fk_pos - pos)),
            orientation_error(fk_quat, quat_xyzw),
        )

    def _seeds(
        self, entry: dict[str, Any], seed_q: np.ndarray | None
    ) -> list[np.ndarray]:
        model: RobotModel = entry["model"]
        seeds = [] if seed_q is None else [seed_q]
        home = np.asarray(model.home_q or (entry["lower"] + entry["upper"]) / 2)
        seeds.append(home)
        for _ in range(self.random_seeds):
            seeds.append(self._rng.uniform(entry["lower"], entry["upper"]))
        return seeds

    def _reaches(self, pos_err: float, ori_err: float) -> bool:
        return pos_err <= self.pos_tol and ori_err <= self.ori_tol

    # -- interface --------------------------------------------------------

    def robots(self) -> dict[str, dict[str, Any]]:
        out = {}
        for name, model in ROBOTS.items():
            info: dict[str, Any] = {
                "description": model.description,
                "ee_link": model.ee_link,
                "tool_offset_xyz": list(model.tool_offset_xyz),
                "arm_joints": list(model.arm_joints),
                "fixed_joints": dict(model.fixed_joints),
                "home_q": list(model.home_q),
                "note": model.note,
                "loaded": name in self._loaded,
            }
            if name in self._loaded:
                info["lower"] = self._loaded[name]["lower"].tolist()
                info["upper"] = self._loaded[name]["upper"].tolist()
            out[name] = info
        return out

    def solve(self, robot: str, target_pose: Any, seed_q: Any = None) -> dict[str, Any]:
        started = time.perf_counter()
        entry = self._load(robot)
        model: RobotModel = entry["model"]
        pos, quat = parse_pose(target_pose, "target_pose")
        seed = None if seed_q is None else _q_vector(seed_q, model, "seed_q")
        best: tuple[np.ndarray, float, float] | None = None
        tried = 0
        for s in self._seeds(entry, seed):
            tried += 1
            q, pos_err, ori_err = self._solve_once(entry, pos, quat, s, rest_w=0.01)
            limits_ok = self._within_limits(entry, q)
            if best is None or (limits_ok and pos_err + ori_err < best[1] + best[2]):
                best = (q, pos_err, ori_err)
            if limits_ok and self._reaches(pos_err, ori_err):
                return solve_result(
                    robot=robot,
                    q=q,
                    ok=True,
                    error=None,
                    position_err=pos_err,
                    orientation_err=ori_err,
                    started=started,
                    seeds_tried=tried,
                )
        assert best is not None
        q, pos_err, ori_err = best
        error = (
            f"unreachable: best of {tried} seeds misses by {pos_err * 1000:.1f} mm "
            f"and {ori_err:.3f} rad (tolerance {self.pos_tol * 1000:.0f} mm, "
            f"{self.ori_tol:.2f} rad)"
        )
        if not self._within_limits(entry, q):
            error += "; solution outside joint limits"
        return solve_result(
            robot=robot,
            q=q,
            ok=False,
            error=error,
            position_err=pos_err,
            orientation_err=ori_err,
            started=started,
            seeds_tried=tried,
        )

    def _path_collisions(
        self, entry: dict[str, Any], path: np.ndarray, geoms: list[Any]
    ) -> float:
        """Smallest robot-to-obstacle distance along the path (negative = penetration)."""
        import jax.numpy as jnp
        import pyroki as pk

        coll = self._collision(entry)
        full = np.stack([self._full_q(entry, q) for q in path])
        robot_geom = coll.at_config(entry["robot"], jnp.asarray(full))
        worst = float("inf")
        for geom in geoms:
            dist = np.asarray(pk.collision.collide(robot_geom, geom))
            worst = min(worst, float(dist.min()))
        return worst

    def plan(
        self,
        robot: str,
        start_q: Any,
        goal_pose: Any = None,
        goal_q: Any = None,
        obstacles: Any = None,
        waypoints: int = 20,
    ) -> dict[str, Any]:
        started = time.perf_counter()
        entry = self._load(robot)
        model: RobotModel = entry["model"]
        start = _q_vector(start_q, model, "start_q")
        if (goal_pose is None) == (goal_q is None):
            raise ValueError("plan takes exactly one of goal_pose or goal_q")
        n = int(waypoints)
        parsed = [parse_obstacle(o) for o in (obstacles or [])]
        geoms = self._geoms(parsed) if parsed else None
        if geoms is not None:
            self._collision(entry)

        if goal_q is not None:
            goal = _q_vector(goal_q, model, "goal_q")
            path = np.linspace(start, goal, n)
        else:
            goal_pos, goal_quat = parse_pose(goal_pose, "goal_pose")
            start_pos, start_quat = self.fk(robot, start)
            path_list = [start]
            prev = start
            for i, (pos, quat) in enumerate(
                interpolate_poses(start_pos, start_quat, goal_pos, goal_quat, n)[1:], 1
            ):
                q, pos_err, ori_err = self._solve_once(
                    entry, pos, quat, prev, rest_w=1.0, geoms=geoms
                )
                if not (
                    self._reaches(pos_err, ori_err) and self._within_limits(entry, q)
                ):
                    why = (
                        "unreachable"
                        if geoms is None
                        else "unreachable clear of the obstacles"
                    )
                    return plan_result(
                        robot=robot,
                        path=None,
                        ok=False,
                        error=(
                            f"waypoint {i}/{n - 1} {why}: misses by "
                            f"{pos_err * 1000:.1f} mm and {ori_err:.3f} rad"
                        ),
                        started=started,
                        failed_waypoint=i,
                        collision_free=None if geoms is None else False,
                    )
                path_list.append(q)
                prev = q
            path = np.stack(path_list)
        jump = _joint_jumps(path)
        if jump > MAX_JOINT_JUMP_RAD:
            return plan_result(
                robot=robot,
                path=path,
                ok=False,
                error=f"joint jump of {jump:.2f} rad between waypoints (limit {MAX_JOINT_JUMP_RAD})",
                started=started,
            )
        extra: dict[str, Any] = {"collision_free": None, "max_joint_step_rad": jump}
        if geoms is not None:
            clearance = self._path_collisions(entry, path, geoms)
            extra["collision_free"] = clearance > 0.0
            extra["min_clearance_m"] = clearance
            if clearance <= 0.0:
                return plan_result(
                    robot=robot,
                    path=path,
                    ok=False,
                    error=f"path penetrates an obstacle by {-clearance * 1000:.1f} mm",
                    started=started,
                    **extra,
                )
        return plan_result(
            robot=robot, path=path, ok=True, error=None, started=started, **extra
        )


# ---------------------------------------------------------------------------
# cuRobo backend
# ---------------------------------------------------------------------------


class CuroboBackend:
    """IK with self-collision checks and collision-free planning with cuRobo (GPU).

    The same ``solve`` / ``plan`` interface and results as :class:`PyrokiBackend`, for
    the robots cuRobo ships a config for (``RobotModel.curobo_config``). ``plan``
    returns cuRobo's interpolated trajectory (``interpolation_dt`` apart); with
    obstacles it plans around them, and every result carries ``collision_free``.

    ``solver_factory`` / ``planner_factory`` build the cuRobo ``IKSolver`` and
    ``MotionGen`` for a model; tests inject doubles, the CLI uses cuRobo's.
    """

    name = "curobo"

    def __init__(
        self,
        *,
        pos_tol: float = DEFAULT_POS_TOL_M,
        ori_tol: float = DEFAULT_ORI_TOL_RAD,
        solver_factory=None,
        planner_factory=None,
        interpolation_dt: float = 0.02,
    ) -> None:
        self.pos_tol = float(pos_tol)
        self.ori_tol = float(ori_tol)
        self.interpolation_dt = float(interpolation_dt)
        self._solver_factory = solver_factory or self._make_solver
        self._planner_factory = planner_factory or self._make_planner
        self._solvers: dict[str, Any] = {}
        self._planners: dict[str, Any] = {}
        if solver_factory is None or planner_factory is None:
            try:
                import curobo  # noqa: F401
                import torch  # noqa: F401
            except ImportError as exc:
                raise RuntimeError(
                    "cuRobo is missing; install the services' `ik-curobo` extra "
                    "(torch first, then `nvidia-curobo` with --no-build-isolation)"
                ) from exc

    # -- models -----------------------------------------------------------

    @staticmethod
    def _model(name: str) -> RobotModel:
        model = ROBOTS.get(name)
        if model is None:
            raise ValueError(f"unknown robot {name!r}; known: {sorted(ROBOTS)}")
        if model.curobo_config is None:
            raise ValueError(
                f"robot {name!r} has no cuRobo config; the pyroki backend solves it"
            )
        return model

    def loaded(self, name: str) -> bool:
        return name in self._solvers

    @staticmethod
    def _robot_cfg(model: RobotModel):
        from curobo.types.base import TensorDeviceType
        from curobo.types.robot import RobotConfig
        from curobo.util_file import get_robot_configs_path, join_path, load_yaml

        tensor_args = TensorDeviceType()
        data = load_yaml(join_path(get_robot_configs_path(), model.curobo_config))[
            "robot_cfg"
        ]
        data["kinematics"]["ee_link"] = model.ee_link
        return RobotConfig.from_dict(data, tensor_args=tensor_args), tensor_args

    def _make_solver(self, model: RobotModel):
        from curobo.geom.types import WorldConfig
        from curobo.wrap.reacher.ik_solver import IKSolver, IKSolverConfig

        robot_cfg, tensor_args = self._robot_cfg(model)
        cfg = IKSolverConfig.load_from_robot_config(
            robot_cfg,
            WorldConfig.from_dict(CUROBO_EMPTY_WORLD),
            position_threshold=self.pos_tol,
            rotation_threshold=self.ori_tol,
            num_seeds=32,
            self_collision_check=True,
            self_collision_opt=True,
            tensor_args=tensor_args,
            use_cuda_graph=True,
        )
        return IKSolver(cfg)

    def _make_planner(self, model: RobotModel):
        from curobo.geom.sdf.world import CollisionCheckerType
        from curobo.geom.types import WorldConfig
        from curobo.wrap.reacher.motion_gen import MotionGen, MotionGenConfig

        robot_cfg, tensor_args = self._robot_cfg(model)
        cfg = MotionGenConfig.load_from_robot_config(
            robot_cfg,
            WorldConfig.from_dict(CUROBO_EMPTY_WORLD),
            collision_checker_type=CollisionCheckerType.PRIMITIVE,
            use_cuda_graph=True,
            collision_cache={"obb": 32, "sphere": 32, "capsule": 8},
            position_threshold=self.pos_tol,
            rotation_threshold=self.ori_tol,
            num_ik_seeds=32,
            num_trajopt_seeds=4,
            num_graph_seeds=4,
            interpolation_dt=self.interpolation_dt,
            collision_activation_distance=0.02,
            tensor_args=tensor_args,
        )
        planner = MotionGen(cfg)
        planner.warmup(enable_graph=True, warmup_js_trajopt=True)
        return planner

    def _solver(self, name: str):
        if name not in self._solvers:
            started = time.perf_counter()
            self._solvers[name] = self._solver_factory(self._model(name))
            logger.info(
                "cuRobo IK solver for %s ready in %.1fs",
                name,
                time.perf_counter() - started,
            )
        return self._solvers[name]

    def _planner(self, name: str):
        if name not in self._planners:
            started = time.perf_counter()
            self._planners[name] = self._planner_factory(self._model(name))
            logger.info(
                "cuRobo planner for %s ready in %.1fs",
                name,
                time.perf_counter() - started,
            )
        return self._planners[name]

    # -- helpers ----------------------------------------------------------

    @staticmethod
    def _world_dict(obstacles: list[dict[str, Any]]) -> dict[str, Any]:
        """cuRobo ``WorldConfig`` dict; a halfspace becomes a 10 m slab below its plane."""
        world: dict[str, dict[str, Any]] = {"cuboid": {}, "sphere": {}, "capsule": {}}
        for i, obs in enumerate(obstacles):
            name = obs["name"] or f"{obs['type']}_{i}"
            if obs["type"] == "box":
                q = np.asarray(obs["quat_xyzw"])[[3, 0, 1, 2]]
                world["cuboid"][name] = {
                    "dims": obs["extent"].tolist(),
                    "pose": [*obs["position"].tolist(), *q.tolist()],
                }
            elif obs["type"] == "sphere":
                world["sphere"][name] = {
                    "radius": obs["radius"],
                    "pose": [*obs["center"].tolist(), 1.0, 0.0, 0.0, 0.0],
                }
            elif obs["type"] == "capsule":
                q = np.asarray(obs["quat_xyzw"])[[3, 0, 1, 2]]
                h = obs["height"] / 2.0
                world["capsule"][name] = {
                    "radius": obs["radius"],
                    "base": [0.0, 0.0, -h],
                    "tip": [0.0, 0.0, h],
                    "pose": [*obs["position"].tolist(), *q.tolist()],
                }
            else:
                normal = obs["normal"] / np.linalg.norm(obs["normal"])
                depth = 10.0
                rot = Rotation.align_vectors([normal], [[0.0, 0.0, 1.0]])[0]
                center = obs["point"] - normal * depth / 2.0
                q = rot.as_quat()[[3, 0, 1, 2]]
                world["cuboid"][name] = {
                    "dims": [depth * 2, depth * 2, depth],
                    "pose": [*center.tolist(), *q.tolist()],
                }
        world = {k: v for k, v in world.items() if v}
        return world or dict(CUROBO_EMPTY_WORLD)

    def _full_q(self, solver: Any, arm_q: np.ndarray) -> np.ndarray:
        """``arm_q`` padded to cuRobo's cspace with the retract values of the other joints
        (franka.yml has 9: the arm and the two fingers)."""
        kin = getattr(solver, "kinematics", None)
        if kin is None:
            return arm_q
        dof = int(kin.get_dof())
        if dof <= len(arm_q):
            return arm_q[:dof]
        retract = self._to_numpy(kin.retract_config).reshape(-1)
        return np.concatenate([arm_q, retract[len(arm_q) : dof]])

    @staticmethod
    def _to_numpy(value: Any) -> np.ndarray:
        if hasattr(value, "detach"):
            value = value.detach().cpu().numpy()
        return np.asarray(value, dtype=np.float64)

    # -- interface --------------------------------------------------------

    def robots(self) -> dict[str, dict[str, Any]]:
        out = {}
        for name, model in ROBOTS.items():
            out[name] = {
                "description": model.description,
                "ee_link": model.ee_link,
                "tool_offset_xyz": list(model.tool_offset_xyz),
                "arm_joints": list(model.arm_joints),
                "fixed_joints": dict(model.fixed_joints),
                "home_q": list(model.home_q),
                "note": model.note,
                "curobo_config": model.curobo_config,
                "supported": model.curobo_config is not None,
                "loaded": name in self._solvers,
            }
        return out

    def _goal(
        self, model: RobotModel, pos: np.ndarray, quat_xyzw: np.ndarray, solver: Any
    ):
        from curobo.types.math import Pose

        link_pos, link_quat = tcp_to_link(model, pos, quat_xyzw)
        tensor_args = solver.tensor_args
        return Pose(
            position=tensor_args.to_device([link_pos.tolist()]),
            quaternion=tensor_args.to_device([link_quat[[3, 0, 1, 2]].tolist()]),
        )

    def solve(self, robot: str, target_pose: Any, seed_q: Any = None) -> dict[str, Any]:
        started = time.perf_counter()
        model = self._model(robot)
        pos, quat = parse_pose(target_pose, "target_pose")
        solver = self._solver(robot)
        goal = self._goal(model, pos, quat, solver)
        seed = None
        if seed_q is not None:
            # cuRobo takes the seed as a [1, n_seeds, dof] tensor over its whole cspace
            # (franka.yml lists the fingers too), not a JointState.
            full = self._full_q(solver, _q_vector(seed_q, model, "seed_q"))
            seed = solver.tensor_args.to_device([[full.tolist()]])
        result = solver.solve_single(goal, seed_config=seed)
        success = bool(self._to_numpy(result.success).reshape(-1)[0])
        q = self._to_numpy(result.solution).reshape(-1)[: len(model.arm_joints)]
        pos_err = float(self._to_numpy(result.position_error).reshape(-1)[0])
        ori_err = float(self._to_numpy(result.rotation_error).reshape(-1)[0])
        error = None
        if not success:
            error = (
                f"unreachable or in self-collision: cuRobo misses by "
                f"{pos_err * 1000:.1f} mm and {ori_err:.3f} rad"
            )
        return solve_result(
            robot=robot,
            q=q,
            ok=success,
            error=error,
            position_err=pos_err,
            orientation_err=ori_err,
            started=started,
            self_collision_checked=True,
        )

    def plan(
        self,
        robot: str,
        start_q: Any,
        goal_pose: Any = None,
        goal_q: Any = None,
        obstacles: Any = None,
        waypoints: int | None = None,
    ) -> dict[str, Any]:
        from curobo.geom.types import WorldConfig
        from curobo.types.state import JointState
        from curobo.wrap.reacher.motion_gen import MotionGenPlanConfig

        started = time.perf_counter()
        model = self._model(robot)
        start = _q_vector(start_q, model, "start_q")
        if (goal_pose is None) == (goal_q is None):
            raise ValueError("plan takes exactly one of goal_pose or goal_q")
        parsed = [parse_obstacle(o) for o in (obstacles or [])]
        planner = self._planner(robot)
        planner.world_coll_checker.clear_cache()
        planner.update_world(WorldConfig.from_dict(self._world_dict(parsed)))
        start_state = JointState.from_position(
            planner.tensor_args.to_device([self._full_q(planner, start).tolist()])
        )
        plan_cfg = MotionGenPlanConfig(
            max_attempts=10, enable_graph=True, enable_graph_attempt=1, timeout=30.0
        )
        if goal_q is not None:
            goal = _q_vector(goal_q, model, "goal_q")
            goal_state = JointState.from_position(
                planner.tensor_args.to_device([self._full_q(planner, goal).tolist()])
            )
            result = planner.plan_single_js(start_state, goal_state, plan_cfg)
        else:
            goal_pos, goal_quat = parse_pose(goal_pose, "goal_pose")
            result = planner.plan_single(
                start_state, self._goal(model, goal_pos, goal_quat, planner), plan_cfg
            )
        success = bool(self._to_numpy(result.success).reshape(-1)[0])
        status = str(result.status) if result.status is not None else None
        if not success:
            return plan_result(
                robot=robot,
                path=None,
                ok=False,
                error=f"cuRobo found no collision-free path ({status})",
                started=started,
                collision_free=False,
                status=status,
                obstacles=len(parsed),
            )
        traj = result.get_interpolated_plan()
        path = self._to_numpy(traj.position)
        if path.ndim == 3:
            path = path[0]
        path = path[:, : len(model.arm_joints)]
        if waypoints is not None and len(path) > int(waypoints) >= 2:
            idx = np.linspace(0, len(path) - 1, int(waypoints)).round().astype(int)
            path = path[idx]
        return plan_result(
            robot=robot,
            path=path,
            ok=True,
            error=None,
            started=started,
            collision_free=True,
            status=status,
            dt=self.interpolation_dt,
            obstacles=len(parsed),
        )


# ---------------------------------------------------------------------------
# Facade
# ---------------------------------------------------------------------------


class IkFacade(RpcFacade):
    """The ``ik.*`` RPC methods over one backend."""

    SERVICE_NAME = "ik"

    def __init__(self, backend: Any) -> None:
        super().__init__()
        self.backend = backend
        self._rpc["ik.solve"] = self.solve
        self._rpc["ik.plan"] = self.plan
        self._rpc["ik.robots"] = self.robots
        self._readonly_methods.update({"ik.solve", "ik.plan", "ik.robots"})

    def robots(self) -> dict[str, Any]:
        return {"backend": self.backend.name, "robots": self.backend.robots()}

    def solve(self, robot: str, target_pose: Any, seed_q: Any = None) -> dict[str, Any]:
        if not isinstance(robot, str) or not robot:
            raise ValueError("robot must be a robot model name (see ik.robots)")
        return self.backend.solve(robot, target_pose, seed_q)

    def plan(
        self,
        robot: str,
        start_q: Any,
        goal_pose: Any = None,
        goal_q: Any = None,
        obstacles: Any = None,
        waypoints: int = 20,
    ) -> dict[str, Any]:
        if not isinstance(robot, str) or not robot:
            raise ValueError("robot must be a robot model name (see ik.robots)")
        if obstacles is not None and not isinstance(obstacles, list):
            raise ValueError("obstacles must be a list of obstacle dicts")
        return self.backend.plan(
            robot,
            start_q,
            goal_pose=goal_pose,
            goal_q=goal_q,
            obstacles=obstacles,
            waypoints=waypoints,
        )

    def warmup(self, robots: list[str]) -> None:
        """Load (and, for PyRoKi, compile) the named robots before serving."""
        for name in robots:
            model = ROBOTS.get(name)
            if model is None:
                raise ValueError(f"unknown robot {name!r}; known: {sorted(ROBOTS)}")
            started = time.perf_counter()
            if hasattr(self.backend, "fk"):
                pos, quat = self.backend.fk(name, model.home_q)
                self.backend.solve(name, pose_dict(pos, quat), model.home_q)
            else:
                self.backend._solver(name)
            logger.info("warm: %s in %.1fs", name, time.perf_counter() - started)


def _build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="pi-embodied IK / path server")
    parser.add_argument("--transport", choices=["http"], default="http")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument(
        "--backend",
        choices=["pyroki", "curobo"],
        default="pyroki",
        help="pyroki (CPU, default) or curobo (GPU; self-collision IK and collision-free planning)",
    )
    parser.add_argument(
        "--robots",
        default="panda,panda_libero",
        help="Comma-separated robot models to load before serving ('' = load on first use); "
        f"known: {','.join(ROBOTS)}",
    )
    parser.add_argument("--pos-tol", type=float, default=DEFAULT_POS_TOL_M, help="m")
    parser.add_argument(
        "--ori-tol", type=float, default=DEFAULT_ORI_TOL_RAD, help="rad"
    )
    parser.add_argument(
        "--cuda-device",
        type=int,
        default=None,
        help="GPU exposed through CUDA_VISIBLE_DEVICES (curobo backend).",
    )
    parser.add_argument(
        "--parent-watch",
        action="store_true",
        help="watch parent process via stdin pipe and exit when it dies",
    )
    return parser


def build_backend(name: str, *, pos_tol: float, ori_tol: float) -> Any:
    if name == "pyroki":
        return PyrokiBackend(pos_tol=pos_tol, ori_tol=ori_tol)
    if name == "curobo":
        return CuroboBackend(pos_tol=pos_tol, ori_tol=ori_tol)
    raise ValueError(f"unknown backend {name!r}")


def main() -> None:
    args = _build_argparser().parse_args()
    logging.basicConfig(level=logging.INFO, format="[%(name)s] %(message)s")
    if args.cuda_device is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.cuda_device)
    if args.backend == "pyroki":
        # JAX on the CPU: the simulators and VLAs own the GPU.
        os.environ.setdefault("JAX_PLATFORMS", "cpu")
    facade = IkFacade(
        build_backend(args.backend, pos_tol=args.pos_tol, ori_tol=args.ori_tol)
    )
    robots = [r.strip() for r in args.robots.split(",") if r.strip()]
    facade.warmup(robots)
    facade.serve(
        transport=args.transport,
        host=args.host,
        port=args.port,
        parent_watch=args.parent_watch,
    )


if __name__ == "__main__":
    main()
