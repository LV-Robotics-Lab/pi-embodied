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

"""What the real Franka servers share on top of their backends: pi's limits, the primitive
manifest's methods and code mode (``code.run``, utils/code_real.py ``RealCodeMode``). Mixed into
the RLinf single-arm facade (../franka/env_server.py), the Polymetis one
(../franka_polymetis/env_server.py) and the dual rig (../dual_franka/env_server.py).

pi's limits. The server takes pi's ``--max-move``, ``--max-rotate``, ``--workspace-xy`` and
``--z-floor`` at spawn and applies them in ``env.move_delta`` / ``env.rotate_delta`` for every
caller (pi's tools, a program, a manual call), on top of the backend's own limits (Polymetis'
per-call caps, workspace and floor; RLinf's pose clip): a translation beyond the per-call limit, a
move that ends outside the box or below the floor (unless it moves back toward it) and a rotation
beyond the per-call limit are refused before anything is commanded.

The primitives (packages/embodied/src/primitives/manifests/franka.json, dual_franka.json):

- ``env.open_gripper`` / ``env.close_gripper``: ``env.set_gripper`` open / closed (the tools and
  CaP-X's high tier).
- Single arm, CaP-X's high tier (FrankaControlApi): ``get_object_pose`` (SAM3 + depth through the
  hand-eye calibration), ``sample_grasp_pose`` (the grasp server's best candidate, else the object's
  point with the current orientation), ``goto_pose`` (bounded legs of the registered
  ``env.rotate_delta`` / ``env.move_delta``, so every leg passes pi's limits, the reach check and the
  detection-id expiry), ``home_pose`` (goto_pose to the TCP pose of the last reset).
- Single arm, CaP-X's joint-space parts (FrankaControlApiReduced): ``solve_ik`` and ``traj_plan``
  (read-only, the ik service), ``move_to_joints`` and ``move_along_trajectory`` only where the
  backend streams joints (Polymetis: ``joints``), each call bounded by the per-joint step
  (``limits.max_joint_step_rad``), pi's translation and rotation limits and the forward-kinematics
  path in the workspace. RLinf's action space is a Cartesian twist (no joint command) and the dual
  rig's joint reset is not bounded per call, so neither serves them.

Quaternions: the servers' TCP poses are xyzw; CaP-X's functions take and return wxyz.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from pi_embodied_services.robots.franka.perception import quat_xyzw_to_matrix
from pi_embodied_services.utils.code_real import (
    RealCodeMode,
    check_rotation,
    check_translation,
    vec3,
    workspace_refusal,
)

#: Most bounded legs (rotations + translations) one goto_pose / home_pose / trajectory runs.
MAX_LEGS = 8
#: A leg's share of the per-call limit (headroom for the TCP's settling error).
LEG_SHARE = 0.95
#: Most waypoints of traj_plan / move_along_trajectory, and traj_plan's step (m).
MAX_TRAJECTORY = 100
TRAJ_STEP_M = 0.02
#: Forward-kinematics samples of one joint move's path.
FK_SAMPLES = 10


def unit_wxyz(q: Any, name: str = "quaternion_wxyz") -> np.ndarray:
    a = np.asarray(q, dtype=np.float64).reshape(-1)
    if a.shape != (4,) or not np.all(np.isfinite(a)) or np.linalg.norm(a) < 1e-9:
        raise ValueError(f"{name} must be 4 finite numbers (a wxyz quaternion)")
    return a / np.linalg.norm(a)


def wxyz_matrix(q: Any) -> np.ndarray:
    w, x, y, z = unit_wxyz(q)
    return quat_xyzw_to_matrix(np.array([x, y, z, w]))


def matrix_xyzw(R: np.ndarray) -> np.ndarray:
    """A rotation matrix as a unit xyzw quaternion (w >= 0)."""
    m = np.asarray(R, dtype=np.float64)
    t = float(np.trace(m))
    if t > 0:
        s = math.sqrt(t + 1.0) * 2
        q = [
            (m[2, 1] - m[1, 2]) / s,
            (m[0, 2] - m[2, 0]) / s,
            (m[1, 0] - m[0, 1]) / s,
            0.25 * s,
        ]
    else:
        i = int(np.argmax(np.diag(m)))
        j, k = (i + 1) % 3, (i + 2) % 3
        s = math.sqrt(1.0 + m[i, i] - m[j, j] - m[k, k]) * 2
        q = [0.0, 0.0, 0.0, (m[k, j] - m[j, k]) / s]
        q[i] = 0.25 * s
        q[j] = (m[j, i] + m[i, j]) / s
        q[k] = (m[k, i] + m[i, k]) / s
    a = np.asarray(q, dtype=np.float64)
    a /= np.linalg.norm(a)
    return -a if a[3] < 0 else a


def xyzw_to_wxyz(q: Any) -> list[float]:
    x, y, z, w = (float(v) for v in np.asarray(q, dtype=np.float64).reshape(4))
    return [w, x, y, z]


def rotvec_of(R: np.ndarray) -> np.ndarray:
    """The axis-angle vector of a rotation matrix."""
    q = matrix_xyzw(R)
    angle = 2 * math.atan2(float(np.linalg.norm(q[:3])), float(q[3]))
    n = float(np.linalg.norm(q[:3]))
    return np.zeros(3) if n < 1e-12 else q[:3] / n * angle


def rotvec_matrix(v: Any) -> np.ndarray:
    v = np.asarray(v, dtype=np.float64).reshape(3)
    angle = float(np.linalg.norm(v))
    if angle < 1e-12:
        return np.eye(3)
    k = v / angle
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + math.sin(angle) * K + (1 - math.cos(angle)) * K @ K


def euler_xyz(R: np.ndarray) -> np.ndarray:
    """Extrinsic xyz Euler angles (rotate_delta's delta_rpy: R = Rz(yaw) Ry(pitch) Rx(roll))."""
    m = np.asarray(R, dtype=np.float64)
    pitch = -math.asin(max(-1.0, min(1.0, float(m[2, 0]))))
    roll = math.atan2(float(m[2, 1]), float(m[2, 2]))
    yaw = math.atan2(float(m[1, 0]), float(m[0, 0]))
    return np.array([roll, pitch, yaw])


class FrankaCodeMode(RealCodeMode):
    """The real Franka servers' limits, manifest methods and code mode (see the module doc)."""

    #: pi's flags (packages/embodied/src/robots/franka/index.ts): --max-move, --max-rotate; the
    #: box and the floor are off unless pi (or the operator's command line) passes them.
    _LIMIT_DEFAULTS = {
        "max_move_m": 0.1,
        "max_rotate_rad": 0.5,
        "z_floor_m": None,
        "workspace_xy": None,
    }
    #: Whether motion methods take an ``arm`` first (the dual rig).
    _ARMED = False
    #: The TCP pose (xyz + xyzw) after the last successful reset: home_pose's target.
    _home_tcp: np.ndarray | None = None

    # ---- the facade's hooks ----

    def _manifest_name(self) -> str:
        return "dual_franka" if self._ARMED else "franka"

    def _tcp_pose(self, arm: str | None = None) -> np.ndarray:
        """The TCP pose (xyz + xyzw) in the frame pi's limits are in (single arm: the base)."""
        state = self._rpc["env.get_robot_state"]()
        return np.asarray(state["raw_base_state"]["tcp_pose"], dtype=np.float64)[:7]

    def _code_tcp(self, arm: str | None) -> np.ndarray:
        return self._tcp_pose(arm)[:3]

    def _joints(self) -> np.ndarray:
        state = self._rpc["env.get_robot_state"]()
        return np.asarray(
            state["raw_base_state"]["arm_joint_position"], dtype=np.float64
        ).reshape(-1)[:7]

    def _view_backend(self) -> Any:
        """What answers get_observation / get_camera_meta / get_robot_state for the camera views."""
        return getattr(self, "_backend", self)

    def _joint_mover(self) -> Any:
        """``move(q) -> dict``: a bounded, stop-polled joint stream (None: not on this backend)."""
        return None

    def _joint_step_rad(self) -> float | None:
        """The backend's per-call joint step (rad per joint); None: no joint moves."""
        return None

    def _box_violation(self, p: np.ndarray) -> float:
        """How far ``p`` lies outside the backend's own workspace (0 inside)."""
        return 0.0

    def _has(self, capability: str) -> bool:
        """What this server can serve of the manifest's ``requires``."""
        perception = getattr(self, "_perception", None)
        caps = perception.capabilities() if perception is not None else {}
        grasp = getattr(self, "_grasp", None)
        return {
            "sam3": bool(caps.get("segment")),
            "unidepth": bool(caps.get("enhance_depth")),
            "ik": getattr(self, "_reach", None) is not None,
            "grasp": grasp is not None,
            "place": grasp is not None and bool(grasp.capabilities().get("place")),
            "geometry": bool(getattr(self, "_geometry_on", False)),
            "joints": self._joint_mover() is not None,
            "vla": bool(getattr(self, "_has_vla", False)),
        }.get(capability, False)

    # ---- pi's limits in the motion methods ----

    def _check_motion(self, name: str, arm: str | None, delta: Any) -> None:
        """Refuse a motion beyond pi's limits before it is commanded."""
        if name == "rotate_delta":
            check_rotation(delta, self._limit("max_rotate_rad"))
            return
        d = vec3(delta, "delta_xyz")
        check_translation(d, self._limit("max_move_m"))
        floor, box = self._limit("z_floor_m"), self._limit("workspace_xy")
        if floor is None and box is None:
            return
        tcp = self._code_tcp(arm)
        where = f" ({arm} arm)" if arm else ""
        why = workspace_refusal(tcp, tcp + d, floor, box, where)
        if why:
            raise ValueError(why)

    def _limited(self, name: str, handler: Any) -> Any:
        """``handler`` (env.move_delta / env.rotate_delta) behind pi's limits."""
        key = "delta_xyz" if name == "move_delta" else "delta_rpy"

        def call(*args: Any, **kwargs: Any) -> Any:
            rest = list(args)
            arm = None
            if self._ARMED:
                arm = kwargs.get("arm")
                if arm is None and rest:
                    arm = rest.pop(0)
            delta = kwargs.get(key, rest[0] if rest else None)
            if delta is None:
                raise ValueError(f"env.{name} takes {key}")
            self._check_motion(name, arm, delta)
            return handler(*args, **kwargs)

        return call

    def _install_franka(self) -> None:
        """At the end of ``_register_rpc`` (after perception, geometry and the grasp planner):
        the manifest's methods, then code.api / code.run (``_install_real``)."""
        rpc = self._rpc
        rpc["env.open_gripper"] = self.open_gripper
        rpc["env.close_gripper"] = self.close_gripper
        if not self._ARMED:
            reset = rpc["env.reset"]

            def reset_and_home(*args: Any, **kwargs: Any) -> Any:
                out = reset(*args, **kwargs)
                try:
                    self._home_tcp = self._tcp_pose()
                except Exception:  # noqa: BLE001 - no home until a state reads back
                    self._home_tcp = None
                return out

            rpc["env.reset"] = reset_and_home
            rpc.update(
                {
                    "env.get_object_pose": self.get_object_pose,
                    "env.sample_grasp_pose": self.sample_grasp_pose,
                    "env.goto_pose": self.goto_pose,
                    "env.home_pose": self.home_pose,
                }
            )
            self._readonly_methods.update(
                {"env.get_object_pose", "env.sample_grasp_pose"}
            )
            if getattr(self, "_reach", None) is not None:
                rpc["env.solve_ik"] = self.solve_ik
                rpc["env.traj_plan"] = self.traj_plan
                self._readonly_methods.update({"env.solve_ik", "env.traj_plan"})
                if self._joint_mover() is not None:
                    rpc["env.move_to_joints"] = self.move_to_joints
                    rpc["env.move_along_trajectory"] = self.move_along_trajectory
        self._install_real(self._manifest_name(), have=self._has)

    # ---- grippers ----

    def open_gripper(self, arm: str | None = None) -> dict[str, Any]:
        """Open the gripper and wait for it to settle (env.set_gripper)."""
        lead = [arm] if self._ARMED else []
        return self._rpc["env.set_gripper"](*lead, open=True)

    def close_gripper(self, arm: str | None = None) -> dict[str, Any]:
        """Close the gripper and wait for it to settle (env.set_gripper)."""
        lead = [arm] if self._ARMED else []
        return self._rpc["env.set_gripper"](*lead, open=False)

    # ---- CaP-X's high tier (single arm) ----

    def _object_points(self, object_name: str) -> np.ndarray:
        """The object's base-frame points: the SAM3 mask (env.segment, so its id is booked)
        through the depth of the first camera that finds it (third person, then the wrist)."""
        from pi_embodied_services.robots.franka import perception as franka_perception
        from pi_embodied_services.robots.franka.grasp_views import franka_view

        if not self._has("sam3"):
            raise RuntimeError("get_object_pose needs SAM3 (the server's --sam3)")
        cache: dict[str, Any] = {}

        def calibration() -> dict[str, Any]:
            if "bundle" not in cache:
                cache["bundle"] = franka_perception.load_calibration_bundle()
            return cache["bundle"]

        view = franka_view(self._view_backend(), calibration)
        tried = []
        for camera in ("third_person", "wrist"):
            seg = self._rpc["env.segment"](camera=camera, prompt=str(object_name))
            if not seg.get("found"):
                tried.append(f"{camera}: {seg.get('reason') or 'no mask'}")
                continue
            mask = np.asarray(self._perception.book.get(seg["ids"][0])["mask"], bool)
            v = view(camera)
            depth = np.asarray(v["depth"], dtype=np.float64)
            if depth.shape != mask.shape:
                tried.append(f"{camera}: mask {mask.shape} vs depth {depth.shape}")
                continue
            rows, cols = np.nonzero(
                mask & np.isfinite(depth) & (depth > 0) & (depth < 5)
            )
            if len(rows) < 10:
                tried.append(f"{camera}: {len(rows)} valid depth pixels")
                continue
            K = np.asarray(v["intrinsic_K"], dtype=np.float64)
            z = depth[rows, cols]
            cam = np.stack(
                [(cols - K[0, 2]) * z / K[0, 0], (rows - K[1, 2]) * z / K[1, 1], z], 1
            )
            T = np.asarray(v["extrinsic_cam2world"], dtype=np.float64)
            return cam @ T[:3, :3].T + T[:3, 3]
        raise ValueError(
            f"no SAM3 detection with depth for {object_name!r} ({'; '.join(tried)})"
        )

    def get_object_pose(
        self, object_name: str, return_bbox_extent: bool = False
    ) -> list:
        """CaP-X's get_object_pose: [position, quaternion_wxyz (identity), extent or None]."""
        pts = self._object_points(object_name)
        pos = np.median(pts, axis=0)
        extent = np.percentile(pts, 95, axis=0) - np.percentile(pts, 5, axis=0)
        return [
            [round(float(v), 4) for v in pos],
            [1.0, 0.0, 0.0, 0.0],
            [round(float(v), 4) for v in extent] if return_bbox_extent else None,
        ]

    def sample_grasp_pose(self, object_name: str) -> list:
        """CaP-X's sample_grasp_pose: [position, quaternion_wxyz] of a TCP grasp pose."""
        if getattr(self, "_grasp", None) is not None:
            plan = self._rpc["env.plan_grasp"](object=str(object_name))
            if not plan.get("candidates"):
                raise ValueError(f"no grasp candidates for {object_name!r}")
            best = plan["candidates"][0]
            return [
                [float(v) for v in best["eef_position"]],
                xyzw_to_wxyz(best["eef_quat_xyzw"]),
            ]
        pts = self._object_points(object_name)
        return [
            [round(float(v), 4) for v in np.median(pts, axis=0)],
            xyzw_to_wxyz(self._tcp_pose()[3:]),
        ]

    def _route(
        self, stops: list[np.ndarray], R_target: np.ndarray
    ) -> tuple[list[np.ndarray], list[np.ndarray], np.ndarray]:
        """The legs of a goto_pose (rotation steps as delta_rpy, then translation leg ends),
        checked whole against pi's limits and the backend's box: ValueError before any
        motion when the route needs more than MAX_LEGS legs or leaves the workspace."""
        tcp = self._tcp_pose()
        here, R_now = tcp[:3], quat_xyzw_to_matrix(tcp[3:])
        max_rot = self._limit("max_rotate_rad") or 0.5
        max_move = self._limit("max_move_m") or 0.1
        turn = rotvec_of(R_target @ R_now.T)
        angle = float(np.linalg.norm(turn))
        n_rot = math.ceil(angle / (LEG_SHARE * max_rot)) if angle > 1e-3 else 0
        rotations: list[np.ndarray] = []
        prev = R_now
        for k in range(1, n_rot + 1):
            R_k = rotvec_matrix(turn * k / n_rot) @ R_now
            rotations.append(euler_xyz(R_k @ prev.T))
            prev = R_k
        legs: list[np.ndarray] = []
        start = here
        for stop in stops:
            dist = float(np.linalg.norm(stop - start))
            n = math.ceil(dist / (LEG_SHARE * max_move)) if dist > 1e-4 else 0
            legs += [start + (stop - start) * k / n for k in range(1, n + 1)]
            start = stop
        if len(rotations) + len(legs) > MAX_LEGS:
            raise ValueError(
                f"the pose is {len(rotations)} rotation and {len(legs)} translation legs away "
                f"(at most {MAX_LEGS} per call within --max-move {max_move} m / --max-rotate "
                f"{max_rot} rad): move closer first"
            )
        floor, box = self._limit("z_floor_m"), self._limit("workspace_xy")
        prev_p = here
        for p in legs:
            why = workspace_refusal(prev_p, p, floor, box)
            if why is None and self._box_violation(p) > 1e-6:
                why = f"the route passes {np.round(p, 3).tolist()}, outside the server's workspace"
            if why:
                raise ValueError(f"goto_pose refused before moving: {why}")
            prev_p = p
        return rotations, legs, here

    def goto_pose(
        self, position: Any, quaternion_wxyz: Any, z_approach: float = 0.0
    ) -> dict[str, Any]:
        """CaP-X's goto_pose over bounded legs (see the module doc)."""
        target = vec3(position, "position")
        R_target = wxyz_matrix(quaternion_wxyz)
        za = float(z_approach or 0.0)
        if not (0.0 <= za <= 0.3):
            raise ValueError("z_approach must be within 0..0.3 m")
        stops = [target + R_target @ np.array([0.0, 0.0, -za])] if za > 1e-4 else []
        rotations, legs, _ = self._route([*stops, target], R_target)
        out: dict[str, Any] = {"ok": True}
        done = 0

        def ended(r: Any) -> bool:
            return (
                not isinstance(r, dict)
                or bool(r.get("cancelled"))
                or r.get("ok") is False
            )

        for rpy in rotations:
            if self.stop_requested():
                return {**out, "cancelled": True, "legs": done, "reached": False}
            out = self._rpc["env.rotate_delta"](delta_rpy=[float(v) for v in rpy])
            done += 1
            if ended(out):
                return {**out, "legs": done, "reached": False}
        for p in legs:
            if self.stop_requested():
                return {**out, "cancelled": True, "legs": done, "reached": False}
            delta = p - self._tcp_pose()[:3]
            out = self._rpc["env.move_delta"](delta_xyz=[float(v) for v in delta])
            done += 1
            if ended(out):
                return {**out, "legs": done, "reached": False}
        return {**out, "legs": done, "reached": True}

    def home_pose(self) -> dict[str, Any]:
        """CaP-X's home_pose: goto_pose to the TCP pose of the last reset."""
        if self._home_tcp is None:
            raise ValueError("home_pose: no reset has recorded a home pose yet")
        home = self._home_tcp
        return self.goto_pose(home[:3], xyzw_to_wxyz(home[3:]))

    # ---- CaP-X's joint-space parts (single arm) ----

    def solve_ik(self, position: Any, quaternion_wxyz: Any) -> list[float]:
        """7 joint angles for a base-frame TCP pose (the ik service, seeded with the joints)."""
        w, x, y, z = unit_wxyz(quaternion_wxyz)
        r = self.preview_reach(vec3(position, "position").tolist(), [x, y, z, w])
        if r.get("status") != "reachable" or r.get("q") is None:
            raise ValueError(f"no IK solution: {r.get('message') or r.get('status')}")
        return [float(v) for v in np.asarray(r["q"], dtype=np.float64).reshape(-1)[:7]]

    def traj_plan(self, start_pose_wxyz_xyz: Any, end_pose_wxyz_xyz: Any) -> list:
        """Joint waypoints along a straight TCP line (IK from each predecessor); nothing moves."""
        a = np.asarray(start_pose_wxyz_xyz, dtype=np.float64).reshape(7)
        b = np.asarray(end_pose_wxyz_xyz, dtype=np.float64).reshape(7)
        if not (np.all(np.isfinite(a)) and np.all(np.isfinite(b))):
            raise ValueError("poses must be 7 finite numbers (wxyz then xyz)")
        n = max(
            1,
            min(MAX_TRAJECTORY, math.ceil(np.linalg.norm(b[4:] - a[4:]) / TRAJ_STEP_M)),
        )
        Ra, Rb = wxyz_matrix(a[:4]), wxyz_matrix(b[:4])
        turn = rotvec_of(Rb @ Ra.T)
        seed = self._joints()
        out = []
        for k in range(1, n + 1):
            t = k / n
            q = matrix_xyzw(rotvec_matrix(turn * t) @ Ra)
            r = self._reach.preview(seed, a[4:] + (b[4:] - a[4:]) * t, q)
            if r.get("status") != "reachable" or r.get("q") is None:
                raise ValueError(
                    f"no IK solution at waypoint {k}/{n}: {r.get('message') or r.get('status')}"
                )
            seed = np.asarray(r["q"], dtype=np.float64).reshape(-1)[:7]
            out.append([float(v) for v in seed])
        return out

    def _joint_goal(self, joints: Any) -> np.ndarray:
        q = np.asarray(joints, dtype=np.float64).reshape(-1)
        if q.shape != (7,) or not np.all(np.isfinite(q)):
            raise ValueError("joints must be 7 finite joint angles (rad)")
        return q

    def _check_joint_move(self, q0: np.ndarray, q1: np.ndarray) -> float:
        """Refuse a joint move beyond the per-call bounds; returns its TCP translation (m)."""
        step = self._joint_step_rad()
        if step is None or self._reach is None:
            raise RuntimeError(
                "joint moves need the Polymetis backend and the ik service (--ik)"
            )
        jump = float(np.max(np.abs(q1 - q0)))
        if jump > step + 1e-9:
            raise ValueError(
                f"the move turns a joint by {jump:.3f} rad; the limit is {step} rad per call "
                "(limits.max_joint_step_rad). Split it into smaller moves"
            )
        path = self._reach.joint_path(q0, q1, FK_SAMPLES)
        if not path["ok"] or not path.get("tcp_path"):
            raise ValueError(f"the joint path cannot be checked: {path.get('error')}")
        poses = np.asarray(path["tcp_path"], dtype=np.float64).reshape(-1, 7)
        start = poses[0]
        move = float(np.linalg.norm(poses[-1][:3] - start[:3]))
        check_translation(poses[-1][:3] - start[:3], self._limit("max_move_m"))
        turn = rotvec_of(
            quat_xyzw_to_matrix(poses[-1][3:]) @ quat_xyzw_to_matrix(start[3:]).T
        )
        check_rotation(turn, self._limit("max_rotate_rad"), "the TCP rotation")
        floor, box = self._limit("z_floor_m"), self._limit("workspace_xy")
        prev = start[:3]
        for p in poses[1:, :3]:
            why = workspace_refusal(prev, p, floor, box)
            if why is None and self._box_violation(p) > 1e-6:
                why = f"the path passes {np.round(p, 3).tolist()}, outside the server's workspace"
            if why:
                raise ValueError(f"move_to_joints refused before moving: {why}")
            prev = p
        return move

    def move_to_joints(self, joints: Any) -> dict[str, Any]:
        """Stream to a joint configuration within the per-call bounds (Polymetis)."""
        q1 = self._joint_goal(joints)
        self._check_joint_move(self._joints(), q1)
        return self._joint_mover()(q1)

    def move_along_trajectory(self, trajectory: Any) -> dict[str, Any]:
        """move_to_joints through every waypoint; the whole path checked first."""
        traj = np.asarray(trajectory, dtype=np.float64)
        if traj.ndim != 2 or traj.shape[1] != 7 or not 1 <= len(traj) <= MAX_TRAJECTORY:
            raise ValueError(
                f"trajectory must be [N, 7] with 1 <= N <= {MAX_TRAJECTORY}"
            )
        q = self._joints()
        total = 0.0
        for goal in traj:
            total += self._check_joint_move(q, self._joint_goal(goal))
            q = goal
        cap = MAX_LEGS * (self._limit("max_move_m") or 0.1)
        if total > cap + 1e-9:
            raise ValueError(
                f"the trajectory moves the TCP {total:.3f} m; at most {cap:.3f} m per call"
            )
        out: dict[str, Any] = {}
        for i, goal in enumerate(traj):
            if self.stop_requested():
                return {**out, "cancelled": True, "waypoints_done": i}
            out = self.move_to_joints(goal)
            if (
                not isinstance(out, dict)
                or out.get("cancelled")
                or out.get("ok") is False
            ):
                return {**(out or {}), "waypoints_done": i}
        return {**out, "waypoints_done": len(traj)}

    # ---- code mode ----

    def _video_frame(self, obs: dict) -> np.ndarray | None:
        """The first external view, else the wrist (pi's episode video)."""
        extra = obs.get("extra_view_images")
        if isinstance(extra, np.ndarray) and extra.ndim == 4 and len(extra):
            return extra[0]
        main = obs.get("main_images")
        return main if isinstance(main, np.ndarray) else None

    def _code_move_m(self, method: str, kwargs: dict) -> float:
        """A program call's TCP translation (the run's move budget); rotations and the gripper
        move none."""
        if method == "env.move_delta":
            return float(np.linalg.norm(vec3(kwargs["delta_xyz"], "delta_xyz")))
        if method in ("env.goto_pose", "env.home_pose"):
            here = self._tcp_pose()[:3]
            if method == "env.home_pose":
                if self._home_tcp is None:
                    return 0.0
                return float(np.linalg.norm(self._home_tcp[:3] - here))
            target = vec3(kwargs["position"], "position")
            za = float(kwargs.get("z_approach") or 0.0)
            return float(np.linalg.norm(target - here) + 2 * abs(za))
        if method == "env.move_to_joints":
            return self._check_joint_move(
                self._joints(), self._joint_goal(kwargs["joints"])
            )
        if method == "env.move_along_trajectory":
            q, total = self._joints(), 0.0
            for goal in np.asarray(kwargs["trajectory"], dtype=np.float64)[
                :MAX_TRAJECTORY
            ]:
                total += self._check_joint_move(q, self._joint_goal(goal))
                q = goal
            return total
        return 0.0


__all__ = ["MAX_LEGS", "FrankaCodeMode"]
