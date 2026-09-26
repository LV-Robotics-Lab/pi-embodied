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
# OpenETA's ik_preview_check (sim/mcp_server/server.py): a tri-state reachability
# answer where "unknown" is never presented as approval; here the IK runs in the ik
# service instead of a copy of the simulator, so the sim is never touched.

"""Reach preview for env servers: is a TCP pose reachable from the current joints?

An env server started with ``--ik <url>`` builds a :class:`ReachPreview` and serves
``env.preview_reach``; its motion primitives call :func:`require_reachable` before
moving and refuse a target the ik service cannot reach. Without ``--ik`` nothing
changes (``no_service()`` answers ``env.preview_reach``).

Wiring a new env server (the Robosuite server, for instance)::

    add_ik_argument(parser)                       # --ik <url>
    reach = reach_from_args(args, "panda_libero")  # None without --ik

    def preview_reach(self, pos, quat_xyzw=None):  # RPC env.preview_reach
        if reach is None:
            return no_service()
        q = <robot0_joint_pos>                     # the arm joints, rad
        quat = quat_xyzw or <robot0_eef_quat>      # keep the current orientation
        return reach.preview(q, pos, quat, base_pose=<robot0_base world pose>)

    def move_to(self, xyz, ...):                   # any move_to-style primitive
        require_reachable(self.preview_reach(xyz), "move_to")  # raises when unreachable
        ...step the sim...

``base_pose`` (``{"pos", "quat_xyzw"}`` of the robot base in the world) converts a
world-frame target into the robot's base frame, which is what the ik service solves
in; the Franka servers report base-frame poses already and pass none.

Result of :meth:`ReachPreview.preview` (also of ``env.preview_reach``)::

    {"status": "reachable" | "unreachable" | "unknown",
     "reachable": true | false | null,
     "q": [...] | null,                # the IK solution (arm joints, rad)
     "position_err": m, "orientation_err": rad,
     "message": str,
     "target": {"frame": "world" | "base", "pos": [...], "quat_xyzw": [...]},
     "robot": str, "backend": str | null, "path_checked": false}

``unknown`` (the ik service is down, errored or missing) is not approval: it says the
check could not run. :func:`require_reachable` refuses only ``unreachable``.
"""

from __future__ import annotations

import argparse
from typing import Any

import numpy as np
from scipy.spatial.transform import Rotation

from pi_embodied_services.utils.logging import get_logger
from pi_embodied_services.utils.rpc import RpcError, make_rpc_client
from pi_embodied_services.utils.transforms import invert_transform, transform_pose

logger = get_logger("reach")

DEFAULT_TIMEOUT_S = 15.0

IK_HELP = (
    "IK service URL (components/ik_server.py) for env.preview_reach and the reach "
    "check before each motion primitive; without it targets are not checked"
)


def add_ik_argument(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--ik", default=None, help=IK_HELP)


def reach_from_args(args: argparse.Namespace, robot: str) -> "ReachPreview | None":
    """The env server's :class:`ReachPreview` for ``--ik``, or None without it."""
    url = getattr(args, "ik", None)
    return ReachPreview(url, robot) if url else None


def no_service() -> dict[str, Any]:
    """``env.preview_reach`` without ``--ik``: the check cannot run."""
    return {
        "status": "unknown",
        "reachable": None,
        "q": None,
        "position_err": None,
        "orientation_err": None,
        "message": "no IK service: start the env server with --ik <url> to check reach",
        "target": None,
        "robot": None,
        "backend": None,
        "path_checked": False,
    }


def pose_matrix(pos: Any, quat_xyzw: Any) -> np.ndarray:
    """4x4 transform of a pose."""
    matrix = np.eye(4)
    matrix[:3, :3] = Rotation.from_quat(
        np.asarray(quat_xyzw, dtype=np.float64)
    ).as_matrix()
    matrix[:3, 3] = np.asarray(pos, dtype=np.float64).reshape(3)
    return matrix


def pose_in_base(
    pos: Any, quat_xyzw: Any, base_pos: Any, base_quat_xyzw: Any
) -> tuple[np.ndarray, np.ndarray]:
    """A world pose expressed in the frame of a base whose world pose is given."""
    world_pose = np.concatenate(
        [
            np.asarray(pos, dtype=np.float64).reshape(3),
            _unit(np.asarray(quat_xyzw, dtype=np.float64).reshape(4)),
        ]
    )
    base_from_world = invert_transform(pose_matrix(base_pos, _unit(base_quat_xyzw)))
    local = transform_pose(base_from_world, world_pose)
    return local[:3], local[3:]


def delta_target(
    tcp_pose: Any, *, delta_xyz: Any = None, delta_rpy: Any = None
) -> tuple[np.ndarray, np.ndarray]:
    """The TCP pose after a Franka ``move_delta`` / ``rotate_delta`` from ``tcp_pose``
    (xyz + xyzw, base frame): the translation is added in the base frame; the rotation
    is ``Rotation.from_euler("xyz", delta_rpy)`` applied in the base frame, as the
    franka servers' ``rotate_delta`` does."""
    pose = np.asarray(tcp_pose, dtype=np.float64).reshape(-1)
    if pose.shape != (7,):
        raise ValueError("tcp_pose must be xyz + xyzw")
    pos, quat = pose[:3].copy(), _unit(pose[3:])
    if delta_xyz is not None:
        d = np.asarray(delta_xyz, dtype=np.float64).reshape(-1)
        if d.shape != (3,) or not np.isfinite(d).all():
            raise ValueError("delta_xyz must be 3 finite values")
        pos = pos + d
    if delta_rpy is not None:
        r = np.asarray(delta_rpy, dtype=np.float64).reshape(-1)
        if r.shape != (3,) or not np.isfinite(r).all():
            raise ValueError("delta_rpy must be 3 finite values")
        quat = (Rotation.from_euler("xyz", r) * Rotation.from_quat(quat)).as_quat()
    return pos, quat


def _unit(quat: Any) -> np.ndarray:
    q = np.asarray(quat, dtype=np.float64).reshape(4)
    norm = float(np.linalg.norm(q))
    if norm < 1e-9 or not np.isfinite(norm):
        raise ValueError("quaternion must be finite and nonzero")
    return q / norm


class ReachPreview:
    """Asks the ik service whether a TCP pose is reachable from a joint state.

    ``robot`` is the ik service's robot model (``ik.robots``): ``panda`` for the
    Franka servers' TCP, ``panda_libero`` for robosuite's grip site. The optional
    ``client`` (anything with ``call(method, args, kwargs, timeout_s=)``) replaces
    the HTTP client in tests.
    """

    def __init__(
        self,
        url: str | None,
        robot: str,
        *,
        client: Any = None,
        timeout_s: float = DEFAULT_TIMEOUT_S,
    ) -> None:
        if client is None:
            if not url:
                raise ValueError("ReachPreview needs the ik service URL")
            client = make_rpc_client(url)
        self.url = url
        self.robot = robot
        self.timeout_s = float(timeout_s)
        self._client = client
        self._backend: str | None = None

    def _call(self, method: str, **kwargs: Any) -> Any:
        return self._client.call(method, (), kwargs, timeout_s=self.timeout_s)

    def backend(self) -> str | None:
        """The ik service's backend name (cached; None while unreachable)."""
        if self._backend is None:
            try:
                self._backend = str(self._call("ik.robots")["backend"])
            except Exception as exc:  # reported through preview()'s "unknown"
                logger.warning("ik service %s: %s", self.url, exc)
        return self._backend

    def preview(
        self,
        q: Any,
        pos: Any,
        quat_xyzw: Any,
        *,
        base_pose: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Whether the TCP can reach ``pos`` / ``quat_xyzw`` from joints ``q``.

        With ``base_pose`` the target is a world pose and is converted into the
        robot's base frame first; without it, it is a base-frame pose.
        """
        target_pos = np.asarray(pos, dtype=np.float64).reshape(-1)
        if target_pos.shape != (3,) or not np.isfinite(target_pos).all():
            raise ValueError("pos must be 3 finite values")
        target_quat = _unit(quat_xyzw)
        frame = "base"
        solve_pos, solve_quat = target_pos, target_quat
        if base_pose is not None:
            frame = "world"
            solve_pos, solve_quat = pose_in_base(
                target_pos, target_quat, base_pose["pos"], base_pose["quat_xyzw"]
            )
        seed = [float(v) for v in np.asarray(q, dtype=np.float64).reshape(-1)]
        out: dict[str, Any] = {
            "target": {
                "frame": frame,
                "pos": [round(float(v), 5) for v in target_pos],
                "quat_xyzw": [round(float(v), 5) for v in target_quat],
                # The robot base's world pose a world target was converted with.
                **({"base_pose": base_pose} if base_pose is not None else {}),
            },
            "robot": self.robot,
            "backend": self._backend,
            "path_checked": False,
        }
        try:
            result = self._call(
                "ik.solve",
                robot=self.robot,
                target_pose={
                    "pos": solve_pos.tolist(),
                    "quat_xyzw": solve_quat.tolist(),
                },
                seed_q=seed,
            )
        except (RpcError, OSError, ValueError, KeyError, TypeError) as exc:
            logger.warning("reach preview unavailable: %s", exc)
            out.update(
                status="unknown",
                reachable=None,
                q=None,
                position_err=None,
                orientation_err=None,
                message=f"IK service unavailable, reach not checked: {exc}",
            )
            return out
        if not isinstance(result, dict) or "ok" not in result:
            out.update(
                status="unknown",
                reachable=None,
                q=None,
                position_err=None,
                orientation_err=None,
                message=f"IK service returned an invalid result: {result!r}",
            )
            return out
        ok = bool(result["ok"])
        pos_err = result.get("position_err")
        ori_err = result.get("orientation_err")
        if ok:
            message = (
                f"reachable (IK error {float(pos_err) * 1000:.1f} mm, "
                f"{float(ori_err):.3f} rad)"
                if pos_err is not None and ori_err is not None
                else "reachable"
            )
        else:
            error = str(result.get("error") or "no IK solution")
            message = (
                error if error.startswith("unreachable") else f"unreachable: {error}"
            )
        out.update(
            status="reachable" if ok else "unreachable",
            reachable=ok,
            q=result.get("q"),
            position_err=pos_err,
            orientation_err=ori_err,
            message=message,
        )
        return out


def require_reachable(preview: dict[str, Any], action: str) -> dict[str, Any]:
    """Refuse ``action`` (raise ``ValueError``) when the preview says unreachable.

    ``unknown`` passes with a warning: an unavailable ik service must not stop a
    robot that ran without one before.
    """
    status = preview.get("status")
    if status == "unreachable":
        target = preview.get("target") or {}
        raise ValueError(
            f"{action} refused: target {target.get('pos')} ({target.get('frame')} frame) "
            f"is out of reach from the current joints; {preview.get('message')}"
        )
    if status != "reachable":
        logger.warning("%s: reach not checked (%s)", action, preview.get("message"))
    return preview


__all__ = [
    "IK_HELP",
    "ReachPreview",
    "add_ik_argument",
    "delta_target",
    "no_service",
    "pose_in_base",
    "pose_matrix",
    "reach_from_args",
    "require_reachable",
]
