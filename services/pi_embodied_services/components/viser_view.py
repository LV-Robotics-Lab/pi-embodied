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

"""A live 3D view of a robot's env server in the browser (Viser), after CaP-X.

CaP-X (github.com/capgym/cap-x @53e9966) draws its scenes with Viser when an env runs
with ``viser_debug`` (``capx/envs/simulators/robosuite_base.py`` ``_update_viser_server``):
the point cloud back-projected from each RGB-D camera, the camera frustum with its image,
the end-effector frame and the chosen grasp. Here the same view is a separate process
that follows a running env server over its RPC (it changes nothing there): every
``--interval`` seconds it reads the cameras (rgb, metric depth, K, cam2world) and the
robot state, and redraws.

    PYTHONPATH=services python -m pi_embodied_services.components.viser_view \\
        --robot libero --env http://127.0.0.1:PORT --viser-port 8080

pi starts it with ``--viser`` (``packages/embodied/src/viser.ts``), which also pushes each
``plan_grasp`` / ``plan_place`` result here (``viser.grasps``): every candidate as a frame,
the active one larger. Sources:

* ``libero``: ``env.get_observation`` (``agentview`` and ``wrist``, each with depth and
  calibration, and the EEF pose).
* ``franka``: ``env.get_observation`` (``main_*`` = wrist, ``extra_view_*[0]`` = third
  person), ``env.get_camera_meta`` for K, ``env.get_robot_state`` for the TCP, and the
  hand-eye calibration bundle (``robots/franka/perception.py``) for cam2world. Without the
  bundle only the EEF is drawn.

Env calls queue behind the robot's own (the env server runs one call at a time), so the
interval bounds what the view costs the episode. The process is an RPC service
(``viser.status``, ``viser.refresh``, ``viser.grasps``, ``viser.attach``) and exits with
its parent under ``--parent-watch``. Viser itself is the ``viser`` extra.
"""

from __future__ import annotations

import argparse
import logging
import math
import threading
from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np

from pi_embodied_services.utils.logging import get_logger
from pi_embodied_services.utils.rpc import RpcFacade

logger = get_logger("viser_view")

Call = Callable[[str], Any]


@dataclass
class CameraView:
    """One RGB-D camera: rgb uint8 [H, W, 3], depth float32 [H, W] metres (0 = none),
    K [3, 3] and cam2world [4, 4] (OpenCV camera: +z forward, +y down), or None."""

    name: str
    rgb: np.ndarray
    depth: np.ndarray | None
    K: np.ndarray | None
    cam2world: np.ndarray | None


@dataclass
class Snapshot:
    cameras: list[CameraView] = field(default_factory=list)
    #: EEF position and wxyz quaternion in the world frame
    eef: tuple[np.ndarray, np.ndarray] | None = None
    gripper_width: float | None = None
    notes: list[str] = field(default_factory=list)


def xyzw_to_wxyz(q: Any) -> np.ndarray:
    x, y, z, w = (float(v) for v in q)
    return np.array([w, x, y, z])


def matrix_to_wxyz(R: np.ndarray) -> np.ndarray:
    from scipy.spatial.transform import Rotation

    return xyzw_to_wxyz(Rotation.from_matrix(np.asarray(R, dtype=np.float64)).as_quat())


def _view(name: str, v: dict[str, Any]) -> CameraView:
    return CameraView(
        name=name,
        rgb=np.ascontiguousarray(np.asarray(v["rgb"])[..., :3], dtype=np.uint8),
        depth=None if v.get("depth") is None else np.asarray(v["depth"], np.float32),
        K=None if v.get("intrinsic_K") is None else np.asarray(v["intrinsic_K"], float),
        cam2world=None
        if v.get("extrinsic_cam2world") is None
        else np.asarray(v["extrinsic_cam2world"], float),
    )


def libero_snapshot(call: Call) -> Snapshot:
    """LIBERO: ``env.get_observation`` carries both views with depth and calibration."""
    obs = call("env.get_observation")
    snap = Snapshot(
        cameras=[_view(k, obs[k]) for k in ("agentview", "wrist") if k in obs]
    )
    if obs.get("eef_pos") is not None and obs.get("eef_quat_xyzw") is not None:
        snap.eef = (
            np.asarray(obs["eef_pos"], float),
            xyzw_to_wxyz(obs["eef_quat_xyzw"]),
        )
    if obs.get("gripper_width") is not None:
        snap.gripper_width = float(obs["gripper_width"])
    return snap


def _load_franka_calibration() -> dict[str, Any] | None:
    try:
        from pi_embodied_services.robots.franka.perception import (
            load_calibration_bundle,
        )

        return load_calibration_bundle()
    except Exception as exc:  # the view still shows the EEF
        logger.warning("no Franka calibration bundle: %s", exc)
        return None


def franka_snapshot(
    call: Call,
    calibration: Callable[[], dict[str, Any] | None] = _load_franka_calibration,
) -> Snapshot:
    """The single Franka (``robots/franka/grasp_views.py``'s layout and calibration)."""
    from pi_embodied_services.robots.franka.grasp_views import (
        FRANKA_CAMERAS,
        franka_intrinsics,
    )

    obs = call("env.get_observation") or {}
    meta = call("env.get_camera_meta")
    tcp = np.asarray(
        call("env.get_robot_state")["raw_base_state"]["tcp_pose"], dtype=float
    ).reshape(-1)
    T_tcp = np.eye(4)
    T_tcp[:3, :3] = _rotation(xyzw_to_wxyz(tcp[3:7]))
    T_tcp[:3, 3] = tcp[:3]
    snap = Snapshot(eef=(tcp[:3], xyzw_to_wxyz(tcp[3:7])))
    cal = calibration()
    if cal is None:
        snap.notes.append("no hand-eye calibration: cameras not placed")
    for name, (image_key, depth_key, index) in FRANKA_CAMERAS.items():
        if obs.get(image_key) is None:
            continue
        rgb = np.asarray(obs[image_key])
        depth = obs.get(depth_key)
        depth = None if depth is None else np.asarray(depth, np.float32)
        if index is not None:
            rgb = rgb[index]
            depth = None if depth is None else depth[index]
        if depth is not None:
            depth = np.squeeze(depth)
            depth = depth if depth.ndim == 2 else None
        K = franka_intrinsics(meta, "main" if index is None else f"extra_{index}")
        cam2world = None
        if cal is not None:
            if name == "wrist":
                cam2world = T_tcp @ np.asarray(cal["wrist"]["matrix"], float)
            else:
                cam2world = np.asarray(cal["external"]["matrix"], float)
        snap.cameras.append(
            CameraView(
                name, np.asarray(rgb)[..., :3].astype(np.uint8), depth, K, cam2world
            )
        )
    return snap


SOURCES: dict[str, Callable[[Call], Snapshot]] = {
    "libero": libero_snapshot,
    "franka": franka_snapshot,
}


def _rotation(wxyz: np.ndarray) -> np.ndarray:
    from scipy.spatial.transform import Rotation

    w, x, y, z = wxyz
    return Rotation.from_quat([x, y, z, w]).as_matrix()


def back_project(
    view: CameraView, stride: int = 4, max_depth: float = 3.0
) -> tuple[np.ndarray, np.ndarray]:
    """World points (float32 [N, 3]) and their colours (uint8 [N, 3]) of every
    ``stride``-th pixel with depth in (0, max_depth]: x = (col - cx) z / fx,
    y = (row - cy) z / fy, then cam2world."""
    if view.depth is None or view.K is None or view.cam2world is None:
        return np.zeros((0, 3), np.float32), np.zeros((0, 3), np.uint8)
    z = view.depth[::stride, ::stride]
    rows, cols = np.mgrid[
        0 : view.depth.shape[0] : stride, 0 : view.depth.shape[1] : stride
    ]
    ok = np.isfinite(z) & (z > 0) & (z <= max_depth)
    z, rows, cols = z[ok], rows[ok], cols[ok]
    K = view.K
    cam = np.stack(
        [(cols - K[0, 2]) * z / K[0, 0], (rows - K[1, 2]) * z / K[1, 1], z], axis=1
    )
    world = cam @ view.cam2world[:3, :3].T + view.cam2world[:3, 3]
    colors = view.rgb[::stride, ::stride][ok]
    return world.astype(np.float32), np.ascontiguousarray(colors, dtype=np.uint8)


def candidate_pose(c: dict[str, Any]) -> tuple[np.ndarray, np.ndarray] | None:
    """A grasp/placement candidate's pose (position, wxyz): its EEF pose when given,
    else its grasp centre with the approach (x) and closing (y) axes."""
    if c.get("eef_position") is not None and c.get("eef_quat_xyzw") is not None:
        return np.asarray(c["eef_position"], float), xyzw_to_wxyz(c["eef_quat_xyzw"])
    if c.get("position") is None:
        return None
    pos = np.asarray(c["position"], float)
    if c.get("approach") is None or c.get("closing") is None:
        return pos, np.array([1.0, 0.0, 0.0, 0.0])
    x = np.asarray(c["approach"], float)
    y = np.asarray(c["closing"], float)
    y = y - x * float(x @ y)
    if np.linalg.norm(x) < 1e-9 or np.linalg.norm(y) < 1e-9:
        return pos, np.array([1.0, 0.0, 0.0, 0.0])
    x, y = x / np.linalg.norm(x), y / np.linalg.norm(y)
    return pos, matrix_to_wxyz(np.stack([x, y, np.cross(x, y)], axis=1))


class Scene:
    """What is drawn on one Viser server (``viser.ViserServer`` or a stand-in)."""

    def __init__(self, server: Any, *, stride: int = 4, max_depth: float = 3.0) -> None:
        self.server = server
        self.stride = int(stride)
        self.max_depth = float(max_depth)
        self._grasps: list[Any] = []
        self._image: Any = None
        self._text: Any = None

    def update(self, snap: Snapshot, status: str = "") -> int:
        """Redraw the cameras and the EEF; returns the number of points drawn."""
        scene = self.server.scene
        total = 0
        for view in snap.cameras:
            points, colors = back_project(view, self.stride, self.max_depth)
            total += len(points)
            if len(points):
                scene.add_point_cloud(
                    f"/cameras/{view.name}/points",
                    points=points,
                    colors=colors,
                    point_size=0.004,
                    point_shape="square",
                )
            if view.cam2world is not None and view.K is not None:
                h, w = view.rgb.shape[:2]
                scene.add_camera_frustum(
                    f"/cameras/{view.name}/frustum",
                    fov=2 * math.atan(h / 2 / float(view.K[1, 1])),
                    aspect=w / h,
                    scale=0.08,
                    image=view.rgb,
                    wxyz=matrix_to_wxyz(view.cam2world[:3, :3]),
                    position=view.cam2world[:3, 3],
                )
        if snap.cameras:
            if self._image is None:
                self._image = self.server.gui.add_image(
                    snap.cameras[0].rgb, label=snap.cameras[0].name
                )
            else:
                self._image.image = snap.cameras[0].rgb
        if snap.eef is not None:
            scene.add_frame(
                "/eef",
                position=snap.eef[0],
                wxyz=snap.eef[1],
                axes_length=0.1,
                axes_radius=0.004,
            )
        lines = [status] if status else []
        if snap.gripper_width is not None:
            lines.append(f"gripper width {snap.gripper_width * 100:.1f} cm")
        lines += snap.notes
        text = "  \n".join(lines) or " "
        if self._text is None:
            self._text = self.server.gui.add_markdown(text)
        else:
            self._text.content = text
        return total

    def grasps(
        self, candidates: list[dict[str, Any]], active: str | None = None
    ) -> int:
        """Replace the drawn candidates; returns how many had a pose."""
        for h in self._grasps:
            h.remove()
        self._grasps = []
        for i, c in enumerate(candidates):
            pose = candidate_pose(c)
            if pose is None:
                continue
            cid = str(c.get("id", i))
            big = cid == active
            self._grasps.append(
                self.server.scene.add_frame(
                    f"/grasps/{cid}",
                    position=pose[0],
                    wxyz=pose[1],
                    axes_length=0.08 if big else 0.04,
                    axes_radius=0.004 if big else 0.0015,
                )
            )
        return len(self._grasps)


class ViserViewFacade(RpcFacade):
    """Polls one env server into a :class:`Scene` on a background thread."""

    SERVICE_NAME = "viser"

    def __init__(
        self,
        scene: Scene,
        source: Callable[[Call], Snapshot],
        *,
        env: str | None,
        interval: float = 1.0,
        url: str = "",
        client: Callable[[str], Any] | None = None,
    ) -> None:
        super().__init__()
        self._scene = scene
        self._source = source
        self._interval = float(interval)
        self._url = url
        self._client_factory = client or _http_client
        self._env_url: str | None = None
        self._env: Any = None
        self._draw = threading.Lock()
        self.updates = 0
        self.points = 0
        self.last_error: str | None = None
        if env:
            self.attach(env)
        self._rpc.update(
            {
                "viser.status": self.status,
                "viser.refresh": self.refresh,
                "viser.grasps": self.grasps,
                "viser.attach": self.attach,
            }
        )

    def attach(self, env: str) -> dict[str, Any]:
        """Follow another env server (a new episode's)."""
        with self._draw:
            self._env_url = env
            self._env = self._client_factory(env)
        return self.status()

    def status(self) -> dict[str, Any]:
        return {
            "url": self._url,
            "env": self._env_url,
            "updates": self.updates,
            "points": self.points,
            "last_error": self.last_error,
        }

    def refresh(self) -> dict[str, Any]:
        """Read the env server once and redraw."""
        with self._draw:
            if self._env is None:
                raise RuntimeError("no env server attached")
            env = self._env
            try:
                snap = self._source(lambda method: env.call(method, timeout_s=60.0))
                self.points = self._scene.update(
                    snap, f"{self._env_url} · update {self.updates + 1}"
                )
                self.updates += 1
                self.last_error = None
            except Exception as exc:
                self.last_error = f"{type(exc).__name__}: {exc}"
                raise
        return self.status()

    def grasps(
        self, candidates: list[dict[str, Any]], active: str | None = None
    ) -> dict[str, Any]:
        """Draw a plan's candidates (``plan_grasp`` / ``plan_place`` results)."""
        with self._draw:
            n = self._scene.grasps(list(candidates or []), active)
        return {"drawn": n}

    def poll_forever(self, stop: threading.Event) -> None:
        while not stop.wait(self._interval):
            if self._env is None:
                continue
            try:
                self.refresh()
            except Exception as exc:  # an episode between envs, a busy robot
                logger.debug("refresh failed: %s", exc)


def _http_client(url: str) -> Any:
    from pi_embodied_services.utils.rpc.client_utils import make_rpc_client

    return make_rpc_client(url)


def _build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="pi-embodied Viser 3D view")
    parser.add_argument("--robot", choices=sorted(SOURCES), default="libero")
    parser.add_argument("--env", default=None, help="the env server, http://host:port")
    parser.add_argument("--viser-host", default="0.0.0.0")
    parser.add_argument("--viser-port", type=int, default=8080)
    parser.add_argument(
        "--interval", type=float, default=1.0, help="seconds between reads"
    )
    parser.add_argument(
        "--stride", type=int, default=4, help="pixel stride of the clouds"
    )
    parser.add_argument("--max-depth", type=float, default=3.0, help="metres")
    parser.add_argument("--transport", choices=["http"], default="http")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=0)
    parser.add_argument(
        "--parent-watch",
        action="store_true",
        help="watch parent process via stdin pipe and exit when it dies",
    )
    return parser


def main() -> None:
    args = _build_argparser().parse_args()
    logging.basicConfig(level=logging.INFO, format="[%(name)s] %(message)s")
    import viser

    server = viser.ViserServer(host=args.viser_host, port=args.viser_port)
    port = server.get_port()
    url = f"http://{args.viser_host}:{port}"
    # pi (src/viser.ts) reads this line for the port it links to.
    print(f"viser listening on {url}", flush=True)
    facade = ViserViewFacade(
        Scene(server, stride=args.stride, max_depth=args.max_depth),
        SOURCES[args.robot],
        env=args.env,
        interval=args.interval,
        url=url,
    )
    stop = threading.Event()
    threading.Thread(target=facade.poll_forever, args=(stop,), daemon=True).start()
    try:
        facade.serve(
            transport=args.transport,
            host=args.host,
            port=args.port,
            parent_watch=args.parent_watch,
        )
    finally:
        stop.set()
        server.stop()


if __name__ == "__main__":
    main()
