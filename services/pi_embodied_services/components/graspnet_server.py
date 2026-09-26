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

"""RPC server owning a Contact-GraspNet (PyTorch port) grasp predictor.

Run manually with::

    CONTACT_GRASPNET_ROOT=/path/to/contact_graspnet_pytorch PYTHONPATH=/path/to/pi/services \
        python -m pi_embodied_services.components.graspnet_server --port 8120

Runs in its own venv (the ``graspnet`` extra: the elchun/contact_graspnet_pytorch checkout
installed with ``pip install -e``, a CUDA torch; the checkpoint ships in the checkout under
``checkpoints/contact_graspnet``). Serves ``contact_graspnet.plan``: the model predicts in its
Panda base frame (Z approach, X closing, origin at the gripper base); the answer is in the
GraspNet grasp frame (``utils/grasp.contact_graspnet_candidates``). CaP-X served the same
model through FastAPI (``capx/serving/launch_contact_graspnet_server.py``); the retry over
random viewpoints it did when no grasp came back is kept as ``--viewpoint-retries``.
"""

from __future__ import annotations

import contextlib
import os
import random
import sys
from pathlib import Path

import numpy as np

from pi_embodied_services.components.grasp_server_base import (
    GraspServer,
    grasp_argparser,
    pin_cuda,
    serve,
)
from pi_embodied_services.utils.grasp import (
    contact_graspnet_candidates,
    object_points,
    rigid,
)
from pi_embodied_services.utils.logging import get_logger

logger = get_logger("graspnet_server")

TARGET_SEGMENT = 1


def _random_viewpoint(
    center: np.ndarray, rng: np.random.Generator, extent: float = 0.25
) -> np.ndarray:
    """A camera pose (4x4, camera->scene) looking at ``center`` from a jittered position above it."""
    offset = rng.uniform(-extent, extent, size=3)
    offset[2] = -abs(offset[2]) - 0.3
    position = center + offset
    z = center - position
    z /= np.linalg.norm(z)
    x = np.cross(np.array([0.0, 1.0, 0.0]), z)
    if np.linalg.norm(x) < 1e-6:
        x = np.array([1.0, 0.0, 0.0])
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    return rigid(np.stack([x, y, z], axis=1), position)


class ContactGraspNetFacade(GraspServer):
    """``contact_graspnet.plan`` over an external contact_graspnet_pytorch checkout."""

    SERVICE_NAME = "contact_graspnet"
    MODEL_NAME = "contact_graspnet_pytorch"
    NATIVE_GRASP_FRAME = "contact_graspnet"

    def __init__(
        self,
        root: str,
        checkpoint_dir: str | None,
        *,
        depth_min: float,
        depth_max: float,
        forward_passes: int,
        viewpoint_retries: int,
        seed: int,
    ) -> None:
        super().__init__()
        self._root = Path(root).expanduser().resolve()
        self._ckpt = Path(
            checkpoint_dir or self._root / "checkpoints" / "contact_graspnet"
        ).resolve()
        self._depth_min = float(depth_min)
        self._depth_max = float(depth_max)
        self._passes = int(forward_passes)
        self._retries = int(viewpoint_retries)
        self._seed = int(seed)
        self._load()

    def _load(self) -> None:
        if not self._root.is_dir():
            raise RuntimeError(f"Contact-GraspNet checkout not found: {self._root}")
        if not (self._ckpt / "config.yaml").is_file():
            raise RuntimeError(
                f"Contact-GraspNet checkpoint dir has no config.yaml: {self._ckpt}"
            )
        for p in (self._root, self._root / "Pointnet_Pointnet2_pytorch"):
            if p.is_dir() and str(p) not in sys.path:
                sys.path.insert(0, str(p))
        try:
            import torch
            from contact_graspnet_pytorch import config_utils
            from contact_graspnet_pytorch.checkpoints import CheckpointIO
            from contact_graspnet_pytorch.contact_grasp_estimator import GraspEstimator
        except ImportError as exc:
            raise RuntimeError(
                "contact_graspnet_pytorch is not importable; install the checkout into this venv "
                "(pip install -e <root>) with a CUDA torch"
            ) from exc
        if not torch.cuda.is_available():
            raise RuntimeError("Contact-GraspNet requires a CUDA-capable GPU")
        config = config_utils.load_config(str(self._ckpt), batch_size=1)
        # The checkpoint holds numpy scalars; torch 2.6+ loads weights-only by default.
        numpy_core = getattr(np, "_core", None) or np.core
        torch.serialization.add_safe_globals(
            [numpy_core.multiarray.scalar, np.dtype, type(np.dtype(np.float64))]
        )
        with contextlib.redirect_stdout(sys.stderr):
            estimator = GraspEstimator(config)
            io = CheckpointIO(
                checkpoint_dir=str(self._ckpt / "checkpoints"), model=estimator.model
            )
            io.load("model.pt")
            estimator.model.eval()
        self._torch = torch
        self._estimator = estimator
        self._gripper_width = float(config["DATA"]["gripper_width"])
        logger.info("Contact-GraspNet loaded from %s", self._ckpt)

    def _predict_raw(self, scene: np.ndarray, obj: np.ndarray):
        with self._torch.inference_mode(), contextlib.redirect_stdout(sys.stderr):
            grasps, scores, contacts, openings = self._estimator.predict_scene_grasps(
                scene,
                pc_segments={TARGET_SEGMENT: obj.copy()},
                local_regions=True,
                filter_grasps=True,
                forward_passes=self._passes,
            )
        return (
            np.asarray(
                grasps.get(TARGET_SEGMENT, np.zeros((0, 4, 4))), dtype=np.float64
            ),
            np.asarray(scores.get(TARGET_SEGMENT, []), dtype=np.float64),
            np.asarray(contacts.get(TARGET_SEGMENT, []), dtype=np.float64),
            np.asarray(openings.get(TARGET_SEGMENT, []), dtype=np.float64),
        )

    def predict(self, *, depth, K, mask, rgb, up_direction_camera, max_candidates):
        random.seed(self._seed)
        np.random.seed(self._seed)
        self._torch.manual_seed(self._seed)
        obj, scene = object_points(depth, K, mask, depth_max=self._depth_max)
        near = depth[mask]
        near = near[np.isfinite(near) & (near > 0)]
        if near.size and float(near.min()) < self._depth_min:
            raise ValueError(
                f"object points start at {float(near.min()):.3f} m, below --depth-min {self._depth_min}"
            )
        scene = np.concatenate([scene, obj], axis=0)
        scene = scene[scene[:, 2] > self._depth_min]
        if len(obj) < 50:
            raise ValueError(
                f"only {len(obj)} object points; the mask or depth is too sparse"
            )
        grasps, scores, contacts, openings = self._predict_raw(scene, obj)
        retries = 0
        rng = np.random.default_rng(self._seed)
        # No grasp from this viewpoint: re-express the cloud from a jittered virtual camera
        # (the model was trained on camera-facing clouds) and map the grasps back.
        while len(grasps) == 0 and retries < self._retries:
            T_cam_virtual = _random_viewpoint(obj.mean(axis=0), rng)
            T_virtual_cam = np.linalg.inv(T_cam_virtual)

            def to_v(p: np.ndarray) -> np.ndarray:
                return (p @ T_virtual_cam[:3, :3].T + T_virtual_cam[:3, 3]).astype(
                    np.float32
                )

            g, s, c, o = self._predict_raw(to_v(scene), to_v(obj))
            if len(g):
                grasps = np.einsum("ij,njk->nik", T_cam_virtual, g)
                contacts = c @ T_cam_virtual[:3, :3].T + T_cam_virtual[:3, 3]
                scores, openings = s, o
            retries += 1
        metadata = {
            "object_points": int(len(obj)),
            "scene_points": int(len(scene)),
            "viewpoint_retries": retries,
            "forward_passes": self._passes,
            "max_gripper_width": self._gripper_width,
            "depth_range_m": [self._depth_min, self._depth_max],
        }
        if len(grasps) == 0:
            return [], metadata
        return contact_graspnet_candidates(grasps, scores, contacts, openings), metadata


def main() -> None:
    parser = grasp_argparser("pi-embodied Contact-GraspNet server", 8120)
    parser.add_argument(
        "--root",
        default=os.environ.get("CONTACT_GRASPNET_ROOT"),
        help="contact_graspnet_pytorch checkout",
    )
    parser.add_argument(
        "--checkpoint-dir",
        default=None,
        help="dir with config.yaml and checkpoints/model.pt (default <root>/checkpoints/contact_graspnet)",
    )
    parser.add_argument("--depth-min", type=float, default=0.1)
    parser.add_argument("--depth-max", type=float, default=1.8)
    parser.add_argument("--forward-passes", type=int, default=1)
    parser.add_argument("--viewpoint-retries", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    if not args.root:
        raise SystemExit("--root (or CONTACT_GRASPNET_ROOT) is required")
    pin_cuda(args)
    facade = ContactGraspNetFacade(
        args.root,
        args.checkpoint_dir,
        depth_min=args.depth_min,
        depth_max=args.depth_max,
        forward_passes=args.forward_passes,
        viewpoint_retries=args.viewpoint_retries,
        seed=args.seed,
    )
    serve(facade, args)


if __name__ == "__main__":
    main()
