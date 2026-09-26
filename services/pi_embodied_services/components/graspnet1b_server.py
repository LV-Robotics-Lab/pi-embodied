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

"""RPC server owning a GraspNet-1Billion grasp detector: graspnet-baseline or GSNet.

Run manually with::

    PYTHONPATH=/path/to/pi/services python -m pi_embodied_services.components.graspnet1b_server \
        --model baseline --root /path/to/graspnet-baseline --checkpoint /path/to/checkpoint-rs.tar
    PYTHONPATH=/path/to/pi/services python -m pi_embodied_services.components.graspnet1b_server \
        --model gsnet --root /path/to/graspness_implementation --checkpoint /path/to/checkp_realsense.tar

``--model baseline`` is graspnet-baseline (Fang et al., CVPR 2020; ``checkpoint-rs.tar`` was
trained on the RealSense split, ``checkpoint-kn.tar`` on the Kinect one); ``--model gsnet`` is
GSNet, "Graspness Discovery in Clutters" (Wang et al., ICCV 2021, rhett-chen's
graspness_implementation). Both run in the ``graspnet1b`` venv (``services/setup.sh
graspnet1b``: the checkouts, their compiled pointnet2 / knn CUDA ops and, for GSNet,
MinkowskiEngine). Serves ``graspnet1b.plan``. Both models predict graspnetAPI ``GraspGroup``
rows (score, width, height, depth, rotation, translation), which are already the GraspNet
grasp frame, so the answer goes through the same conversion as AnyGrasp's
(``utils/grasp.anygrasp_candidates``).

Per request the scene is the valid depth inside a box around the object (``--workspace-margin``),
sampled to the model's point count; the model-free collision check of both repositories removes
grasps whose fingers intersect the scene, a greedy NMS removes duplicates, and only grasps
centered on the object's points are kept.
"""

from __future__ import annotations

import contextlib
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from pi_embodied_services.components.grasp_server_base import (
    GraspServer,
    grasp_argparser,
    pin_cuda,
    serve,
)
from pi_embodied_services.utils.grasp import (
    anygrasp_candidates,
    backproject,
    valid_points,
)
from pi_embodied_services.utils.logging import get_logger

logger = get_logger("graspnet1b_server")

MODELS = ("baseline", "gsnet")
#: Points each model was trained on (graspnet-baseline demo.py; graspness infer_vis_grasp.py).
NUM_POINTS = {"baseline": 20000, "gsnet": 15000}
#: GSNet's sparse-convolution voxel (graspness_implementation's default).
GSNET_VOXEL_M = 0.005
#: A grasp is kept for the object when its center lies this close to an object point.
ON_OBJECT_M = 0.03
#: graspnetAPI GraspGroup row: score, width, height, depth, rotation (9, row-major), translation (3), object id.
GRASP_ROW = 17


@dataclass(frozen=True)
class Grasp:
    """One ``GraspGroup`` row with the attributes ``anygrasp_candidates`` reads."""

    score: float
    width: float
    height: float
    depth: float
    rotation_matrix: np.ndarray
    translation: np.ndarray


def grasps_from_array(rows: np.ndarray) -> list[Grasp]:
    rows = np.asarray(rows, dtype=np.float64).reshape(-1, GRASP_ROW)
    return [
        Grasp(
            score=float(r[0]),
            width=float(r[1]),
            height=float(r[2]),
            depth=float(r[3]),
            rotation_matrix=r[4:13].reshape(3, 3),
            translation=r[13:16].copy(),
        )
        for r in rows
    ]


def grasp_nms(
    rows: np.ndarray,
    translation_thresh: float = 0.03,
    rotation_thresh: float = np.deg2rad(30.0),
) -> np.ndarray:
    """graspnetAPI's ``GraspGroup.nms``: best first, drop a grasp that is within both the
    translation and the rotation threshold of one already kept. Returns the kept rows, best first."""
    rows = np.asarray(rows, dtype=np.float64).reshape(-1, GRASP_ROW)
    order = np.argsort(-rows[:, 0], kind="stable")
    rows = rows[order]
    t = rows[:, 13:16]
    R = rows[:, 4:13].reshape(-1, 3, 3)
    kept: list[int] = []
    for i in range(len(rows)):
        if kept:
            k = np.asarray(kept)
            near = np.linalg.norm(t[k] - t[i], axis=1) < translation_thresh
            # angle between rotations: arccos((trace(R_k^T R_i) - 1) / 2)
            tr = np.einsum("nji,ji->n", R[k], R[i])
            ang = np.arccos(np.clip((tr - 1.0) / 2.0, -1.0, 1.0))
            if np.any(near & (ang < rotation_thresh)):
                continue
        kept.append(i)
    return rows[kept]


def voxel_down_sample(points: np.ndarray, voxel: float) -> np.ndarray:
    """Open3D's ``voxel_down_sample`` (the repositories' collision check uses it; the server's venv
    has open3d), else the centroid of the points in each occupied voxel of a grid anchored at
    the minimum (the same up to Open3D's grid anchoring)."""
    points = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    if not len(points):
        return points
    try:
        import open3d as o3d
    except ImportError:
        pass
    else:
        cloud = o3d.geometry.PointCloud()
        cloud.points = o3d.utility.Vector3dVector(points)
        return np.asarray(cloud.voxel_down_sample(voxel).points)
    keys = np.floor((points - points.min(axis=0)) / voxel).astype(np.int64)
    _, inverse, counts = np.unique(
        keys, axis=0, return_inverse=True, return_counts=True
    )
    inverse = inverse.reshape(-1)
    sums = np.zeros((len(counts), 3))
    np.add.at(sums, inverse, points)
    return sums / counts[:, None]


def collision_mask(
    rows: np.ndarray,
    scene: np.ndarray,
    *,
    voxel: float = 0.01,
    approach_dist: float = 0.05,
    collision_thresh: float = 0.01,
    chunk: int = 64,
) -> np.ndarray:
    """``ModelFreeCollisionDetector.detect`` of graspnet-baseline / graspness (True = collides):
    the fraction of the two fingers', the palm's and the approach corridor's volume that scene
    points (voxel-downsampled) occupy exceeds ``collision_thresh``."""
    finger_width, finger_length = 0.01, 0.06
    rows = np.asarray(rows, dtype=np.float64).reshape(-1, GRASP_ROW)
    pts = voxel_down_sample(scene, voxel)
    approach_dist = max(approach_dist, finger_width)
    out = np.zeros(len(rows), dtype=bool)
    for s in range(0, len(rows), chunk):
        g = rows[s : s + chunk]
        T = g[:, 13:16]
        R = g[:, 4:13].reshape(-1, 3, 3)
        heights, depths, widths = g[:, 2:3], g[:, 3:4], g[:, 1:2]
        targets = np.matmul(pts[None, :, :] - T[:, None, :], R)
        x, y, z = targets[..., 0], targets[..., 1], targets[..., 2]
        m1 = (z > -heights / 2) & (z < heights / 2)
        m2 = (x > depths - finger_length) & (x < depths)
        m3 = y > -(widths / 2 + finger_width)
        m4 = y < -widths / 2
        m5 = y < (widths / 2 + finger_width)
        m6 = y > widths / 2
        m7 = (x <= depths - finger_length) & (x > depths - finger_length - finger_width)
        m8 = (x <= depths - finger_length - finger_width) & (
            x > depths - finger_length - finger_width - approach_dist
        )
        hit = (
            (m1 & m2 & m3 & m4)
            | (m1 & m2 & m5 & m6)
            | (m1 & m3 & m5 & m7)
            | (m1 & m3 & m5 & m8)
        )
        h, w = heights[:, 0], widths[:, 0]
        volume = (
            2 * h * finger_length * finger_width
            + h * (w + 2 * finger_width) * finger_width
            + h * (w + 2 * finger_width) * approach_dist
        ) / voxel**3
        out[s : s + chunk] = hit.sum(axis=1) / (volume + 1e-6) > collision_thresh
    return out


def workspace(
    depth: np.ndarray,
    K: np.ndarray,
    mask: np.ndarray,
    *,
    depth_max: float,
    margin: float,
) -> tuple[np.ndarray, np.ndarray]:
    """(scene points inside the object's box grown by ``margin``, object points), float32."""
    points = backproject(depth, K)
    valid = valid_points(depth, 0.0, depth_max)
    obj = points[valid & mask]
    if not len(obj):
        raise ValueError("the mask has no pixel with depth")
    lo, hi = obj.min(axis=0) - margin, obj.max(axis=0) + margin
    scene = points[valid]
    inside = np.all((scene >= lo) & (scene <= hi), axis=1)
    return scene[inside].astype(np.float32), obj.astype(np.float32)


def sample(points: np.ndarray, n: int, rng: np.random.Generator) -> np.ndarray:
    """``n`` points as the demos sample them: without replacement, or all plus repeats."""
    if len(points) >= n:
        return points[rng.choice(len(points), n, replace=False)]
    extra = rng.choice(len(points), n - len(points), replace=True)
    return np.concatenate([points, points[extra]], axis=0)


class GraspNet1BFacade(GraspServer):
    """``graspnet1b.plan`` over a graspnet-baseline or graspness_implementation checkout."""

    SERVICE_NAME = "graspnet1b"
    MODEL_NAME = "graspnet1b"

    def __init__(
        self,
        model: str,
        root: str,
        checkpoint: str,
        *,
        depth_max: float = 1.5,
        workspace_margin: float = 0.15,
        collision_thresh: float = 0.01,
        seed: int = 0,
        load: bool = True,
    ) -> None:
        super().__init__()
        if model not in MODELS:
            raise ValueError(f"--model must be one of {MODELS}, got {model!r}")
        self._model = model
        self._root = Path(root).expanduser().resolve()
        self._ckpt = Path(checkpoint).expanduser().resolve()
        self._depth_max = float(depth_max)
        self._margin = float(workspace_margin)
        self._collision = float(collision_thresh)
        self._seed = int(seed)
        self.MODEL_NAME = {"baseline": "graspnet-baseline", "gsnet": "gsnet"}[model]
        if load:
            self._load()

    def _load(self) -> None:
        if not self._root.is_dir():
            raise RuntimeError(f"{self._model} checkout not found: {self._root}")
        if not self._ckpt.is_file():
            raise RuntimeError(f"{self._model} checkpoint not found: {self._ckpt}")
        # Each repository imports its own modules by bare name (models/, utils/, pointnet2/, knn/).
        for sub in ("", "models", "utils", "pointnet2", "knn", "dataset"):
            p = str(self._root / sub) if sub else str(self._root)
            if p not in sys.path:
                sys.path.insert(0, p)
        try:
            import torch

            if self._model == "baseline":
                from graspnet import GraspNet, pred_decode

                net = GraspNet(
                    input_feature_dim=0,
                    num_view=300,
                    num_angle=12,
                    num_depth=4,
                    cylinder_radius=0.05,
                    hmin=-0.02,
                    hmax_list=[0.01, 0.02, 0.03, 0.04],
                    is_training=False,
                )
            else:
                from dataset.graspnet_dataset import minkowski_collate_fn
                from models.graspnet import GraspNet, pred_decode

                net = GraspNet(seed_feat_dim=512, is_training=False)
                self._collate = minkowski_collate_fn
        except ImportError as exc:
            raise RuntimeError(
                f"{self._model} is not importable from {self._root}: {exc}. Install it with "
                "services/setup.sh graspnet1b (the compiled pointnet2 / knn ops"
                + (", MinkowskiEngine" if self._model == "gsnet" else "")
                + ")"
            ) from exc
        if not torch.cuda.is_available():
            raise RuntimeError(f"{self._model} requires a CUDA-capable GPU")
        self._torch = torch
        self._device = torch.device("cuda:0")
        ckpt = torch.load(self._ckpt, map_location="cpu", weights_only=True)
        net.load_state_dict(ckpt["model_state_dict"])
        net.to(self._device).eval()
        self._net = net
        self._decode = pred_decode
        self._epoch = int(ckpt.get("epoch", -1))
        logger.info(
            "%s loaded from %s (epoch %d)", self.MODEL_NAME, self._ckpt, self._epoch
        )

    def info(self) -> dict[str, Any]:
        return {
            **super().info(),
            "variant": self._model,
            "checkpoint": self._ckpt.name,
        }

    def _forward(self, cloud: np.ndarray) -> np.ndarray:
        """GraspGroup rows [M, 17] for one sampled cloud [N, 3] (camera frame, metres)."""
        torch = self._torch
        torch.manual_seed(self._seed)
        with torch.inference_mode(), contextlib.redirect_stdout(sys.stderr):
            if self._model == "baseline":
                end_points = {
                    "point_clouds": torch.from_numpy(cloud[None]).to(self._device)
                }
            else:
                batch = self._collate(
                    [
                        {
                            "point_clouds": cloud,
                            "coors": cloud / GSNET_VOXEL_M,
                            "feats": np.ones_like(cloud),
                        }
                    ]
                )
                end_points = {
                    k: (
                        v.to(self._device)
                        if hasattr(v, "to")
                        else [[t.to(self._device) for t in b] for b in v]
                    )
                    for k, v in batch.items()
                }
            preds = self._decode(self._net(end_points))
        return preds[0].detach().float().cpu().numpy()

    def predict(self, *, depth, K, mask, rgb, up_direction_camera, max_candidates):
        rng = np.random.default_rng(self._seed)
        scene, obj = workspace(
            depth, K, mask, depth_max=self._depth_max, margin=self._margin
        )
        if len(obj) < 50:
            raise ValueError(
                f"only {len(obj)} object points; the mask or depth is too sparse"
            )
        cloud = sample(scene, NUM_POINTS[self._model], rng).astype(np.float32)
        rows = self._forward(cloud)
        metadata: dict[str, Any] = {
            "variant": self._model,
            "scene_points": int(len(scene)),
            "object_points": int(len(obj)),
            "model_grasps": int(len(rows)),
        }
        if len(rows) and self._collision > 0:
            hit = collision_mask(rows, scene, collision_thresh=self._collision)
            rows = rows[~hit]
            metadata["collision_free"] = int(len(rows))
        rows = grasp_nms(rows)
        metadata["after_nms"] = int(len(rows))
        if not len(rows):
            return [], metadata
        # Targeted: keep the grasps centered on the object's points.
        d = np.min(
            np.linalg.norm(rows[:, None, 13:16] - obj[None, :, :], axis=-1), axis=1
        )
        metadata["nearest_to_object_m"] = [round(float(v), 4) for v in np.sort(d)[:5]]
        rows = rows[d < ON_OBJECT_M]
        metadata["on_object"] = int(len(rows))
        return (
            anygrasp_candidates(grasps_from_array(rows), source_model="graspnet1b"),
            metadata,
        )


def main() -> None:
    parser = grasp_argparser("pi-embodied GraspNet-1Billion server", 8124)
    parser.add_argument("--model", choices=MODELS, required=True)
    parser.add_argument(
        "--root",
        default=None,
        help="the model's checkout (default $GRASPNET_BASELINE_ROOT / $GRASPNESS_ROOT)",
    )
    parser.add_argument(
        "--checkpoint",
        default=None,
        help="baseline: checkpoint-rs.tar (RealSense) or checkpoint-kn.tar (Kinect); gsnet: the "
        "RealSense or Kinect checkpoint (default $GRASPNET1B_CHECKPOINT)",
    )
    parser.add_argument("--depth-max", type=float, default=1.5)
    parser.add_argument(
        "--workspace-margin",
        type=float,
        default=0.15,
        help="scene points within this many metres of the object's box go to the model",
    )
    parser.add_argument(
        "--collision-thresh",
        type=float,
        default=0.01,
        help="model-free collision check threshold (0 = off)",
    )
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    env_root = {"baseline": "GRASPNET_BASELINE_ROOT", "gsnet": "GRASPNESS_ROOT"}
    root = args.root or os.environ.get(env_root[args.model])
    checkpoint = args.checkpoint or os.environ.get("GRASPNET1B_CHECKPOINT")
    if not root or not checkpoint:
        raise SystemExit(
            f"--root (or {env_root[args.model]}) and --checkpoint (or GRASPNET1B_CHECKPOINT) are required"
        )
    pin_cuda(args)
    facade = GraspNet1BFacade(
        args.model,
        root,
        checkpoint,
        depth_max=args.depth_max,
        workspace_margin=args.workspace_margin,
        collision_thresh=args.collision_thresh,
        seed=args.seed,
    )
    serve(facade, args)


if __name__ == "__main__":
    main()
