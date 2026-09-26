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

"""RPC server owning NVlabs GraspGenX (cross-embodiment grasp generation).

Run manually with::

    GRASPGENX_ROOT=/path/to/GraspGenX GRASPGENX_CHECKPOINT_DIR=/path/to/graspgenx_checkpoints \
    GRASPGENX_GRIPPER_CFG_DIR=/path/to/gripper_descriptions PYTHONPATH=/path/to/pi/services \
        python -m pi_embodied_services.components.graspgenx_server --gripper franka_panda --port 8121

Runs in its own venv (the ``graspgenx`` extra: ``pip install -e <GraspGenX checkout>``; the
checkpoints and gripper descriptions come from Hugging Face and are never downloaded here).
Serves ``graspgenx.plan``: the object's points are aligned so world up is +Z (GraspGenX's
GraspMoE planner expects that), grasps come back in the gripper base frame (Z approach, X
closing) and are converted to the GraspNet grasp frame (``utils/grasp.graspgenx_candidates``);
grasps colliding with the rest of the scene are dropped with the package's own filter.
OpenETA's ``tools/graspgenx_core.py`` is the reference for the planner options.
"""

from __future__ import annotations

import contextlib
import json
import os
import re
import sys
from pathlib import Path
from typing import Any

import numpy as np

from pi_embodied_services.components.grasp_server_base import (
    GraspServer,
    grasp_argparser,
    pin_cuda,
    serve,
)
from pi_embodied_services.utils.grasp import graspgenx_candidates, object_points
from pi_embodied_services.utils.logging import get_logger

logger = get_logger("graspgenx_server")

NUM_GRASPS = 200
MOE = {
    "moe_num_yaws": 36,
    "moe_z_offsets_cm": (-2.0, 0.0),
    "moe_outlier_threshold": 0.014,
    "moe_outlier_k": 20,
    "moe_obb_mode": "advanced",
    "moe_skip_obb_rule": "auto",
    "moe_obb_density": "dense-topandside",
    "moe_obb_position_spacing_cm": 1.0,
}
COLLISION_THRESHOLD = 0.02
MAX_SCENE_POINTS = 8192
NUM_SURFACE_SAMPLES = 2000


def rotation_up_to_z(up: np.ndarray) -> np.ndarray:
    """R with ``R @ up = [0, 0, 1]`` (Rodrigues; identity when already aligned)."""
    a = np.asarray(up, dtype=np.float64)
    a = a / np.linalg.norm(a)
    b = np.array([0.0, 0.0, 1.0])
    c = float(np.clip(a @ b, -1.0, 1.0))
    if c > 1 - 1e-12:
        return np.eye(3)
    if c < -1 + 1e-12:
        return np.diag([1.0, -1.0, -1.0])
    v = np.cross(a, b)
    s = np.linalg.norm(v)
    k = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + k + k @ k * ((1 - c) / (s * s))


def checkpoint_pair(root: Path) -> tuple[Path, Path]:
    """The generator and discriminator checkpoints under ``root`` (``gen/*.pth``, ``dis/*.pth``,
    latest epoch), also when they sit one directory down (a versioned release)."""
    for base in [root, *sorted(p for p in root.iterdir() if p.is_dir())]:
        gen, dis = base / "gen", base / "dis"
        if (gen / "config.yaml").is_file() and (dis / "config.yaml").is_file():

            def pick(d: Path) -> Path:
                def epoch(p: Path) -> tuple[int, str]:
                    m = re.search(r"epoch_(\d+)", p.stem)
                    return (int(m.group(1)) if m else -1, p.name)

                return max(d.glob("*.pth"), key=epoch)

            return pick(gen), pick(dis)
    raise RuntimeError(f"no gen/config.yaml + dis/config.yaml under {root}")


def gripper_geometry(descriptions_root: Path, name: str) -> dict[str, Any]:
    """``fingertip`` and open ``width`` of a gripper from its ``config.json``."""
    cfg = (
        descriptions_root
        / "gripper_descriptions"
        / "assets"
        / "x_grippers"
        / name
        / "config.json"
    )
    if not cfg.is_file():
        cfg = descriptions_root / "assets" / "x_grippers" / name / "config.json"
    if not cfg.is_file():
        raise RuntimeError(
            f"gripper {name!r} has no config.json under {descriptions_root}"
        )
    data = json.loads(cfg.read_text())
    return {
        "fingertip_xyz": [float(v) for v in data["fingertip"]],
        "width": float(data["sweep_volume"]["extents"][0]),
        "type": str(data.get("type")),
    }


class GraspGenXFacade(GraspServer):
    SERVICE_NAME = "graspgenx"
    MODEL_NAME = "graspgenx"
    NATIVE_GRASP_FRAME = "graspgenx"

    def __init__(
        self,
        root: str,
        checkpoints: str,
        grippers: str,
        *,
        gripper: str,
        depth_truncation: float,
    ) -> None:
        super().__init__()
        self._root = Path(root).expanduser().resolve()
        self._ckpt = Path(checkpoints).expanduser().resolve()
        self._grippers = Path(grippers).expanduser().resolve()
        self._gripper = gripper
        self._depth_max = float(depth_truncation)
        self._geometry = gripper_geometry(self._grippers, gripper)
        self._load()

    def _load(self) -> None:
        if not (self._root / "graspgenx" / "__init__.py").is_file():
            raise RuntimeError(f"GraspGenX checkout not found: {self._root}")
        gen, dis = checkpoint_pair(self._ckpt)
        os.environ["GRASPGENX_CHECKPOINT_DIR"] = str(gen.parent.parent)
        os.environ["GRASPGENX_GRIPPER_CFG_DIR"] = str(self._grippers)
        if str(self._root) not in sys.path:
            sys.path.insert(0, str(self._root))
        try:
            with contextlib.redirect_stdout(sys.stderr):
                import torch
                import trimesh
                from graspgenx.grasp_server import (
                    GraspGenXSampler,
                    load_grasp_gen_model,
                )
                from graspgenx.samplers import run_planner_on_object
                from graspgenx.utils.checkpoint_io import load_model_cfg
                from graspgenx.utils.collision_filter import filter_colliding_grasps
        except ImportError as exc:
            raise RuntimeError(
                "graspgenx is not importable; pip install -e the checkout into this venv"
            ) from exc
        if not torch.cuda.is_available():
            raise RuntimeError("GraspGenX requires a CUDA-capable GPU")
        with contextlib.redirect_stdout(sys.stderr):
            cfg = load_model_cfg(str(gen.parent), str(dis.parent), gen.name, dis.name)
            model = load_grasp_gen_model(cfg)
            assets = self._grippers / "gripper_descriptions" / "assets"
            if not assets.is_dir():
                assets = self._grippers / "assets"
            sampler = GraspGenXSampler(
                cfg, self._gripper, assets_dir=str(assets), model=model
            )
            info = sampler.get_gripper_info()
            surface, _ = trimesh.sample.sample_surface(
                info.collision_mesh, NUM_SURFACE_SAMPLES
            )
        self._torch = torch
        self._sampler = sampler
        self._run = run_planner_on_object
        self._filter = filter_colliding_grasps
        self._surface = np.ascontiguousarray(surface, dtype=np.float32)
        self._checkpoints = {"generator": gen.name, "discriminator": dis.name}
        logger.info("GraspGenX loaded (%s) for gripper %s", self._ckpt, self._gripper)

    def info(self) -> dict[str, Any]:
        return {
            **super().info(),
            "gripper": self._gripper,
            **self._geometry,
            "checkpoints": self._checkpoints,
        }

    def predict(self, *, depth, K, mask, rgb, up_direction_camera, max_candidates):
        obj, scene = object_points(depth, K, mask, depth_max=self._depth_max)
        if len(obj) < 100:
            raise ValueError(
                f"only {len(obj)} object points; GraspGenX needs at least 100"
            )
        up = (
            up_direction_camera
            if up_direction_camera is not None
            else np.array([0.0, -1.0, 0.0])
        )
        R = rotation_up_to_z(up)
        aligned = np.ascontiguousarray(obj @ R.T, dtype=np.float32)
        with self._torch.inference_mode(), contextlib.redirect_stdout(sys.stderr):
            grasps, scores, tags, _obb = self._run(
                aligned,
                self._sampler,
                planner="graspmoe",
                grasp_threshold=-1.0,
                num_grasps=NUM_GRASPS,
                topk_num_grasps=-1,
                **MOE,
            )
        grasps = np.asarray(grasps, dtype=np.float64).reshape(-1, 4, 4)
        scores = np.asarray(scores, dtype=np.float64).reshape(-1)
        tags = list(tags)
        back = np.eye(4)
        back[:3, :3] = R.T
        camera_grasps = np.einsum("ij,njk->nik", back, grasps)
        metadata: dict[str, Any] = {
            "object_points": int(len(obj)),
            "scene_points": int(len(scene)),
            "generated": int(len(scores)),
            "diffusion": tags.count("diff"),
            "obb": len(tags) - tags.count("diff"),
        }
        keep = np.arange(len(scores))
        if len(scene) and len(scores):
            pts = (
                scene
                if len(scene) <= MAX_SCENE_POINTS
                else scene[
                    np.random.default_rng(0).choice(
                        len(scene), MAX_SCENE_POINTS, replace=False
                    )
                ]
            )
            with contextlib.redirect_stdout(sys.stderr):
                free = np.asarray(
                    self._filter(
                        scene_pc=np.ascontiguousarray(pts, dtype=np.float32),
                        grasp_poses=camera_grasps,
                        collision_threshold=COLLISION_THRESHOLD,
                        gripper_surface_points=self._surface,
                        batch_size=16,
                        device="cuda",
                    )
                ).astype(bool)
            keep = np.nonzero(free)[0]
            metadata.update(
                {
                    "collision_checked": int(len(free)),
                    "collision_rejected": int((~free).sum()),
                }
            )
        cands = graspgenx_candidates(
            camera_grasps[keep],
            scores[keep],
            fingertip_xyz=self._geometry["fingertip_xyz"],
            width=self._geometry["width"],
            tags=[tags[i] for i in keep],
        )
        return cands, metadata


def main() -> None:
    parser = grasp_argparser("pi-embodied GraspGenX server", 8121)
    parser.add_argument(
        "--root", default=os.environ.get("GRASPGENX_ROOT"), help="GraspGenX checkout"
    )
    parser.add_argument(
        "--checkpoints",
        default=os.environ.get("GRASPGENX_CHECKPOINT_DIR"),
        help="graspgenx_checkpoints dir (gen/, dis/)",
    )
    parser.add_argument(
        "--grippers",
        default=os.environ.get("GRASPGENX_GRIPPER_CFG_DIR"),
        help="gripper_descriptions dir",
    )
    parser.add_argument(
        "--gripper",
        default="franka_panda",
        help="gripper name from gripper_descriptions",
    )
    parser.add_argument("--depth-truncation", type=float, default=2.0)
    args = parser.parse_args()
    for name in ("root", "checkpoints", "grippers"):
        if not getattr(args, name):
            raise SystemExit(
                f"--{name} is required (or its GRASPGENX_* environment variable)"
            )
    pin_cuda(args)
    facade = GraspGenXFacade(
        args.root,
        args.checkpoints,
        args.grippers,
        gripper=args.gripper,
        depth_truncation=args.depth_truncation,
    )
    serve(facade, args)


if __name__ == "__main__":
    main()
