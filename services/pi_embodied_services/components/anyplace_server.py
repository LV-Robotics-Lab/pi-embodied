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

"""RPC server owning AnyPlace (ac-rad/anyplace): object placement prediction.

Run manually with::

    ANYPLACE_ROOT=/path/to/anyplace ANYPLACE_CONFIG=/path/to/multitask_eval.yaml \
    PYTHONPATH=/path/to/pi/services python -m pi_embodied_services.components.anyplace_server --port 8123

Runs in its own venv (the AnyPlace checkout with its torch/CUDA dependencies and the
multi-task checkpoint the config points at). Serves ``anyplace.plan``: from one aligned RGB-D
frame, the object mask (the Placement Object) and the placement-region mask, the model predicts
Object Placement Transforms ``p_placed = R @ p_current + t`` in the camera frame, several
candidates best first. The env server composes them with the pick grasp
(``utils/grasp.compose_placement``). The model loading follows OpenETA's
``tools/anyplace_core.py`` (the official internal policy function, no CLI scripts).
"""

from __future__ import annotations

import argparse
import contextlib
import os
import random
import sys
import threading
import time
from pathlib import Path
from typing import Any

import numpy as np

from pi_embodied_services.components.grasp_server_base import (
    grasp_argparser,
    pin_cuda,
    serve,
)
from pi_embodied_services.utils.grasp import CAMERA_FRAME, object_points
from pi_embodied_services.utils.logging import get_logger
from pi_embodied_services.utils.rpc import RpcFacade

logger = get_logger("anyplace_server")

MIN_POINTS = 1024
MAX_OBJECT_POINTS = 200_000
MAX_REGION_POINTS = 500_000


class _NoOpVisualizer:
    """Meshcat stand-in for AnyPlace helpers that always draw."""

    def __getitem__(self, _name: str) -> "_NoOpVisualizer":
        return self

    def set_object(self, *_a: Any, **_k: Any) -> None:
        return None

    def set_transform(self, *_a: Any, **_k: Any) -> None:
        return None


def validate_placements(raw: Any) -> np.ndarray:
    """The model's transforms as float64 [N, 4, 4], every one rigid."""
    arr = np.asarray(raw, dtype=np.float64)
    if arr.ndim != 3 or arr.shape[1:] != (4, 4) or not np.isfinite(arr).all():
        raise RuntimeError(
            f"AnyPlace returned no [N, 4, 4] transforms: shape {arr.shape}"
        )
    for T in arr:
        R = T[:3, :3]
        if not (
            np.allclose(T[3], [0, 0, 0, 1], atol=1e-6)
            and np.allclose(R.T @ R, np.eye(3), atol=1e-5)
            and np.isclose(np.linalg.det(R), 1, atol=1e-5)
        ):
            raise RuntimeError("AnyPlace returned a non-rigid transform")
    return arr


class AnyPlaceFacade(RpcFacade):
    SERVICE_NAME = "anyplace"

    def __init__(
        self, root: str, config: str, *, seed: int, depth_truncation: float
    ) -> None:
        super().__init__()
        self._root = Path(root).expanduser().resolve()
        self._config = Path(config).expanduser().resolve()
        self._seed = int(seed)
        self._depth_max = float(depth_truncation)
        self._lock = threading.Lock()
        self._load()
        self._rpc["anyplace.plan"] = self.plan
        self._rpc["anyplace.info"] = self.info

    def _load(self) -> None:
        if not self._root.is_dir():
            raise RuntimeError(f"AnyPlace checkout not found: {self._root}")
        if not self._config.is_file():
            raise RuntimeError(f"AnyPlace config not found: {self._config}")
        if str(self._root) not in sys.path:
            sys.path.insert(0, str(self._root))
        os.environ["ANYPLACE_SOURCE_DIR"] = str(self._root / "anyplace")
        os.environ.setdefault("ANYPLACE_DATA_DIR", str(self._root / "anyplace"))
        try:
            import torch
            from anyplace.model.transformer.policy import (
                NSMTransformerImplicit,
                NSMTransformerSingleTransformationRegression,
            )
            from anyplace.utils import config_util, util
            from anyplace.utils.anyplace.multistep_pose_regression_anyplace import (
                policy_inference_methods_dict,
            )
            from anyplace.utils.mesh_util import three_util
        except ImportError as exc:
            raise RuntimeError(
                "anyplace is not importable; install the checkout into this venv"
            ) from exc
        if not torch.cuda.is_available():
            raise RuntimeError("AnyPlace requires a CUDA-capable GPU")
        args = config_util.recursive_attr_dict(
            config_util.load_config(str(self._config), demo_train_eval="eval")
        )
        ckpt = Path(args.experiment.eval.ckpt_path)
        if not ckpt.is_absolute():
            ckpt = self._config.parent / ckpt
        if not ckpt.exists():
            raise RuntimeError(f"AnyPlace checkpoint not found: {ckpt}")
        torch.manual_seed(self._seed)
        random.seed(self._seed)
        np.random.seed(self._seed)
        with contextlib.redirect_stdout(sys.stderr):
            state = torch.load(str(ckpt), map_location="cpu", weights_only=False)
            config_util.update_recursive(
                args.model.refine_pose,
                config_util.recursive_attr_dict(state["args"]["model"]["refine_pose"]),
            )
            kind = args.model.refine_pose.type
            pr_args = config_util.copy_attr_dict(args.model[kind])
            if args.model.refine_pose.get("model_kwargs") is not None:
                config_util.update_recursive(
                    pr_args, args.model.refine_pose.model_kwargs[kind]
                )
            if kind == "nsm_transformer":
                model = NSMTransformerSingleTransformationRegression(
                    mc_vis=_NoOpVisualizer(),
                    feat_dim=args.model.refine_pose.feat_dim,
                    **pr_args,
                ).cuda()
            elif kind == "nsm_implicit":
                model = NSMTransformerImplicit(
                    mc_vis=_NoOpVisualizer(),
                    feat_dim=args.model.refine_pose.feat_dim,
                    is_train=False,
                    **pr_args,
                ).cuda()
            else:
                raise RuntimeError(f"unsupported AnyPlace refine_pose type {kind!r}")
            model.load_state_dict(state["refine_pose_model_state_dict"])
            model.eval()
            if hasattr(model, "eval_sample"):
                model.set_eval_sample(True)
        reso, pad = args.data.voxel_grid.reso_grid, args.data.voxel_grid.padding
        raster = (
            three_util.get_raster_points(reso, padding=pad)
            .reshape(reso, reso, reso, 3)
            .transpose(2, 1, 0, 3)
            .reshape(-1, 3)
        )
        rot_grid = util.generate_healpix_grid(size=args.data.rot_grid_samples)
        args.data.rot_grid_bins = rot_grid.shape[0]
        args.data.coarse_aff.scene_scale = 1.0 / float(
            np.max(np.asarray(args.data.coarse_aff.scene_extents, dtype=np.float64))
        )
        self._torch = torch
        self._args = args
        self._model = model
        self._raster = raster
        self._rot_grid = rot_grid
        self._infer = policy_inference_methods_dict[
            args.experiment.eval.inference_method
        ]
        self._checkpoint = str(ckpt)
        logger.info("AnyPlace loaded from %s", ckpt)

    def info(self) -> dict[str, Any]:
        return {
            "backend": "anyplace",
            "model": "anyplace_multitask",
            "camera_frame": CAMERA_FRAME,
            "convention": "p_placed = R @ p_current + t",
            "checkpoint": self._checkpoint,
        }

    def _predict(self, obj: np.ndarray, region: np.ndarray) -> Any:
        a = self._args
        e = a.experiment
        kwargs: dict[str, Any] = {
            "gt_child_cent": None,
            "export_viz": False,
            "export_viz_dirname": None,
            "export_viz_relative_trans_guess": None,
            "compute_coverage_scores": False,
            "out_coverage_dirname1": None,
            "out_coverage_dirname2": None,
            "iteration": 0,
        }
        if getattr(e.eval, "multi_aff_rot", False):
            kwargs["multi_aff_rot"] = True
        mesh_dict = {
            "parent_file": None,
            "parent_scale": None,
            "parent_pose": None,
            "child_file": None,
            "child_scale": None,
            "child_pose": None,
            "multi": True,
        }
        self._torch.manual_seed(self._seed)
        random.seed(self._seed)
        with self._torch.no_grad(), contextlib.redirect_stdout(sys.stderr):
            preds = self._infer(
                _NoOpVisualizer(),
                region,
                obj,
                None,
                self._model,
                None,
                scene_mean=a.data.coarse_aff.scene_mean,
                scene_scale=a.data.coarse_aff.scene_scale,
                grid_pts=self._raster,
                rot_grid=self._rot_grid,
                viz=False,
                n_iters=e.eval.n_refine_iters,
                no_parent_crop=(not e.parent_crop),
                return_top=(not e.eval.return_rand),
                with_coll=e.eval.with_coll,
                run_affordance=e.eval.run_affordance,
                init_k_val=e.eval.init_k_val,
                no_sc_score=e.eval.no_success_classifier,
                init_parent_mean=e.eval.init_parent_mean_pos,
                init_orig_ori=e.eval.init_orig_ori,
                refine_anneal=e.eval.refine_anneal,
                mesh_dict=mesh_dict,
                add_per_iter_noise=e.eval.add_per_iter_noise,
                per_iter_noise_kwargs=e.eval.per_iter_noise_kwargs,
                variable_size_crop=e.eval.variable_size_crop,
                timestep_emb_decay_factor=e.eval.timestep_emb_decay_factor,
                remove_redundant_pose=e.eval.remove_redundant_pose,
                **kwargs,
            )
            self._torch.cuda.synchronize()
        return preds

    def plan(
        self,
        rgb: Any,
        depth: Any,
        intrinsic_K: Any,
        object_mask: Any,
        region_mask: Any,
        max_candidates: int = 5,
    ) -> dict[str, Any]:
        started = time.perf_counter()
        depth_arr = np.asarray(depth, dtype=np.float32)
        K = np.asarray(intrinsic_K, dtype=np.float64)
        obj_mask = np.asarray(object_mask).astype(bool)
        reg_mask = np.asarray(region_mask).astype(bool)
        if (
            depth_arr.ndim != 2
            or obj_mask.shape != depth_arr.shape
            or reg_mask.shape != depth_arr.shape
        ):
            raise ValueError(
                "depth, object_mask and region_mask must share one [H, W] shape"
            )
        if K.shape != (3, 3):
            raise ValueError("intrinsic_K must be 3x3")
        obj, _ = object_points(depth_arr, K, obj_mask, depth_max=self._depth_max)
        region, _ = object_points(depth_arr, K, reg_mask, depth_max=self._depth_max)
        for name, pts, cap in (
            ("object", obj, MAX_OBJECT_POINTS),
            ("region", region, MAX_REGION_POINTS),
        ):
            if len(pts) < MIN_POINTS:
                raise ValueError(
                    f"the {name} mask has {len(pts)} points with depth; AnyPlace needs at least {MIN_POINTS}"
                )
            if len(pts) > cap:
                raise ValueError(
                    f"the {name} mask has {len(pts)} points; at most {cap}"
                )
        with self._lock:
            preds = validate_placements(self._predict(obj, region))
        n = max(1, int(max_candidates))
        return {
            **self.info(),
            "placements": [
                {"rank": i, "score": None, "transform_matrix": T.tolist()}
                for i, T in enumerate(preds[:n])
            ],
            "model_candidate_count": int(len(preds)),
            "latency_s": round(time.perf_counter() - started, 4),
            "metadata": {
                "object_points": int(len(obj)),
                "region_points": int(len(region)),
            },
        }


def main() -> None:
    parser: argparse.ArgumentParser = grasp_argparser(
        "pi-embodied AnyPlace server", 8123
    )
    parser.add_argument(
        "--root", default=os.environ.get("ANYPLACE_ROOT"), help="anyplace checkout"
    )
    parser.add_argument(
        "--config",
        default=os.environ.get("ANYPLACE_CONFIG"),
        help="eval config yaml (its ckpt_path names the checkpoint)",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--depth-truncation", type=float, default=2.0)
    args = parser.parse_args()
    if not args.root or not args.config:
        raise SystemExit(
            "--root and --config are required (or ANYPLACE_ROOT / ANYPLACE_CONFIG)"
        )
    pin_cuda(args)
    serve(
        AnyPlaceFacade(
            args.root,
            args.config,
            seed=args.seed,
            depth_truncation=args.depth_truncation,
        ),
        args,
    )


if __name__ == "__main__":
    main()
