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

"""What the grasp model servers share: the ``<name>.plan`` RPC and its argument checks.

Every grasp server (Contact-GraspNet, GraspGenX, AnyGrasp) takes the same request from the
env server's ``GraspPlanner`` (``utils/grasp.py``): ``depth`` float32 [H, W] metres (0 = no
depth), ``intrinsic_K`` 3x3, ``mask`` uint8 [H, W] (nonzero = the object), optional ``rgb``
uint8 [H, W, 3], ``max_candidates`` and ``up_direction_camera`` (world up in the camera
frame). It answers ``{"backend", "model", "native_grasp_frame", "grasp_frame": "graspnet",
"camera_frame": "opencv", "candidates": [...], "latency_s", "metadata"}`` with every
candidate already converted to the GraspNet grasp frame (``utils/grasp.make_candidate``).
The subclass implements :meth:`GraspServer.predict` in the model's own terms.
"""

from __future__ import annotations

import argparse
import logging
import os
import threading
import time
from typing import Any

import numpy as np

from pi_embodied_services.utils.grasp import CAMERA_FRAME, GRASP_FRAME, rank
from pi_embodied_services.utils.rpc import RpcFacade


class GraspServer(RpcFacade):
    """Base of the grasp model servers: registers ``<SERVICE_NAME>.plan`` and ``.info``."""

    SERVICE_NAME = "grasp"
    MODEL_NAME = "unknown"
    NATIVE_GRASP_FRAME = GRASP_FRAME

    def __init__(self) -> None:
        super().__init__()
        self._lock = threading.Lock()
        self._rpc[f"{self.SERVICE_NAME}.plan"] = self.plan
        self._rpc[f"{self.SERVICE_NAME}.info"] = self.info

    def info(self) -> dict[str, Any]:
        return {
            "backend": self.SERVICE_NAME,
            "model": self.MODEL_NAME,
            "native_grasp_frame": self.NATIVE_GRASP_FRAME,
            "grasp_frame": GRASP_FRAME,
            "camera_frame": CAMERA_FRAME,
        }

    def predict(
        self,
        *,
        depth: np.ndarray,
        K: np.ndarray,
        mask: np.ndarray,
        rgb: np.ndarray | None,
        up_direction_camera: np.ndarray | None,
        max_candidates: int,
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        """(normalized candidates, metadata) for one masked RGB-D frame. Subclasses override."""
        raise NotImplementedError

    def plan(
        self,
        depth: Any,
        intrinsic_K: Any,
        mask: Any,
        rgb: Any | None = None,
        max_candidates: int = 10,
        up_direction_camera: Any | None = None,
    ) -> dict[str, Any]:
        started = time.perf_counter()
        depth_arr = np.asarray(depth, dtype=np.float32)
        if depth_arr.ndim != 2:
            raise ValueError(f"depth must be [H, W], got shape {depth_arr.shape}")
        K = np.asarray(intrinsic_K, dtype=np.float64)
        if (
            K.shape != (3, 3)
            or not np.isfinite(K).all()
            or K[0, 0] <= 0
            or K[1, 1] <= 0
        ):
            raise ValueError("intrinsic_K must be a finite 3x3 camera matrix")
        mask_arr = np.asarray(mask).astype(bool)
        if mask_arr.shape != depth_arr.shape:
            raise ValueError(
                f"mask {mask_arr.shape} does not match depth {depth_arr.shape}"
            )
        if not mask_arr.any():
            raise ValueError("mask selects no pixel")
        rgb_arr = None
        if rgb is not None:
            rgb_arr = np.asarray(rgb, dtype=np.uint8)
            if rgb_arr.shape[:2] != depth_arr.shape or rgb_arr.ndim != 3:
                raise ValueError(
                    f"rgb {rgb_arr.shape} does not match depth {depth_arr.shape}"
                )
        up = None
        if up_direction_camera is not None:
            up = np.asarray(up_direction_camera, dtype=np.float64).reshape(3)
            if not np.isfinite(up).all() or np.linalg.norm(up) < 1e-9:
                raise ValueError("up_direction_camera must be a finite nonzero vector")
            up = up / np.linalg.norm(up)
        n = int(max_candidates)
        if n <= 0:
            raise ValueError("max_candidates must be positive")
        with self._lock:
            candidates, metadata = self.predict(
                depth=depth_arr,
                K=K,
                mask=mask_arr,
                rgb=rgb_arr,
                up_direction_camera=up,
                max_candidates=n,
            )
        return {
            **self.info(),
            "candidates": rank(candidates, n),
            "model_candidate_count": len(candidates),
            "latency_s": round(time.perf_counter() - started, 4),
            "metadata": metadata,
        }


def grasp_argparser(description: str, default_port: int) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--transport", choices=["http"], default="http")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=default_port)
    parser.add_argument(
        "--cuda-device",
        type=int,
        default=None,
        help="GPU device exposed through CUDA_VISIBLE_DEVICES.",
    )
    parser.add_argument(
        "--parent-watch",
        action="store_true",
        help="watch parent process via stdin pipe and exit when it dies",
    )
    return parser


def serve(facade: RpcFacade, args: argparse.Namespace) -> None:
    logging.basicConfig(level=logging.INFO, format="[%(name)s] %(message)s")
    facade.serve(
        transport=args.transport,
        host=args.host,
        port=args.port,
        parent_watch=args.parent_watch,
    )


def pin_cuda(args: argparse.Namespace) -> None:
    """Apply ``--cuda-device`` before torch is imported."""
    if args.cuda_device is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.cuda_device)


__all__ = ["GraspServer", "grasp_argparser", "pin_cuda", "serve"]
