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

"""RPC server owning the AnyGrasp SDK detector (graspnet-baseline's ``gsnet``).

Run manually with::

    ANYGRASP_SDK_ROOT=/path/to/anygrasp_sdk ANYGRASP_CHECKPOINT=/path/to/checkpoint_detection.tar \
    PYTHONPATH=/path/to/pi/services python -m pi_embodied_services.components.anygrasp_server --port 8122

AnyGrasp is not open source: the authors hand out a compiled ``gsnet`` module and a license
file bound to one machine's hardware id (``license/licenseCfg.json`` plus the ``*.lic`` files
under ``<sdk>/grasp_detection/license/``). Without that license the SDK refuses to load, and so
does this server, with a message that says so; there is no public workaround and this
repository ships none. The wire contract is the same as the other grasp servers'; AnyGrasp
already predicts in the GraspNet grasp frame (``utils/grasp.anygrasp_candidates``).
"""

from __future__ import annotations

import os
import sys
from argparse import Namespace
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

logger = get_logger("anygrasp_server")


def license_status(sdk_root: Path) -> tuple[bool, str]:
    """Whether the SDK's machine-bound license is present (``license/`` beside ``gsnet``)."""
    lic = sdk_root / "grasp_detection" / "license"
    if not lic.is_dir():
        return False, f"{lic} does not exist"
    files = sorted(p.name for p in lic.iterdir())
    if not any(p.endswith(".lic") for p in files) or "licenseCfg.json" not in files:
        return (
            False,
            f"{lic} has {files or 'no files'}; expected licenseCfg.json and *.lic from the AnyGrasp authors",
        )
    return True, "present"


class AnyGraspFacade(GraspServer):
    SERVICE_NAME = "anygrasp"
    MODEL_NAME = "anygrasp_sdk"

    def __init__(
        self,
        sdk_root: str,
        checkpoint: str,
        *,
        max_gripper_width: float,
        gripper_height: float,
        depth_truncation: float,
        collision_detection: bool,
    ) -> None:
        super().__init__()
        self._sdk = Path(sdk_root).expanduser().resolve()
        self._ckpt = Path(checkpoint).expanduser().resolve()
        self._max_width = float(max_gripper_width)
        self._height = float(gripper_height)
        self._depth_max = float(depth_truncation)
        self._collision = bool(collision_detection)
        self._load()

    def _load(self) -> None:
        if not self._sdk.is_dir():
            raise RuntimeError(f"AnyGrasp SDK not found: {self._sdk}")
        ok, why = license_status(self._sdk)
        if not ok:
            raise RuntimeError(
                "AnyGrasp license file missing: "
                + why
                + ". The SDK is bound to one machine's hardware "
                "id; request a license from the AnyGrasp authors (graspnet.net) for this host and put "
                "the files under <sdk>/grasp_detection/license/."
            )
        if not self._ckpt.is_file():
            raise RuntimeError(f"AnyGrasp checkpoint not found: {self._ckpt}")
        detection = self._sdk / "grasp_detection"
        if str(detection) not in sys.path:
            sys.path.insert(0, str(detection))
        try:
            from gsnet import AnyGrasp
        except ImportError as exc:
            raise RuntimeError(
                "the AnyGrasp SDK's gsnet module is not importable (needs the authors' compiled module, "
                "MinkowskiEngine 0.5.4, pointnet2 and graspnetAPI in this venv)"
            ) from exc
        detector = AnyGrasp(
            Namespace(
                checkpoint_path=str(self._ckpt),
                max_gripper_width=self._max_width,
                gripper_height=self._height,
                top_down_grasp=False,
                debug=False,
            )
        )
        detector.load_net()
        self._detector = detector
        logger.info("AnyGrasp loaded from %s", self._ckpt)

    def info(self) -> dict[str, Any]:
        return {
            **super().info(),
            "max_gripper_width_m": self._max_width,
            "gripper_height_m": self._height,
        }

    def predict(self, *, depth, K, mask, rgb, up_direction_camera, max_candidates):
        points = backproject(depth, K)
        valid = valid_points(depth, 0.0, self._depth_max)
        if not (valid & mask).any():
            raise ValueError("the mask has no pixel with depth")
        pts = np.ascontiguousarray(points[valid], dtype=np.float32)
        colors = (
            np.ascontiguousarray(rgb[valid][:, :3], dtype=np.float32) / 255.0
            if rgb is not None
            else np.zeros_like(pts)
        )
        lims = [
            float(pts[:, 0].min()),
            float(pts[:, 0].max()),
            float(pts[:, 1].min()),
            float(pts[:, 1].max()),
            0.0,
            self._depth_max,
        ]
        grasps, _cloud = self._detector.get_grasp(
            pts,
            colors,
            lims=lims,
            apply_object_mask=True,
            dense_grasp=False,
            collision_detection=self._collision,
        )
        metadata: dict[str, Any] = {
            "scene_points": int(len(pts)),
            "collision_detection": self._collision,
        }
        if grasps is None or len(grasps) == 0:
            return [], metadata
        grasps = grasps.nms().sort_by_score()
        # Targeted mode: keep the grasps whose center lies on the object's points.
        obj = points[valid & mask]
        centers = np.asarray([g.translation for g in grasps])
        d = (
            np.min(
                np.linalg.norm(centers[:, None, :] - obj[None, :, :], axis=-1), axis=1
            )
            if len(obj)
            else np.zeros(len(centers))
        )
        keep = [g for g, dist in zip(grasps, d) if dist < 0.03]
        metadata.update({"scene_grasps": int(len(grasps)), "on_object": len(keep)})
        return anygrasp_candidates(keep), metadata


def main() -> None:
    parser = grasp_argparser("pi-embodied AnyGrasp server", 8122)
    parser.add_argument(
        "--sdk-root",
        default=os.environ.get("ANYGRASP_SDK_ROOT"),
        help="anygrasp_sdk checkout (with grasp_detection/gsnet*.so and license/)",
    )
    parser.add_argument(
        "--checkpoint",
        default=os.environ.get("ANYGRASP_CHECKPOINT"),
        help="checkpoint_detection.tar",
    )
    parser.add_argument("--max-gripper-width", type=float, default=0.1)
    parser.add_argument("--gripper-height", type=float, default=0.03)
    parser.add_argument("--depth-truncation", type=float, default=2.0)
    parser.add_argument("--no-collision-detection", action="store_true")
    args = parser.parse_args()
    if not args.sdk_root or not args.checkpoint:
        raise SystemExit(
            "--sdk-root and --checkpoint are required (or ANYGRASP_SDK_ROOT / ANYGRASP_CHECKPOINT)"
        )
    pin_cuda(args)
    facade = AnyGraspFacade(
        args.sdk_root,
        args.checkpoint,
        max_gripper_width=args.max_gripper_width,
        gripper_height=args.gripper_height,
        depth_truncation=args.depth_truncation,
        collision_detection=not args.no_collision_detection,
    )
    serve(facade, args)


if __name__ == "__main__":
    main()
