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
# The solver follows OpenETA real/calibration/eye_to_hand.py (cv2.calibrateHandEye in
# both configurations, findChessboardCornersSB detection, two-pass outlier rejection on
# reprojection and board-pose consistency). Modified by pi-embodied: the result is
# written to a NEW easy_handeye-layout YAML bound to an arm id and a camera, and
# replaces the robot's calibration only through a separate ``apply`` step a human
# runs; scipy rotations; a sample directory the ``capture`` step writes.

"""Hand-eye calibration for a UR5e (or Franka) camera, in three separate steps.

1. ``capture``: the human jogs the arm (freedrive / pendant) to 15-30 poses that
   keep the checkerboard in view; each Enter records the TCP pose and a frame under
   ``--samples``. The tool never moves the arm.
2. ``solve``: detects the board, solves ``cv2.calibrateHandEye`` (eye-to-hand for a
   fixed camera -> ``T_base_cam``; eye-in-hand for a wrist camera -> ``T_tcp_cam``),
   rejects outliers and re-solves, prints the residuals and writes
   ``<calibration>.new.yaml`` next to the camera's calibration file named in the
   robot config. The file names the arm (``--arm-id`` / ``calibration.arm_id``) and
   the camera; the env server refuses a calibration made on another arm.
3. ``apply``: shows the new file's residuals and the change from the current one,
   then (only with ``--yes``) backs the old file up and moves the new one in place.

The YAML has the easy_handeye layout (``parameters`` + ``transformation`` x/y/z +
qx/qy/qz/qw) the Franka robot already reads, plus ``parameters.arm_id``,
``camera``, ``camera_serial``, ``mode``, ``residuals`` and ``samples``.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from scipy.spatial.transform import Rotation

EYE_TO_HAND = "eye_to_hand"
EYE_IN_HAND = "eye_in_hand"
MODES = (EYE_TO_HAND, EYE_IN_HAND)
METHODS = ("TSAI", "PARK", "HORAUD", "ANDREFF", "DANIILIDIS")


# -- transforms -------------------------------------------------------------


def transform(R: np.ndarray, t: np.ndarray) -> np.ndarray:
    T = np.eye(4)
    T[:3, :3] = np.asarray(R, dtype=np.float64)
    T[:3, 3] = np.asarray(t, dtype=np.float64).reshape(3)
    return T


def invert(T: np.ndarray) -> np.ndarray:
    R, t = T[:3, :3], T[:3, 3]
    return transform(R.T, -R.T @ t)


def pose_matrix(pose: Any) -> np.ndarray:
    """4x4 of ``[x, y, z, qx, qy, qz, qw]`` (7) or UR ``[x, y, z, rx, ry, rz]`` (6)."""
    p = np.asarray(pose, dtype=np.float64).reshape(-1)
    if p.shape == (7,):
        return transform(Rotation.from_quat(p[3:]).as_matrix(), p[:3])
    if p.shape == (6,):
        return transform(Rotation.from_rotvec(p[3:]).as_matrix(), p[:3])
    raise ValueError("a pose is 7 numbers (xyz + xyzw) or 6 (xyz + rotation vector)")


def mean_transform(Ts: list[np.ndarray]) -> np.ndarray:
    if not Ts:
        raise ValueError("no transforms to average")
    R = Rotation.from_matrix(np.stack([T[:3, :3] for T in Ts])).mean().as_matrix()
    return transform(R, np.mean([T[:3, 3] for T in Ts], axis=0))


def rotation_deg(Ra: np.ndarray, Rb: np.ndarray) -> float:
    return math.degrees(
        (Rotation.from_matrix(Ra).inv() * Rotation.from_matrix(Rb)).magnitude()
    )


def summarize(values: list[float]) -> dict[str, float | None]:
    if not values:
        return {"mean": None, "median": None, "max": None}
    a = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(a.mean()),
        "median": float(np.median(a)),
        "max": float(a.max()),
    }


# -- samples ------------------------------------------------------------------


@dataclass
class Sample:
    """One (robot pose, board observation) pair."""

    id: str
    T_base_tcp: np.ndarray
    T_cam_board: np.ndarray | None = None
    reprojection_px: float | None = None
    reject: str | None = None

    @property
    def valid(self) -> bool:
        return self.reject is None and self.T_cam_board is not None


@dataclass(frozen=True)
class Board:
    """A checkerboard: inner corners (cols, rows) and the square side (m)."""

    cols: int = 11
    rows: int = 8
    square_m: float = 0.02

    @classmethod
    def parse(cls, spec: str) -> Board:
        try:
            c, r, s = spec.lower().split("x")
            return cls(int(c), int(r), float(s))
        except ValueError:
            raise ValueError(
                f"--board must be <cols>x<rows>x<square_m>, got {spec!r}"
            ) from None

    def object_points(self) -> np.ndarray:
        grid = np.zeros((self.rows * self.cols, 3), dtype=np.float64)
        grid[:, :2] = np.mgrid[0 : self.cols, 0 : self.rows].T.reshape(-1, 2)
        return grid * self.square_m


def detect_board(
    image: np.ndarray, K: np.ndarray, D: np.ndarray, board: Board
) -> tuple[np.ndarray, float]:
    """(``T_cam_board``, mean reprojection error px) of the checkerboard in ``image``
    (RGB or grey). Raises ValueError with a short reason when it is not found."""
    import cv2

    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) if image.ndim == 3 else image
    # SB keeps the corner order consistent across frames (the legacy detector flips
    # the board on wide views, which injects a systematic hand-eye error).
    found, corners = cv2.findChessboardCornersSB(
        gray, (board.cols, board.rows), cv2.CALIB_CB_NORMALIZE_IMAGE
    )
    if not found:
        raise ValueError("checkerboard_not_found")
    corners = np.asarray(corners, dtype=np.float64).reshape(-1, 2)
    obj = board.object_points()
    ok, rvec, tvec = cv2.solvePnP(obj, corners, K, D, flags=cv2.SOLVEPNP_ITERATIVE)
    if not ok:
        raise ValueError("solvepnp_failed")
    projected, _ = cv2.projectPoints(obj, rvec, tvec, K, D)
    reproj = float(np.mean(np.linalg.norm(projected.reshape(-1, 2) - corners, axis=1)))
    R, _ = cv2.Rodrigues(rvec)
    return transform(R, tvec.reshape(3)), reproj


# -- solver -------------------------------------------------------------------


def solve_hand_eye(
    samples: list[Sample], mode: str, method: str = "PARK"
) -> np.ndarray:
    """``T_base_cam`` (eye_to_hand) or ``T_tcp_cam`` (eye_in_hand) from the valid samples.

    cv2.calibrateHandEye solves the eye-in-hand problem from (gripper->base,
    target->camera) pairs. For a fixed camera the robot poses are inverted
    (base->gripper), which turns the returned camera->gripper transform into
    camera->base.
    """
    import cv2

    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}")
    if method not in METHODS:
        raise ValueError(f"method must be one of {METHODS}")
    valid = [s for s in samples if s.valid]
    if len(valid) < 3:
        raise ValueError(
            f"hand-eye calibration needs at least 3 valid samples, have {len(valid)}"
        )
    R_g, t_g, R_t, t_t = [], [], [], []
    for s in valid:
        T_robot = invert(s.T_base_tcp) if mode == EYE_TO_HAND else s.T_base_tcp
        R_g.append(T_robot[:3, :3])
        t_g.append(T_robot[:3, 3].reshape(3, 1))
        R_t.append(s.T_cam_board[:3, :3])
        t_t.append(s.T_cam_board[:3, 3].reshape(3, 1))
    R, t = cv2.calibrateHandEye(
        R_g, t_g, R_t, t_t, method=getattr(cv2, f"CALIB_HAND_EYE_{method}")
    )
    return transform(np.asarray(R), np.asarray(t).reshape(3))


def implied_board(
    samples: list[Sample], T_cam: np.ndarray, mode: str
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    """The board transform every sample implies (``T_tcp_board`` for eye_to_hand,
    ``T_base_board`` for eye_in_hand), its mean, and each sample's deviation from it:
    a constant in a consistent calibration, so the spread is the quality signal."""
    valid = [s for s in samples if s.valid]
    implied = [
        (invert(s.T_base_tcp) if mode == EYE_TO_HAND else s.T_base_tcp)
        @ T_cam
        @ s.T_cam_board
        for s in valid
    ]
    T_mean = mean_transform(implied)
    deviations = [
        {
            "id": s.id,
            "translation_m": float(np.linalg.norm(T[:3, 3] - T_mean[:3, 3])),
            "rotation_deg": rotation_deg(T_mean[:3, :3], T[:3, :3]),
        }
        for s, T in zip(valid, implied)
    ]
    return T_mean, deviations


def reject_outliers(
    samples: list[Sample],
    deviations: list[dict[str, Any]],
    *,
    max_reprojection_px: float,
    max_translation_mm: float,
    max_rotation_deg: float,
) -> None:
    """Flag (in place) samples whose detection or implied board pose is off."""
    by_id = {d["id"]: d for d in deviations}
    for s in samples:
        if not s.valid:
            continue
        if s.reprojection_px is not None and s.reprojection_px > max_reprojection_px:
            s.reject = (
                f"reprojection {s.reprojection_px:.2f} px > {max_reprojection_px}"
            )
            continue
        d = by_id.get(s.id)
        if d is None:
            s.reject = "no consistency estimate"
        elif d["translation_m"] * 1000 > max_translation_mm:
            s.reject = f"board pose off by {d['translation_m'] * 1000:.1f} mm > {max_translation_mm}"
        elif d["rotation_deg"] > max_rotation_deg:
            s.reject = (
                f"board pose off by {d['rotation_deg']:.2f} deg > {max_rotation_deg}"
            )


def consistency(
    samples: list[Sample], T_cam: np.ndarray, T_board: np.ndarray, mode: str
) -> list[dict[str, Any]]:
    """Per valid sample: the predicted vs observed ``T_cam_board`` error."""
    out = []
    for s in samples:
        if not s.valid:
            continue
        if mode == EYE_TO_HAND:
            T_pred = invert(T_cam) @ s.T_base_tcp @ T_board
        else:
            T_pred = invert(T_cam) @ invert(s.T_base_tcp) @ T_board
        out.append(
            {
                "id": s.id,
                "translation_m": float(
                    np.linalg.norm(T_pred[:3, 3] - s.T_cam_board[:3, 3])
                ),
                "rotation_deg": rotation_deg(s.T_cam_board[:3, :3], T_pred[:3, :3]),
                "reprojection_px": s.reprojection_px,
            }
        )
    return out


def calibrate(
    samples: list[Sample],
    mode: str,
    *,
    method: str = "PARK",
    max_reprojection_px: float = 3.0,
    max_translation_mm: float = 40.0,
    max_rotation_deg: float = 6.0,
) -> dict[str, Any]:
    """Two-pass solve: estimate, reject outliers, re-solve. Returns the transform, the
    implied board transform, the residuals and the sample bookkeeping."""
    if len([s for s in samples if s.valid]) < 3:
        raise ValueError(
            f"only {len([s for s in samples if s.valid])} valid detections; need >= 3"
        )
    T_first = solve_hand_eye(samples, mode, method)
    _, deviations = implied_board(samples, T_first, mode)
    reject_outliers(
        samples,
        deviations,
        max_reprojection_px=max_reprojection_px,
        max_translation_mm=max_translation_mm,
        max_rotation_deg=max_rotation_deg,
    )
    if len([s for s in samples if s.valid]) < 3:
        raise ValueError("outlier rejection left fewer than 3 samples")
    T = solve_hand_eye(samples, mode, method)
    T_board, deviations = implied_board(samples, T, mode)
    errors = consistency(samples, T, T_board, mode)
    used = [s for s in samples if s.valid]
    rejected = [s for s in samples if s.reject is not None]
    return {
        "mode": mode,
        "method": method,
        "transform": "T_base_cam" if mode == EYE_TO_HAND else "T_tcp_cam",
        "matrix": T,
        "board": "T_tcp_board" if mode == EYE_TO_HAND else "T_base_board",
        "board_matrix": T_board,
        "residuals": {
            "translation_mm": summarize([e["translation_m"] * 1000 for e in errors]),
            "rotation_deg": summarize([e["rotation_deg"] for e in errors]),
            "reprojection_px": summarize(
                [s.reprojection_px for s in used if s.reprojection_px is not None]
            ),
            "board_spread_mm": summarize(
                [d["translation_m"] * 1000 for d in deviations]
            ),
            "board_spread_deg": summarize([d["rotation_deg"] for d in deviations]),
        },
        "samples": {
            "total": len(samples),
            "used": len(used),
            "rejected": len(rejected),
        },
        "rejected": [{"id": s.id, "reason": s.reject} for s in rejected],
        "per_sample": errors,
    }


# -- calibration YAML (easy_handeye layout + binding) --------------------------------


def matrix_to_transformation(T: np.ndarray) -> dict[str, float]:
    q = Rotation.from_matrix(np.asarray(T)[:3, :3]).as_quat()
    x, y, z = np.asarray(T)[:3, 3]
    return {
        "x": float(x),
        "y": float(y),
        "z": float(z),
        "qx": float(q[0]),
        "qy": float(q[1]),
        "qz": float(q[2]),
        "qw": float(q[3]),
    }


def transformation_to_matrix(t: dict[str, Any]) -> np.ndarray:
    return transform(
        Rotation.from_quat([t["qx"], t["qy"], t["qz"], t["qw"]]).as_matrix(),
        [t["x"], t["y"], t["z"]],
    )


def load_calibration_yaml(path: str | Path) -> dict[str, Any]:
    """One calibration YAML: ``eye_on_hand``, the 4x4 ``matrix`` (camera -> tcp or
    camera -> base), ``arm_id``, ``camera``, ``camera_serial``, the raw sections."""
    p = Path(path).expanduser()
    if not p.exists():
        raise ValueError(
            f"calibration YAML not found: {p}. Run robots/ur5e/calibrate.py (capture, "
            "solve, then apply) for this camera"
        )
    try:
        data = yaml.safe_load(p.read_text(errors="replace"))
    except yaml.YAMLError as exc:
        raise ValueError(f"invalid calibration YAML {p}: {exc}") from exc
    if (
        not isinstance(data, dict)
        or not isinstance(data.get("parameters"), dict)
        or not isinstance(data.get("transformation"), dict)
    ):
        raise ValueError(
            f"{p} must be a calibration YAML (a mapping with 'parameters' and "
            "'transformation' sections)"
        )
    params, tr = data["parameters"], data["transformation"]
    missing = {"x", "y", "z", "qx", "qy", "qz", "qw"} - set(tr)
    if missing:
        raise ValueError(f"{p} missing transform fields: {sorted(missing)}")
    if "eye_on_hand" not in params:
        raise ValueError(f"{p}: parameters.eye_on_hand is required")
    arm_id = params.get("arm_id")
    return {
        "path": str(p),
        "eye_on_hand": bool(params["eye_on_hand"]),
        "matrix": transformation_to_matrix(tr),
        "arm_id": None if arm_id in (None, "") else str(arm_id),
        "camera": params.get("camera"),
        "camera_serial": params.get("camera_serial"),
        "residuals": params.get("residuals"),
        "parameters": params,
        "transformation": dict(tr),
    }


def write_calibration_yaml(
    path: str | Path,
    result: dict[str, Any],
    *,
    arm_id: str,
    camera: str,
    camera_serial: str | None,
    target_path: str,
    robot: str = "ur5e",
    base_frame: str = "base",
    effector_frame: str = "tcp",
) -> Path:
    """Write ``result`` (from :func:`calibrate`) as a calibration YAML bound to ``arm_id``."""
    if not arm_id:
        raise ValueError("arm_id is required: the calibration must name its arm")
    eye_on_hand = result["mode"] == EYE_IN_HAND
    tracking = f"{camera}_color_optical_frame"
    data = {
        "parameters": {
            "eye_on_hand": eye_on_hand,
            "robot_base_frame": base_frame,
            "robot_effector_frame": effector_frame,
            "tracking_base_frame": tracking,
            "tracking_marker_frame": "checkerboard",
            "freehand_robot_movement": True,
            "robot": robot,
            "arm_id": str(arm_id),
            "camera": camera,
            "camera_serial": camera_serial,
            "mode": result["mode"],
            "method": result["method"],
            "transform": result["transform"],
            "maps": "camera_frame_point -> "
            + ("effector_frame_point" if eye_on_hand else "base_frame_point"),
            "created": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "target_path": str(target_path),
            "samples": result["samples"],
            "rejected": result["rejected"],
            "residuals": result["residuals"],
        },
        "transformation": matrix_to_transformation(result["matrix"]),
    }
    out = Path(path).expanduser()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(yaml.safe_dump(data, sort_keys=False))
    return out


def new_path(target: str | Path) -> Path:
    """``<calibration>.new.yaml`` next to the robot config's calibration file."""
    t = Path(target).expanduser()
    stem = t.name[: -len(t.suffix)] if t.suffix else t.name
    return t.with_name(f"{stem}.new.yaml")


def format_residuals(res: dict[str, Any]) -> str:
    lines = []
    for key, unit in (
        ("translation_mm", "mm"),
        ("rotation_deg", "deg"),
        ("reprojection_px", "px"),
        ("board_spread_mm", "mm"),
        ("board_spread_deg", "deg"),
    ):
        s = res.get(key) or {}
        if s.get("mean") is None:
            continue
        lines.append(
            f"  {key:16s} mean {s['mean']:.3f}  median {s['median']:.3f}  max {s['max']:.3f} {unit}"
        )
    return "\n".join(lines)


# -- sample directories -------------------------------------------------------


def load_intrinsics(path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    """K (3x3) and D (distortion coefficients) from ``intrinsics.json`` (RealSense
    layout ``fx, fy, ppx, ppy[, coeffs]``)."""
    data = json.loads(Path(path).expanduser().read_text())
    K = np.array(
        [
            [data["fx"], 0.0, data["ppx"]],
            [0.0, data["fy"], data["ppy"]],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    D = np.asarray(data.get("coeffs") or [0.0] * 5, dtype=np.float64)
    return K, D


def load_samples(
    directory: str | Path, K: np.ndarray, D: np.ndarray, board: Board
) -> list[Sample]:
    """The ``NNNN.json`` (+ ``NNNN.png``) pairs of a capture directory as samples: the
    JSON holds ``tcp_pose`` (7) or ``tcp_pose_rotvec`` (6) and, optionally, a
    precomputed ``T_cam_board``; otherwise the board is detected in the PNG."""
    d = Path(directory).expanduser()
    samples: list[Sample] = []
    metadata = {"intrinsics.json", "camera.json"}
    for meta in sorted(p for p in d.glob("*.json") if p.name not in metadata):
        data = json.loads(meta.read_text())
        pose = data.get("tcp_pose", data.get("tcp_pose_rotvec"))
        if pose is None:
            samples.append(Sample(meta.stem, np.eye(4), reject="no tcp_pose"))
            continue
        s = Sample(meta.stem, pose_matrix(pose))
        if data.get("T_cam_board") is not None:
            s.T_cam_board = np.asarray(data["T_cam_board"], dtype=np.float64).reshape(
                4, 4
            )
            s.reprojection_px = data.get("reprojection_px")
        else:
            image_path = d / data.get("image", f"{meta.stem}.png")
            if not image_path.exists():
                s.reject = "no image"
            else:
                import cv2

                bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
                if bgr is None:
                    s.reject = "unreadable image"
                else:
                    try:
                        s.T_cam_board, s.reprojection_px = detect_board(
                            bgr[:, :, ::-1], K, D, board
                        )
                    except ValueError as exc:
                        s.reject = str(exc)
        samples.append(s)
    return samples


# -- CLI ----------------------------------------------------------------------


def _config_and_camera(
    args: argparse.Namespace,
) -> tuple[dict[str, Any], dict[str, Any], str]:
    """The robot config, the named camera's device mapping and its calibration path."""
    from pi_embodied_services.robots.ur5e.env_server import DEFAULT_CONFIG

    cfg = (
        yaml.safe_load(
            Path(args.robot_config or DEFAULT_CONFIG).expanduser().read_text()
        )
        or {}
    )
    devices = (cfg.get("cameras") or {}).get("devices") or {}
    dev = devices.get(args.camera)
    if dev is None:
        raise SystemExit(
            f"camera {args.camera!r} is not in cameras.devices (have: {', '.join(devices) or 'none'})"
        )
    target = dev.get("calibration")
    if not target:
        raise SystemExit(
            f"cameras.devices.{args.camera}.calibration names no file: set the path the "
            "calibration should be written to"
        )
    return cfg, dev, str(Path(target).expanduser())


def cmd_capture(args: argparse.Namespace) -> int:
    """Record (TCP pose, frame) pairs; the human moves the arm between captures."""
    import cv2

    from pi_embodied_services.components.cameras import open_camera
    from pi_embodied_services.robots.ur5e.env_server import build_arm

    cfg, dev, _ = _config_and_camera(args)
    out = Path(args.samples).expanduser()
    out.mkdir(parents=True, exist_ok=True)
    defaults = {k: v for k, v in (cfg.get("cameras") or {}).items() if k != "devices"}
    arm, gripper = build_arm(cfg)
    cam = open_camera(args.camera, dev, defaults)
    try:
        intr = cam.intrinsics()
        if intr is None:
            raise SystemExit(
                f"camera {args.camera} reports no intrinsics; set cameras.devices."
                f"{args.camera}.intrinsics (a one-off checkerboard calibration) first"
            )
        (out / "intrinsics.json").write_text(json.dumps(intr, indent=1))
        (out / "camera.json").write_text(
            json.dumps(
                {
                    "camera": args.camera,
                    "serial": getattr(cam, "serial", None),
                    **cam.describe(),
                }
            )
        )
        n = len(list(out.glob("*.png")))
        print(
            f"Move the arm so the board is fully visible, then press Enter to record a "
            f"sample (q + Enter finishes). The tool never moves the arm. Samples under {out}"
        )
        while True:
            answer = input(f"[{n} recorded] > ").strip().lower()
            if answer in ("q", "quit", "exit"):
                break
            pose = np.asarray(arm.tcp_pose(), dtype=np.float64)
            frame = cam.read()
            stem = f"{n:04d}"
            cv2.imwrite(str(out / f"{stem}.png"), frame.rgb[:, :, ::-1])
            (out / f"{stem}.json").write_text(
                json.dumps(
                    {
                        "tcp_pose_rotvec": pose.tolist(),
                        "joints": np.asarray(arm.joints()).tolist(),
                        "image": f"{stem}.png",
                        "timestamp_s": frame.timestamp_s,
                    }
                )
            )
            n += 1
            print(f"  recorded {stem}: tcp {np.round(pose[:3], 4).tolist()}")
    finally:
        cam.close()
        if gripper is not None:
            gripper.close()
        arm.close()
    return 0


def cmd_solve(args: argparse.Namespace) -> int:
    cfg, dev, target = _config_and_camera(args)
    arm_id = args.arm_id or (cfg.get("calibration") or {}).get("arm_id")
    if not arm_id:
        raise SystemExit(
            "--arm-id is required (or calibration.arm_id in the robot config): the "
            "calibration must name the arm it belongs to (env_server --print-identity)"
        )
    mount = dev.get("mount")
    mode = args.mode or (
        EYE_IN_HAND if mount == "wrist" else EYE_TO_HAND if mount == "fixed" else None
    )
    if mode is None:
        raise SystemExit(
            "--mode is required when cameras.devices.<name>.mount is not set"
        )
    samples_dir = Path(args.samples).expanduser()
    K, D = load_intrinsics(args.intrinsics or samples_dir / "intrinsics.json")
    samples = load_samples(samples_dir, K, D, Board.parse(args.board))
    result = calibrate(
        samples,
        mode,
        method=args.method,
        max_reprojection_px=args.max_reprojection_px,
        max_translation_mm=args.max_translation_mm,
        max_rotation_deg=args.max_rotation_deg,
    )
    serial = None
    cam_json = samples_dir / "camera.json"
    if cam_json.exists():
        serial = json.loads(cam_json.read_text()).get("serial")
    out = write_calibration_yaml(
        args.out or new_path(target),
        result,
        arm_id=str(arm_id),
        camera=args.camera,
        camera_serial=serial,
        target_path=target,
    )
    print(
        f"{result['transform']} ({mode}, {args.method}) from {result['samples']['used']} of "
        f"{result['samples']['total']} samples ({result['samples']['rejected']} rejected):"
    )
    for r in result["rejected"]:
        print(f"  rejected {r['id']}: {r['reason']}")
    print(format_residuals(result["residuals"]))
    tr = matrix_to_transformation(result["matrix"])
    print(
        f"  translation {[round(tr[k], 4) for k in 'xyz']} m, quaternion "
        f"{[round(tr[k], 4) for k in ('qx', 'qy', 'qz', 'qw')]}"
    )
    print(f"written to {out} (arm_id {arm_id}); review it, then apply with:")
    print(
        f"  python -m pi_embodied_services.robots.ur5e.calibrate apply "
        f"--robot-config {args.robot_config or 'config/example.yaml'} --camera {args.camera} --yes"
    )
    return 0


def cmd_apply(args: argparse.Namespace) -> int:
    """Replace the camera's calibration with the reviewed ``.new.yaml``."""
    cfg, _, target = _config_and_camera(args)
    new_file = Path(args.new or new_path(target)).expanduser()
    if not new_file.exists():
        raise SystemExit(f"{new_file} does not exist; run `solve` first")
    new = load_calibration_yaml(new_file)
    want = (cfg.get("calibration") or {}).get("arm_id")
    problems = []
    if want in (None, ""):
        problems.append("the robot config has no calibration.arm_id")
    elif new["arm_id"] != str(want):
        problems.append(
            f"the new file is bound to arm {new['arm_id']!r}, the config to {want!r}"
        )
    if new["camera"] != args.camera:
        problems.append(
            f"the new file was solved for camera {new['camera']!r}, not {args.camera!r}"
        )
    recorded = new["parameters"].get("target_path")
    if recorded and str(Path(recorded).expanduser()) != target:
        problems.append(f"the new file was solved for {recorded}, not {target}")
    if problems:
        raise SystemExit("refusing to apply:\n  " + "\n  ".join(problems))
    print(
        f"new calibration {new_file} ({new['parameters'].get('mode')}, arm {new['arm_id']}):"
    )
    print(format_residuals(new.get("residuals") or {}))
    old_path = Path(target)
    if old_path.exists():
        old = load_calibration_yaml(old_path)
        dt = float(np.linalg.norm(new["matrix"][:3, 3] - old["matrix"][:3, 3]))
        dr = rotation_deg(old["matrix"][:3, :3], new["matrix"][:3, :3])
        print(f"change from the current calibration: {dt * 1000:.1f} mm, {dr:.2f} deg")
    else:
        print("no current calibration at the target path (first calibration)")
    if not args.yes:
        print(f"dry run: re-run with --yes to replace {target}")
        return 2
    if old_path.exists():
        backup = old_path.with_name(
            f"{old_path.name}.bak-{time.strftime('%Y%m%d-%H%M%S')}"
        )
        os.replace(old_path, backup)
        print(f"backed up the old calibration to {backup}")
    old_path.parent.mkdir(parents=True, exist_ok=True)
    os.replace(new_file, old_path)
    print(f"applied: {old_path}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--robot-config", default=None, help="the robot YAML")
    common.add_argument("--camera", required=True, help="cameras.devices.<name>")
    cap = sub.add_parser(
        "capture", parents=[common], help="record TCP poses and frames"
    )
    cap.add_argument("--samples", required=True, help="output directory")
    cap.set_defaults(run=cmd_capture)
    solve = sub.add_parser(
        "solve", parents=[common], help="solve and write <calibration>.new.yaml"
    )
    solve.add_argument("--samples", required=True, help="capture directory")
    solve.add_argument(
        "--mode", choices=MODES, default=None, help="default: from the camera's mount"
    )
    solve.add_argument("--method", choices=METHODS, default="PARK")
    solve.add_argument(
        "--board", default="11x8x0.02", help="inner corners cols x rows x square (m)"
    )
    solve.add_argument(
        "--intrinsics", default=None, help="intrinsics.json (default: in --samples)"
    )
    solve.add_argument(
        "--arm-id", default=None, help="default: calibration.arm_id of the robot config"
    )
    solve.add_argument("--out", default=None, help="default: <calibration>.new.yaml")
    solve.add_argument("--max-reprojection-px", type=float, default=3.0)
    solve.add_argument("--max-translation-mm", type=float, default=40.0)
    solve.add_argument("--max-rotation-deg", type=float, default=6.0)
    solve.set_defaults(run=cmd_solve)
    apply = sub.add_parser(
        "apply", parents=[common], help="replace the calibration with the .new.yaml"
    )
    apply.add_argument("--new", default=None, help="default: <calibration>.new.yaml")
    apply.add_argument(
        "--yes", action="store_true", help="really replace (without it: dry run)"
    )
    apply.set_defaults(run=cmd_apply)
    args = parser.parse_args(argv)
    return int(args.run(args))


if __name__ == "__main__":
    sys.exit(main())
