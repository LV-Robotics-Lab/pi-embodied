# Copyright 2026 The Show-Harness Authors (github.com/showlab/Show-Harness @137d571).
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
# Modified by pi-embodied: scripts/trajectory/rebuild_video.py and step_timing.py for
# pi-embodied's GUMI run dirs (packages/embodied/src/gumi: images/agentview, images/wrist
# or images/wrist_left + wrist_right, steps.jsonl with per-arm records on two arms);
# the annotated header is drawn with PIL instead of core.record's renderer, the video
# is encoded by piping RGB into ffmpeg (on PATH, else imageio-ffmpeg's), and the
# timing report groups by who acted (src) since a pi run's model is in its session.

"""Tools over GUMI recordings (``--gumi-record <root>/<MMDD>/task_<id>/<HH-MM-SS>/``).

python -m pi_embodied_services.flywheel.gumi_tools rebuild-video <run> [--view annotated|side|agentview|wrist] [--fps 2] [--output f.mp4]
    re-encode a run's video from its saved PNG frames (an interrupted run keeps every frame)
python -m pi_embodied_services.flywheel.gumi_tools step-timing <run or parent>...
    per-step cycle time from steps.jsonl ``ts`` (period = ts[i+1] - ts[i]), per run and per source;
    Show-Harness runner fields (t_obs_ms, t_decide_ms, t_exec_ms, vlm_ms) are reported when present
"""

from __future__ import annotations

import argparse
import json
import shutil
import statistics
import subprocess
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw

FPS_DEFAULT, FPS_MIN, FPS_MAX = 2.0, 0.5, 30.0
HEADER_PX = 34

# ---------------------------------------------------------------------------
# rebuild-video


def run_dirs_of(path: Path) -> tuple[Path, Path]:
    """(run dir, images dir) from a run dir, its images/ or images/agentview/."""
    path = path.resolve()
    for run in (path, path.parent, path.parent.parent):
        if (run / "images" / "agentview").is_dir():
            return run, run / "images"
    raise SystemExit(f"no images/agentview/*.png under {path}")


def records(run: Path) -> dict[int, dict[str, Any]]:
    """step index -> steps.jsonl record (steps.json when the run was closed)."""
    rows: list[Any] = []
    if (run / "steps.json").exists():
        try:
            rows = json.loads((run / "steps.json").read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            rows = []
    if not rows and (run / "steps.jsonl").exists():
        for line in (run / "steps.jsonl").read_text(encoding="utf-8").splitlines():
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return {int(r.get("i", n)): r for n, r in enumerate(rows) if isinstance(r, dict)}


def header(record: dict[str, Any]) -> str:
    """#i src act grip, per arm on two arms."""
    arms = [a for a in ("left", "right") if isinstance(record.get(a), dict)]
    if arms:
        parts = [
            f"{a[0].upper()}:{record[a].get('act', '?')} {record[a].get('grip', '')}"
            for a in arms
        ]
        src = record[arms[0]].get("src", "")
        return f"#{record.get('i', '?')} {src} " + "  ".join(parts)
    n = f" x{record['n']}" if record.get("n", 1) != 1 else ""
    return f"#{record.get('i', '?')} {record.get('src', '')} {record.get('act', '')}{n} {record.get('grip', '')}".strip()


def load(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("RGB"))


def hstack(images: list[np.ndarray]) -> np.ndarray:
    """Side by side at the first image's height."""
    h = images[0].shape[0]
    out = []
    for img in images:
        if img.shape[0] != h:
            w = max(1, round(img.shape[1] * h / img.shape[0]))
            img = np.asarray(
                Image.fromarray(img).resize((w, h), Image.Resampling.BILINEAR)
            )
        out.append(img)
    return np.concatenate(out, axis=1)


def even(img: np.ndarray) -> np.ndarray:
    """h.264 needs even dimensions."""
    h, w = img.shape[:2]
    return np.ascontiguousarray(img[: h - h % 2, : w - w % 2])


def build_frames(view: str, run: Path, images: Path) -> list[np.ndarray]:
    frames = sorted((images / "agentview").glob("*.png"), key=lambda p: p.stem)
    if not frames:
        raise SystemExit(f"no agentview frames in {images / 'agentview'}")
    wrists = [
        d
        for d in (images / "wrist", images / "wrist_left", images / "wrist_right")
        if d.is_dir()
    ]
    if view == "wrist" and not wrists:
        raise SystemExit(f"no wrist frames under {images}")
    recs = records(run) if view == "annotated" else {}
    out = []
    for f in frames:
        wrist = [load(d / f.name) for d in wrists if (d / f.name).exists()]
        if view == "agentview":
            img = load(f)
        elif view == "wrist":
            img = hstack(wrist) if wrist else None
            if img is None:
                continue
        else:
            img = hstack([load(f), *wrist])
        if view == "annotated":
            idx = int(f.stem) if f.stem.isdigit() else -1
            text = header(recs.get(idx, {"i": idx}))
            canvas = Image.new(
                "RGB", (img.shape[1], img.shape[0] + HEADER_PX), (0, 0, 0)
            )
            canvas.paste(Image.fromarray(img), (0, HEADER_PX))
            ImageDraw.Draw(canvas).text((6, 10), text, fill=(255, 255, 255))
            img = np.asarray(canvas)
        out.append(even(img))
    return out


def auto_fps(run: Path) -> float:
    for name in ("summary.json", "metadata.json"):
        try:
            value = json.loads((run / name).read_text(encoding="utf-8")).get(
                "video_fps"
            )
        except (OSError, json.JSONDecodeError, AttributeError):
            continue
        if value is not None:
            return float(value)
    return FPS_DEFAULT


def ffmpeg_exe() -> str:
    exe = shutil.which("ffmpeg")
    if exe:
        return exe
    try:
        import imageio_ffmpeg

        return imageio_ffmpeg.get_ffmpeg_exe()
    except ImportError as exc:
        raise SystemExit(
            "no ffmpeg on PATH and imageio-ffmpeg is not installed"
        ) from exc


def write_mp4(path: Path, frames: list[np.ndarray], fps: float) -> None:
    """H.264 yuv420p from RGB frames (one size: the first frame's; others are resized to it)."""
    h, w = frames[0].shape[:2]
    cmd = [
        ffmpeg_exe(),
        "-y",
        "-loglevel",
        "error",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "-s",
        f"{w}x{h}",
    ]
    cmd += [
        "-r",
        f"{fps:g}",
        "-i",
        "-",
        "-an",
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        str(path),
    ]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
    assert proc.stdin is not None
    for f in frames:
        if f.shape[:2] != (h, w):
            f = np.asarray(Image.fromarray(f).resize((w, h), Image.Resampling.BILINEAR))
        proc.stdin.write(np.ascontiguousarray(f, dtype=np.uint8).tobytes())
    proc.stdin.close()
    err = proc.stderr.read().decode() if proc.stderr else ""
    if proc.wait() != 0:
        raise SystemExit(f"ffmpeg failed: {err.strip()}")


def rebuild_video(argv: list[str]) -> int:
    p = argparse.ArgumentParser(prog="gumi_tools rebuild-video")
    p.add_argument(
        "path", type=Path, help="run dir (or its images/ or images/agentview/)"
    )
    p.add_argument(
        "--view",
        choices=("annotated", "side", "agentview", "wrist"),
        default="annotated",
    )
    p.add_argument(
        "--fps",
        type=float,
        default=None,
        help=f"default: metadata video_fps, else {FPS_DEFAULT}",
    )
    p.add_argument(
        "--output", type=Path, default=None, help="default: <run>/rollout_rebuilt.mp4"
    )
    args = p.parse_args(argv)
    run, images = run_dirs_of(args.path)
    fps = min(
        FPS_MAX, max(FPS_MIN, args.fps if args.fps is not None else auto_fps(run))
    )
    frames = build_frames(args.view, run, images)
    if not frames:
        raise SystemExit("no frames to encode")
    out = args.output or run / "rollout_rebuilt.mp4"
    out.parent.mkdir(parents=True, exist_ok=True)
    write_mp4(out, frames, fps)
    print(
        json.dumps(
            {
                "run": str(run),
                "view": args.view,
                "frames": len(frames),
                "fps": fps,
                "output": str(out),
            }
        )
    )
    return 0


# ---------------------------------------------------------------------------
# step-timing


def stats(xs: list[float]) -> str:
    if not xs:
        return "-"
    xs = sorted(xs)
    p90 = xs[round(0.9 * (len(xs) - 1))]
    return f"mean {statistics.mean(xs):7.0f}  median {statistics.median(xs):7.0f}  p90 {p90:7.0f}  n={len(xs)}"


def src_of(record: dict[str, Any]) -> str:
    if "src" in record:
        return str(record["src"])
    for arm in ("left", "right"):
        if isinstance(record.get(arm), dict) and record[arm].get("src"):
            return str(record[arm]["src"])
    return "?"


FIELDS = (
    ("period", "period ms"),
    ("obs", "camera ms"),
    ("decide", "decide ms"),
    ("vlm", "vlm ms"),
    ("exec", "motion ms"),
)


def analyze(run: Path) -> dict[str, dict[str, list[float]]]:
    """Per source: the step periods (a pause over 10 min is no step) and any runner timing fields."""
    rows = [r for _, r in sorted(records(run).items())]
    out: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for r in rows:
        src = src_of(r)
        for key, field in (
            ("obs", "t_obs_ms"),
            ("decide", "t_decide_ms"),
            ("vlm", "vlm_ms"),
            ("exec", "t_exec_ms"),
        ):
            if field in r:
                out[src][key].append(float(r[field]))
    for a, b in zip(rows, rows[1:]):
        if "ts" in a and "ts" in b:
            period = (float(b["ts"]) - float(a["ts"])) * 1000.0
            if 0 < period < 600_000:
                out[src_of(a)]["period"].append(period)
    return out


def step_timing(argv: list[str]) -> int:
    p = argparse.ArgumentParser(prog="gumi_tools step-timing")
    p.add_argument("paths", nargs="+", type=Path, help="run dirs, or parents to scan")
    args = p.parse_args(argv)
    runs = sorted({f.parent for path in args.paths for f in path.rglob("steps.jsonl")})
    if not runs:
        print("no steps.jsonl found under the given paths")
        return 1
    total: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for run in runs:
        print(f"\n{run}")
        for src, agg in sorted(analyze(run).items()):
            for key, label in FIELDS:
                if agg[key]:
                    print(f"  {src:6} {label:10} {stats(agg[key])}")
                    total[src][key] += agg[key]
    if len(runs) > 1:
        print("\n=== all runs, per source ===")
        for src, agg in sorted(total.items()):
            for key, label in FIELDS:
                if agg[key]:
                    print(f"  {src:6} {label:10} {stats(agg[key])}")
    return 0


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    tools = {"rebuild-video": rebuild_video, "step-timing": step_timing}
    if not argv or argv[0] not in tools:
        print(__doc__)
        return 2
    return tools[argv[0]](argv[1:])


if __name__ == "__main__":
    sys.exit(main())
