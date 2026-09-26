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

"""A LeRobot v3.0 dataset of ours (flywheel/export.py, flywheel/gumi.py) as Show-Harness rollouts.

    python -m pi_embodied_services.finetuned.lerobot_to_rollouts <dataset dir> --out <rollouts dir> \\
        [--prompt v3|v5] [--include-failures] [--robot R] [--agentview spec --wrist spec] [--task T]

Writes the layout ``finetuned/prepare.ts`` writes from GUMI recordings and ``train.sh`` feeds to
Show-Harness's ``rollouts_to_alpaca.py``: ``<out>/<task>/rollout_NNN/{agentview,wrist}/NNNN.png``,
``actions.jsonl`` ({step, token, gripper_closed, agentview, wrist, ...}) and ``metadata.json``
(``task_text``, the source dataset and episode). The frames are the dataset's PNG bytes put
through the provider's own camera transform by ``packages/embodied/src/finetuned/transform.ts``
(one node call per episode), so a training frame here is byte for byte what ``prepare.ts`` would
have written from the recording and what the provider sends at inference.

Tokens:

- a GUMI export names its units (``action`` is a one-hot with names ``arm.<unit>``): the
  recorded unit is the token, ``actor``/``dagger``/``action_repeat`` come back as GUMI's
  ``src``/``dagger``/``n``. Dual-arm datasets (``left.<unit>``) are refused.
- a Flywheel export of a VLA run has no units, only per-step deltas: the eef path
  (``observation.state``'s eef_x/y/z) is quantized onto the 2 cm lattice with real2sim's
  ``manhattan_tokens`` (a sample whenever the eef has moved a whole step from the last lattice
  point, its frame the observation where that motion began), each axis named by the robot's
  own MV_* base-frame vectors (``UNIT_VECTORS``, the ``units.vectors`` of the robot's TS
  extension: LIBERO's MV_LEFT is -y, RoboCasa's +y), and a flip of the ``gripper`` action's
  sign is GRASP/RELEASE, sampled on the observation before the command; the samples after it
  start from the observation after it. The remaining frames of the run are not samples;
  ``metadata.json`` says ``token_source: quantized``.

Episodes: those with ``success`` set (a GUMI export marks each run; a Flywheel export only holds
successes) unless ``--include-failures``. ``--prompt v5`` keeps RT_* turns, any other version
drops them like ``prepare.ts`` (v3/v4 offer no turn). Needs pyarrow, numpy, pillow and node.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

import numpy as np

from pi_embodied_services.robots.maniskill.real2sim import MOVE_DIRS, manhattan_tokens

RT_UNITS = (
    "RT_ROLL_LEFT",
    "RT_ROLL_RIGHT",
    "RT_PITCH_FWD",
    "RT_PITCH_BACK",
    "RT_YAW_CW",
    "RT_YAW_CCW",
)
#: finetuned/index.ts PRESETS: the prompt versions, and the one whose vocabulary has turns.
PROMPT_VERSIONS = ("v3", "v4-franka", "v4-piper", "v5")
PI_ROOT = Path(__file__).resolve().parents[3]
TRANSFORM = PI_ROOT / "packages/embodied/src/finetuned/transform.ts"
#: Show-Harness's lattice (real2sim's step_m, the units' default step).
STEP_M = 0.02
#: The MV_* base-frame directions of the robots whose Flywheel exports are quantized, as each
#: one's TS extension grounds them (packages/embodied/src/<robot>/index.ts ``units.vectors``):
#: what a unit means on that robot, so a recorded -y move on RoboCasa is its MV_RIGHT.
UNIT_VECTORS: dict[str, dict[str, tuple[int, int, int]]] = {
    "libero": {
        "MV_FWD": (1, 0, 0),
        "MV_BACK": (-1, 0, 0),
        "MV_LEFT": (0, -1, 0),
        "MV_RIGHT": (0, 1, 0),
        "MV_UP": (0, 0, 1),
        "MV_DOWN": (0, 0, -1),
    },
    "robocasa": {
        "MV_FWD": (1, 0, 0),
        "MV_BACK": (-1, 0, 0),
        "MV_LEFT": (0, 1, 0),
        "MV_RIGHT": (0, -1, 0),
        "MV_UP": (0, 0, 1),
        "MV_DOWN": (0, 0, -1),
    },
}


def slug(task: str) -> str:
    """prepare.ts's task directory name."""
    s = re.sub(r"[^a-z0-9]+", "_", task.lower()).strip("_")[:80]
    return s or "task"


def read_dataset(root: Path) -> dict[str, Any]:
    """The dataset's info, flywheel manifest, tasks and frames grouped by episode (pyarrow only:
    no torch, and the image bytes untouched)."""
    import pyarrow.parquet as pq

    info = json.loads((root / "meta/info.json").read_text(encoding="utf-8"))
    if info.get("codebase_version") != "v3.0":
        raise ValueError(
            f"{root}: not a LeRobot v3.0 dataset ({info.get('codebase_version')})"
        )
    manifest_path = root / "meta/flywheel.json"
    manifest = (
        json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest_path.is_file()
        else {}
    )
    tasks_table = pq.read_table(root / "meta/tasks.parquet").to_pydict()
    text_column = next(k for k in tasks_table if k != "task_index")
    tasks = dict(zip(tasks_table["task_index"], tasks_table[text_column]))
    episodes: dict[int, list[dict[str, Any]]] = {}
    data_dir = root / info["data_path"].split("/")[0]
    for file in sorted(data_dir.rglob("*.parquet")):
        for row in pq.read_table(file).to_pylist():
            episodes.setdefault(int(row["episode_index"]), []).append(row)
    for frames in episodes.values():
        frames.sort(key=lambda r: int(r["frame_index"]))
    return {
        "root": root,
        "info": info,
        "manifest": manifest,
        "tasks": tasks,
        "episodes": episodes,
    }


def recorded_tokens(
    frames: list[dict[str, Any]], action_names: list[str]
) -> list[dict[str, Any]]:
    """GUMI rows from a units export: the one-hot's unit, the state and GUMI's per-step fields."""
    arms = {n.split(".", 1)[0] for n in action_names}
    if arms != {"arm"}:
        raise ValueError(
            f"dual-arm dataset (arms {sorted(arms)}); the single-arm converter needs 'arm.<unit>'"
        )
    units = [n.split(".", 1)[1] for n in action_names]
    rows = []
    for i, f in enumerate(frames):
        one_hot = np.asarray(f["action"], np.float32)
        hot = np.flatnonzero(one_hot == 1.0)
        if len(hot) != 1:
            raise ValueError(f"frame {i}: action is not a one-hot ({one_hot.tolist()})")
        state = [float(v) for v in f["observation.state"]]
        row: dict[str, Any] = {
            "step": i,
            "token": units[int(hot[0])],
            "gripper_closed": bool(state[4] > 0.5),
            "ee_pose": [round(v, 4) for v in state[:3]],
            "gripper_width": round(state[3], 4),
            "src": "human" if int(f.get("actor", 0)) else "agent",
        }
        if int(f.get("action_repeat", 1)) > 1:
            row["n"] = int(f["action_repeat"])
        if int(f.get("dagger", 0)):
            row["dagger"] = True
        rows.append(row)
    return rows


def unit_names(vectors: dict[str, tuple[int, int, int]]) -> dict[tuple[int, int], str]:
    """``(axis, sign) -> unit`` of a robot's MV_* vectors (each one a signed base axis)."""
    names: dict[tuple[int, int], str] = {}
    for unit, v in vectors.items():
        axes = [k for k in range(3) if v[k]]
        if len(axes) != 1 or abs(v[axes[0]]) != 1:
            raise ValueError(f"{unit}: {v} is not a signed base axis")
        names[(axes[0], v[axes[0]])] = unit
    if len(names) != 6:
        raise ValueError(f"the vectors cover {len(names)} of the 6 signed axes")
    return names


def quantized_tokens(
    frames: list[dict[str, Any]],
    state_names: list[str],
    action_names: list[str],
    step_m: float = STEP_M,
    *,
    vectors: dict[str, tuple[int, int, int]],
) -> list[dict[str, Any]]:
    """Units read off a VLA run: the eef path on the ``step_m`` lattice, its axes named by the
    robot's MV_* ``vectors``, gripper flips as GRASP/RELEASE. Each row's ``step`` is the frame
    whose observation it samples: a move's the frame the motion began at, a gripper flip's the
    frame before the command, and what follows the flip starts from the frame after it."""
    units = unit_names(vectors)
    eef = None
    for prefix in ("eef_", "eef_rel_"):
        names = [f"{prefix}{a}" for a in "xyz"]
        if all(n in state_names for n in names):
            eef = [state_names.index(n) for n in names]
            break
    if eef is None:
        raise ValueError(f"no eef_x/y/z in observation.state names {state_names}")
    if "gripper" not in action_names:
        raise ValueError(f"no 'gripper' in action names {action_names}")
    grip = action_names.index("gripper")
    xyz = np.asarray(
        [[f["observation.state"][k] for k in eef] for f in frames], np.float64
    )
    closed_cmd = [float(f["action"][grip]) > 0 for f in frames]
    src = lambda f: "vla" if int(f.get("action_source", 1)) else "scripted"  # noqa: E731
    rows: list[dict[str, Any]] = []
    anchor, a_frame, closed = xyz[0].copy(), 0, closed_cmd[0]
    for i in range(len(frames)):
        # Whole steps since the last lattice point; the residual carries over.
        tokens = manhattan_tokens(xyz[i] - anchor, step_m, step_m)
        for token in tokens:
            axis, sign = MOVE_DIRS[
                token
            ]  # real2sim's names decode manhattan_tokens's axes
            anchor[axis] += sign * step_m
            rows.append(
                {
                    "step": a_frame,
                    "token": units[(axis, sign)],
                    "kind": "move",
                    "gripper_closed": closed,
                    "ee_pose": [round(float(v), 4) for v in xyz[a_frame]],
                    "src": src(frames[a_frame]),
                }
            )
        if tokens:
            a_frame = i
        if closed_cmd[i] != closed:
            closed = closed_cmd[i]
            rows.append(
                {
                    "step": i,
                    "token": "GRASP" if closed else "RELEASE",
                    "kind": "gripper",
                    "gripper_closed": closed,
                    "ee_pose": [round(float(v), 4) for v in xyz[i]],
                    "src": src(frames[i]),
                }
            )
            # The gripper command is action i: the next sample sees the observation after it,
            # not the flip's own frame (two samples of one frame with different labels).
            a_frame = min(i + 1, len(frames) - 1)
            anchor = xyz[a_frame].copy()
    return rows


def transform_frames(
    jobs: list[dict[str, str]],
    views: dict[str, str] | None,
    robot: str,
    env_id: str,
    node: str,
) -> dict[str, str]:
    """Run transform.ts over ``jobs`` (src, dst, camera); the view specs it applied."""
    manifest: dict[str, Any] = {"jobs": jobs, "robot": robot, "env_id": env_id}
    if views:
        manifest["views"] = views
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
        json.dump(manifest, f)
    try:
        proc = subprocess.run(
            [node, "--experimental-strip-types", str(TRANSFORM), f.name],
            capture_output=True,
            text=True,
        )
    finally:
        os.unlink(f.name)
    if proc.returncode:
        raise RuntimeError(f"transform.ts failed:\n{proc.stderr.strip()}")
    reply = json.loads(proc.stdout.strip().splitlines()[-1])
    if reply.get("count") != len(jobs):
        raise RuntimeError(
            f"transform.ts wrote {reply.get('count')} of {len(jobs)} frames"
        )
    return reply["views"]


def convert(
    dataset: Path | str,
    out: Path | str,
    *,
    prompt: str = "v3",
    include_failures: bool = False,
    robot: str | None = None,
    views: dict[str, str] | None = None,
    task: str | None = None,
    cameras: tuple[str, str] = (
        "observation.images.agentview",
        "observation.images.wrist",
    ),
    step_m: float = STEP_M,
    node: str = "node",
    log=sys.stderr,
) -> dict[str, Any]:
    """Convert one dataset; a summary {rollouts, steps, skipped, task_dirs}."""
    if prompt not in PROMPT_VERSIONS:
        raise ValueError(
            f"unknown prompt version {prompt} ({', '.join(PROMPT_VERSIONS)})"
        )
    ds = read_dataset(Path(dataset).expanduser().resolve())
    info, manifest = ds["info"], ds["manifest"]
    features = info["features"]
    for key in cameras:
        if key not in features:
            raise ValueError(
                f"no camera {key}; the dataset has {[k for k in features if k.startswith('observation.images.')]}"
            )
    robot = robot or manifest.get("robot") or info.get("robot_type") or ""
    action_names = list(features["action"]["names"] or [])
    recorded = bool(action_names) and all("." in n for n in action_names)
    # Quantized units are named in the recorded robot's frame (the camera transform's --robot
    # override does not change what the data's MV_LEFT is).
    data_robot = str(manifest.get("robot") or info.get("robot_type") or "")
    if not recorded and data_robot not in UNIT_VECTORS:
        raise ValueError(
            f"no MV_* vectors for robot {data_robot!r} to name its quantized units; "
            f"have {', '.join(UNIT_VECTORS)}"
        )
    out_root = Path(out).expanduser().resolve()
    counters: dict[Path, int] = {}
    summary: dict[str, Any] = {
        "rollouts": 0,
        "steps": 0,
        "skipped": [],
        "task_dirs": [],
    }
    for ep, frames in sorted(ds["episodes"].items()):
        if (
            not include_failures
            and "success" in features
            and not int(frames[0]["success"])
        ):
            summary["skipped"].append(ep)
            print(
                f"skip episode {ep}: not successful (--include-failures converts it)",
                file=log,
            )
            continue
        text = task or ds["tasks"][int(frames[0]["task_index"])]
        if recorded:
            rows = recorded_tokens(frames, action_names)
            token_source = "recorded"
        else:
            rows = quantized_tokens(
                frames,
                list(features["observation.state"]["names"] or []),
                action_names,
                step_m,
                vectors=UNIT_VECTORS[data_robot],
            )
            token_source = "quantized"
        turns = (
            [r["step"] for r in rows if r["token"] in RT_UNITS]
            if prompt != "v5"
            else []
        )
        rows = [r for r in rows if prompt == "v5" or r["token"] not in RT_UNITS]
        task_dir = out_root / slug(text)
        # Numbering continues after rollouts already there (several datasets into one root).
        n = counters.get(
            task_dir, len(list(task_dir.glob("rollout_*"))) if task_dir.is_dir() else 0
        )
        counters[task_dir] = n + 1
        rollout = task_dir / f"rollout_{n:03d}"
        with tempfile.TemporaryDirectory() as raw:
            jobs = []
            for k, row in enumerate(rows):
                for cam, key in zip(("agentview", "wrist"), cameras):
                    src = Path(raw) / cam / f"{k:04d}.png"
                    src.parent.mkdir(parents=True, exist_ok=True)
                    src.write_bytes(frames[row["step"]][key]["bytes"])
                    row[cam] = f"{cam}/{k:04d}.png"
                    jobs.append(
                        {"src": str(src), "dst": str(rollout / row[cam]), "camera": cam}
                    )
            for cam in ("agentview", "wrist"):
                (rollout / cam).mkdir(parents=True, exist_ok=True)
            applied = transform_frames(
                jobs, views, robot, str(manifest.get("env_id") or ""), node
            )
        (rollout / "actions.jsonl").write_text(
            "".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8"
        )
        source = {"source_dataset": str(ds["root"]), "source_episode_index": ep}
        if "repo_id" in manifest:
            source["repo_id"] = manifest["repo_id"]
        for key in ("source_runs", "source_episode_ids"):
            if ep < len(manifest.get(key) or []):
                source[key[:-1]] = manifest[key][ep]
        meta = {
            "robot": robot,
            "task_text": text,
            "prompt_version": prompt,
            "source": "lerobot",
            **source,
            "token_source": token_source,
            **({"step_m": step_m} if token_source == "quantized" else {}),
            "views": applied,
        }
        (rollout / "metadata.json").write_text(
            json.dumps(meta, indent=2), encoding="utf-8"
        )
        if turns:
            print(
                f"warning: episode {ep}: dropped RT_* steps {', '.join(map(str, turns))} "
                f"({prompt} has no turns; --prompt v5 keeps them)",
                file=log,
            )
        print(
            f'episode {ep} -> {rollout} ({len(rows)} steps, {token_source}, "{text}")',
            file=log,
        )
        summary["rollouts"] += 1
        summary["steps"] += len(rows)
    summary["task_dirs"] = [str(d) for d in counters]
    print(
        f"{summary['steps']} steps in {len(counters)} task dir(s) under {out_root}"
        + (
            f"; {len(summary['skipped'])} episode(s) skipped as not successful"
            if summary["skipped"]
            else ""
        ),
        file=log,
    )
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="lerobot_to_rollouts", description=__doc__.split("\n\n")[0]
    )
    parser.add_argument("dataset", type=Path, help="a LeRobot v3.0 dataset directory")
    parser.add_argument("--out", type=Path, required=True, help="the rollouts root")
    parser.add_argument("--prompt", default="v3", choices=PROMPT_VERSIONS)
    parser.add_argument("--include-failures", action="store_true")
    parser.add_argument(
        "--robot", help="camera transform to apply (default: the dataset's robot)"
    )
    parser.add_argument(
        "--agentview", help="view spec overriding the robot's, e.g. square=256"
    )
    parser.add_argument("--wrist", help="view spec overriding the robot's")
    parser.add_argument("--task", help="task text overriding the dataset's")
    parser.add_argument(
        "--cameras",
        default="observation.images.agentview,observation.images.wrist",
        help="the agentview and wrist features, comma-separated",
    )
    parser.add_argument(
        "--step-m", type=float, default=STEP_M, help="lattice for VLA runs"
    )
    parser.add_argument("--node", default=os.environ.get("NODE", "node"))
    args = parser.parse_args(argv)
    if bool(args.agentview) != bool(args.wrist):
        parser.error("--agentview and --wrist go together")
    cameras = tuple(args.cameras.split(","))
    if len(cameras) != 2:
        parser.error("--cameras takes exactly two features")
    convert(
        args.dataset,
        args.out,
        prompt=args.prompt,
        include_failures=args.include_failures,
        robot=args.robot,
        views={"agentview": args.agentview, "wrist": args.wrist}
        if args.agentview
        else None,
        task=args.task,
        cameras=cameras,
        step_m=args.step_m,
        node=args.node,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
