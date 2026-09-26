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

"""GUMI rollouts (packages/embodied/src/gumi's recorder) as a LeRobot v3.0 dataset, with the
feature names every pi-embodied dataset shares (flywheel/export.py):

- ``observation.images.<view>``: ``agentview`` and ``wrist`` (two arms: ``wrist_left``,
  ``wrist_right``), the PNGs the policy saw;
- ``observation.state``: per arm the measured eef xyz, gripper width and the commanded gripper
  (1 closed), measured at the observation the unit was decided on;
- ``action``: per arm a one-hot over the unit vocabulary, dimensions named ``<arm>.<unit>``;
- ``action_repeat``: how many times the unit ran; ``actor``: 1 for a human's unit, 0 for the
  agent's; ``dagger``: 1 where the human took over from a running agent.

One dataset holds one robot's runs of one task with one vocabulary and camera set.
"""

from __future__ import annotations

import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]*")


def _runs(root: Path) -> list[Path]:
    """Closed runs (summary.json written) under ``root``: one run dir, or a task dir of them."""
    if (root / "summary.json").is_file():
        return [root]
    return sorted(p.parent for p in root.rglob("summary.json"))


def _arms(meta: dict[str, Any]) -> list[str]:
    arms = meta.get("arms") or []
    return list(arms) if len(arms) > 1 else ["arm"]


def _views(arms: list[str]) -> list[str]:
    return (
        ["agentview", "wrist"]
        if arms == ["arm"]
        else ["agentview", *(f"wrist_{a}" for a in arms)]
    )


def _task(meta: dict[str, Any]) -> str:
    for key in ("instruction", "language", "task_language", "task"):
        if isinstance(meta.get(key), str) and meta[key]:
            return meta[key]
    raise ValueError(f"run {meta.get('run_dir')} names no task")


def _image(path: Path) -> np.ndarray:
    from PIL import Image

    with Image.open(path) as img:
        return np.asarray(img.convert("RGB"), dtype=np.uint8)


def load_runs(
    root: Path | str, *, include_failed: bool = False
) -> list[dict[str, Any]]:
    """The runs to export, each ``{dir, meta, steps}``, all of one robot, task, vocabulary and arms."""
    runs = []
    for run in _runs(Path(root).expanduser().resolve()):
        summary = json.loads((run / "summary.json").read_text(encoding="utf-8"))
        if not summary.get("success") and not include_failed:
            continue
        meta = json.loads((run / "metadata.json").read_text(encoding="utf-8"))
        steps = [
            json.loads(line)
            for line in (run / "actions.jsonl").read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        if steps:
            runs.append({"dir": run, "meta": meta, "steps": steps})
    if not runs:
        raise ValueError(
            f"no {'closed' if include_failed else 'successful'} GUMI runs under {root}"
        )
    first = runs[0]["meta"]
    if not first.get("vocabulary"):
        raise ValueError(
            f"run {runs[0]['dir']} records no unit vocabulary; re-record it"
        )
    for r in runs:
        m = r["meta"]
        for key in ("vocabulary", "arms"):
            if m.get(key) != first.get(key):
                raise ValueError(
                    f"run {r['dir']} has another {key} than {runs[0]['dir']}"
                )
        if _task(m) != _task(first):
            raise ValueError(f"run {r['dir']} is of another task than {runs[0]['dir']}")
    return runs


def features(runs: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    meta, step = runs[0]["meta"], runs[0]["steps"][0]
    arms, vocabulary = _arms(meta), list(meta["vocabulary"])
    out: dict[str, dict[str, Any]] = {
        f"observation.images.{v}": {
            "dtype": "image",
            "shape": _image(runs[0]["dir"] / step[v]).shape,
            "names": ["height", "width", "channel"],
        }
        for v in _views(arms)
    }
    out["observation.state"] = {
        "dtype": "float32",
        "shape": (5 * len(arms),),
        "names": [
            f"{a}.{d}"
            for a in arms
            for d in ("eef_x", "eef_y", "eef_z", "gripper_width", "gripper_closed")
        ],
    }
    out["action"] = {
        "dtype": "float32",
        "shape": (len(vocabulary) * len(arms),),
        "names": [f"{a}.{u}" for a in arms for u in vocabulary],
    }
    for name in ("action_repeat", "actor", "dagger"):
        out[name] = {"dtype": "int64", "shape": (1,), "names": None}
    return out


def frame(run: dict[str, Any], step: dict[str, Any]) -> dict[str, Any]:
    meta = run["meta"]
    arms, vocabulary = _arms(meta), list(meta["vocabulary"])
    per_arm = [step] if arms == ["arm"] else [step[a] for a in arms]
    state, action = [], np.zeros(len(vocabulary) * len(arms), np.float32)
    for k, rec in enumerate(per_arm):
        eef = list(rec.get("ee_pose") or [])[:3]
        eef += [0.0] * (3 - len(eef))
        state += [
            *eef,
            float(rec.get("gripper_width") or 0.0),
            float(bool(rec.get("gripper_closed"))),
        ]
        if rec["token"] not in vocabulary:
            raise ValueError(f"unit {rec['token']!r} is not in the run's vocabulary")
        action[k * len(vocabulary) + vocabulary.index(rec["token"])] = 1.0
    src = step.get("src") if arms == ["arm"] else per_arm[0].get("src")
    return {
        **{
            f"observation.images.{v}": _image(run["dir"] / step[v])
            for v in _views(arms)
        },
        "observation.state": np.asarray(state, np.float32),
        "action": action,
        "action_repeat": np.array([int(step.get("n", 1))], np.int64),
        "actor": np.array([1 if src == "human" else 0], np.int64),
        "dagger": np.array([1 if step.get("dagger") else 0], np.int64),
        "task": _task(meta),
    }


def export_gumi(
    root: Path | str,
    *,
    output_root: Path | str,
    dataset_id: str | None = None,
    include_failed: bool = False,
) -> dict[str, Any]:
    """Export the GUMI runs under ``root`` (a run dir or a task dir) to a LeRobot v3.0 dataset."""
    dataset_id = dataset_id or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    if not _NAME.fullmatch(dataset_id):
        raise ValueError(f"invalid dataset ID: {dataset_id!r}")
    runs = load_runs(root, include_failed=include_failed)
    from lerobot.datasets.lerobot_dataset import CODEBASE_VERSION, LeRobotDataset

    parent = Path(output_root).expanduser().resolve()
    destination, partial = parent / dataset_id, parent / f"{dataset_id}.partial"
    if destination.exists() or partial.exists():
        raise FileExistsError(f"dataset already exists: {destination}")
    parent.mkdir(parents=True, exist_ok=True)
    robot = str(runs[0]["meta"].get("robot") or "unknown")
    repo_id = f"pi-embodied/gumi-{re.sub(r'[^A-Za-z0-9_.-]+', '-', robot)}-{dataset_id}"
    dataset = LeRobotDataset.create(
        repo_id=repo_id,
        root=partial,
        robot_type=robot,
        # GUMI steps are decisions, not a control rate: one frame per unit.
        fps=1,
        features=features(runs),
        use_videos=False,
        image_writer_threads=2,
    )
    frames = 0
    for run in runs:
        for step in run["steps"]:
            dataset.add_frame(frame(run, step))
        dataset.save_episode()
        frames += len(run["steps"])
    dataset.finalize()
    if len(LeRobotDataset(repo_id, root=partial)) != frames:
        raise RuntimeError("LeRobot frame count changed after reopening")
    manifest = {
        "schema_version": 2,
        "codebase_version": CODEBASE_VERSION,
        "repo_id": repo_id,
        "robot": robot,
        "source": "gumi",
        "task": _task(runs[0]["meta"]),
        "source_runs": [str(r["dir"]) for r in runs],
        "episode_count": len(runs),
        "frame_count": frames,
    }
    (partial / "meta" / "flywheel.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(partial, destination)
    return {"dataset_path": str(destination), **manifest}
