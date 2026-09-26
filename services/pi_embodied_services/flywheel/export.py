# Copyright 2026 The RPent Authors.
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
#
# Modified by pi-embodied: import paths rewritten; LeRobot v3.0 (lerobot 0.4) with the shared
# feature names, any robot's spec, a selection of raw episodes by path.

"""Export successful episodes to a LeRobot v3.0 dataset using a robot's data rules.

Every robot's dataset has the same shape of features, whatever its embodiment:

- ``observation.images.<camera>``: one per camera the spec names (``spec["cameras"]``);
- ``observation.state`` and ``action``: float32 vectors, their dimensions named by the spec;
- ``action_source``: 1 where a VLA chunk produced the action, 0 for a scripted primitive;
- ``task``: the episode's task language (LeRobot's own column).

Their lengths and meanings are the robot's: one dataset holds one embodiment and action space.

A spec may derive its state and action from other recorded arrays (``spec["columns"]``), and may
ask for XPolicyLab's layout (``spec["layout"] == "xpolicylab"``, e.g. RoboTwin's joint space): the
features of XPolicyLab's scripts/transform_lerobot_v30_format.py exactly, so its training scripts
read the dataset as one of their own. There the state and action come first, their motor names in
one nested list, the cameras are channel-first ``(3, H, W)`` mp4 at CRF 18, and there is no
``action_source``.
"""

from __future__ import annotations

import json
import os
import re
from collections.abc import Iterable
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from pi_embodied_services.flywheel.episode import validate_episode

_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]*")


#: XPolicyLab's video quality (its DatasetConfig.video_crf; LeRobot's own default is 30).
XPOLICYLAB_CRF = 18


def _xpolicylab(spec: dict[str, Any]) -> bool:
    return spec.get("layout") == "xpolicylab"


def features(spec: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """The LeRobot features of a robot's dataset."""
    arrays = spec["arrays"]
    if _xpolicylab(spec):
        out = {
            name: {"dtype": "float32", "shape": (len(names),), "names": [list(names)]}
            for name, names in (
                ("observation.state", spec["state_names"]),
                ("action", spec["action_names"]),
            )
        }
        for key, camera in spec["cameras"].items():
            height, width, channels = arrays[key]["shape"]
            out[f"observation.images.{camera}"] = {
                "dtype": "video",
                "shape": (channels, height, width),
                "names": ["channels", "height", "width"],
            }
        return out
    out: dict[str, dict[str, Any]] = {
        f"observation.images.{camera}": {
            "dtype": "image",
            "shape": tuple(arrays[key]["shape"]),
            "names": ["height", "width", "channel"],
        }
        for key, camera in spec["cameras"].items()
    }
    for name, key, names in (
        ("observation.state", "states", spec["state_names"]),
        ("action", "actions", spec["action_names"]),
    ):
        shape = tuple(arrays[key]["shape"])
        if len(names) != shape[0]:
            raise ValueError(f"{name}: {len(names)} dimension names for shape {shape}")
        out[name] = {"dtype": "float32", "shape": shape, "names": list(names)}
    out["action_source"] = {"dtype": "int64", "shape": (1,), "names": None}
    return out


def columns(data: Any, spec: dict[str, Any]) -> tuple[Any, Any]:
    """An episode's state (one per observation) and action (one per step) columns: the recorded
    ``states`` and ``actions``, or the ones the spec derives from its other arrays."""
    if "columns" in spec:
        return spec["columns"](data)
    return data["states"], data["actions"]


def frame(data: Any, index: int, spec: dict[str, Any], task: str) -> dict[str, Any]:
    """Frame ``index`` of an episode's transitions: the observation before action ``index``."""
    states, actions = columns(data, spec)
    out = {
        **{
            f"observation.images.{camera}": data[key][index]
            for key, camera in spec["cameras"].items()
        },
        "observation.state": states[index].astype(np.float32),
        "action": actions[index].astype(np.float32),
        "action_source": np.array([data["action_source"][index]], dtype=np.int64),
        "task": task,
    }
    if _xpolicylab(spec):
        del out["action_source"]
    return out


def _successful_episodes(
    paths: Iterable[Path], *, spec: dict[str, Any], group: tuple[str, ...]
) -> list[tuple[Path, dict[str, Any]]]:
    episodes = []
    for path in paths:
        if not path.is_dir() or path.name.endswith(".partial"):
            continue
        metadata = validate_episode(path, spec=spec)
        if metadata["is_success"]:
            episodes.append((path, metadata))
    if not episodes:
        raise ValueError("no successful episodes found")
    first = episodes[0][1]
    for path, metadata in episodes:
        if any(metadata.get(key) != first.get(key) for key in group):
            raise ValueError(
                f"episode {path} is of another {'/'.join(group)} than {first}"
            )
    return episodes


def export_lerobot(
    episode_paths: Iterable[Path],
    *,
    spec: dict[str, Any],
    repo_id_prefix: str,
    output_root: Path | str,
    dataset_id: str | None = None,
    videos: bool = False,
) -> dict[str, Any]:
    """Validate selected episodes and export their successful training prefixes."""
    dataset_id = dataset_id or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    if not _NAME.fullmatch(dataset_id):
        raise ValueError(f"invalid dataset ID: {dataset_id!r}")
    group = tuple(spec["group"])
    episodes = _successful_episodes(episode_paths, spec=spec, group=group)
    # Image sizes the spec leaves open are the first episode's; LeRobot refuses any other.
    with np.load(episodes[0][0] / "transitions.npz", allow_pickle=False) as data:
        spec = {
            **spec,
            "arrays": {
                key: {**field, "shape": field["shape"] or data[key].shape[1:]}
                for key, field in spec["arrays"].items()
            },
        }

    parent = Path(output_root).expanduser().resolve()
    destination = parent / dataset_id
    partial = parent / f"{dataset_id}.partial"
    if destination.exists() or partial.exists():
        raise FileExistsError(f"dataset already exists: {destination}")

    try:
        from lerobot.datasets.lerobot_dataset import CODEBASE_VERSION, LeRobotDataset
    except ImportError as exc:
        raise RuntimeError(
            "install pi-embodied-services with the 'flywheel' extra"
        ) from exc
    if CODEBASE_VERSION != "v3.0":
        raise RuntimeError(
            f"lerobot writes {CODEBASE_VERSION}; the flywheel extra pins the v3.0 one"
        )
    parent.mkdir(parents=True, exist_ok=True)

    feats = features(spec)
    # XPolicyLab's datasets are video only.
    videos = videos or _xpolicylab(spec)
    if videos:
        for f in feats.values():
            if f["dtype"] == "image":
                f["dtype"] = "video"
    repo_id = f"{repo_id_prefix}-{dataset_id}"
    dataset = LeRobotDataset.create(
        repo_id=repo_id,
        root=partial,
        robot_type=spec["robot_type"],
        fps=spec["fps"],
        features=feats,
        use_videos=videos,
        image_writer_threads=2,
        # XPolicyLab encodes while it adds frames, to set its CRF (configure_video_encoding).
        streaming_encoding=_xpolicylab(spec),
    )
    if _xpolicylab(spec):
        dataset._streaming_encoder.crf = XPOLICYLAB_CRF
    frame_count = 0
    source_ids = []
    tasks = set()
    for path, metadata in episodes:
        count = metadata["training_step_count"]
        task = metadata["task_language"]
        tasks.add(task)
        with np.load(path / "transitions.npz", allow_pickle=False) as npz:
            # One read of each array, not one per frame.
            data = {key: npz[key] for key in npz.files}
            for index in range(count):
                dataset.add_frame(frame(data, index, spec, task))
        dataset.save_episode()
        frame_count += count
        source_ids.append(metadata["episode_id"])
    dataset.finalize()

    reopened = LeRobotDataset(repo_id, root=partial)
    if len(reopened) != frame_count:
        raise RuntimeError("LeRobot frame count changed after reopening")
    first = episodes[0][1]
    manifest = {
        "schema_version": 2,
        "codebase_version": CODEBASE_VERSION,
        "repo_id": repo_id,
        "robot": spec["robot"],
        **{key: first.get(key) for key in group},
        "tasks": sorted(tasks),
        "source_episode_ids": source_ids,
        "episode_count": len(source_ids),
        "frame_count": frame_count,
    }
    (partial / "meta" / "flywheel.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(partial, destination)
    return {"dataset_path": str(destination), **manifest}
