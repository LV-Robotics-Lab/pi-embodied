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


"""Flywheel raw episodes and their LeRobot v3.0 export (the export runs where lerobot 0.4 is
installed: the flywheel extra's own environment)."""

from __future__ import annotations

import json

import numpy as np
import pytest

from pi_embodied_services.flywheel import cli
from pi_embodied_services.flywheel.episode import EpisodeWriter, validate_episode
from pi_embodied_services.flywheel.export import features
from pi_embodied_services.flywheel.specs import ROBOTS, select, spec


def obs(s: dict, step: int) -> dict:
    """An observation of ``s``'s arrays: images filled with ``step``, the state counting up."""
    out = {}
    for key, field in s["arrays"].items():
        if key == "actions":
            continue
        shape = field["shape"] or (6, 8, 3)
        out[key] = (
            np.full(shape, step, np.uint8)
            if key in s["image_fields"]
            else np.arange(shape[0], dtype=np.float32) + step
        )
    return out


def record(root, robot: str, path: list[str], metadata: dict, steps: int, solved: bool):
    s = spec(robot)
    writer = EpisodeWriter(
        root / "raw" / robot / "/".join(path),
        metadata={**metadata, "robot": robot},
        spec=s,
        initial_observation=obs(s, 0),
    )
    width = s["arrays"]["actions"]["shape"][0]
    writer.begin_primitive("vla")
    vla = writer.add_proposal("go", np.zeros((steps, width), np.float32))
    for i in range(steps):
        writer.add_transition(
            np.full(width, i, np.float32),
            obs(s, i + 1),
            0.0,
            solved and i == steps - 2,
            False,
            vla_id=vla,
            proposal_index=i,
        )
    return writer.finalize()


def test_every_robot_spec_names_its_dimensions_and_shared_features():
    for robot in ROBOTS:
        s = spec(robot)
        assert s["robot"] == robot
        # Open image sizes resolve from the data; the features are the shared names.
        resolved = {
            **s,
            "arrays": {
                k: {**f, "shape": f["shape"] or (6, 8, 3)}
                for k, f in s["arrays"].items()
            },
        }
        names = set(features(resolved))
        assert {"observation.state", "action", "action_source"} <= names
        assert all(
            n.startswith("observation.images.")
            for n in names - {"observation.state", "action", "action_source"}
        )


def test_select_refuses_paths_outside_the_robot(tmp_path):
    for bad in ("", "..", "../franka", "/etc"):
        with pytest.raises(ValueError):
            select(tmp_path, "libero", bad)


def test_an_open_image_size_is_the_episodes_own(tmp_path):
    path = record(
        tmp_path,
        "robotwin",
        ["demo_randomized", "beat_block_hammer", "seed_000"],
        {
            "task_config": "demo_randomized",
            "task_name": "beat_block_hammer",
            "seed": 0,
            "task_language": "beat the block",
        },
        steps=3,
        solved=True,
    )
    meta = validate_episode(path, spec=spec("robotwin"))
    assert meta["training_step_count"] == 2
    with np.load(path / "transitions.npz") as data:
        assert data["head_images"].shape == (4, 6, 8, 3)


def test_export_writes_lerobot_v3_with_the_shared_feature_names(tmp_path, capsys):
    lerobot = pytest.importorskip("lerobot.datasets.lerobot_dataset")
    meta = {"suite": "libero_10", "task_id": 2, "task_language": "put the bowl away"}
    for seed, solved in ((0, True), (1, False), (2, True)):
        record(
            tmp_path,
            "libero",
            ["libero_10", "task_02", f"seed_{seed:03d}"],
            {**meta, "seed": seed},
            steps=4,
            solved=solved,
        )
    argv = ["export-lerobot", "--data-root", str(tmp_path), "--robot", "libero"]
    assert cli.main([*argv, "--select", "libero_10/task_02", "--dataset-id", "d1"]) == 0
    out = json.loads(capsys.readouterr().out)
    assert out["codebase_version"] == "v3.0"
    assert (out["episode_count"], out["frame_count"]) == (2, 6)
    root = tmp_path / "datasets/lerobot/libero/libero_10/task_02/d1"
    info = json.loads((root / "meta/info.json").read_text())
    assert info["codebase_version"] == "v3.0"
    assert info["features"]["action"]["names"] == spec("libero")["action_names"]
    ds = lerobot.LeRobotDataset(out["repo_id"], root=root)
    item = ds[1]
    assert set(item) >= {
        "observation.images.agentview",
        "observation.images.wrist",
        "observation.state",
        "action",
        "action_source",
        "task",
    }
    assert item["task"] == "put the bowl away"
    assert item["action"].tolist() == [1.0] * 7
    # Episodes of another task in the selection are refused, not mixed in.
    record(
        tmp_path,
        "libero",
        ["libero_10", "task_02", "seed_009"],
        {**meta, "task_id": 3, "seed": 9},
        steps=2,
        solved=True,
    )
    with pytest.raises(ValueError, match="another suite/task_id"):
        cli.main([*argv, "--select", "libero_10/task_02", "--dataset-id", "d2"])


def test_export_takes_open_image_sizes_from_the_episodes(tmp_path, capsys):
    pytest.importorskip("lerobot.datasets.lerobot_dataset")
    meta = {
        "task_config": "demo_randomized",
        "task_name": "beat_block_hammer",
        "task_language": "beat the block",
    }
    for seed in (0, 1):
        record(
            tmp_path,
            "robotwin",
            ["demo_randomized", "beat_block_hammer", f"seed_{seed:03d}"],
            {**meta, "seed": seed},
            steps=3,
            solved=True,
        )
    argv = ["export-lerobot", "--data-root", str(tmp_path), "--robot", "robotwin"]
    assert cli.main([*argv, "--select", "demo_randomized/beat_block_hammer"]) == 0
    out = json.loads(capsys.readouterr().out)
    info = json.loads((tmp_path / out["dataset_path"] / "meta/info.json").read_text())
    assert info["features"]["observation.images.cam_high"]["shape"] == [6, 8, 3]
    assert out["frame_count"] == 4


def gumi_run(root, name: str, success: bool, tokens: list[str], vocabulary: list[str]):
    """A single-arm GUMI run as packages/embodied/src/gumi's Recorder writes it."""
    from PIL import Image

    run = root / "0926" / "task_0" / name
    for view in ("agentview", "wrist"):
        (run / "images" / view).mkdir(parents=True)
    steps = []
    for i, token in enumerate(tokens):
        files = {}
        for view in ("agentview", "wrist"):
            files[view] = f"images/{view}/{i:04d}.png"
            Image.new("RGB", (8, 6), (i, i, i)).save(run / files[view])
        steps.append(
            {
                "step": i,
                "token": token,
                "gripper_closed": token == "GRASP",
                "ee_pose": [0.1 * i, 0.2, 0.3],
                "gripper_width": 0.04,
                **files,
                "src": "human" if i else "agent",
                **({"dagger": True} if i else {}),
            }
        )
    (run / "actions.jsonl").write_text("".join(json.dumps(s) + "\n" for s in steps))
    meta = {"robot": "maniskill", "task": "pick the cube", "vocabulary": vocabulary}
    (run / "metadata.json").write_text(json.dumps(meta))
    (run / "summary.json").write_text(json.dumps({"success": success}))


def test_gumi_runs_export_as_lerobot_v3_with_one_hot_units(tmp_path, capsys):
    lerobot = pytest.importorskip("lerobot.datasets.lerobot_dataset")
    vocabulary = ["MV_FWD", "MV_DOWN", "GRASP", "DONE"]
    gumi_run(tmp_path, "10-00-00", True, ["MV_FWD", "MV_DOWN", "GRASP"], vocabulary)
    gumi_run(tmp_path, "10-05-00", False, ["MV_FWD"], vocabulary)
    out_root = tmp_path / "out"
    argv = [
        "export-gumi",
        str(tmp_path / "0926" / "task_0"),
        "--output-root",
        str(out_root),
    ]
    assert cli.main([*argv, "--dataset-id", "g1"]) == 0
    out = json.loads(capsys.readouterr().out)
    assert (out["codebase_version"], out["episode_count"], out["frame_count"]) == (
        "v3.0",
        1,
        3,
    )
    info = json.loads((out_root / "g1" / "meta/info.json").read_text())
    assert info["features"]["action"]["names"] == [f"arm.{u}" for u in vocabulary]
    ds = lerobot.LeRobotDataset(out["repo_id"], root=out_root / "g1")
    item = ds[2]
    assert item["action"].tolist() == [0.0, 0.0, 1.0, 0.0]
    assert item["observation.state"].tolist()[3:] == pytest.approx([0.04, 1.0])
    assert (int(item["actor"]), int(item["dagger"])) == (1, 1)
    assert int(ds[0]["actor"]) == 0
    assert set(item) >= {"observation.images.agentview", "observation.images.wrist"}
    # Failed runs join only on request.
    assert cli.main([*argv, "--dataset-id", "g2", "--include-failed"]) == 0
    assert json.loads(capsys.readouterr().out)["episode_count"] == 2
