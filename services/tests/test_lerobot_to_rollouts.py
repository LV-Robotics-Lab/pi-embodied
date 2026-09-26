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

"""LeRobot v3.0 datasets -> Show-Harness rollouts (finetuned/lerobot_to_rollouts.py). The
dataset-backed tests need lerobot 0.4 (the flywheel extra's environment) and node (the frames go
through packages/embodied/src/finetuned/transform.ts)."""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest

from pi_embodied_services.finetuned import lerobot_to_rollouts as l2r
from pi_embodied_services.flywheel import cli
from pi_embodied_services.flywheel.episode import EpisodeWriter
from pi_embodied_services.flywheel.specs import spec

PREPARE = l2r.PI_ROOT / "packages/embodied/src/finetuned/prepare.ts"
VOCABULARY = ["MV_FWD", "MV_DOWN", "ROTATE_CW", "RT_YAW_CW", "GRASP", "RELEASE", "DONE"]


def needs_node():
    if shutil.which("node") is None:
        pytest.skip("node is not on PATH")


def frame_png(h: int, w: int, seed: int) -> "np.ndarray":
    """A frame every pixel of which differs from its neighbours, so a transform slip shows."""
    y, x = np.mgrid[0:h, 0:w]
    return np.stack(
        [(x * 3 + y + seed) % 256, (x * y + seed) % 256, (x ^ y) & 255], -1
    ).astype(np.uint8)


def gumi_run(
    root: Path,
    name: str,
    success: bool,
    tokens: list[str],
    robot="maniskill",
    env_id: str | None = None,
):
    """A single-arm GUMI run (packages/embodied/src/gumi's Recorder), frames 64x48 / 48x64."""
    from PIL import Image

    run = root / "0926" / "task_0" / name
    for view in ("agentview", "wrist"):
        (run / "images" / view).mkdir(parents=True)
    steps = []
    for i, token in enumerate(tokens):
        files = {}
        for view, (h, w) in (("agentview", (48, 64)), ("wrist", (60, 80))):
            files[view] = f"images/{view}/{i:04d}.png"
            Image.fromarray(frame_png(h, w, 7 * i + len(view))).save(run / files[view])
        steps.append(
            {
                "step": i,
                "token": token,
                "kind": "move",
                "gripper_closed": token == "GRASP",
                "ee_pose": [round(0.1 * i, 4), 0.2, 0.3],
                "gripper_width": 0.04,
                **files,
                "time": 1.5 * i,
                "src": "human" if i else "agent",
                **({"dagger": True} if i else {}),
                **({"n": 2} if token == "MV_DOWN" else {}),
            }
        )
    (run / "actions.jsonl").write_text("".join(json.dumps(s) + "\n" for s in steps))
    meta = {"robot": robot, "task": "pick the cube", "vocabulary": VOCABULARY}
    if env_id:
        meta["robot_task"] = {"env-id": env_id}
    (run / "metadata.json").write_text(json.dumps(meta))
    (run / "summary.json").write_text(json.dumps({"success": success}))
    return run


def rows_of(rollout: Path) -> list[dict]:
    return [json.loads(l) for l in (rollout / "actions.jsonl").read_text().splitlines()]


def test_recorded_tokens_read_the_one_hot_and_gumis_fields():
    names = [f"arm.{u}" for u in VOCABULARY]
    frames = [
        {
            "action": [1.0 if u == t else 0.0 for u in VOCABULARY],
            "observation.state": [0.1, 0.2, 0.3, 0.04, float(t == "GRASP")],
            "actor": i,
            "dagger": i,
            "action_repeat": 1 + i,
        }
        for i, t in enumerate(["MV_FWD", "GRASP"])
    ]
    rows = l2r.recorded_tokens(frames, names)
    assert [r["token"] for r in rows] == ["MV_FWD", "GRASP"]
    assert rows[0] == {
        "step": 0,
        "token": "MV_FWD",
        "gripper_closed": False,
        "ee_pose": [0.1, 0.2, 0.3],
        "gripper_width": 0.04,
        "src": "agent",
    }
    assert (
        rows[1]["gripper_closed"],
        rows[1]["src"],
        rows[1]["n"],
        rows[1]["dagger"],
    ) == (
        True,
        "human",
        2,
        True,
    )
    with pytest.raises(ValueError, match="dual-arm"):
        l2r.recorded_tokens(frames, [f"left.{u}" for u in VOCABULARY])
    with pytest.raises(ValueError, match="not a one-hot"):
        l2r.recorded_tokens([{**frames[0], "action": [0.0] * len(VOCABULARY)}], names)


def test_quantized_tokens_walk_the_2cm_lattice_and_flip_the_gripper():
    state_names = spec("libero")["state_names"]
    action_names = spec("libero")["action_names"]

    def frame(x, y, z, gripper, source=1):
        return {
            "observation.state": [x, y, z, 0, 0, 0, 0, 0],
            "action": [0, 0, 0, 0, 0, 0, gripper],
            "action_source": source,
        }

    frames = [
        frame(0.000, 0.0, 0.5, -1),  # start, open
        frame(0.007, 0.0, 0.5, -1),  # 7 mm: below half a step, no token yet
        frame(0.012, 0.0, 0.5, -1),  # 12 mm: rounds to one MV_FWD, decided at frame 0
        frame(
            0.041, -0.021, 0.5, -1
        ),  # +29 mm x, -21 mm y from the lattice point: FWD, LEFT
        frame(0.041, -0.021, 0.5, 1),  # gripper command closes: GRASP at this frame
        frame(0.041, -0.021, 0.5, 1),
        frame(0.041, -0.021, 0.545, 1, 0),  # 45 mm up: two MV_UP from the grasp frame
        frame(0.041, -0.021, 0.549, 1),  # 4 mm residual: dropped
    ]
    rows = l2r.quantized_tokens(frames, state_names, action_names)
    assert [(r["token"], r["step"], r["gripper_closed"]) for r in rows] == [
        ("MV_FWD", 0, False),
        ("MV_FWD", 2, False),
        ("MV_LEFT", 2, False),
        ("GRASP", 4, True),
        ("MV_UP", 4, True),
        ("MV_UP", 4, True),
    ]
    assert rows[0]["ee_pose"] == [0.0, 0.0, 0.5] and rows[3]["kind"] == "gripper"
    assert {r["src"] for r in rows} == {"vla"}
    with pytest.raises(ValueError, match="no eef_x/y/z"):
        l2r.quantized_tokens(frames, ["a"] * 8, action_names)
    with pytest.raises(ValueError, match="no 'gripper'"):
        l2r.quantized_tokens(frames, state_names, ["dx"] * 7)


def test_slug_is_prepare_ts_task_dir():
    assert l2r.slug("Pick up the banana, place it in the bowl!") == (
        "pick_up_the_banana_place_it_in_the_bowl"
    )
    assert l2r.slug("***") == "task"


def test_gumi_dataset_round_trips_to_prepare_ts_bytes(tmp_path, capsys):
    pytest.importorskip("lerobot.datasets.lerobot_dataset")
    needs_node()
    tokens = ["MV_FWD", "MV_DOWN", "RT_YAW_CW", "ROTATE_CW", "GRASP", "RELEASE"]
    good = gumi_run(tmp_path, "10-00-00", True, tokens)
    gumi_run(tmp_path, "10-05-00", False, ["MV_FWD", "GRASP"])
    argv = [
        "export-gumi",
        str(tmp_path / "0926/task_0"),
        "--output-root",
        str(tmp_path / "ds"),
    ]
    assert cli.main([*argv, "--dataset-id", "g", "--include-failed"]) == 0
    out = json.loads(capsys.readouterr().out)
    assert (out["episode_count"], out["successes"]) == (2, [True, False])
    ds = tmp_path / "ds/g"

    # Success only, v3: the RT turn is dropped; ROTATE_CW stays for rollouts_to_alpaca to skip.
    summary = l2r.convert(ds, tmp_path / "v3", node="node")
    assert (summary["rollouts"], summary["steps"], summary["skipped"]) == (1, 5, [1])
    rollout = tmp_path / "v3/pick_the_cube/rollout_000"
    rows = rows_of(rollout)
    assert [r["token"] for r in rows] == [
        "MV_FWD",
        "MV_DOWN",
        "ROTATE_CW",
        "GRASP",
        "RELEASE",
    ]
    assert [r["agentview"] for r in rows] == [
        f"agentview/{i:04d}.png" for i in range(5)
    ]
    assert rows[1]["n"] == 2 and rows[0]["src"] == "agent" and rows[1]["dagger"] is True
    assert rows[3]["gripper_closed"] is True and rows[0]["ee_pose"] == [0.0, 0.2, 0.3]
    meta = json.loads((rollout / "metadata.json").read_text())
    assert meta["task_text"] == "pick the cube" and meta["prompt_version"] == "v3"
    assert (meta["source_episode_index"], meta["source_run"]) == (0, str(good))
    assert (meta["repo_id"], meta["token_source"]) == (out["repo_id"], "recorded")
    assert meta["views"]["wrist"] == "rot=0,flip=vertical,crop=1.3333,square=256"

    # prepare.ts on the recording itself writes the same PNG bytes and the same rows, up to the
    # fields the dataset does not hold (kind, time).
    subprocess.run(
        [
            "node",
            "--experimental-strip-types",
            str(PREPARE),
            "--out",
            str(tmp_path / "ref"),
            str(good),
        ],
        check=True,
        capture_output=True,
    )
    ref = tmp_path / "ref/pick_the_cube/rollout_000"
    ref_rows = rows_of(ref)
    assert len(ref_rows) == 5
    for mine, theirs in zip(rows, ref_rows):
        for view in ("agentview", "wrist"):
            assert (rollout / mine[view]).read_bytes() == (
                ref / theirs[view]
            ).read_bytes()
        assert {k: v for k, v in theirs.items() if k not in ("kind", "time")} == mine

    # v5 keeps the turn; failures join on request, numbered after the successes.
    summary = l2r.convert(
        ds, tmp_path / "v5", prompt="v5", include_failures=True, node="node"
    )
    assert (summary["rollouts"], summary["steps"]) == (2, 8)
    assert [
        r["token"] for r in rows_of(tmp_path / "v5/pick_the_cube/rollout_000")
    ] == tokens
    assert [r["token"] for r in rows_of(tmp_path / "v5/pick_the_cube/rollout_001")] == [
        "MV_FWD",
        "GRASP",
    ]
    assert (
        json.loads(
            (tmp_path / "v5/pick_the_cube/rollout_001/metadata.json").read_text()
        )["prompt_version"]
        == "v5"
    )
    with pytest.raises(ValueError, match="unknown prompt version"):
        l2r.convert(ds, tmp_path / "v9", prompt="v9")
    with pytest.raises(ValueError, match="no camera observation.images.head"):
        l2r.convert(
            ds,
            tmp_path / "x",
            cameras=("observation.images.head", "observation.images.wrist"),
        )


def test_vla_dataset_is_quantized_with_explicit_views(tmp_path, capsys):
    pytest.importorskip("lerobot.datasets.lerobot_dataset")
    needs_node()
    s = spec("libero")
    xs = [0.0, 0.011, 0.025, 0.025, 0.025]
    grip = [-1, -1, -1, 1, 1]

    def obs(i):
        return {
            "main_images": frame_png(256, 256, i),
            "wrist_images": frame_png(256, 256, 100 + i),
            "states": np.array([xs[i], 0.1, 0.5, 0, 0, 0, 0, 0], np.float32),
        }

    meta = {
        "suite": "libero_10",
        "task_id": 2,
        "seed": 0,
        "task_language": "put the bowl away",
    }
    writer = EpisodeWriter(
        tmp_path / "raw/libero/libero_10/task_02/seed_000",
        metadata=meta,
        spec=s,
        initial_observation=obs(0),
    )
    writer.begin_primitive("vla")
    vla = writer.add_proposal("go", np.zeros((4, 7), np.float32))
    for i in range(4):
        action = np.array([0, 0, 0, 0, 0, 0, grip[i]], np.float32)
        writer.add_transition(
            action, obs(i + 1), 0.0, i == 3, False, vla_id=vla, proposal_index=i
        )
    writer.finalize()
    argv = ["export-lerobot", "--data-root", str(tmp_path), "--robot", "libero"]
    assert cli.main([*argv, "--select", "libero_10/task_02", "--dataset-id", "d"]) == 0
    out = json.loads(capsys.readouterr().out)
    ds = Path(out["dataset_path"])
    views = {"agentview": "square=64", "wrist": "flip=both,crop=1.3333,square=64"}
    summary = l2r.convert(ds, tmp_path / "roll", views=views, node="node")
    assert (summary["rollouts"], summary["steps"]) == (1, 2)
    rollout = tmp_path / "roll/put_the_bowl_away/rollout_000"
    rows = rows_of(rollout)
    # 25 mm of +x rounds to one MV_FWD decided at frame 0; the gripper closes on action 3.
    assert [(r["token"], r["step"], r["gripper_closed"]) for r in rows] == [
        ("MV_FWD", 0, False),
        ("GRASP", 3, True),
    ]
    meta = json.loads((rollout / "metadata.json").read_text())
    assert (meta["token_source"], meta["step_m"], meta["robot"]) == (
        "quantized",
        0.02,
        "libero",
    )
    assert meta["source_episode_id"] == out["source_episode_ids"][0]
    assert meta["views"] == {
        "agentview": "rot=0,flip=none,square=64",
        "wrist": "rot=0,flip=both,crop=1.3333,square=64",
    }
    from PIL import Image

    with Image.open(rollout / "wrist/0001.png") as img:
        assert img.size == (64, 64)
    # The sampled frame is the dataset's frame 3, not the first one.
    with (
        Image.open(rollout / "agentview/0001.png") as img,
        Image.open(rollout / "agentview/0000.png") as first,
    ):
        assert np.asarray(img).tobytes() != np.asarray(first).tobytes()


def test_real2sim_rig_keeps_its_raw_views(tmp_path, capsys):
    """prepare.ts picks the transform by the run's robot_task env-id (a real2sim rig's frames are
    already the training frames); the GUMI export carries it for the converter to do the same."""
    pytest.importorskip("lerobot.datasets.lerobot_dataset")
    needs_node()
    gumi_run(tmp_path, "10-00-00", True, ["MV_FWD", "GRASP"], env_id="BlockPAP-v1")
    argv = ["export-gumi", str(tmp_path / "0926/task_0"), "--output-root"]
    assert cli.main([*argv, str(tmp_path / "ds"), "--dataset-id", "r"]) == 0
    assert json.loads(capsys.readouterr().out)["env_id"] == "BlockPAP-v1"
    l2r.convert(tmp_path / "ds/r", tmp_path / "roll", node="node")
    rollout = tmp_path / "roll/pick_the_cube/rollout_000"
    meta = json.loads((rollout / "metadata.json").read_text())
    assert meta["views"] == {"agentview": "rot=0,flip=none", "wrist": "rot=0,flip=none"}
    from PIL import Image

    with Image.open(rollout / "wrist/0000.png") as img:
        assert np.asarray(img).tobytes() == frame_png(60, 80, len("wrist")).tobytes()
