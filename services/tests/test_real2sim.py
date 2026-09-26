"""Real2sim datagen + standalone training prep: the atomic core, Scheme D on a fake pick-and-place
sim, the dataset tools (make_dataset / merge_shards / check_dataset), train.sh on the vendored
Show-Harness files, the pinned dataset listing and the RoboLab plan/termination glue."""

from __future__ import annotations

import functools
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from pi_embodied_services.finetuned import (
    atomic,
    check_dataset,
    download_dataset,
    make_dataset,
    merge_shards,
)
from pi_embodied_services.robots.maniskill import real2sim_follow as ms_follow
from pi_embodied_services.robots.robolab import real2sim as rl
from pi_embodied_services.robots.robolab import sim as rl_sim

FINETUNED = Path(atomic.__file__).resolve().parent
TABLE = 0.02  # block centre height resting on the table


class FakeSim:
    """A point-mass hand over a table with one block and a coaster: the block rides with the
    closed fingers when they close within 1.5 cm of it and falls to the table (or the coaster)
    when they open. ``success`` = block at rest within 2 cm of the coaster."""

    def __init__(self, block=(0.5, 0.1), coaster=(0.55, -0.1), lag: float = 1.0):
        self.tcp = np.array([0.45, 0.0, 0.2])
        self.block = np.array([*block, TABLE])
        self.coaster = np.array([*coaster, 0.0])
        self.grip = 1.0
        self.held = False
        self.lag = lag
        self.steps = 0
        self.frozen_flag = False
        self.env = SimpleNamespace(
            COASTER_THICKNESS=0.004,
            BLOCK_HALF_SIZE=[0.02, 0.02, 0.02],
            cube=self._actor("block"),
            target=self._actor("coaster"),
        )

    def _actor(self, attr):
        sim = self

        class _P:
            @property
            def pose(self):
                return SimpleNamespace(p=getattr(sim, attr))

        return _P()

    # backend protocol
    def tcp_pos(self):
        return self.tcp.copy()

    def tcp_pose7(self):
        return [*self.tcp.tolist(), 1.0, 0.0, 0.0, 0.0]

    def gripper_width(self):
        if self.grip > 0:
            return 0.08
        return 0.04 if self.held else 0.0

    def apply_delta(self, delta, grip_cmd, max_cmd_m):
        self.steps += 1
        d = np.clip(np.asarray(delta, dtype=float), -max_cmd_m, max_cmd_m) * self.lag
        closing = grip_cmd < 0 <= self.grip
        opening = grip_cmd >= 0 > self.grip
        self.grip = grip_cmd
        if closing and np.linalg.norm(self.tcp - self.block) < 0.015:
            self.held = True
        if opening and self.held:
            self.held = False
            on_coaster = np.linalg.norm(self.block[:2] - self.coaster[:2]) < 0.03
            self.block[2] = TABLE + (0.004 if on_coaster else 0.0)
        self.tcp = self.tcp + d
        self.tcp[2] = max(self.tcp[2], TABLE - 0.005)
        if self.held:
            self.block = self.tcp.copy()

    def grab_frames(self):
        img = np.zeros((8, 8, 3), np.uint8)
        img[0, 0] = int(self.steps) % 255
        return img, img

    def success(self):
        return (not self.held) and float(
            np.linalg.norm(self.block[:2] - self.coaster[:2])
        ) < 0.02

    def frozen(self):
        return self.frozen_flag


class FakeFacadeBackend(ms_follow.FacadeBackend):
    """real2sim_follow's backend with the facade replaced by FakeSim."""

    def __init__(self, sim: FakeSim):
        self.s, self.env, self.env_id = sim, sim.env, "BlockPAP-v1"

    def tcp_pos(self):
        return self.s.tcp_pos()

    def tcp_pose7(self):
        return self.s.tcp_pose7()

    def gripper_width(self):
        return self.s.gripper_width()

    def apply_delta(self, d, g, m):
        self.s.apply_delta(d, g, m)

    def grab_frames(self):
        return self.s.grab_frames()

    def success(self):
        return self.s.success()

    def actor_pos(self, name):
        return (
            {"cube": self.s.block, "target": self.s.coaster}[name].astype(float).copy()
        )


# -- the atomic core -------------------------------------------------------------------------


def test_manhattan_rdp_and_events():
    assert atomic.manhattan_tokens([0.041, -0.012, 0.0], 0.02, 0.011) == [
        "MV_FWD",
        "MV_FWD",
        "MV_LEFT",
    ]
    assert atomic.manhattan_tokens([0.0, 0.0, -0.005], 0.02, 0.011) == []
    line = np.array([[0, 0, 0], [0.05, 0, 0], [0.1, 0, 0], [0.1, 0.1, 0]], float)
    assert atomic.rdp(line) == [0, 2, 3]
    assert atomic.gripper_events([1, 1, -1, -1, 1]) == [[2, "GRASP"], [4, "RELEASE"]]


def test_chase_emits_single_axis_units_and_records_frame_before(tmp_path):
    sim = FakeSim(
        lag=0.5
    )  # half of each command lands: the executor must close the loop
    w = atomic.RolloutWriter(tmp_path / "rollout_000")
    ep = atomic.TokenEpisode(sim, w, atomic.AtomicExec(sim))
    ep.chase(np.array([0.51, -0.04, 0.2]), tol=0.008)
    recs = [json.loads(x) for x in (tmp_path / "rollout_000/actions.jsonl").open()]
    toks = [r["token"] for r in recs]
    assert toks[:3] == ["MV_FWD"] * 3 and set(toks) == {"MV_FWD", "MV_LEFT"}
    for a, b in zip(recs, recs[1:]):
        d = np.subtract(b["ee_pose"][:3], a["ee_pose"][:3])
        ax, sg = atomic.TOKEN_AXIS[a["token"]]
        assert abs(d[ax] * sg - 0.02) < 0.0015 and np.abs(np.delete(d, ax)).max() < 1e-9
    assert (tmp_path / "rollout_000/wrist/0000.png").exists()
    w.close({"task_text": "t"})
    meta = json.loads((tmp_path / "rollout_000/metadata.json").read_text())
    assert meta["num_steps"] == len(toks) and meta["token_counts"]["MV_FWD"] == 3


def test_emit_stops_on_frozen_sim_and_budget(tmp_path):
    sim = FakeSim()
    ep = atomic.TokenEpisode(
        sim, atomic.RolloutWriter(tmp_path / "r"), atomic.AtomicExec(sim), max_tokens=1
    )
    ep.emit("MV_UP")
    with pytest.raises(atomic.TokenBudgetExceeded):
        ep.emit("MV_UP")
    ep.max_tokens = 5
    sim.frozen_flag = True
    with pytest.raises(atomic.EpisodeComplete):
        ep.emit("MV_UP")
    assert ep.writer.step == 1


# -- ManiSkill Scheme D on the fake sim ------------------------------------------------------


def test_scheme_d_records_then_follows_to_a_passing_rollout(tmp_path):
    backend = FakeFacadeBackend(FakeSim())
    rec = atomic.DemoRecorder(backend)
    rec.capture()
    assert ms_follow.scripted_demo(rec, backend, hover=0.06)
    track = {"task": "put the block on the coaster", **rec.track()}
    assert [e[1] for e in track["events"]] == ["GRASP", "RELEASE"]

    backend2 = FakeFacadeBackend(FakeSim())
    out = tmp_path / "blockpap_follow/rollout_000"
    result = ms_follow.follow_track(backend2, track, out, 0.02)
    writer = result.pop("writer")
    writer.close({"task_text": track["task"], "step_m": 0.02, **result})
    assert result["success"], result
    toks = writer.tokens
    assert toks.count("GRASP") == 1 and toks.count("RELEASE") == 1
    assert toks[-1] == "MV_UP"
    assert check_dataset.check(tmp_path) == 0
    assert ms_follow.layout_drift(backend2, {"cube": {"p": [0, 0, 0]}}) > 0.1


# -- dataset tools ---------------------------------------------------------------------------


def fake_rollout(d: Path, tokens: list[str], success=True, task="put it in the bowl"):
    d.mkdir(parents=True)
    w = atomic.RolloutWriter(d)
    z = 0.3
    x = 0.4
    for t in tokens:
        closed = w.tokens.count("GRASP") > w.tokens.count("RELEASE")
        img = np.full((4, 4, 3), len(w.tokens), np.uint8)
        w.add_step(
            t, atomic.token_kind(t), img, img, closed, [x, 0, z, 1, 0, 0, 0], 0.03
        )
        if t in atomic.TOKEN_AXIS:
            ax, sg = atomic.TOKEN_AXIS[t]
            if ax == 2:
                z += 0.02 * sg
            elif ax == 0:
                x += 0.02 * sg
    w.close({"task_text": task, "task": task, "step_m": 0.02, "success": success})


GOOD = ["MV_DOWN", "MV_DOWN", "GRASP", "MV_UP", "MV_FWD", "RELEASE", "MV_UP", "MV_UP"]


def test_check_dataset_gates(tmp_path):
    fake_rollout(tmp_path / "ok/rollout_000", GOOD)
    assert check_dataset.check(tmp_path / "ok") == 0
    fake_rollout(tmp_path / "bad/rollout_000", GOOD[:-2])  # ends on RELEASE
    fake_rollout(tmp_path / "bad/rollout_001", GOOD, success=False)
    problems = check_dataset.check_episode(tmp_path / "bad/rollout_000")
    assert any("ends on RELEASE" in p for p in problems)
    assert check_dataset.check(tmp_path) == 1


def test_merge_shards_renumbers_and_traces(tmp_path):
    for shard in ("s0", "s1"):
        for i in range(2):
            fake_rollout(tmp_path / shard / f"rollout_{i:03d}", GOOD)
    n = merge_shards.merge([tmp_path / "s0", tmp_path / "s1"], tmp_path / "all")
    assert n == 4
    meta = json.loads((tmp_path / "all/rollout_003/metadata.json").read_text())
    assert (meta["shard"], meta["shard_rollout"]) == ("s1", "rollout_001")


def test_make_dataset_runs_the_vendored_converter(tmp_path):
    fake_rollout(tmp_path / "root/cube/rollout_000", GOOD)
    fake_rollout(tmp_path / "root/cube/rollout_001", GOOD)
    assert make_dataset.main(["--root", str(tmp_path / "root")]) == 0
    stats = json.loads((tmp_path / "root/stats.json").read_text())["cube"]
    assert stats["episodes"] == 2 and stats["samples"] == 2 * (len(GOOD) + 1)
    assert stats["lf_samples"] == stats["samples"]
    assert (
        stats["per_token_disp_mm_mean"] == 20.0
        and stats["adjacent_opposite_pairs"] == 0
    )
    sample = json.loads((tmp_path / "root/cube/rollout_lite.json").read_text())[0]
    assert sample["output"] == "MV_DOWN" and sample["instruction"].startswith(
        "<image><image>"
    )


def test_train_sh_prepares_from_rollouts_with_the_vendored_files(tmp_path):
    fake_rollout(tmp_path / "gen/cube_follow/rollout_000", GOOD)
    env = {
        **os.environ,
        "NAME": "t",
        "DATA": str(tmp_path / "data"),
        "STEP": "prepare",
        "PYTHON": sys.executable,
        "LOCK": "",
    }
    env.pop("SH", None)
    proc = subprocess.run(
        [
            "bash",
            str(FINETUNED / "train.sh"),
            "--from-rollouts",
            str(tmp_path / "gen/cube_follow"),
        ],
        env=env,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, proc.stderr
    samples = json.loads((tmp_path / "data/rollouts.json").read_text())
    assert len(samples) == len(GOOD) + 1 and samples[-1]["output"] == "DONE"
    assert "put it in the bowl" in samples[0]["instruction"]


def test_vendored_files_carry_provenance():
    root = FINETUNED / "showharness"
    for f in ("train/data_preparation/rollouts_to_alpaca.py", "train/scripts/train.sh"):
        head = (root / f).read_text().splitlines()[:3]
        assert any("Show-Harness Authors" in x for x in head)
        assert any("Modified by pi-embodied" in x and "@137d571" in x for x in head)
    assert (root / "LICENSE").read_text().startswith("\n") or "Apache" in (
        root / "LICENSE"
    ).read_text()


# -- the pinned dataset listing --------------------------------------------------------------


def test_dataset_listing_must_match_the_pinned_digest(monkeypatch):
    sib = [
        {
            "rfilename": "sim/rollouts.json",
            "size": 3,
            "blobId": "x",
            "lfs": {"sha256": "ab"},
        },
        {"rfilename": "sim/a.png", "size": 2, "blobId": "cd"},
        {"rfilename": "real/b.png", "size": 1, "blobId": "ef"},
    ]
    lines = "sim/a.png\t2\tgit-sha1:cd\nsim/rollouts.json\t3\tsha256:ab\n"
    digest = hashlib.sha256(lines.encode()).hexdigest()
    monkeypatch.setitem(download_dataset.SPLITS, "sim", (2, 5, digest))
    assert [e[0] for e in download_dataset.pinned_listing(sib, "sim")] == [
        "sim/rollouts.json",
        "sim/a.png",
    ]
    sib[1]["blobId"] = "tampered"
    with pytest.raises(SystemExit, match="pinned"):
        download_dataset.pinned_listing(sib, "sim")


# -- RoboLab glue ----------------------------------------------------------------------------


def test_task_targets_and_plan():
    def pick_and_place(**_):
        return True

    cfg = SimpleNamespace(
        subtasks=[
            SimpleNamespace(
                conditions={
                    "banana": [functools.partial(pick_and_place, container="bowl")]
                }
            )
        ]
    )
    targets = rl_sim.task_targets(cfg)
    assert targets == {"objects": ["banana"], "container": "bowl"}
    assert rl.resolve_plan(targets) == {
        "object": "banana",
        "target": "bowl",
        "target_kind": "container",
    }
    with pytest.raises(rl.UnsupportedTask):
        rl.resolve_plan({"objects": ["a", "b"], "container": "bowl"})
    assert rl_sim.task_targets(SimpleNamespace()) == {}


def test_place_and_carry_heights_use_live_geometry():
    geo = {
        "cube": (np.array([0.5, 0.1, 0.25]), 0.03),
        "bowl": (np.array([0.6, -0.1, 0.03]), 0.03),
    }
    b = SimpleNamespace(
        object_centroid=lambda n: geo[n][0],
        object_extent=lambda n: (geo[n][0][2] - geo[n][1], geo[n][0][2] + geo[n][1]),
        tcp_pos=lambda: np.array([0.51, 0.1, 0.38]),
    )
    place = rl.place_tcp(b, "cube", "bowl")
    # the object (1 cm behind the flange in x) lands on the bowl centre; underside 2 cm over the rim
    assert np.allclose(place[:2], [0.61, -0.1])
    assert place[2] == pytest.approx(0.06 + 0.02 + (0.38 - 0.22))
    assert rl.carry_z(b, "cube", "bowl", grasp_flange_z=0.16) == pytest.approx(
        0.06 + 0.13 + 0.12
    )


def test_suspended_termination_is_evaluated_on_demand():
    done = SimpleNamespace(
        time_out=False, func=lambda env, flag: flag["v"], params={"flag": {"v": 0}}
    )
    tout = SimpleNamespace(time_out=True)
    manager = SimpleNamespace(
        _term_names=["success", "time_out"],
        _term_cfgs=[done, tout],
        _class_term_cfgs=[done],
    )
    backend = rl.FacadeBackend.__new__(rl.FacadeBackend)
    backend.env = SimpleNamespace(termination_manager=manager)
    backend._suspended = []
    assert backend.suspend_task_termination() == ["success"]
    assert manager._term_names == ["time_out"] and manager._class_term_cfgs == []
    assert backend.success() is False
    done.params["flag"]["v"] = 1
    assert backend.success() is True
