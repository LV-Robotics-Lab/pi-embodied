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

"""The RoboCasa365 task table and the env server's manifest resets (no simulator)."""

from __future__ import annotations

import json
import os

import numpy as np
import pytest

from pi_embodied_services.robots.robocasa import env_server, tasks


def test_table_shape():
    """317 tasks, sorted and unique; 65 atomic; the 50 Target50 tasks; 50 distinct
    int32 scene seeds per task; 634 env ids."""
    t = tasks.load_table()
    names = [x["name"] for x in t["tasks"]]
    assert len(names) == 317 and names == sorted(set(names))
    assert sum(x["kind"] == "atomic" for x in t["tasks"]) == 65
    assert sum(x["target50"] for x in t["tasks"]) == 50
    for x in t["tasks"]:
        assert len(set(x["manifest"])) == 50 == t["scenes_per_task"]
        assert all(0 <= s < 2**31 for s in x["manifest"])
        assert x["horizon"] > 0 and x["instruction"]
        assert x["manifest"] == tasks.manifest_seeds(x["name"])
    ids = {tasks.env_id(n, s) for n in names for s in t["splits"]}
    assert len(ids) == 634
    assert tasks.dump_table(t) == tasks.TABLE.read_text(encoding="utf-8")


def test_manifest_seeds_are_openetas_sampling():
    """A SeedSequence of the master seed and the task name's SHA-256, 50 draws without
    replacement: stable whatever else is in the table, and equal to OpenETA's manifest."""
    assert tasks.manifest_seeds("OpenDrawer")[:2] == [1995668784, 658165347]
    assert tasks.manifest_seeds("OpenDrawer") != tasks.manifest_seeds("CloseDrawer")
    assert len(set(tasks.manifest_seeds("OpenDrawer", 7))) == 7
    assert tasks.manifest_seeds("OpenDrawer", master_seed=1) != tasks.manifest_seeds(
        "OpenDrawer"
    )


def test_build_table_from_a_registry():
    registry = {
        "all_tasks": ["B", "A", "C"],
        "all_atomic_tasks": ["A"],
        "target50": ["C"],
    }
    t = tasks.build_table(
        registry, horizon=lambda n: 100, instruction=lambda n: f"do {n}", version="9"
    )
    assert [x["name"] for x in t["tasks"]] == ["A", "B", "C"]
    assert [x["kind"] for x in t["tasks"]] == ["atomic", "composite", "composite"]
    assert [x["target50"] for x in t["tasks"]] == [False, False, True]
    assert t["tasks"][0] == {
        "name": "A",
        "kind": "atomic",
        "target50": False,
        "horizon": 100,
        "instruction": "do A",
        "manifest": tasks.manifest_seeds("A"),
    }
    assert t["robocasa_version"] == "9" and t["schema"] == tasks.SCHEMA
    assert json.loads(tasks.dump_table(t)) == t
    with pytest.raises(ValueError):
        tasks.build_table(
            {**registry, "all_tasks": ["A", "A"]}, lambda n: 1, lambda n: "", ""
        )


def test_describe_takes_the_docstring_up_to_args():
    class Env:
        """
        Prepare Coffee: composite task.

        Steps:
            Pick the mug,
            press the button.

        Args:
            cab_id (str): ignored
        """

    assert (
        tasks.describe(Env)
        == "Prepare Coffee: composite task.\nSteps: Pick the mug, press the button."
    )
    assert tasks.describe(type("AirDryFruit", (), {})) == "Air dry fruit."
    assert tasks.describe(type("Task2Go", (), {"__doc__": "  \n"})) == "Task2 go."


def test_list_tasks_and_scene_seed():
    t = tasks.load_table()
    pre = tasks.list_tasks(t, "pretrain")
    assert len(pre) == 317
    assert pre[0]["split"] == "pretrain"
    assert pre[0]["env_id"] == f"robocasa365/pretrain/{pre[0]['name']}"
    assert {"name", "kind", "horizon", "instruction", "manifest"} <= set(pre[0])
    open_ = tasks.find_task(t, "OpenDrawer")
    assert tasks.scene_seed(t, "OpenDrawer", "target", 0) == open_["manifest"][0]
    assert tasks.scene_seed(t, "OpenDrawer", "pretrain", 49) == open_["manifest"][49]
    with pytest.raises(ValueError):
        tasks.list_tasks(t, "all")
    with pytest.raises(ValueError):
        tasks.scene_seed(t, "OpenDrawer", "all", 0)
    with pytest.raises(ValueError):
        tasks.scene_seed(t, "OpenDrawer", "target", 50)
    with pytest.raises(KeyError):
        tasks.scene_seed(t, "OpenDrawr", "target", 0)


class _Env:
    """A robosuite env stand-in: records the rng it was reset with."""

    def __init__(self):
        self.rng = np.random.default_rng(123)
        self.seed = None
        self.resets = []
        self.closed = False

    def reset(self):
        self.resets.append(int(self.rng.integers(2**31)))
        return {"obs": len(self.resets)}

    def close(self):
        self.closed = True


def _facade(monkeypatch, task="OpenDrawer", split="target", seed=5, scene=None):
    """A RoboCasaEnvFacade whose ``_make`` builds a stand-in env instead of robosuite."""
    made = []

    def make(self, task_name, split, seed, scene):
        if scene is not None:
            seed = tasks.scene_seed(self.table, task_name, split, scene)
        else:
            tasks.find_task(self.table, task_name)
        if self.env is not None:
            self.env.close()
        self.task_name, self.split, self.seed, self.scene = (
            task_name,
            split,
            seed,
            scene,
        )
        self.env = _Env()
        made.append((task_name, split, seed, scene))
        self._meta = {
            "task_name": task_name,
            "split": split,
            "seed": seed,
            "scene": scene,
            "env_id": tasks.env_id(task_name, split),
        }

    monkeypatch.setattr(env_server.RoboCasaEnvFacade, "_make", make)
    monkeypatch.delenv("RLDX_RESET_SEED", raising=False)
    f = env_server.RoboCasaEnvFacade(task, split=split, seed=seed, scene=scene)
    return f, made


def test_reset_reseeds_every_manifest_scene_reset(monkeypatch):
    """With a scene, every reset samples the same kitchen: the env rng is reseeded with
    the scene's seed before each reset, as RLDX_RESET_SEED does."""
    f, made = _facade(monkeypatch, scene=2)
    t = f.table
    seed = tasks.scene_seed(t, "OpenDrawer", "target", 2)
    assert made == [("OpenDrawer", "target", seed, 2)]
    assert f.get_env_meta()["seed"] == seed and f.get_env_meta()["scene"] == 2
    f.reset()
    f.reset()
    assert f.env.resets[0] == f.env.resets[1]
    assert f.env.seed == seed
    # Another scene of the same task: no rebuild, a new seed.
    f.reset(scene=3)
    assert len(made) == 1
    assert f.seed == tasks.scene_seed(t, "OpenDrawer", "target", 3)
    assert f.get_env_meta()["scene"] == 3 and f.get_env_meta()["seed"] == f.seed
    assert f.env.resets[2] != f.env.resets[1]
    # Another task or split rebuilds the env (closing the old one) at the same scene.
    old = f.env
    f.reset(task="CloseDrawer", split="pretrain")
    assert old.closed
    assert made[-1] == (
        "CloseDrawer",
        "pretrain",
        tasks.scene_seed(t, "CloseDrawer", "pretrain", 3),
        3,
    )
    assert f.get_env_meta()["env_id"] == "robocasa365/pretrain/CloseDrawer"
    with pytest.raises(ValueError):
        f.reset(scene=50)
    with pytest.raises(KeyError):
        f.reset(task="Nope")


def test_reset_without_a_scene_seeds_once(monkeypatch):
    """Seed mode is unchanged: the env is seeded at construction and successive resets
    walk its rng (the second reset is the episode's scene), unless RLDX_RESET_SEED."""
    f, made = _facade(monkeypatch, seed=5)
    assert made == [("OpenDrawer", "target", 5, None)]
    assert f.get_env_meta()["scene"] is None
    f.reset()
    f.reset()
    assert f.env.resets[0] != f.env.resets[1]
    monkeypatch.setenv("RLDX_RESET_SEED", "77")
    f.reset()
    f.reset()
    assert f.env.resets[2] == f.env.resets[3] and f.env.seed == 77
    os.environ.pop("RLDX_RESET_SEED", None)
    assert f.list_tasks("target")[0]["env_id"].startswith("robocasa365/target/")
    with pytest.raises(ValueError):
        f.list_tasks("all")
