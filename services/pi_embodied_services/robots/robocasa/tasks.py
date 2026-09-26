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

"""The RoboCasa365 task table: every registered kitchen task, in both splits, with a
manifest of 50 scene seeds per task.

The table (``eval/robocasa365.json``) is generated from the installed robocasa package
(``python -m ...robocasa.tasks --write``) and read by the env server, pi's robocasa robot
and eval.sh, none of which need robocasa importable for that. A task's env id is
``robocasa365/<split>/<Task>`` (317 tasks x 2 splits = 634). Its manifest is OpenETA's
sampling (``sim/robocasa_benchmark.py``): a numpy ``SeedSequence`` of the master seed and
the task name's SHA-256, drawing 50 distinct int32 seeds, so a scene index names the same
scene whatever else is in the table. ``--check`` compares the table with the installed
package and its kitchen assets (the layouts and styles each split samples from).
"""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import re
import sys
from pathlib import Path

import numpy as np

SPLITS = ("pretrain", "target")
SCENES_PER_TASK = 50
MASTER_SEED = 0
SCHEMA = "robocasa365-tasks/1"
TABLE = Path(__file__).with_name("eval") / "robocasa365.json"
# robocasa's ``create_env(split=...)`` layout groups (scene_registry.LAYOUT_GROUPS_TO_IDS):
# target kitchens are layouts 1-10 paired with styles 1-10, pretrain samples layouts 11-60.
LAYOUT_GROUP = {"target": -1, "pretrain": -2}


def env_id(task: str, split: str) -> str:
    return f"robocasa365/{split}/{task}"


def manifest_seeds(
    task: str, n: int = SCENES_PER_TASK, master_seed: int = MASTER_SEED
) -> list[int]:
    """The task's scene seeds, in scene-index order."""
    entropy = int.from_bytes(
        hashlib.sha256(task.encode("utf-8")).digest()[:8], "little"
    )
    rng = np.random.default_rng(np.random.SeedSequence([int(master_seed), entropy]))
    return [int(s) for s in rng.choice(np.iinfo(np.int32).max, size=n, replace=False)]


def describe(env_class) -> str:
    """A task class's docstring up to its ``Args:`` section, one line per paragraph; a
    class without one is described by its name (``AirDryFruit`` -> ``Air dry fruit.``)."""
    doc = inspect.cleandoc(env_class.__doc__ or "")
    doc = doc.split("\nArgs:")[0].strip()
    if not doc:
        words = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", " ", env_class.__name__).lower()
        return words[:1].upper() + words[1:] + "."
    return "\n".join(" ".join(p.split()) for p in doc.split("\n\n") if p.strip())


def _installed():
    """(task set registry, horizon lookup, description lookup, version) of the installed package."""
    import robocasa  # noqa: F401 — registers the kitchen envs
    from robocasa.utils.dataset_registry import TASK_SET_REGISTRY
    from robocasa.utils.dataset_registry_utils import get_task_horizon
    from robosuite.environments.base import REGISTERED_ENVS

    return (
        TASK_SET_REGISTRY,
        get_task_horizon,
        lambda name: describe(REGISTERED_ENVS[name]),
        str(getattr(robocasa, "__version__", "")),
    )


def build_table(registry=None, horizon=None, instruction=None, version="") -> dict:
    """The table from a task set registry (``all_tasks``, ``all_atomic_tasks``,
    ``target50``), else from the installed robocasa package."""
    if registry is None:
        registry, horizon, instruction, version = _installed()
    atomic = set(registry["all_atomic_tasks"])
    target50 = set(registry["target50"])
    names = sorted(registry["all_tasks"])
    if len(set(names)) != len(names):
        raise ValueError("all_tasks lists a task twice")
    return {
        "schema": SCHEMA,
        "benchmark": "RoboCasa365",
        "robocasa_version": version,
        "splits": list(SPLITS),
        "scenes_per_task": SCENES_PER_TASK,
        "master_seed": MASTER_SEED,
        "tasks": [
            {
                "name": name,
                "kind": "atomic" if name in atomic else "composite",
                "target50": name in target50,
                "horizon": int(horizon(name)),
                "instruction": instruction(name),
                "manifest": manifest_seeds(name),
            }
            for name in names
        ],
    }


def dump_table(table: dict) -> str:
    """The table as JSON, one line per manifest."""
    text = json.dumps(table, indent=1)
    return (
        re.sub(
            r'"manifest": \[[^\]]*\]',
            lambda m: (
                '"manifest": ['
                + ", ".join(x.rstrip(",") for x in m.group(0)[13:-1].split())
                + "]"
            ),
            text,
        )
        + "\n"
    )


def load_table(path: Path | str = TABLE) -> dict:
    table = json.loads(Path(path).read_text(encoding="utf-8"))
    if table.get("schema") != SCHEMA:
        raise ValueError(f"{path}: not a {SCHEMA} table")
    return table


def find_task(table: dict, name: str) -> dict:
    for task in table["tasks"]:
        if task["name"] == name:
            return task
    raise KeyError(f"unknown RoboCasa365 task {name!r}")


def list_tasks(table: dict, split: str) -> list[dict]:
    """The split's 317 tasks: name, split, env_id, kind, horizon, instruction, manifest."""
    if split not in table["splits"]:
        raise ValueError(f"split must be one of {table['splits']}, not {split!r}")
    return [
        {"name": t["name"], "split": split, "env_id": env_id(t["name"], split), **t}
        for t in table["tasks"]
    ]


def scene_seed(table: dict, task: str, split: str, scene: int) -> int:
    """The seed of manifest scene ``scene`` (0-based) of ``task`` in ``split``."""
    if split not in table["splits"]:
        raise ValueError(f"split must be one of {table['splits']}, not {split!r}")
    manifest = find_task(table, task)["manifest"]
    if not 0 <= int(scene) < len(manifest):
        raise ValueError(f"scene must be 0..{len(manifest) - 1}, not {scene}")
    return int(manifest[int(scene)])


def check_installed(table: dict) -> list[str]:
    """Differences between the table and the installed package: tasks missing from or
    added to the registry, horizons, manifests, and the kitchen layout and style files
    each split samples from (the ``ROBOCASA_ASSETS_PATH`` assets)."""
    from robocasa.models.scenes import scene_registry as sr

    registry, horizon, instruction, version = _installed()
    installed = build_table(registry, horizon, instruction, version)
    errors = []
    mine = {t["name"]: t for t in table["tasks"]}
    theirs = {t["name"]: t for t in installed["tasks"]}
    for name in sorted(set(mine) ^ set(theirs)):
        errors.append(
            f"{name}: {'not in' if name in mine else 'missing from'} the table"
        )
    for name in sorted(set(mine) & set(theirs)):
        for key in ("kind", "target50", "horizon", "manifest"):
            if mine[name][key] != theirs[name][key]:
                errors.append(f"{name}: {key} differs from the installed package")
    for split in table["splits"]:
        ids = sr.LAYOUT_GROUPS_TO_IDS[LAYOUT_GROUP[split]]
        for i in ids:
            for kind, path in (
                ("layout", sr.get_layout_path(i)),
                ("style", sr.get_style_path(i)),
            ):
                if not Path(path).is_file():
                    errors.append(f"{split}: {kind} {i} missing ({path})")
    return errors


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--table", type=Path, default=TABLE)
    p.add_argument("--write", action="store_true", help="rebuild the table")
    p.add_argument(
        "--check", action="store_true", help="compare with the installed package"
    )
    p.add_argument("--list", metavar="SPLIT", help="print the split's tasks as JSON")
    args = p.parse_args(argv)
    if args.write:
        table = build_table()
        args.table.write_text(dump_table(table), encoding="utf-8")
        print(
            f"{args.table}: {len(table['tasks'])} tasks x {len(table['splits'])} splits"
        )
    table = load_table(args.table)
    if args.list:
        print(json.dumps(list_tasks(table, args.list), indent=1))
    if args.check:
        errors = check_installed(table)
        for e in errors:
            print(e, file=sys.stderr)
        print(f"{len(table['tasks'])} tasks, {len(errors)} differences")
        return 1 if errors else 0
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
