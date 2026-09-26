# Copyright 2026 The Show-Harness Authors.
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
# Modified by pi-embodied: scripts/robolab/check_dataset.py of github.com/showlab/Show-Harness
# @137d571, as a module over any generator's output (ManiSkill and RoboLab).

"""Contract check over a generated dataset: exit code 1 means do not train on it.

    python -m pi_embodied_services.finetuned.check_dataset <root>

``<root>`` holds one directory per task (``<root>/<task>/rollout_*``), or is itself one task
directory. Plain Python, no simulator. Checks what has gone wrong on this pipeline and no
aggregate statistic shows:

* the episode must end on MV_UP, with >= 2 tokens after the last RELEASE: the converter builds
  the terminal DONE sample from the last frame, so ending on RELEASE trains one image to mean
  both "open the fingers" and "the task is over";
* the post-release retreat must have moved (``retreat_travel_m`` when the generator records it);
* no more than ``MAX_STALLED_FRAC`` of the move tokens may be recorded over < 6 mm of travel
  (a blocked arm, or a frozen env, labelled with 2 cm moves);
* between the last GRASP and the last RELEASE the closed fingers must hold something;
* ``metadata.json`` must say success.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

MIN_TOKEN_TRAVEL_M = 0.006
MAX_STALLED_FRAC = 0.05
#: Finger opening below which a gripper commanded closed is holding nothing.
GRASPED_WIDTH_M = 0.005


def check_episode(ep: Path) -> list[str] | None:
    """The episode's problems; None while it is still being written (no metadata.json)."""
    recs = [json.loads(x) for x in (ep / "actions.jsonl").open() if x.strip()]
    if not recs:
        return [f"{ep}: empty actions.jsonl"]
    meta_path = ep / "metadata.json"
    if not meta_path.exists():
        return None
    meta = json.loads(meta_path.read_text())
    tokens = [r["token"] for r in recs]
    problems: list[str] = []
    if tokens[-1] != "MV_UP":
        problems.append(f"{ep}: ends on {tokens[-1]}, expected MV_UP")
    if "RELEASE" not in tokens:
        problems.append(f"{ep}: never released")
    else:
        last_release = len(tokens) - 1 - tokens[::-1].index("RELEASE")
        after = len(tokens) - 1 - last_release
        if after < 2:
            problems.append(f"{ep}: only {after} token(s) after RELEASE, expected >= 2")
    stalled = [
        round(t, 5)
        for t in meta.get("retreat_travel_m") or []
        if t < MIN_TOKEN_TRAVEL_M
    ]
    if stalled:
        problems.append(f"{ep}: retreat token(s) barely moved: {stalled} m")
    blocked = moves = 0
    for cur, nxt in zip(recs, recs[1:]):
        if cur["kind"] != "move":
            continue
        moves += 1
        d = (
            sum((a - b) ** 2 for a, b in zip(cur["ee_pose"][:3], nxt["ee_pose"][:3]))
            ** 0.5
        )
        blocked += d < MIN_TOKEN_TRAVEL_M
    if moves and blocked / moves > MAX_STALLED_FRAC:
        problems.append(
            f"{ep}: {blocked}/{moves} move tokens ({100 * blocked / moves:.0f}%) over "
            f"< {MIN_TOKEN_TRAVEL_M * 1000:.0f} mm of travel"
        )
    # The LAST grasp: an empty-grasp retry (GRASP, RELEASE, MV_DOWN, GRASP) legitimately
    # contains a closed-and-empty frame before it.
    if "GRASP" in tokens and "RELEASE" in tokens:
        g = len(tokens) - 1 - tokens[::-1].index("GRASP")
        r = len(tokens) - 1 - tokens[::-1].index("RELEASE")
        if g < r:
            empty = [
                rec["step"]
                for rec in recs[g + 1 : r + 1]
                if rec["gripper_closed"] and rec["gripper_width"] <= GRASPED_WIDTH_M
            ]
            if empty:
                problems.append(
                    f"{ep}: gripper closed on nothing at steps {empty[:5]}"
                    f"{'...' if len(empty) > 5 else ''}"
                )
    if not meta.get("success"):
        problems.append(f"{ep}: metadata says success=false ({meta.get('reason')})")
    return problems


def task_dirs(root: Path) -> list[Path]:
    if any(root.glob("rollout_*")):
        return [root]
    return sorted(d for d in root.iterdir() if d.is_dir() and any(d.glob("rollout_*")))


def check(root: Path, quiet: bool = False) -> int:
    tasks = task_dirs(root)
    if not tasks:
        print(f"no task directories with rollouts under {root}")
        return 1
    all_problems: list[str] = []
    total = skipped = 0
    for task in tasks:
        problems: list[str] = []
        endings: Counter = Counter()
        lengths: list[int] = []
        n_skip = 0
        for ep in sorted(task.glob("rollout_*")):
            if not (ep / "actions.jsonl").exists():
                continue
            found = check_episode(ep)
            if found is None:
                n_skip += 1
                continue
            total += 1
            recs = [json.loads(x) for x in (ep / "actions.jsonl").open() if x.strip()]
            if recs:
                endings[recs[-1]["token"]] += 1
                lengths.append(len(recs))
            problems.extend(found)
        skipped += n_skip
        span = (
            f"tokens {min(lengths)}-{max(lengths)} (mean {sum(lengths) / len(lengths):.1f})"
            if lengths
            else "no complete episodes"
        )
        print(
            f"[{'OK ' if not problems else 'BAD'}] {task.name:26s} {len(lengths):3d} eps  "
            f"{span}  endings={dict(endings)}"
            f"{f'  (+{n_skip} still generating)' if n_skip else ''}"
        )
        if problems and not quiet:
            for p in problems[:10]:
                print(f"        {p}")
            if len(problems) > 10:
                print(f"        ... and {len(problems) - 10} more")
        all_problems.extend(problems)
    print()
    if skipped:
        print(f"({skipped} episode(s) skipped -- still being generated)")
    if all_problems:
        print(f"FAILED: {len(all_problems)} problem(s) across {total} episodes")
        return 1
    print(f"PASSED: {total} episodes across {len(tasks)} tasks satisfy the contract")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("root")
    ap.add_argument("--quiet", action="store_true", help="only print the verdict")
    args = ap.parse_args(argv)
    return check(Path(args.root), args.quiet)


if __name__ == "__main__":
    raise SystemExit(main())
