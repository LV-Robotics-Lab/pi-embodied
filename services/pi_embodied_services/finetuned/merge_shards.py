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
# Modified by pi-embodied: scripts/trajectory/real2sim/maniskill/merge_shards.py of
# github.com/showlab/Show-Harness @137d571, as a module.

"""Merge sharded generator runs into one dataset directory, renumbering the rollouts.

    python -m pi_embodied_services.finetuned.merge_shards --shards <d1> <d2> ... --out <dir> [--move]

Generation is parallelised by running generators over disjoint seed ranges, each numbering from
``rollout_000``; this concatenates them in order and records the source shard and its original
rollout name in each episode's metadata, so any rollout stays traceable.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path


def merge(shards: list[Path], out: Path, move: bool = False) -> int:
    out.mkdir(parents=True, exist_ok=True)
    n = len([d for d in out.glob("rollout_*") if d.is_dir()])
    for sd in shards:
        eps = sorted(
            d
            for d in Path(sd).iterdir()
            if d.is_dir() and (d / "actions.jsonl").exists()
        )
        for ep in eps:
            dst = out / f"rollout_{n:03d}"
            (shutil.move if move else shutil.copytree)(str(ep), str(dst))
            meta_path = dst / "metadata.json"
            meta = json.loads(meta_path.read_text())
            meta["shard"] = Path(sd).name
            meta["shard_rollout"] = ep.name
            meta_path.write_text(json.dumps(meta, indent=2, sort_keys=True))
            n += 1
        print(f"[merge] {Path(sd).name}: {len(eps)} episodes", flush=True)
    return n


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--shards", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument(
        "--move", action="store_true", help="move instead of copy (consumes the shards)"
    )
    args = ap.parse_args(argv)
    n = merge([Path(s) for s in args.shards], Path(args.out), args.move)
    print(f"[merge] total {n} episodes -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
