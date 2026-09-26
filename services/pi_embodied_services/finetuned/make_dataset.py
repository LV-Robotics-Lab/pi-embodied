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
# Modified by pi-embodied: scripts/trajectory/real2sim/maniskill/make_dataset.py of
# github.com/showlab/Show-Harness @137d571, as a module running the vendored converter.

"""Convert generated rollout datasets to LLaMA-Factory samples and gate them on their stats.

    python -m pi_embodied_services.finetuned.make_dataset --root <root> [--version v3] [--skip-convert]

For every dataset directory under ``--root`` (each holding ``rollout_NNN/``), runs the vendored
``showharness/train/data_preparation/rollouts_to_alpaca.py --version <v> --task <text>`` into
``<dataset>/rollout_lite.json`` and writes per-dataset stats to ``<root>/stats.json``: episodes,
samples, token histogram, per-token displacement from the ``ee_pose`` deltas, blocked tokens and
adjacent opposite pairs. A closed-loop dataset must show ~``step_m`` per token with a sub-mm sigma
and zero adjacent opposite pairs; ``tokens_blocked`` counts tokens that moved under 0.75 step
(the arm was blocked while the label claims a full step).
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from collections import Counter
from pathlib import Path

import numpy as np

from pi_embodied_services.finetuned.atomic import OPPOSITE, TOKEN_AXIS

CONVERTER = (
    Path(__file__).resolve().parent
    / "showharness/train/data_preparation/rollouts_to_alpaca.py"
)


def rollout_stats(ds_dir: Path) -> dict:
    episodes = sorted(
        d for d in ds_dir.iterdir() if d.is_dir() and (d / "actions.jsonl").exists()
    )
    tokens: Counter = Counter()
    lengths: list[int] = []
    disp: list[float] = []
    flips = move_pairs = 0
    task, step_m = "", 0.02
    for ep in episodes:
        recs = [json.loads(x) for x in (ep / "actions.jsonl").open() if x.strip()]
        lengths.append(len(recs))
        toks = [r["token"] for r in recs]
        tokens.update(toks)
        flips += sum(1 for a, b in zip(toks, toks[1:]) if OPPOSITE.get(a) == b)
        move_pairs += sum(
            1 for a, b in zip(toks, toks[1:]) if a in TOKEN_AXIS and b in TOKEN_AXIS
        )
        meta = json.loads((ep / "metadata.json").read_text())
        task = meta.get("task_text") or meta.get("task", task)
        step_m = float(meta.get("step_m", step_m))
        for a, b in zip(recs, recs[1:]):
            if a["token"] in TOKEN_AXIS:
                ax, sg = TOKEN_AXIS[a["token"]]
                disp.append(float((b["ee_pose"][ax] - a["ee_pose"][ax]) * sg))
    d = np.asarray(disp) if disp else np.zeros(1)
    blocked = int((d < 0.75 * step_m).sum()) if disp else 0
    return {
        "episodes": len(episodes),
        "samples": int(sum(lengths))
        + len(episodes),  # + the synthesized DONE per episode
        "tokens_per_episode_mean": round(float(np.mean(lengths)), 1) if lengths else 0,
        "token_hist": dict(sorted(tokens.items())),
        "per_token_disp_mm_mean": round(float(d.mean() * 1000), 2),
        "per_token_disp_mm_std": round(float(d.std() * 1000), 2),
        "tokens_blocked": blocked,
        "tokens_blocked_pct": round(100 * blocked / max(1, len(disp)), 2),
        "adjacent_opposite_pairs": int(flips),
        "adjacent_move_pairs": int(move_pairs),
        "task": task,
    }


def datasets(root: Path) -> list[Path]:
    return sorted(
        d
        for d in root.iterdir()
        if d.is_dir()
        and d.name != "tracks"
        and any((c / "actions.jsonl").exists() for c in d.iterdir() if c.is_dir())
    )


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--root", required=True)
    ap.add_argument("--version", default="v3")
    ap.add_argument("--skip-convert", action="store_true")
    ap.add_argument("--converter", default=str(CONVERTER))
    args = ap.parse_args(argv)
    root = Path(args.root)
    all_stats: dict = {}
    failed = 0
    for ds in datasets(root):
        stats = rollout_stats(ds)
        if not args.skip_convert:
            out_json = ds / "rollout_lite.json"
            cmd = [sys.executable, args.converter, str(ds), "--version", args.version]
            cmd += ["--task", stats["task"], "--output", str(out_json)]
            r = subprocess.run(cmd, capture_output=True, text=True)
            if r.returncode != 0:
                failed += 1
                print(f"[convert] {ds.name} FAILED:\n{r.stderr[-800:]}", flush=True)
            else:
                stats["lf_samples"] = len(json.loads(out_json.read_text()))
                print(f"[convert] {ds.name}: {stats['lf_samples']} samples", flush=True)
        all_stats[ds.name] = stats
    (root / "stats.json").write_text(json.dumps(all_stats, indent=2, sort_keys=True))
    print(json.dumps(all_stats, indent=2, sort_keys=True))
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
