#!/usr/bin/env python3
# Copyright 2026 The Show-Harness Authors. Licensed under the Apache License, Version 2.0.
# Modified by pi-embodied: github.com/showlab/Show-Harness @137d571
# (train/data_preparation/generate_affordance.py) made standalone: the call through
# ./vlm_planning.py (no Show-Harness imports, no robot config: --vlm-url / --model name the
# endpoint), and pi's GUMI rollout layout read too.
"""Call a focused grasp-affordance VLM role on the first frame and write affordance_config.json.

A lightweight cousin of generate_subgoals.py: instead of a full ordered plan it asks the VLM
for ONE thing -- which object to grasp first and the best visible grasp point on it -- so the
training prompt can carry a stable "where to grasp" hint without the subgoal machinery. The
``affordance`` semantics match the SubgoalPlanner's (a visible graspable part / contact
region; for hollow objects the left/right side wall).

The written affordance_config.json is read by rollouts_to_alpaca.py --use-affordance
(prompts/<version>/mvtoken_generator_affordance.txt).

Schema:
    {"task": "...", "target": "...", "affordance": "..."}

Examples
--------
python generate_affordance.py <gumi-record>/0927/task_3/10-11-12 \\
    --task "pick up the banana and place it on the blue plate" \\
    --vlm-url http://localhost:8000/v1 --model qwen35
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from vlm_planning import (  # noqa: E402
    Client,
    VlmError,
    add_vlm_args,
    client_from,
    first_frames,
    plan_affordance,
    resolve_target,
)


def generate_for_rollout(
    image_dir: Path,
    out_dir: Path,
    task: str,
    client: Client,
    dry_run: bool = False,
) -> dict | None:
    """Plan the grasp point from ``image_dir``; write affordance_config.json to ``out_dir``."""
    tag = out_dir.name
    agentview, wrist = first_frames(image_dir)
    if agentview is None or not agentview.exists():
        print(f"[skip] {tag}: no first agentview frame in {image_dir}", file=sys.stderr)
        return None
    src = "" if image_dir == out_dir else f" (from {image_dir.name})"
    print(f"[{tag}] calling GraspAffordance{src} ...", flush=True)
    out = plan_affordance(client, task, [agentview] + ([wrist] if wrist else []))
    if not out["target"] or not out["affordance"]:
        print(f"[{tag}] WARNING: empty target/affordance", file=sys.stderr)
        print(f"  raw: {out['raw'][:300]}", file=sys.stderr)
        return None
    print(f"[{tag}] target={out['target']!r}  affordance={out['affordance']!r}")
    config = {"task": task, "target": out["target"], "affordance": out["affordance"]}
    out_path = out_dir / "affordance_config.json"
    if dry_run:
        print(f"[{tag}] [dry-run] would write {out_path}:")
        print(json.dumps(config, indent=2, ensure_ascii=False))
    else:
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(config, indent=2, ensure_ascii=False))
        print(f"[{tag}] wrote {out_path}")
    return config


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Generate affordance_config.json via GraspAffordance"
    )
    parser.add_argument("rollout_dirs", nargs="+", type=Path)
    parser.add_argument(
        "--task", required=True, help="Natural language task description"
    )
    add_vlm_args(parser)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Write affordance_config.json here instead of the location derived from the "
        "input path (the images still come from the input path).",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    client = client_from(args)
    print(f"model   : {client.model}\nurl     : {client.base_url}")
    failed = 0
    for d in args.rollout_dirs:
        image_dir, out_dir = resolve_target(d)
        if image_dir is None:
            print(f"[skip] {d}: not a rollout and no rollout subdirs", file=sys.stderr)
            failed += 1
            continue
        try:
            done = generate_for_rollout(
                image_dir, args.out_dir or out_dir, args.task, client, args.dry_run
            )
        except VlmError as exc:
            print(f"[{d}] WARNING: {exc}", file=sys.stderr)
            done = None
        failed += done is None
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
