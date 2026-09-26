#!/usr/bin/env python3
# Copyright 2026 The Show-Harness Authors. Licensed under the Apache License, Version 2.0.
# Modified by pi-embodied: github.com/showlab/Show-Harness @137d571
# (train/data_preparation/generate_subgoals.py) made standalone: the planner through
# ./vlm_planning.py (no Show-Harness imports, no robot config: --vlm-url / --model name the
# endpoint), pi's GUMI rollout layout read too, and the planner's route recorded.
"""Call the SubgoalPlanner on the first frame of each rollout and write task_config.json.

Every subgoal field (id/target/affordance/motion/description/completion) is produced by
the VLM, then validated and pre-grasp-merged exactly as the Show-Harness runtime does
(vlm_planning.plan_subgoals), so the stored plan is what the runtime would consume. No
keyword-based stage bucketing is applied here; the ordered subgoal list is written verbatim.

task_config.json is read by rollouts_to_alpaca.py --use-subgoal, which aligns each
recorded action step to one of these subgoals (anchored on the recorded GRASP/RELEASE) and
fills the per-step prompt (prompts/<version>/mvtoken_generator.txt) with that subgoal's
motion/target/affordance/description/completion.

Schema:
    {"task": "...", "subgoals": [{"id","target","affordance","motion","description",
                                  "completion"}, ...], "planner": {"route", ...}}

Examples
--------
# A served planner (vLLM, OpenAI-compatible):
python generate_subgoals.py <gumi-record>/0927/task_3/10-11-12 \\
    --task "pick up the banana and place it on the blue plate" \\
    --vlm-url http://localhost:8000/v1 --model qwen35

# A task folder: plan once from its first rollout, write task_config.json into the folder.
python generate_subgoals.py <gumi-record>/0927/task_3 --task "..." --model qwen35

# Dry run: print the plan without writing it.
    --dry-run
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
    plan_subgoals,
    resolve_target,
)


def generate_for_rollout(
    image_dir: Path,
    out_dir: Path,
    task: str,
    client: Client,
    dry_run: bool = False,
) -> dict | None:
    """Plan from ``image_dir``'s first frame; write task_config.json to ``out_dir``."""
    tag = out_dir.name
    agentview, wrist = first_frames(image_dir)
    if agentview is None or not agentview.exists():
        print(f"[skip] {tag}: no first agentview frame in {image_dir}", file=sys.stderr)
        return None
    src = "" if image_dir == out_dir else f" (from {image_dir.name})"
    print(f"[{tag}] calling SubgoalPlanner{src} ...", flush=True)
    try:
        subgoals, record = plan_subgoals(
            client, task, [agentview] + ([wrist] if wrist else [])
        )
    except (RuntimeError, ValueError) as exc:
        print(f"[{tag}] WARNING: planner failed: {exc}", file=sys.stderr)
        return None
    print(f"[{tag}] {len(subgoals)} subgoals ({record['route']}):")
    for sg in subgoals:
        print(f"  [{sg.motion:>14s}] {sg.id} -- {sg.target} / {sg.affordance}")
    task_config = {
        "task": task,
        "subgoals": [sg.to_prompt_dict() for sg in subgoals],
        "planner": {
            "model": client.model,
            "route": record["route"],
            "errors": record["errors"],
        },
    }
    out_path = out_dir / "task_config.json"
    if dry_run:
        print(f"[{tag}] [dry-run] would write {out_path}:")
        print(json.dumps(task_config, indent=2, ensure_ascii=False))
    else:
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(task_config, indent=2, ensure_ascii=False))
        print(f"[{tag}] wrote {out_path}")
    return task_config


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Generate task_config.json via the SubgoalPlanner"
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
        help="Write task_config.json here instead of the location derived from the input "
        "path (the images still come from the input path).",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    client = client_from(args)
    print(f"model   : {client.model}\nurl     : {client.base_url}")
    if not client.guided:
        print("note: LlamaFactory backend -- prompt-only JSON (no guided decoding)")
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
