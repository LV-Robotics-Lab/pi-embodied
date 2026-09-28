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

"""Download an XPolicyLab policy's RoboDojo checkpoint by policy and action type.

The public RoboDojo checkpoints live in the HF dataset ``RoboDojo-Benchmark/RoboDojo`` (apache-2.0, not
gated) under ``ckpt/RoboDojo/<policy>/RoboDojo-sim-arx_x5-<action_type>-<seed>/``; most policy READMEs do
not say so. ``--list`` shows what a policy has. Honours ``HF_ENDPOINT`` (e.g. https://hf-mirror.com)::

    python -m pi_embodied_services.robots.robodojo.download_ckpt G05 --action-type joint --dest ~/ckpt
    python -m pi_embodied_services.robots.robodojo.download_ckpt G05 --list
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO = "RoboDojo-Benchmark/RoboDojo"
PREFIX = "ckpt/RoboDojo"


def checkpoint_dir(policy: str, action_type: str, seed: int) -> str:
    """The dataset directory of one checkpoint."""
    if action_type not in ("joint", "ee"):
        raise ValueError(f"action_type must be joint or ee, got {action_type!r}")
    return f"{PREFIX}/{policy}/RoboDojo-sim-arx_x5-{action_type}-{seed}"


def available(files: list[str], policy: str) -> list[str]:
    """The checkpoint directories a policy has, from the dataset's file list."""
    root = f"{PREFIX}/{policy}/"
    return sorted(
        {
            f[len(root) :].split("/", 1)[0]
            for f in files
            if f.startswith(root) and "/" in f[len(root) :]
        }
    )


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument(
        "policy", help="XPolicyLab policy directory name, e.g. G05, InternVLA_A1, Pi_05"
    )
    p.add_argument("--action-type", default="joint", choices=["joint", "ee"])
    p.add_argument(
        "--seed", type=int, default=0, help="training seed of the checkpoint"
    )
    p.add_argument(
        "--dest", default=".", help="local root; the dataset path is kept below it"
    )
    p.add_argument(
        "--list", action="store_true", help="list the policy's checkpoints and exit"
    )
    args = p.parse_args(argv)

    from huggingface_hub import HfApi, hf_hub_download
    from huggingface_hub.hf_api import RepoFile

    # Only the policy's subtree: one page (a whole-dataset listing pages through huggingface.co links,
    # which a mirror like hf-mirror.com hands back unrewritten).
    tree = HfApi().list_repo_tree(
        REPO,
        path_in_repo=f"{PREFIX}/{args.policy}",
        repo_type="dataset",
        recursive=True,
    )
    files = [f.path for f in tree if isinstance(f, RepoFile)]
    have = available(files, args.policy)
    if args.list or not have:
        print(f"{args.policy}: {have or 'no checkpoints'}")
        return 0 if have else 1
    want = checkpoint_dir(args.policy, args.action_type, args.seed)
    if want.rsplit("/", 1)[1] not in have:
        print(f"{want} is not in {REPO}; {args.policy} has {have}", file=sys.stderr)
        return 1
    # File by file (resumable): snapshot_download would list the whole dataset.
    for f in (f for f in files if f.startswith(want + "/")):
        hf_hub_download(REPO, f, repo_type="dataset", local_dir=args.dest)
    print(Path(args.dest) / want)
    return 0


if __name__ == "__main__":
    sys.exit(main())
