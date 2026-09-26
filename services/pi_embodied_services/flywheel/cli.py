# Copyright 2026 The RPent Authors.
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
# Modified by pi-embodied: import paths rewritten.

"""Command-line entry point for Flywheel data: validate one raw episode, or export a selection of
a robot's raw episodes to a LeRobot v3.0 dataset."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from pi_embodied_services.flywheel.specs import ROBOTS, select, spec


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="pi-embodied-flywheel")
    commands = parser.add_subparsers(dest="command", required=True)

    validate = commands.add_parser("validate", help="validate one raw episode")
    validate.add_argument("episode", type=Path)
    validate.add_argument("--robot", choices=ROBOTS, default="libero")

    export = commands.add_parser(
        "export-lerobot", help="export successful episodes to LeRobot v3.0"
    )
    export.add_argument(
        "--data-root",
        type=Path,
        default=Path("datacollection"),
        help="root containing raw Flywheel episodes",
    )
    export.add_argument("--robot", choices=ROBOTS, required=True)
    export.add_argument(
        "--select",
        required=True,
        help="path below raw/<robot>/ whose episodes make the dataset, e.g. libero_10/task_02",
    )
    export.add_argument("--dataset-id")
    export.add_argument(
        "--output-root",
        type=Path,
        help="parent directory for the exported dataset (default <data-root>/datasets/lerobot/<robot>/<select>)",
    )
    export.add_argument(
        "--videos",
        action="store_true",
        help="store the cameras as mp4 instead of images",
    )

    gumi = commands.add_parser(
        "export-gumi", help="export GUMI teleop/DAgger runs to LeRobot v3.0"
    )
    gumi.add_argument(
        "runs",
        type=Path,
        help="a GUMI run dir, or a dir of them (<root>/<MMDD>/task_<id>)",
    )
    gumi.add_argument("--output-root", type=Path, required=True)
    gumi.add_argument("--dataset-id")
    gumi.add_argument(
        "--include-failed", action="store_true", help="also export unsuccessful runs"
    )

    args = parser.parse_args(argv)
    if args.command == "export-gumi":
        from pi_embodied_services.flywheel.gumi import export_gumi

        result = export_gumi(
            args.runs,
            output_root=args.output_root,
            dataset_id=args.dataset_id,
            include_failed=args.include_failed,
        )
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0
    rules = spec(args.robot)
    if args.command == "validate":
        from pi_embodied_services.flywheel.episode import validate_episode

        result = validate_episode(args.episode, spec=rules)
    else:
        from pi_embodied_services.flywheel.export import export_lerobot

        root = args.data_root.expanduser().resolve()
        result = export_lerobot(
            select(root, args.robot, args.select),
            spec=rules,
            repo_id_prefix=f"pi-embodied/{args.robot}-{re.sub(r'[^A-Za-z0-9_.-]+', '-', args.select)}",
            output_root=args.output_root
            or root / "datasets" / "lerobot" / args.robot / args.select,
            dataset_id=args.dataset_id,
            videos=args.videos,
        )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
