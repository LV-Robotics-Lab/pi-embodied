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

"""The robots whose Flywheel data the CLI knows: each one's data rules (``SPEC``) live in
``robots/<robot>/flywheel.py``, next to the TS recorder's matching ``FlywheelSpec``."""

from __future__ import annotations

import importlib
from pathlib import Path
from typing import Any

#: The simulators whose VLA runs agent-side, so every env step is seen and recorded. A real Franka's
#: motions servo inside its env server, where no per-step observation reaches the recorder.
ROBOTS = ("libero", "robocasa", "robotwin")


def spec(robot: str) -> dict[str, Any]:
    """The data rules of ``robot``."""
    if robot not in ROBOTS:
        raise ValueError(f"no Flywheel spec for {robot!r}; have {', '.join(ROBOTS)}")
    return importlib.import_module(f"pi_embodied_services.robots.{robot}.flywheel").SPEC


def select(data_root: Path | str, robot: str, selection: str) -> list[Path]:
    """The raw episodes under ``raw/<robot>/<selection>`` (any depth), in path order."""
    parts = Path(selection).parts
    if (
        not parts
        or any(p in ("", ".", "..") for p in parts)
        or Path(selection).is_absolute()
    ):
        raise ValueError(f"invalid selection: {selection!r}")
    base = Path(data_root).expanduser().resolve() / "raw" / robot / selection
    return sorted(p for p in base.rglob("episode_*") if p.is_dir())
