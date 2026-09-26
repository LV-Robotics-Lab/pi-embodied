# Copyright 2026 The Show-Harness Authors (github.com/showlab/Show-Harness @137d571).
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
# Modified by pi-embodied: the in-place config edits of scripts/franka/capture_z_floor.py,
# capture_pose.py and scripts/piper/capture_z_floor.py (_edit_named_floor,
# _write_arm_scalar, ...) generalized to one nested key path.

"""Set one nested YAML value in place, keeping the file's comments and layout.

The calibration capture scripts write a measured value (a Z floor, a pose) into the robot
config the env server reads. The edit is textual (block by indentation), so comments
survive; the result is re-parsed and written only when it loads with the new value.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import yaml


def _flow(value: Any) -> str:
    """A scalar or a flat list as one YAML flow value."""
    if isinstance(value, (list, tuple)):
        return "[" + ", ".join(_flow(v) for v in value) + "]"
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float):
        return repr(round(value, 6))
    return str(value)


def _child_span(
    text: str, start: int, end: int, key: str, parent_indent: int
) -> tuple[int, int, int] | None:
    """(body start, body end, indent) of `key:` directly inside text[start:end], or None."""
    for m in re.finditer(
        rf"^(?P<ind>[ \t]*){re.escape(key)}:(?P<rest>.*)$",
        text[start:end],
        re.MULTILINE,
    ):
        indent = len(m.group("ind"))
        if indent <= parent_indent:
            continue
        body = start + m.end()
        stop = end
        for line in re.finditer(r"^(?P<ind>[ \t]*)\S.*$", text[body:end], re.MULTILINE):
            if len(line.group("ind")) <= indent and not line.group(
                0
            ).lstrip().startswith("#"):
                stop = body + line.start()
                break
        return body, stop, indent
    return None


def set_value(text: str, keys: list[str], value: Any) -> str:
    """`text` with keys[0].keys[1]...keys[-1] set to `value` (missing blocks and the key are appended)."""
    start, end, indent = 0, len(text), -1
    for depth, key in enumerate(keys[:-1]):
        span = _child_span(text, start, end, key, indent)
        if span is None:
            pad = " " * (2 * depth)
            rest = "".join(
                f"\n{' ' * (2 * (depth + 1 + i))}{k}:"
                for i, k in enumerate(keys[depth + 1 : -1])
            )
            leaf = f"\n{' ' * (2 * (len(keys) - 1))}{keys[-1]}: {_flow(value)}\n"
            block = text[start:end].rstrip("\n")
            insert = (
                f"{block}\n{pad}{key}:{rest}{leaf}"
                if block
                else f"{pad}{key}:{rest}{leaf}"
            )
            return text[:start] + insert + text[end:]
        start, end, indent = span
    leaf = keys[-1]
    for m in re.finditer(
        rf"^(?P<ind>[ \t]*){re.escape(leaf)}:(?P<val>[^#\n]*)(?P<cmt>#.*)?$",
        text[start:end],
        re.MULTILINE,
    ):
        if len(m.group("ind")) <= indent:
            continue
        comment = f"  {m.group('cmt')}" if m.group("cmt") else ""
        line = f"{m.group('ind')}{leaf}: {_flow(value)}{comment}"
        return text[: start + m.start()] + line + text[start + m.end() :]
    child = " " * (indent + 2 if indent >= 0 else 0)
    body = text[start:end].rstrip("\n")
    return text[:start] + f"{body}\n{child}{leaf}: {_flow(value)}\n" + text[end:]


def _same(a: Any, b: Any) -> bool:
    if isinstance(b, (list, tuple)):
        return (
            isinstance(a, list)
            and len(a) == len(b)
            and all(_same(x, y) for x, y in zip(a, b))
        )
    if isinstance(b, float) and not isinstance(a, bool) and isinstance(a, (int, float)):
        return abs(a - b) < 1e-6
    return a == b


def get_value(data: Any, keys: list[str]) -> Any:
    for k in keys:
        data = data.get(k) if isinstance(data, dict) else None
    return data


def write_value(path: str | Path, keys: list[str], value: Any) -> None:
    """Set the value in the file; raises ValueError (file untouched) when the edit would not load back."""
    path = Path(path)
    text = path.read_text(encoding="utf-8") if path.exists() else ""
    new = set_value(text, keys, value)
    try:
        got = get_value(yaml.safe_load(new) or {}, keys)
    except yaml.YAMLError as exc:
        raise ValueError(
            f"editing {'.'.join(keys)} would corrupt {path}: {exc}"
        ) from exc
    if not _same(got, value):
        raise ValueError(
            f"editing {'.'.join(keys)} in {path} did not load back as {value!r}; set it by hand"
        )
    path.write_text(new, encoding="utf-8")
