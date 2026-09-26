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

"""Planner sessions of eval runs as training data (CaP-X's prepare_verl_dataset /
verl_agent_reward, with the environment's success as the reward): full trajectories for SFT, or
task prompts for RL.

An episode is a directory holding a pi session file (``*.jsonl`` whose first line is the
``session`` header), as eval.sh / eval-parallel.sh leave it, usually with ``result.json`` next to
it. Its outcome is ``result.json``'s ``status`` (eval.sh's rule), or, without one, the session's
``robot_result`` entry judged by the same rule: ``env_error`` / ``planner_error`` are invalid,
else ``success`` (``terminated`` for LIBERO) decides. Invalid, timed-out or unfinished episodes
are never exported.

The conversation is the session's branch (the path to its last entry): the system prompt the
planner saw (the ``robot_system_prompt`` entry ../../packages/embodied/src/robot.ts writes, since
pi sends a robot's forced prompt without recording it; sessions recorded before that entry
existed fall back to the recorded system message, and ``system_prompt_source`` says which), the
tools the transcript declared, user and extension messages, the assistant turns that completed
(errored or aborted ones were retried or ended the run, and thinking is dropped), and the tool
results.

Images: every row holds exactly one ``<image>`` placeholder per entry of ``images``, in the same
order (export asserts it): an image is written, as ``images/<episode>/<n>.png`` under the output
directory, only when the turn holding it is kept, and a literal ``<image>`` in recorded text is
written as ``&lt;image&gt;``. ``keep_images`` (SFT) keeps only the newest images, as the robot's
``--keep-images`` does, with its stub in place of the rest.

SFT formats (successes only unless ``include_failures``; ``reward`` is the recorded episode's
environment-judged success, 1.0 or 0.0, never a planner claim or a program that ran):

- ``sharegpt`` (default): LLaMA-Factory's ShareGPT layout with tool calling:
  ``conversations`` of ``human`` / ``function_call`` / ``observation`` / ``gpt`` turns, ``system``,
  ``tools`` (a JSON string) and ``images`` (paths relative to the output directory), plus
  ``dataset_info.json``, so LLaMA-Factory loads it with ``dataset_dir`` = the output directory.
  A function call is one JSON object (a list for parallel calls); text the assistant wrote
  alongside calls goes in front as a thought, in the function formatter's default thought words
  (``<think>``, a newline, the text, a newline, ``</think>`` and a blank line), which templates
  with bare ``<think>`` / ``</think>`` match too. Consecutive results merge into one observation;
  a leading observation and trailing ones (the ``finish`` result, or the last results of an
  episode the budget ended) are dropped with their images, since turns must alternate and end on
  the model.
- ``openai``: OpenAI chat ``messages`` (``tool_calls`` with JSON-string arguments, ``tool``
  messages with ``tool_call_id``, trailing non-assistant messages dropped), ``tools`` and
  ``images`` as ``{"image": <absolute path>}``: the layout of VeRL's multi-turn SFT dataset.

RL format ``verl-rl`` (every validly finished episode's task, success or failure: a row is a
prompt, not a trajectory), in VeRL's RLHFDataset / agent-loop layout:

- ``prompt``: the system prompt and the first user turn (with extension notes that preceded the
  first reply), with that turn's own images only; the robot's camera frames reach a rollout
  through its tool results, as in pi;
- ``images``: ``{"image": <absolute path>}`` for the prompt's placeholders;
- ``data_source`` ``pi_embodied/<robot>``, ``ability``, ``agent_name`` ``tool_agent``;
- ``reward_model``: ``{"style": "env_success", "ground_truth": {"task": {...}}}``, the task the
  robot resets to: its ``robot_task`` entry (robot and task flags, e.g. suite/task/seed, or
  env-id/seed/scene), which fixes the initial state;
- ``extra_info``: ``index``, ``id``, the same ``task``, the declared ``tools`` (VeRL takes its
  tool classes from its tool config; this is their schema), ``tools_kwargs`` (each tool's
  ``create_kwargs`` carry the task, so an env tool can reset to it), ``interaction_kwargs``, and
  ``source``: the recorded episode's status/success/model/session, as metadata only.

The reward of an RL rollout is the environment's verdict on that rollout: ``compute_score`` below
is the reference ``custom_reward_function``; the rollout's env tools must hand it the rollout's
``robot_result`` in ``extra_info["rollout_result"]``.

Every row also carries ``id`` and ``robot``; loaders ignore columns they do not map.
"""

from __future__ import annotations

import base64
import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any

FORMATS = ("sharegpt", "openai", "verl-rl")
IMAGE_PLACEHOLDER = "<image>"
# What a literal "<image>" in recorded text becomes, so that only real images are placeholders.
ESCAPED_PLACEHOLDER = "&lt;image&gt;"
IMAGE_STUB = "[older camera frame omitted]"
SYSTEM_PROMPT_ENTRY = "robot_system_prompt"
TASK_ENTRY = "robot_task"
RESULT_ENTRY = "robot_result"


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    out = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                out.append(json.loads(line))
    return out


def _is_session(path: Path) -> bool:
    try:
        with path.open(encoding="utf-8") as f:
            return json.loads(f.readline()).get("type") == "session"
    except (OSError, ValueError, AttributeError):
        return False


def find_sessions(roots: Iterable[Path]) -> list[Path]:
    """Every pi session file below ``roots`` (sorted)."""
    found: set[Path] = set()
    for root in roots:
        root = Path(root)
        files = [root] if root.is_file() else root.rglob("*.jsonl")
        found.update(p for p in files if _is_session(p))
    return sorted(found)


def branch(entries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """The entries on the path from the root to the last entry (pi's current branch)."""
    with_id = [e for e in entries if isinstance(e.get("id"), str)]
    by_id = {e["id"]: e for e in with_id}
    out: list[dict[str, Any]] = []
    e = with_id[-1] if with_id else None
    while e is not None:
        out.append(e)
        parent = e.get("parentId")
        e = by_id.get(parent) if isinstance(parent, str) else None
    return out[::-1]


def outcome(session: Path, entries: list[dict[str, Any]]) -> tuple[str, dict[str, Any]]:
    """(status, robot_result) of the episode: result.json's status, else eval.sh's rule on the entry."""
    results = [
        e["data"]
        for e in entries
        if e.get("type") == "custom" and e.get("customType") == RESULT_ENTRY
    ]
    last = results[-1] if len(results) == 1 else {}
    result_file = session.parent / "result.json"
    if result_file.is_file():
        try:
            recorded = json.loads(result_file.read_text(encoding="utf-8"))
            return str(recorded.get("status")), {**last, **recorded}
        except ValueError:
            return "missing", last
    if len(results) > 1:
        return "duplicate_result", last
    if not results:
        return "missing", last
    if last.get("env_error"):
        return "env_error", last
    if last.get("planner_error"):
        return "planner_error", last
    done = last.get("success") if "success" in last else last.get("terminated")
    return ("success" if done else "failure"), last


def _parts(content: Any) -> list[dict[str, Any]]:
    if isinstance(content, str):
        return [{"type": "text", "text": content}] if content else []
    return [p for p in content or [] if isinstance(p, dict)]


def conversation(entries: list[dict[str, Any]]) -> dict[str, Any]:
    """System prompt, tools and the context messages (``user`` / ``assistant`` / ``tool``) of a branch."""
    system_sections: dict[str, str] = {}
    system_content: list[str] = []
    tools: dict[str, dict[str, Any]] = {}
    forced: str | None = None
    compacted = False
    messages: list[dict[str, Any]] = []
    for e in entries:
        kind = e.get("type")
        if kind == "custom" and e.get("customType") == SYSTEM_PROMPT_ENTRY:
            forced = str((e.get("data") or {}).get("text", ""))
        elif kind == "compaction":
            compacted = True
        elif kind == "custom_message":
            parts = _parts(e.get("content"))
            if parts:
                messages.append({"role": "user", "parts": parts})
        elif kind == "message":
            m = e["message"]
            role = m.get("role")
            if role == "system":
                text = "".join(p.get("text", "") for p in _parts(m.get("content")))
                if text:
                    system_content.append(text)
                for name, value in (m.get("sections") or {}).items():
                    if value is None:
                        system_sections.pop(name, None)
                    else:
                        system_sections[name] = value
                for t in m.get("toolsRemoved") or []:
                    tools.pop(t.get("name"), None)
                for t in m.get("toolsAdded") or []:
                    tools[t["name"]] = {
                        "name": t["name"],
                        "description": t.get("description", ""),
                        "parameters": t.get("parameters", {}),
                    }
            elif role == "user":
                messages.append({"role": "user", "parts": _parts(m.get("content"))})
            elif role == "assistant":
                if m.get("stopReason") in ("error", "aborted"):
                    continue
                parts = m.get("content") or []
                messages.append(
                    {
                        "role": "assistant",
                        "text": "\n".join(
                            p["text"] for p in parts if p.get("type") == "text"
                        ).strip(),
                        "calls": [
                            {
                                "id": p["id"],
                                "name": p["name"],
                                "arguments": p.get("arguments") or {},
                            }
                            for p in parts
                            if p.get("type") == "toolCall"
                        ],
                    }
                )
            elif role == "toolResult":
                messages.append(
                    {
                        "role": "tool",
                        "id": m.get("toolCallId"),
                        "name": m.get("toolName"),
                        "parts": _parts(m.get("content")),
                    }
                )
    recorded = "\n\n".join(
        p for p in ["\n\n".join(system_content), *system_sections.values()] if p
    )
    return {
        "system": forced if forced is not None else recorded,
        "system_prompt_source": "robot_entry" if forced is not None else "session",
        "tools": list(tools.values()),
        "messages": messages,
        "compacted": compacted,
    }


def _images(messages: list[dict[str, Any]], keep: int | None) -> None:
    """Replace the image parts beyond the newest ``keep`` by the robot's stub (in place)."""
    total = sum(
        1 for m in messages for p in m.get("parts", []) if p.get("type") == "image"
    )
    drop = max(0, total - keep) if keep is not None and keep >= 0 else 0
    n = 0
    for m in messages:
        new_parts = []
        for p in m.get("parts", []):
            if p.get("type") == "image":
                n += 1
                if n <= drop:
                    p = {"type": "text", "text": IMAGE_STUB}
            new_parts.append(p)
        if "parts" in m:
            m["parts"] = new_parts


def _escape(text: str) -> str:
    """Text with any literal ``<image>`` defused: loaders count every occurrence as a placeholder."""
    return text.replace(IMAGE_PLACEHOLDER, ESCAPED_PLACEHOLDER)


def _text(parts: list[dict[str, Any]]) -> tuple[str, list[dict[str, Any]]]:
    """Text parts joined with an ``<image>`` placeholder where each image was, and those images in order."""
    shown = [p for p in parts if p.get("type") in ("text", "image")]
    text = "\n".join(
        IMAGE_PLACEHOLDER if p["type"] == "image" else _escape(str(p.get("text", "")))
        for p in shown
    )
    return text, [p for p in shown if p["type"] == "image"]


def _write(images: list[dict[str, Any]], out_dir: Path, rel: str) -> list[str]:
    """Write ``images`` (the ones the rendered row kept, in placeholder order) to ``out_dir/rel``."""
    paths = []
    for i, p in enumerate(images):
        ext = {"image/jpeg": "jpg", "image/webp": "webp"}.get(
            p.get("mimeType", ""), "png"
        )
        path = f"{rel}/{i:04d}.{ext}"
        (out_dir / rel).mkdir(parents=True, exist_ok=True)
        (out_dir / path).write_bytes(base64.b64decode(p["data"]))
        paths.append(path)
    return paths


def placeholders(value: Any) -> int:
    """``<image>`` placeholders in every string of ``value`` (the count loaders match against images)."""
    if isinstance(value, str):
        return value.count(IMAGE_PLACEHOLDER)
    if isinstance(value, dict):
        return sum(placeholders(v) for v in value.values())
    if isinstance(value, list):
        return sum(placeholders(v) for v in value)
    return 0


def to_sharegpt(
    conv: dict[str, Any],
) -> tuple[list[dict[str, str]], list[dict[str, Any]]]:
    """LLaMA-Factory ShareGPT turns (alternating human|observation and gpt|function_call, ending on
    the model) and the images of the turns kept, in placeholder order."""
    turns: list[dict[str, Any]] = []
    for m in conv["messages"]:
        if m["role"] == "assistant":
            if m["calls"]:
                calls = [
                    {"name": c["name"], "arguments": c["arguments"]} for c in m["calls"]
                ]
                value = json.dumps(
                    calls[0] if len(calls) == 1 else calls, ensure_ascii=False
                )
                if m["text"]:
                    value = (
                        f"<think>\n{_escape(m['text'])}\n</think>\n\n{_escape(value)}"
                    )
                else:
                    value = _escape(value)
                turn = {"from": "function_call", "value": value, "images": []}
            else:
                turn = {"from": "gpt", "value": _escape(m["text"]), "images": []}
            if turns and turns[-1]["from"] in ("gpt", "function_call"):
                turns[-1] = (
                    turn  # a reply with no context between: the later one is what ran
                )
            else:
                turns.append(turn)
            continue
        side = "observation" if m["role"] == "tool" else "human"
        value, images = _text(m["parts"])
        if turns and turns[-1]["from"] in ("human", "observation"):
            turns[-1]["value"] += "\n" + value
            turns[-1]["images"] += images
        elif not turns and side == "observation":
            continue
        else:
            turns.append({"from": side, "value": value, "images": images})
    while turns and turns[-1]["from"] in ("human", "observation"):
        turns.pop()
    images = [p for t in turns for p in t.pop("images")]
    return turns, images


def to_openai(
    conv: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """OpenAI chat messages (images as ``<image>`` placeholders in the text), ending on the model,
    and the images of the messages kept, in placeholder order."""
    out: list[tuple[dict[str, Any], list[dict[str, Any]]]] = [
        ({"role": "system", "content": _escape(conv["system"])}, [])
    ]
    for m in conv["messages"]:
        if m["role"] == "assistant":
            msg: dict[str, Any] = {"role": "assistant", "content": _escape(m["text"])}
            if m["calls"]:
                msg["tool_calls"] = [
                    {
                        "id": c["id"],
                        "type": "function",
                        "function": {
                            "name": c["name"],
                            "arguments": _escape(
                                json.dumps(c["arguments"], ensure_ascii=False)
                            ),
                        },
                    }
                    for c in m["calls"]
                ]
            out.append((msg, []))
            continue
        text, images = _text(m["parts"])
        if m["role"] == "tool":
            out.append(
                ({"role": "tool", "tool_call_id": m["id"], "content": text}, images)
            )
        else:
            out.append(({"role": "user", "content": text}, images))
    while out and out[-1][0]["role"] != "assistant":
        out.pop()
    return [m for m, _ in out], [p for _, images in out for p in images]


def rl_prompt(
    conv: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """The RL prompt: the system prompt and the first user turn (the task as the episode began,
    extension notes before the first reply included), and that turn's own images."""
    parts: list[dict[str, Any]] = []
    for m in conv["messages"]:
        if m["role"] != "user":
            break
        parts += m["parts"]
    text, images = _text(parts)
    return [
        {"role": "system", "content": _escape(conv["system"])},
        {"role": "user", "content": text},
    ], images


def compute_score(
    data_source: str,
    solution_str: str,
    ground_truth: dict[str, Any],
    extra_info: dict[str, Any] | None = None,
    **_: Any,
) -> float:
    """Reference VeRL reward hook (``custom_reward_function.path`` = this file, ``name`` =
    ``compute_score``) for ``verl-rl`` rows: 1.0 when the environment judged the new rollout a
    success, else 0.0. The rollout's environment (the agent loop's env tools, reset with
    ``ground_truth["task"]``) must put its ``robot_result`` (the robot's own result row) into
    ``extra_info["rollout_result"]``; the text of the rollout never earns a reward, and neither
    does the recorded episode's outcome (``extra_info["source"]``)."""
    result = (extra_info or {}).get("rollout_result")
    if not isinstance(result, dict):
        raise ValueError(
            "compute_score needs extra_info['rollout_result'] from the rollout's environment"
        )
    task = ground_truth.get("task") or {}
    for key, want in task.items():
        if key in result and str(result[key]) != str(want):
            raise ValueError(
                f"rollout_result {key}={result[key]!r} is not the prompt's {want!r}"
            )
    if result.get("env_error") or result.get("planner_error"):
        return 0.0
    done = result.get("success") if "success" in result else result.get("terminated")
    return 1.0 if done is True else 0.0


def dataset_info(name: str, file_name: str) -> dict[str, Any]:
    """The LLaMA-Factory ``dataset_info.json`` entry for a ``sharegpt`` export."""
    return {
        name: {
            "file_name": file_name,
            "formatting": "sharegpt",
            "columns": {
                "messages": "conversations",
                "system": "system",
                "tools": "tools",
                "images": "images",
            },
            "tags": {
                "role_tag": "from",
                "content_tag": "value",
                "user_tag": "human",
                "assistant_tag": "gpt",
                "observation_tag": "observation",
                "function_tag": "function_call",
            },
        }
    }


def export_planner(
    roots: Iterable[Path],
    output: Path,
    *,
    fmt: str = "sharegpt",
    include_failures: bool = False,
    keep_images: int | None = None,
    name: str = "pi_embodied_planner",
) -> dict[str, Any]:
    """Export the sessions below ``roots`` to ``output/planner.jsonl``; returns counts and skip reasons."""
    if fmt not in FORMATS:
        raise ValueError(f"format must be one of {FORMATS}")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    skipped: dict[str, list[str]] = {}
    ids: set[str] = set()
    for session in find_sessions(roots):
        entries = branch(_read_jsonl(session))
        status, result = outcome(session, entries)
        # RL rows are prompts: any validly finished episode's task is one, whatever its outcome.
        wanted = (
            status in ("success", "failure")
            if fmt == "verl-rl" or include_failures
            else status == "success"
        )
        if not wanted:
            skipped.setdefault(status, []).append(str(session))
            continue
        conv = conversation(entries)
        episode = session.parent.name if session.parent.name else session.stem
        episode_id = f"{episode}-{session.stem[-8:]}"
        while episode_id in ids:
            episode_id += "_"
        success = status == "success"
        task = (
            next(
                (
                    e.get("data")
                    for e in entries
                    if e.get("type") == "custom" and e.get("customType") == TASK_ENTRY
                ),
                {},
            )
            or {}
        )
        model = next(
            (
                f"{e.get('provider')}/{e.get('modelId')}"
                for e in reversed(entries)
                if e.get("type") == "model_change"
            ),
            result.get("model"),
        )
        robot = result.get("robot") or task.get("robot")
        source = {
            "status": status,
            "success": success,
            "model": model,
            "session": str(session),
            "system_prompt_source": conv["system_prompt_source"],
            "compacted": conv["compacted"],
        }
        rel = f"images/{episode_id}"
        if fmt == "verl-rl":
            prompt, kept = rl_prompt(conv)
            env = {**task, "robot": robot}
            tool_names = [t["name"] for t in conv["tools"]]
            row: dict[str, Any] = {
                "data_source": f"pi_embodied/{robot}",
                "prompt": prompt,
                "images": [
                    {"image": str((output / p).resolve())}
                    for p in _write(kept, output, rel)
                ],
                "ability": "embodied_planning",
                "agent_name": "tool_agent",
                "reward_model": {"style": "env_success", "ground_truth": {"task": env}},
                "extra_info": {
                    "index": len(rows),
                    "id": episode_id,
                    "task": env,
                    "tools": conv["tools"],
                    "tools_kwargs": {
                        n: {"create_kwargs": {"task": env}} for n in tool_names
                    },
                    "interaction_kwargs": {"name": "pi_embodied", "task": env},
                    # The recorded episode: metadata, never the reward of a new rollout.
                    "source": source,
                },
                "id": episode_id,
                "robot": robot,
            }
            content = row["prompt"]
        else:
            _images(conv["messages"], keep_images)
            meta = {
                "id": episode_id,
                "robot": robot,
                "task": task,
                **source,
                "reward": 1.0 if success else 0.0,
            }
            if fmt == "sharegpt":
                turns, kept = to_sharegpt(conv)
                if not turns:
                    skipped.setdefault("empty", []).append(str(session))
                    continue
                row = {
                    "conversations": turns,
                    "system": _escape(conv["system"]),
                    "tools": json.dumps(conv["tools"], ensure_ascii=False),
                    "images": _write(kept, output, rel),
                    **meta,
                }
                content = turns
            else:
                messages, kept = to_openai(conv)
                if len(messages) < 2:
                    skipped.setdefault("empty", []).append(str(session))
                    continue
                row = {
                    "messages": messages,
                    "tools": [
                        {"type": "function", "function": t} for t in conv["tools"]
                    ],
                    "images": [
                        {"image": str((output / p).resolve())}
                        for p in _write(kept, output, rel)
                    ],
                    **meta,
                }
                content = messages
        if placeholders(content) != len(row["images"]):
            raise AssertionError(
                f"{session}: {placeholders(content)} <image> placeholders for {len(row['images'])} images"
            )
        ids.add(episode_id)
        rows.append(row)
    file_name = "planner.jsonl"
    with (output / file_name).open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    if fmt == "sharegpt":
        (output / "dataset_info.json").write_text(
            json.dumps(dataset_info(name, file_name), indent=2) + "\n", encoding="utf-8"
        )
    sources = [r["extra_info"]["source"] if fmt == "verl-rl" else r for r in rows]
    return {
        "output": str(output / file_name),
        "format": fmt,
        "episodes": len(rows),
        "successes": sum(1 for s in sources if s["success"]),
        "failures": sum(1 for s in sources if not s["success"]),
        "images": sum(len(r["images"]) for r in rows),
        "skipped": {k: len(v) for k, v in sorted(skipped.items())},
        "system_prompt_sources": sorted({s["system_prompt_source"] for s in sources}),
    }
