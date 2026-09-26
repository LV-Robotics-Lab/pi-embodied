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

"""Planner sessions of eval runs as supervised fine-tuning data (CaP-X's prepare_verl_dataset /
verl_agent_reward, with the environment's success as the reward).

An episode is a directory holding a pi session file (``*.jsonl`` whose first line is the
``session`` header), as eval.sh / eval-parallel.sh leave it, usually with ``result.json`` next to
it. Its outcome is ``result.json``'s ``status`` (eval.sh's rule), or, without one, the session's
``robot_result`` entry judged by the same rule: ``env_error`` / ``planner_error`` are invalid,
else ``success`` (``terminated`` for LIBERO) decides. Only successes are exported unless
``include_failures``; invalid, timed-out or unfinished episodes never are. ``reward`` is 1.0 for
an environment-judged success and 0.0 otherwise; a planner claim or a program that ran without
error earns nothing.

The conversation is the session's branch (the path to its last entry): the system prompt the
planner saw (the ``robot_system_prompt`` entry ../../packages/embodied/src/robot.ts writes, since
pi sends a robot's forced prompt without recording it; sessions recorded before that entry
existed fall back to the recorded system message, and ``system_prompt_source`` says which), the
tools the transcript declared, user and extension messages, the assistant turns that completed
(errored or aborted ones were retried or ended the run, and thinking is dropped), and the tool
results. Images are written to ``images/<episode>/<n>.png`` under the output directory and
referenced by that relative path; ``keep_images`` keeps only the newest ones as the robot's
``--keep-images`` does and puts its stub in place of the rest.

Formats (one combined ``planner.jsonl``; ``sharegpt`` also writes ``dataset_info.json``):

- ``sharegpt`` (default): LLaMA-Factory's ShareGPT layout with tool calling:
  ``conversations`` of ``human`` / ``function_call`` / ``observation`` / ``gpt`` turns, ``system``,
  ``tools`` (a JSON string) and ``images``, with an ``<image>`` placeholder per image. Chosen as
  the default because LLaMA-Factory loads it directly (``dataset_info.json`` is written next to it,
  so ``dataset_dir`` is the output directory) and it is the format its multimodal tool-calling
  templates train on. A function call is one JSON object (a list for parallel calls); text the
  assistant wrote alongside calls goes in front as a thought, in the function formatter's default
  thought words (``<think>``, a newline, the text, a newline, ``</think>`` and a blank line),
  which templates with bare ``<think>`` / ``</think>`` match too. Consecutive results merge into
  one observation, and a trailing observation (the ``finish`` result) is dropped, since turns must
  alternate and end on the model.
- ``openai``: OpenAI chat ``messages`` (``tool_calls`` with JSON-string arguments, ``tool`` messages
  with ``tool_call_id``) plus ``tools`` and ``images``, which VeRL's multi-turn SFT dataset reads,
  and the VeRL RL columns ``data_source``, ``prompt`` (system + first user message), ``ability``,
  ``reward_model`` ``{"style": "env_success", "ground_truth": {"success": ...}}`` and
  ``extra_info``; convert to parquet (``datasets``/pandas) for VeRL.

Every row also carries ``id``, ``robot``, ``task``, ``status``, ``success``, ``reward``, ``model``
and ``session``; loaders ignore columns they do not map.
"""

from __future__ import annotations

import base64
import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any

FORMATS = ("sharegpt", "openai")
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


def _images(
    messages: list[dict[str, Any]], out_dir: Path, rel: str, keep: int | None
) -> list[str]:
    """Write the images to ``out_dir/rel``, turning each image part into ``{"type": "image", "path"}``
    (or the stub text beyond the newest ``keep``); returns the paths in order."""
    total = sum(
        1 for m in messages for p in m.get("parts", []) if p.get("type") == "image"
    )
    drop = max(0, total - keep) if keep is not None and keep >= 0 else 0
    paths: list[str] = []
    n = 0
    for m in messages:
        new_parts = []
        for p in m.get("parts", []):
            if p.get("type") != "image":
                new_parts.append(p)
                continue
            n += 1
            if n <= drop:
                new_parts.append({"type": "text", "text": IMAGE_STUB})
                continue
            ext = {"image/jpeg": "jpg", "image/webp": "webp"}.get(
                p.get("mimeType", ""), "png"
            )
            path = f"{rel}/{len(paths):04d}.{ext}"
            (out_dir / rel).mkdir(parents=True, exist_ok=True)
            (out_dir / path).write_bytes(base64.b64decode(p["data"]))
            paths.append(path)
            new_parts.append({"type": "image", "path": path})
        if "parts" in m:
            m["parts"] = new_parts
    return paths


def _text(parts: list[dict[str, Any]]) -> str:
    """Text parts joined, with an ``<image>`` placeholder where each image was."""
    return "\n".join(
        "<image>" if p.get("type") == "image" else str(p.get("text", ""))
        for p in parts
        if p.get("type") in ("text", "image")
    )


def to_sharegpt(conv: dict[str, Any]) -> list[dict[str, str]]:
    """LLaMA-Factory ShareGPT turns: alternating human|observation and gpt|function_call, ending on the model."""
    turns: list[dict[str, str]] = []
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
                    value = f"<think>\n{m['text']}\n</think>\n\n{value}"
                turn = {"from": "function_call", "value": value}
            else:
                turn = {"from": "gpt", "value": m["text"]}
            if turns and turns[-1]["from"] in ("gpt", "function_call"):
                turns[-1] = (
                    turn  # a reply with no context between: the later one is what ran
                )
            else:
                turns.append(turn)
            continue
        side = "observation" if m["role"] == "tool" else "human"
        value = _text(m["parts"])
        if turns and turns[-1]["from"] in ("human", "observation"):
            turns[-1]["value"] += "\n" + value
        elif not turns and side == "observation":
            continue
        else:
            turns.append({"from": side, "value": value})
    while turns and turns[-1]["from"] in ("human", "observation"):
        turns.pop()
    return turns


def to_openai(conv: dict[str, Any]) -> list[dict[str, Any]]:
    """OpenAI chat messages; images are ``<image>`` placeholders in the text."""
    out: list[dict[str, Any]] = [{"role": "system", "content": conv["system"]}]
    for m in conv["messages"]:
        if m["role"] == "assistant":
            msg: dict[str, Any] = {"role": "assistant", "content": m["text"]}
            if m["calls"]:
                msg["tool_calls"] = [
                    {
                        "id": c["id"],
                        "type": "function",
                        "function": {
                            "name": c["name"],
                            "arguments": json.dumps(c["arguments"], ensure_ascii=False),
                        },
                    }
                    for c in m["calls"]
                ]
            out.append(msg)
        elif m["role"] == "tool":
            out.append(
                {"role": "tool", "tool_call_id": m["id"], "content": _text(m["parts"])}
            )
        else:
            out.append({"role": "user", "content": _text(m["parts"])})
    while out and out[-1]["role"] != "assistant":
        out.pop()
    return out


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
        if status != "success" and not (include_failures and status == "failure"):
            skipped.setdefault(status, []).append(str(session))
            continue
        conv = conversation(entries)
        episode = session.parent.name if session.parent.name else session.stem
        episode_id = f"{episode}-{session.stem[-8:]}"
        while episode_id in ids:
            episode_id += "_"
        ids.add(episode_id)
        images = _images(conv["messages"], output, f"images/{episode_id}", keep_images)
        success = status == "success"
        task = next(
            (
                e.get("data")
                for e in entries
                if e.get("type") == "custom" and e.get("customType") == TASK_ENTRY
            ),
            {},
        )
        model = next(
            (
                f"{e.get('provider')}/{e.get('modelId')}"
                for e in reversed(entries)
                if e.get("type") == "model_change"
            ),
            result.get("model"),
        )
        meta = {
            "id": episode_id,
            "robot": result.get("robot") or (task or {}).get("robot"),
            "task": task,
            "status": status,
            "success": success,
            "reward": 1.0 if success else 0.0,
            "model": model,
            "session": str(session),
            "system_prompt_source": conv["system_prompt_source"],
            "compacted": conv["compacted"],
        }
        if fmt == "sharegpt":
            turns = to_sharegpt(conv)
            if not turns:
                skipped.setdefault("empty", []).append(str(session))
                continue
            rows.append(
                {
                    "conversations": turns,
                    "system": conv["system"],
                    "tools": json.dumps(conv["tools"], ensure_ascii=False),
                    "images": images,
                    **meta,
                }
            )
        else:
            messages = to_openai(conv)
            if len(messages) < 2:
                skipped.setdefault("empty", []).append(str(session))
                continue
            first_user = next((m for m in messages if m["role"] == "user"), None)
            rows.append(
                {
                    "messages": messages,
                    "tools": [
                        {"type": "function", "function": t} for t in conv["tools"]
                    ],
                    "images": images,
                    "data_source": f"pi_embodied/{meta['robot']}",
                    "prompt": [messages[0], *([first_user] if first_user else [])],
                    "ability": "embodied_planning",
                    "reward_model": {
                        "style": "env_success",
                        "ground_truth": {"success": success},
                    },
                    "extra_info": {
                        "task": task,
                        "status": status,
                        "session": str(session),
                    },
                    **meta,
                }
            )
    file_name = "planner.jsonl"
    with (output / file_name).open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    if fmt == "sharegpt":
        (output / "dataset_info.json").write_text(
            json.dumps(dataset_info(name, file_name), indent=2) + "\n", encoding="utf-8"
        )
    return {
        "output": str(output / file_name),
        "format": fmt,
        "episodes": len(rows),
        "successes": sum(1 for r in rows if r["success"]),
        "failures": sum(1 for r in rows if not r["success"]),
        "images": sum(len(r["images"]) for r in rows),
        "skipped": {k: len(v) for k, v in sorted(skipped.items())},
        "system_prompt_sources": sorted({r["system_prompt_source"] for r in rows}),
    }
