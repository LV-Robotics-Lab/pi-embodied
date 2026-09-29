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
are never exported. Left out unless ``include`` names them: ``privileged`` runs (simulator
ground truth was on offer) and ``operator`` runs (a human judged or ended the episode). Only
sessions a model planned are exported unless ``planners`` names other types: the result's
``planner`` field, or the session's planner provider without it (flash and replay re-run a
recording, human is a person in the model's place, scripted stand-ins answer from a script);
every row records its ``planner`` (under ``extra_info.source`` for ``verl-rl``).

The conversation is what the planner saw at the end of the session's branch (the path to its
last entry), projected as pi projects it (coding-agent session-manager.ts): after a compaction,
its summary and the entries from ``firstKeptEntryId`` on; ``context_edit`` replacements and
removals applied; custom messages and branch summaries as user messages. It holds the system
prompt the planner saw (the ``robot_system_prompt`` entry ../../packages/embodied/src/robot.ts writes, since
pi sends a robot's forced prompt without recording it; sessions recorded before that entry
existed fall back to the recorded system message, and ``system_prompt_source`` says which), the
tools the transcript declared, user and extension messages, the assistant turns that completed
(errored or aborted ones were retried or ended the run, and thinking is dropped), and the tool
results. For SFT, an explore session keeps only its last attempt: the prompt, then the result of
the last successful ``reset`` (the restored scene) as a user turn, and what followed
(``explore_attempts`` in ``include`` keeps them all).

Images: every row holds exactly one ``<image>`` placeholder per entry of ``images``, in the same
order (export asserts it): an image is written, as ``images/<episode>/<n>.png`` under the output
directory, only when the turn holding it is kept, and a literal ``<image>`` in recorded text is
written as ``&lt;image&gt;``. ``keep_images`` (SFT) prunes as the robot's ``--keep-images`` does
(robot.ts ``context`` hook): tool results only, newest result first and each result's images in
order, the anchor frame kept with ``--anchor-image`` (default: the run's recorded setting), and
``image_stub`` in place of the rest.

SFT formats (successes only unless ``include_failures``; ``reward`` is the recorded episode's
environment-judged success, 1.0 or 0.0, never a planner claim or a program that ran):

- ``sharegpt`` (default): LLaMA-Factory's ShareGPT layout with tool calling:
  ``conversations`` of ``human`` / ``function_call`` / ``observation`` / ``gpt`` turns, ``system``,
  ``tools`` (a JSON string) and ``images`` (paths relative to the output directory), plus
  ``dataset_info.json``, so LLaMA-Factory loads it with ``dataset_dir`` = the output directory.
  A function call is one JSON object (a list for parallel calls); text the assistant wrote
  alongside calls goes in front as a thought, in the function formatter's default thought words
  (``<think>``, a newline, the text, a newline, ``</think>`` and a blank line), which templates
  with bare ``<think>`` / ``</think>`` match too. Everything before the first reply is the human
  prompt; consecutive results merge into one observation. A user message after a result (a
  steering message, an extension note, a summary) cannot follow an observation in ShareGPT, so
  such an episode is skipped (``steering``) unless ``merge_steering`` folds it into the
  observation. Trailing observations (the ``finish`` result, or the last results of an
  episode the budget ended) are dropped with their images, since turns must alternate and end on
  the model.
- ``openai``: OpenAI chat ``messages`` (``tool_calls`` with JSON-string arguments, ``tool``
  messages with ``tool_call_id``, user messages as their own messages, trailing non-assistant
  messages dropped), ``tools`` and
  ``images`` as ``{"image": <absolute path>}``: the layout of VeRL's multi-turn SFT dataset.

RL format ``verl-rl`` (every validly finished episode's task, success or failure: a row is a
prompt, not a trajectory), in VeRL's RLHFDataset / agent-loop layout:

- ``prompt``: the system prompt and the first user turn (with extension notes that preceded the
  first reply), with that turn's own images only; the robot's camera frames reach a rollout
  through its tool results, as in pi;
- ``images``: ``{"image": <absolute path>}`` for the prompt's placeholders;
- ``data_source`` ``pi_embodied/<robot>``, ``ability``, ``agent_name`` ``tool_agent``;
- ``reward_model``: ``{"style": "env_success", "ground_truth": {"robot", "seed", "init_state"}}``,
  what reproduces the episode's start, as CaP-X's prepare_verl_dataset stores its env seed: the
  ``seed`` task flag as an int, and ``init_state``, the robot's other task flags from its
  ``robot_task`` entry, which it resets with (LIBERO: suite and task, the init state being the
  seed modulo the task's init states; ManiSkill: env-id and scene). An episode without a seed
  cannot be reproduced and is skipped (``no_seed``);
- ``extra_info``: ``index``, ``id``, ``seed``, the recorded ``task`` flags, the declared ``tools``
  (VeRL takes its tool classes from its tool config; this is their schema), ``tools_kwargs``
  (each tool's ``create_kwargs`` carry the ground truth, so an env tool can reset to it),
  ``interaction_kwargs``, and ``source``: the recorded episode's status/success/model/session, as
  metadata only.

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
# What is left out unless asked for (export_planner's ``include``).
INCLUDES = ("privileged", "operator", "explore_attempts")
# Planner providers that are not a model planning (flash and replay re-run a recording, human is a
# person answering in the model's place); their sessions are not the planner's own data.
NON_MODEL_PROVIDERS = ("flash", "replay", "human")
# pi's summary framing (coding-agent/src/core/messages.ts).
COMPACTION_SUMMARY_PREFIX = "The conversation history before this point was compacted into the following summary:\n\n<summary>\n"
COMPACTION_SUMMARY_SUFFIX = "\n</summary>"
BRANCH_SUMMARY_PREFIX = "The following is a summary of a branch that this conversation came back from:\n\n<summary>\n"
BRANCH_SUMMARY_SUFFIX = "</summary>"


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


def project(path: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """pi's model context for a branch (coding-agent session-manager.ts ``buildSessionProjection``):
    the newest compaction's system message and summary, then the entries it kept and those after
    it; ``context_edit`` entries replace (or, with ``null``, drop) their target's content; custom
    messages, branch and compaction summaries become user messages (messages.ts
    ``convertToLlm``). Returns pi messages (``role`` system / user / assistant / toolResult)."""
    compaction = next(
        (e for e in reversed(path) if e.get("type") == "compaction"), None
    )
    entries = path
    if compaction is not None:
        at = path.index(compaction)
        kept: list[dict[str, Any]] = []
        found = False
        for e in path[:at]:
            found = found or e.get("id") == compaction.get("firstKeptEntryId")
            is_system = (
                e.get("type") == "message" and e["message"].get("role") == "system"
            )
            if found and not is_system:
                kept.append(e)
        entries = [compaction, *kept, *path[at + 1 :]]
    edits = {e.get("targetId"): e for e in entries if e.get("type") == "context_edit"}
    out: list[dict[str, Any]] = []
    for i, e in enumerate(entries):
        kind = e.get("type")
        if kind == "message":
            msgs = [dict(e["message"])]
        elif kind == "custom_message":
            msgs = [
                {
                    "role": "user",
                    "content": e.get("content") or [],
                    "note": e.get("customType"),
                }
            ]
        elif kind == "branch_summary" and e.get("summary"):
            text = BRANCH_SUMMARY_PREFIX + e["summary"] + BRANCH_SUMMARY_SUFFIX
            msgs = [{"role": "user", "content": text, "note": "branch_summary"}]
        elif kind == "compaction" and i == 0:
            text = (
                COMPACTION_SUMMARY_PREFIX
                + e.get("summary", "")
                + COMPACTION_SUMMARY_SUFFIX
            )
            msgs = [{"role": "user", "content": text, "note": "compaction_summary"}]
            if e.get("systemMessage"):
                msgs.insert(0, dict(e["systemMessage"]))
        else:
            continue
        edit = edits.get(e.get("id"))
        if edit is not None:
            if edit.get("replacement") is None:
                continue
            for m in msgs:
                if m.get("role") in ("user", "assistant", "toolResult"):
                    m["content"] = edit["replacement"].get("content")
        out.extend(msgs)
    return out


def conversation(path: list[dict[str, Any]]) -> dict[str, Any]:
    """System prompt, tools and the context messages (``user`` / ``assistant`` / ``tool``) the
    planner saw at the end of a branch (pi's projection, see ``project``)."""
    system_sections: dict[str, str] = {}
    system_content: list[str] = []
    tools: dict[str, dict[str, Any]] = {}
    forced = next(
        (
            str((e.get("data") or {}).get("text", ""))
            for e in reversed(path)
            if e.get("type") == "custom" and e.get("customType") == SYSTEM_PROMPT_ENTRY
        ),
        None,
    )
    messages: list[dict[str, Any]] = []
    for m in project(path):
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
            parts = _parts(m.get("content"))
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
                    "error": bool(m.get("isError")),
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
        "compacted": any(e.get("type") == "compaction" for e in path),
    }


def last_attempt(conv: dict[str, Any]) -> int:
    """Keep only the attempt after the last successful explore ``reset`` (in place): the prompt (what
    preceded the first reply), then the reset's result, as a user turn showing the restored scene,
    and everything after it. Returns the number of attempts dropped."""
    msgs = conv["messages"]
    resets = [
        i
        for i, m in enumerate(msgs)
        if m["role"] == "tool" and m["name"] == "reset" and not m["error"]
    ]
    if not resets:
        return 0
    first_reply = next(i for i, m in enumerate(msgs) if m["role"] == "assistant")
    r = resets[-1]
    conv["messages"] = [
        *msgs[:first_reply],
        {"role": "user", "parts": msgs[r]["parts"]},
        *[
            m
            for m in msgs[r + 1 :]
            if not (m["role"] == "tool" and m["id"] == msgs[r]["id"])
        ],
    ]
    return len(resets)


def prune_images(
    messages: list[dict[str, Any]],
    keep: int | None,
    *,
    anchor: bool = False,
    stub: str = IMAGE_STUB,
) -> None:
    """The robot's ``--keep-images`` pruning (packages/embodied/src/robot.ts ``context`` hook), in
    place: only tool results lose images; walking from the newest result back, each result's images
    in order, the first ``keep`` survive and the rest become ``stub``; with ``anchor``
    (``--anchor-image``) the first image of the earliest result with one always stays."""
    if keep is None or keep < 0:
        return
    tools = [m for m in messages if m["role"] == "tool"]
    first = next(
        (m for m in tools if any(p.get("type") == "image" for p in m["parts"])), None
    )
    left = keep
    for m in reversed(tools):
        anchored = anchor and m is first
        new_parts = []
        for p in m["parts"]:
            if p.get("type") == "image":
                if anchored:
                    anchored = False
                elif left > 0:
                    left -= 1
                else:
                    p = {"type": "text", "text": stub}
            new_parts.append(p)
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


class Steering(ValueError):
    """A user message after the first reply, which ShareGPT's alternation cannot hold."""


def to_sharegpt(
    conv: dict[str, Any], *, merge_steering: bool = False
) -> tuple[list[dict[str, str]], list[dict[str, Any]]]:
    """LLaMA-Factory ShareGPT turns (alternating human|observation and gpt|function_call, ending on
    the model) and the images of the turns kept, in placeholder order. Everything before the first
    reply is the human prompt. A user message after a reply (steering, an extension's note, a
    summary) raises ``Steering`` unless ``merge_steering`` folds it into the observation before it."""
    turns: list[dict[str, Any]] = []
    replied = False
    for m in conv["messages"]:
        if m["role"] == "assistant":
            replied = True
            if m["calls"]:
                calls = [
                    {"name": c["name"], "arguments": c["arguments"]} for c in m["calls"]
                ]
                value = _escape(
                    json.dumps(
                        calls[0] if len(calls) == 1 else calls, ensure_ascii=False
                    )
                )
                if m["text"]:
                    value = f"<think>\n{_escape(m['text'])}\n</think>\n\n{value}"
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
        value, images = _text(m["parts"])
        if not replied:
            if turns:
                turns[0]["value"] += "\n" + value
                turns[0]["images"] += images
            else:
                turns.append({"from": "human", "value": value, "images": images})
            continue
        if (
            m["role"] == "user"
            and turns[-1]["from"] == "observation"
            and not merge_steering
        ):
            raise Steering("a user message follows a tool result")
        if turns[-1]["from"] in ("human", "observation"):
            turns[-1]["value"] += "\n" + value
            turns[-1]["images"] += images
        else:
            side = "observation" if m["role"] == "tool" else "human"
            turns.append({"from": side, "value": value, "images": images})
    while turns and turns[-1]["from"] in ("human", "observation"):
        turns.pop()
    images = [p for t in turns for p in t.pop("images")]
    return turns, images


def to_openai(
    conv: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """OpenAI chat messages (images as ``<image>`` placeholders in the text; user messages stay
    their own messages), ending on the model, and the images of the messages kept, in order."""
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
    """The RL prompt: the system prompt and the first user turn (everything before the first reply),
    and that turn's own images."""
    parts: list[dict[str, Any]] = []
    for m in conv["messages"]:
        if m["role"] == "assistant":
            break
        parts += m["parts"]
    text, images = _text(parts)
    return [
        {"role": "system", "content": _escape(conv["system"])},
        {"role": "user", "content": text},
    ], images


def reset_spec(
    task: dict[str, Any], result: dict[str, Any]
) -> tuple[int | None, dict[str, Any]]:
    """(seed, init_state) that reproduce the episode's initial state: the ``seed`` task flag (or the
    result's), and the robot's other task flags, which it resets with (LIBERO: suite + task, the
    init state being ``seed`` modulo the task's init states; ManiSkill: env-id + scene)."""
    raw = task.get("seed", result.get("seed"))
    try:
        seed = int(raw) if raw not in (None, "") else None
    except (TypeError, ValueError):
        seed = None
    return seed, {k: v for k, v in task.items() if k not in ("robot", "seed")}


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
    ``ground_truth``'s seed and init state) must put its ``robot_result`` (the robot's own result
    row) into ``extra_info["rollout_result"]``; the text of the rollout never earns a reward, and
    neither does the recorded episode's outcome (``extra_info["source"]``)."""
    result = (extra_info or {}).get("rollout_result")
    if not isinstance(result, dict):
        raise ValueError(
            "compute_score needs extra_info['rollout_result'] from the rollout's environment"
        )
    want = {**(ground_truth.get("init_state") or {}), "seed": ground_truth.get("seed")}
    for key, value in want.items():
        if value is not None and key in result and str(result[key]) != str(value):
            raise ValueError(
                f"rollout_result {key}={result[key]!r} is not the prompt's {value!r}"
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


def planner_type(result: dict[str, Any], entries: list[dict[str, Any]]) -> str:
    """Who planned the episode: the result's ``planner`` field (``model``, ``flash``, ``replay``,
    ``human``, ``scripted``, ...) when the run recorded one; otherwise inferred from the session's
    planner provider: ``flash/...`` is flash, ``replay/...`` replay, ``human/...`` a human, and
    anything else a model (a scripted stand-in behind an OpenAI-compatible endpoint is only
    recognized through the field)."""
    recorded = result.get("planner")
    if isinstance(recorded, dict):
        recorded = recorded.get("type") or recorded.get("kind")
    if isinstance(recorded, str) and recorded:
        return recorded
    provider = next(
        (
            e.get("provider")
            for e in reversed(entries)
            if e.get("type") == "model_change"
        ),
        None,
    ) or next(
        (
            e["message"].get("provider")
            for e in reversed(entries)
            if e.get("type") == "message" and e["message"].get("role") == "assistant"
        ),
        None,
    )
    return provider if provider in NON_MODEL_PROVIDERS else "model"


def excluded(
    result: dict[str, Any], conv: dict[str, Any], include: set[str]
) -> str | None:
    """Why an episode is left out by default: ``privileged`` (simulator ground truth was on offer),
    ``operator`` (a human judged or ended it), or None."""
    if result.get("privileged") and "privileged" not in include:
        return "privileged"
    judged = (
        result.get("operator_verdict")
        or result.get("operator_finished")
        or result.get("verdict_source")
    )
    if judged and "operator" not in include:
        return "operator"
    return None


def export_planner(
    roots: Iterable[Path],
    output: Path,
    *,
    fmt: str = "sharegpt",
    include_failures: bool = False,
    include: Iterable[str] = (),
    keep_images: int | None = None,
    anchor_image: bool | None = None,
    image_stub: str = IMAGE_STUB,
    merge_steering: bool = False,
    planners: Iterable[str] = ("model",),
    name: str = "pi_embodied_planner",
) -> dict[str, Any]:
    """Export the sessions below ``roots`` to ``output/planner.jsonl``; returns counts and skip
    reasons. ``include`` opts into what is left out by default: ``privileged``, ``operator``,
    ``explore_attempts`` (the attempts before an explore ``reset``). ``planners``: the planner
    types exported (``planner_type``; default only ``model``; ``all`` takes every type).
    ``anchor_image`` defaults to the run's recorded ``--anchor-image``."""
    if fmt not in FORMATS:
        raise ValueError(f"format must be one of {FORMATS}")
    include = set(include)
    planners = set(planners)
    unknown = include - set(INCLUDES)
    if unknown:
        raise ValueError(f"include takes {INCLUDES}, not {sorted(unknown)}")
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
        planner = planner_type(result, entries)
        why = (
            f"planner_{planner}"
            if "all" not in planners and planner not in planners
            else excluded(result, conv, include)
        )
        if why:
            skipped.setdefault(why, []).append(str(session))
            continue
        # An RL prompt is the episode's start; SFT trains on the attempt that the export keeps.
        keep_all = fmt == "verl-rl" or "explore_attempts" in include
        dropped = 0 if keep_all else last_attempt(conv)
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
            "explore_attempts_dropped": dropped,
            "planner": planner,
        }
        rel = f"images/{episode_id}"
        if fmt == "verl-rl":
            seed, init_state = reset_spec(task, result)
            if seed is None:
                skipped.setdefault("no_seed", []).append(str(session))
                continue
            prompt, kept = rl_prompt(conv)
            env = {"robot": robot, "seed": seed, "init_state": init_state}
            row: dict[str, Any] = {
                "data_source": f"pi_embodied/{robot}",
                "prompt": prompt,
                "images": [
                    {"image": str((output / p).resolve())}
                    for p in _write(kept, output, rel)
                ],
                "ability": "embodied_planning",
                "agent_name": "tool_agent",
                "reward_model": {"style": "env_success", "ground_truth": env},
                "extra_info": {
                    "index": len(rows),
                    "id": episode_id,
                    "seed": seed,
                    "task": task,
                    "tools": conv["tools"],
                    "tools_kwargs": {
                        t["name"]: {"create_kwargs": env} for t in conv["tools"]
                    },
                    "interaction_kwargs": {"name": "pi_embodied", **env},
                    # The recorded episode: metadata, never the reward of a new rollout.
                    "source": source,
                },
                "id": episode_id,
                "robot": robot,
            }
            content: Any = row["prompt"]
        else:
            anchor = (
                bool(result.get("anchor_image"))
                if anchor_image is None
                else anchor_image
            )
            prune_images(conv["messages"], keep_images, anchor=anchor, stub=image_stub)
            meta = {
                "id": episode_id,
                "robot": robot,
                "task": task,
                **source,
                "reward": 1.0 if success else 0.0,
            }
            if fmt == "sharegpt":
                try:
                    turns, kept = to_sharegpt(conv, merge_steering=merge_steering)
                except Steering:
                    skipped.setdefault("steering", []).append(str(session))
                    continue
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
