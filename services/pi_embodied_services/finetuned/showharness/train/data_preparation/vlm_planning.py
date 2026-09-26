# Copyright 2026 The Show-Harness Authors. Licensed under the Apache License, Version 2.0.
# Modified by pi-embodied: the parts of github.com/showlab/Show-Harness @137d571 that
# generate_subgoals.py / generate_affordance.py import (core/v0_types.Subgoal,
# plugins/subgoal/{plugin.py,agent.py,subgoal_planner.txt} and the JSON call of
# core/vlm/vlm_client.py), reduced to what the two generators use and made standalone: an
# OpenAI-compatible chat call over urllib instead of VLMClient, no Show-Harness imports.
"""The subgoal planner and the grasp-affordance role for offline data generation.

``plan_subgoals`` is Show-Harness's SubgoalPlanner as the runtime runs it: the planner
prompt on the first frame(s), guided JSON first (vLLM ``structured_outputs``), then free
JSON with the strict suffix, then a text retry, and a single whole-task stage when all
fail; the items are validated as ``Subgoal.from_dict`` and the pre-grasp merge folds a
pure reach/align stage into the GRASP stage of the same target. ``plan_affordance`` is
generate_affordance.py's GraspAffordance call (guided, then prompt-only JSON).
"""

from __future__ import annotations

import base64
import io
import json
import re
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# No-think regime (plugins/subgoal/agent.py NO_THINK_CHAT_TEMPLATE_KWARGS).
NO_THINK = {"enable_thinking": False, "thinking": False}
# The planner's output budget (agent.py max_tokens; 2048 truncated long plans upstream).
PLAN_MAX_TOKENS = 4096
AFFORDANCE_MAX_TOKENS = 512
STRICT_SUFFIX = (
    "\n\nReturn only the JSON object in the format shown above. "
    "Do not include thought, reasoning, markdown, or prose."
)

# plugins/subgoal/subgoal_planner.txt, verbatim ({video_ref} renders empty offline).
SUBGOAL_PROMPT = """ROLE: SubgoalPlanner
TASK: {task}



Return an ordered JSON plan:
{{
  "subgoals": [
    {{
      "id": "short_snake_case_id",
      "target": "object or destination",
      "affordance": "visible part or placement region",
      "motion": "semantic stage label",
      "description": "visual strategy for this stage",
      "completion": "visible condition that means this stage is complete"
    }}
  ]
}}

### STRICT RULES

1. Stage Segmentation

-Break the task down into meaningful visual milestones (e.g., GRASP, LIFT, MOVE, PLACE, RELEASE, RETREAT)
-MERGE: Do NOT split immediate pre-grasp steps. Combine approach, align, lower, and close into a single `GRASP` stage
-SEPARATE: Keep lift/clearance after a successful grasp as a separate `LIFT` stage
-RETREAT: After every `RELEASE`, add a `RETREAT` stage that lifts the gripper up

2. Affordance Selection

-ONE specific part -- never alternatives like "left end or middle"
-Must be visible in AgentView & bracketable by open fingers
-Containers/Hollow objects (cups, bowls): Target left/right rim or edge.
-Simple Solid objects (blocks): main body

3. Completion Criteria

-ALL completion conditions MUST be strictly judgeable from raw 2D images
-Movement stages: End with a stable visual spatial relation, NOT a gripper event
-Set-aside placements: If moving an object to another location, ensure it is placed away from the origin point on the table
-MUST distinguish among similar objects

Return JSON ONLY."""

_FIELD = {"type": "string"}
SUBGOAL_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "subgoals": {
            "type": "array",
            "minItems": 1,
            "items": {
                "type": "object",
                "properties": {
                    k: _FIELD
                    for k in (
                        "id",
                        "target",
                        "affordance",
                        "motion",
                        "description",
                        "completion",
                    )
                },
                "required": [
                    "id",
                    "target",
                    "affordance",
                    "motion",
                    "description",
                    "completion",
                ],
            },
        }
    },
    "required": ["subgoals"],
}

# generate_affordance.py's _AFFORDANCE_PROMPT, verbatim.
AFFORDANCE_PROMPT = """ROLE: GraspAffordance

Task: {task}

Identify the single object to grasp first for this task, and the best visible grasp point on
it for a parallel-jaw gripper.

Return JSON only:
{{
  "target": "the object to grasp, with a distinguishing relation if similar objects exist",
  "affordance": "the visible part or contact region to put between the gripper fingers"
}}

Rules:
- Pick a grasp point clearly visible in AgentView and reachable by the open fingers
- For simple solid objects (blocks, fruit), the affordance is the object body itself
- For hollow/container objects (bowls, cups), pick a visible side/rim contact region,
  preferably the left or right wall
- Return JSON only, no thought, no markdown, no prose
"""
AFFORDANCE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {"target": _FIELD, "affordance": _FIELD},
    "required": ["target", "affordance"],
    "additionalProperties": False,
}


class VlmError(RuntimeError):
    """The VLM returned nothing usable (the upstream ``VLM returned ...`` errors)."""


# ── the chat call ────────────────────────────────────────────────────────────


def _data_url(path: Path) -> str:
    return "data:image/png;base64," + base64.b64encode(_png_bytes(path)).decode()


def _png_bytes(path: Path) -> bytes:
    raw = path.read_bytes()
    if raw[:8] == b"\x89PNG\r\n\x1a\n":
        return raw
    # Not a PNG (a JPEG frame): re-encode, as VLMClient encodes arrays to PNG.
    from PIL import Image

    buf = io.BytesIO()
    Image.open(path).convert("RGB").save(buf, format="PNG")
    return buf.getvalue()


@dataclass
class Client:
    """One OpenAI-compatible endpoint (vLLM, LLaMA-Factory's API server, a hosted API)."""

    base_url: str
    model: str
    api_key: str = "EMPTY"
    timeout_s: float = 60.0
    # LLaMA-Factory's API server has no guided decoding: skip the schema attempt.
    guided: bool = True

    def complete(
        self,
        prompt: str,
        images: list[Path],
        max_tokens: int,
        schema: dict[str, Any] | None = None,
        think: bool = False,
    ) -> str:
        body: dict[str, Any] = {
            "model": self.model,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        *(
                            {"type": "image_url", "image_url": {"url": _data_url(p)}}
                            for p in images
                        ),
                        {"type": "text", "text": prompt},
                    ],
                }
            ],
            "temperature": 0.0,
            "max_tokens": max_tokens,
        }
        if not think:
            body["chat_template_kwargs"] = dict(NO_THINK)
        if schema is not None:
            body["structured_outputs"] = {"json": schema}
        req = urllib.request.Request(
            self.base_url.rstrip("/") + "/chat/completions",
            data=json.dumps(body).encode(),
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key or 'EMPTY'}",
            },
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout_s) as res:
                data = json.loads(res.read())
        except urllib.error.HTTPError as exc:
            raise RuntimeError(
                f"VLM request failed: HTTP {exc.code}: {exc.read()[:300]!r}"
            ) from exc
        try:
            message = data["choices"][0]["message"]
        except (KeyError, IndexError, TypeError) as exc:
            raise VlmError(f"VLM returned no message: {str(data)[:300]}") from exc
        text = message.get("content") or ""
        if not str(text).strip():
            text = message.get("reasoning_content") or ""
        return re.sub(r"<think>[\s\S]*?</think>", "", str(text)).strip()


def parse_json_object(raw: str) -> dict[str, Any]:
    """The first JSON object in ``raw`` (agent.py _parse_json_object)."""
    text = str(raw or "").strip()
    try:
        value = json.loads(text)
    except json.JSONDecodeError:
        value = None
    if isinstance(value, dict):
        return value
    decoder = json.JSONDecoder()
    for i, char in enumerate(text):
        if char != "{":
            continue
        try:
            candidate, _ = decoder.raw_decode(text[i:])
        except json.JSONDecodeError:
            continue
        if isinstance(candidate, dict):
            return candidate
    raise VlmError(f"VLM returned non-JSON text: {text[:300]!r}")


# ── subgoals (core/v0_types.Subgoal, plugins/subgoal/plugin.py) ─────────────────


@dataclass
class Subgoal:
    id: str
    target: str
    affordance: str
    motion: str
    description: str
    completion: str

    @classmethod
    def from_dict(cls, data: dict[str, Any], index: int) -> Subgoal:
        text = str(data.get("motion") or "").strip()
        if not text:
            raise ValueError(f"Subgoal {index} is missing required motion")
        motion = re.sub(r"[^A-Za-z0-9_]+", "_", text).strip("_").upper()
        if not motion:
            raise ValueError(f"Subgoal {index} has invalid motion {text!r}")
        affordance = str(data.get("affordance", "")).strip()
        if not affordance:
            raise ValueError(f"Subgoal {index} is missing required affordance")
        description = str(data.get("description") or "").strip()
        if not description:
            raise ValueError(f"Subgoal {index} is missing required description")
        return cls(
            id=str(data.get("id") or f"subgoal_{index}"),
            target=str(data.get("target") or ""),
            affordance=affordance,
            motion=motion,
            description=description,
            completion=str(data.get("completion") or "").strip(),
        )

    def to_prompt_dict(self) -> dict[str, str]:
        return {
            "id": self.id,
            "target": self.target,
            "affordance": self.affordance,
            "motion": self.motion,
            "description": self.description,
            "completion": self.completion,
        }


def subgoal_items(payload: dict[str, Any]) -> list[dict[str, Any]]:
    items = payload.get("subgoals")
    if isinstance(items, list):
        return [item for item in items if isinstance(item, dict)]
    if isinstance(items, dict):
        return [items]
    item = payload.get("subgoal")
    if isinstance(item, dict):
        return [item]
    required = {"id", "target", "affordance", "motion", "description", "completion"}
    return [payload] if required.issubset(payload) else []


def _words(text: str, terms: tuple[str, ...]) -> bool:
    return any(
        re.search(rf"(?<![A-Za-z0-9]){re.escape(t)}(?![A-Za-z0-9])", text)
        for t in terms
    )


def _text(s: Subgoal) -> str:
    return " ".join(
        str(getattr(s, f) or "").lower()
        for f in ("id", "target", "affordance", "motion", "description", "completion")
    )


def _norm_target(value: str) -> str:
    text = re.sub(r"\b(the|a|an)\b", " ", str(value or "").lower())
    return re.sub(r"\s+", " ", text).strip()


def _same_target(a: str, b: str) -> bool:
    left, right = _norm_target(a), _norm_target(b)
    return bool(left and right) and (left == right or left in right or right in left)


_REACH = (
    "approach",
    "reach",
    "align",
    "center",
    "position",
    "move to",
    "move toward",
    "above",
    "over",
)
_ACTS = (
    "grasp",
    "close",
    "clamp",
    "secure",
    "lift",
    "release",
    "open",
    "place",
    "drop",
)
_GRASP = ("grasp", "close", "clamp", "secure", "pick")


def _should_merge(current: Subgoal, nxt: Subgoal) -> bool:
    # A DeepPlan REASON checkpoint is a sentinel, never folded.
    if "REASON" in (current.motion, nxt.motion):
        return False
    if not _same_target(current.target, nxt.target) or not _words(_text(nxt), _GRASP):
        return False
    text = _text(current)
    return _words(text, _REACH) and not _words(text, _ACTS)


def _join(first: str, second: str) -> str:
    parts = [
        " ".join(str(t or "").split()).rstrip(".")
        for t in (first, second)
        if " ".join(str(t or "").split())
    ]
    return ". Then ".join(parts) + "." if parts else ""


def merge_pregrasp(subgoals: list[Subgoal]) -> list[Subgoal]:
    """Fold a pure reach/align stage into the GRASP stage of the same target."""
    out: list[Subgoal] = []
    i = 0
    while i < len(subgoals):
        cur = subgoals[i]
        if i + 1 < len(subgoals) and _should_merge(cur, subgoals[i + 1]):
            grasp = subgoals[i + 1]
            out.append(
                Subgoal(
                    id=grasp.id,
                    target=grasp.target or cur.target,
                    affordance=grasp.affordance or cur.affordance,
                    motion=grasp.motion,
                    description=_join(cur.description, grasp.description),
                    completion=grasp.completion,
                )
            )
            i += 2
            continue
        out.append(cur)
        i += 1
    return out


def _has_subgoals(parsed: dict[str, Any]) -> bool:
    items = parsed.get("subgoals")
    if isinstance(items, list):
        return any(isinstance(item, dict) for item in items)
    return isinstance(items, dict)


def fallback_plan(task: str, errors: list[str]) -> dict[str, Any]:
    """agent.py _fallback_response: one whole-task stage after every retry failed."""
    return {
        "planner_fallback": "subgoal planner VLM returned empty/non-JSON after retries",
        "planner_errors": list(errors),
        "subgoals": [
            {
                "id": "task_fallback",
                "target": task,
                "affordance": "task-relevant visible object or region",
                "motion": "TASK",
                "description": f"Complete the task directly: {task}",
                "completion": f"the task is visibly complete: {task}",
            }
        ],
    }


def plan_subgoals(
    client: Client, task: str, images: list[Path]
) -> tuple[list[Subgoal], dict[str, Any]]:
    """The validated, pre-grasp-merged plan and its record (route, raw text, errors)."""
    started = time.monotonic()
    prompt = SUBGOAL_PROMPT.format(task=task)
    errors: list[str] = []
    attempts: list[tuple[str, str, dict[str, Any] | None, bool]] = []
    if client.guided:
        attempts.append(("guided_json", prompt, SUBGOAL_SCHEMA, False))
    attempts.append(("free_json", prompt + STRICT_SUFFIX, None, False))
    retry = (
        prompt
        + "\n\nRETRY FORMAT: Return exactly one JSON object matching the requested "
        "subgoals format. Preserve every task and scene-specific rule above. "
        "Do not include thought, reasoning, markdown, or prose."
    )
    attempts += [
        ("text_no_think", retry, None, False),
        ("text_default", retry, None, True),
    ]
    parsed: dict[str, Any] | None = None
    route = "task_fallback"
    raw = ""
    for label, text, schema, think in attempts:
        try:
            raw = client.complete(text, images, PLAN_MAX_TOKENS, schema, think)
            candidate = parse_json_object(raw)
            if not _has_subgoals(candidate):
                raise VlmError(
                    f"VLM returned JSON without non-empty subgoals: {raw[:200]}"
                )
            parsed, route = candidate, label
            break
        except VlmError as exc:
            errors.append(f"{label}: {' '.join(str(exc).split())}")
    if parsed is None:
        parsed = fallback_plan(task, errors)
    subgoals = [Subgoal.from_dict(d, i) for i, d in enumerate(subgoal_items(parsed))]
    if not subgoals:
        raise VlmError(f"Planner JSON has no subgoals: {raw[:300]!r}")
    return merge_pregrasp(subgoals), {
        "route": route,
        "errors": errors,
        "raw": raw,
        "latency_s": round(time.monotonic() - started, 3),
    }


def plan_affordance(client: Client, task: str, images: list[Path]) -> dict[str, str]:
    """generate_affordance.py _plan_affordance: {target, affordance, raw}."""
    prompt = AFFORDANCE_PROMPT.format(task=task)
    loose = prompt + "\n\nReturn only the JSON object in the format shown above."
    tries = [(prompt, AFFORDANCE_SCHEMA)] if client.guided else []
    tries.append((loose, None))
    last: Exception | None = None
    for text, schema in tries:
        try:
            raw = client.complete(text, images, AFFORDANCE_MAX_TOKENS, schema)
            data = parse_json_object(raw)
            return {
                "target": str(data.get("target", "")).strip(),
                "affordance": str(data.get("affordance", "")).strip(),
                "raw": raw,
            }
        except VlmError as exc:
            last = exc
    raise VlmError(str(last))


# ── rollouts ──────────────────────────────────────────────────────────────────


def first_frames(rollout: Path) -> tuple[Path | None, Path | None]:
    """The first agentview and wrist frames of a rollout: Show-Harness's
    ``agentview/0000.png`` layout, pi's GUMI ``images/agentview/0000.png`` layout, or
    the first ``actions.jsonl`` row's image paths."""
    for base in (rollout, rollout / "images"):
        agent = base / "agentview" / "0000.png"
        if agent.exists():
            wrist = base / "wrist" / "0000.png"
            return agent, wrist if wrist.exists() else None
    actions = rollout / "actions.jsonl"
    if actions.exists():
        for line in actions.read_text().splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            agent = row.get("agentview")
            wrist = row.get("wrist")
            return (
                rollout / agent if agent else None,
                rollout / wrist if wrist and (rollout / wrist).exists() else None,
            )
    return None, None


def is_rollout(path: Path) -> bool:
    return first_frames(path)[0] is not None or (path / "actions.jsonl").exists()


def resolve_target(path: Path) -> tuple[Path | None, Path | None]:
    """``(image_rollout_dir, out_dir)``: a rollout infers from itself and writes into itself;
    a task folder infers once from its first rollout and writes into the task folder."""
    if is_rollout(path):
        return path, path
    if path.is_dir():
        rollouts = sorted(c for c in path.iterdir() if c.is_dir() and is_rollout(c))
        if rollouts:
            return rollouts[0], path
    return None, None


def add_vlm_args(parser: Any) -> None:
    """The flags rollouts_to_alpaca.py forwards (--vlm-backend, --vlm-url, --model, --api-key)."""
    parser.add_argument(
        "--vlm-backend",
        default=None,
        help="llamafactory = no guided decoding (LLaMA-Factory's API server); anything "
        "else is an OpenAI-compatible server with vLLM structured outputs",
    )
    parser.add_argument(
        "--vlm-url",
        default="http://localhost:8000/v1",
        help="OpenAI-compatible base URL",
    )
    parser.add_argument("--model", required=True, help="Served model name")
    parser.add_argument("--api-key", default="EMPTY")
    parser.add_argument("--timeout-s", type=float, default=60.0)


def client_from(args: Any) -> Client:
    return Client(
        base_url=args.vlm_url,
        model=args.model,
        api_key=args.api_key or "EMPTY",
        timeout_s=args.timeout_s,
        guided=str(args.vlm_backend or "").lower() != "llamafactory",
    )
