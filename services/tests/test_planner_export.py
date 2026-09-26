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

"""Planner session export (flywheel/planner_export.py) on synthetic pi sessions."""

from __future__ import annotations

import base64
import json
from pathlib import Path

import pytest

from pi_embodied_services.flywheel import cli
from pi_embodied_services.flywheel.planner_export import (
    FORMATS,
    compute_score,
    export_planner,
    placeholders,
)

PNG_A = b"\x89PNG\r\n\x1a\nAAAA"
PNG_B = b"\x89PNG\r\n\x1a\nBBBB"
PNG_C = b"\x89PNG\r\n\x1a\nCCCC"
TOOLS = [
    {
        "name": "move_to",
        "description": "Move.",
        "parameters": {"type": "object", "properties": {"x": {"type": "number"}}},
    },
    {"name": "finish", "description": "End.", "parameters": {"type": "object"}},
]


def image(data: bytes) -> dict:
    return {
        "type": "image",
        "data": base64.b64encode(data).decode(),
        "mimeType": "image/png",
    }


def assistant(content: list, stop: str = "toolUse") -> dict:
    return {"role": "assistant", "content": content, "stopReason": stop, "usage": {}}


def call(i: str, name: str, **args) -> dict:
    return {"type": "toolCall", "id": i, "name": name, "arguments": args}


def result(i: str, name: str, *content) -> dict:
    return {
        "role": "toolResult",
        "toolCallId": i,
        "toolName": name,
        "content": list(content),
        "isError": False,
    }


def text(t: str) -> dict:
    return {"type": "text", "text": t}


def write_session(
    episode: Path,
    *,
    outcome: dict | None,
    result_json: dict | None = None,
    forced_prompt: bool = True,
    fork: bool = False,
    prompt_image: bool = False,
    literal: bool = False,
    finish_image: bool = False,
    cut: bool = False,
) -> Path:
    """A pi session like an eval episode's: header, task, system message, a user prompt, turns.
    ``prompt_image``: the user prompt carries an image; ``literal``: a tool result's text holds a
    literal ``<image>``; ``finish_image``: the finish result carries a frame; ``cut``: the budget
    ended the episode on a tool result with a frame, before any finish."""
    episode.mkdir(parents=True, exist_ok=True)
    entries: list[dict] = [
        {"type": "session", "version": 3, "id": "s", "cwd": "/"},
    ]
    last: str | None = None

    def add(entry: dict) -> None:
        nonlocal last
        entry = {"id": f"e{len(entries)}", "parentId": last, **entry}
        entries.append(entry)
        last = entry["id"]

    add({"type": "model_change", "provider": "selfhost", "modelId": "muse"})
    add(
        {
            "type": "custom",
            "customType": "robot_task",
            "data": {"robot": "toy", "task": "1"},
        }
    )
    if forced_prompt:
        add(
            {
                "type": "custom",
                "customType": "robot_system_prompt",
                "data": {"text": "You drive the toy arm."},
            }
        )
    add(
        {
            "type": "message",
            "message": {
                "role": "system",
                "content": "",
                "sections": {"preamble": "You are pi.", "cwd": "<cwd>\n/\n</cwd>"},
                "toolsAdded": TOOLS,
            },
        }
    )
    add(
        {
            "type": "message",
            "message": {
                "role": "user",
                "content": [text("Solve the task.")]
                + ([image(PNG_C)] if prompt_image else []),
            },
        }
    )
    # An errored reply (pi retried it): not part of the data.
    add({"type": "message", "message": assistant([text("partial")], stop="error")})
    add(
        {
            "type": "message",
            "message": assistant([text("look first"), call("c1", "move_to", x=1)]),
        }
    )
    add(
        {
            "type": "message",
            "message": result(
                "c1",
                "move_to",
                text("moved <image> tag" if literal else "moved"),
                image(PNG_A),
            ),
        }
    )
    if fork:
        # An abandoned branch: the session goes on from the result above.
        fork_parent = last
        add({"type": "message", "message": assistant([call("cx", "move_to", x=99)])})
        last = fork_parent
    add(
        {
            "type": "custom_message",
            "customType": "vdm",
            "content": "the block moved",
            "display": True,
        }
    )
    add(
        {
            "type": "message",
            "message": assistant(
                [
                    {"type": "thinking", "thinking": "secret"},
                    call("c2", "move_to", x=2),
                    call("c3", "move_to", x=3),
                ]
            ),
        }
    )
    add(
        {
            "type": "message",
            "message": result("c2", "move_to", text("ok2"), image(PNG_B)),
        }
    )
    add({"type": "message", "message": result("c3", "move_to", text("ok3"))})
    if cut:
        add({"type": "message", "message": assistant([call("c5", "move_to", x=5)])})
        add(
            {
                "type": "message",
                "message": result("c5", "move_to", text("budget"), image(PNG_C)),
            }
        )
    else:
        add(
            {
                "type": "message",
                "message": assistant([call("c4", "finish", status="success")]),
            }
        )
        extra = [image(PNG_C)] if finish_image else []
        add(
            {
                "type": "message",
                "message": result("c4", "finish", text("success"), *extra),
            }
        )
    if outcome is not None:
        add(
            {
                "type": "custom",
                "customType": "robot_result",
                "data": {"robot": "toy", **outcome},
            }
        )
    path = episode / "2026-09-26T00-00-00-000Z_0000abcd.jsonl"
    path.write_text("".join(json.dumps(e) + "\n" for e in entries))
    (episode / "stdout.log").write_text("")
    if result_json is not None:
        (episode / "result.json").write_text(json.dumps(result_json))
    return path


def rows(out: Path) -> list[dict]:
    return [
        json.loads(line) for line in (out / "planner.jsonl").read_text().splitlines()
    ]


def test_sharegpt_export_of_a_success(tmp_path: Path) -> None:
    runs = tmp_path / "runs"
    write_session(
        runs / "ep_ok",
        outcome={"success": True, "planner_error": None, "env_error": False},
        result_json={"status": "success", "robot": "toy", "model": "selfhost/muse"},
        fork=True,
    )
    out = tmp_path / "out"
    summary = export_planner([runs], out)
    assert summary["episodes"] == 1 and summary["images"] == 2
    [row] = rows(out)
    assert row["system"] == "You drive the toy arm."
    assert row["system_prompt_source"] == "robot_entry"
    assert [t["name"] for t in json.loads(row["tools"])] == ["move_to", "finish"]
    assert (
        row["reward"] == 1.0 and row["success"] is True and row["status"] == "success"
    )
    assert (
        row["task"] == {"robot": "toy", "task": "1"} and row["model"] == "selfhost/muse"
    )
    turns = row["conversations"]
    assert [t["from"] for t in turns] == [
        "human",
        "function_call",
        "observation",
        "function_call",
        "observation",
        "function_call",
    ], "alternating, ending on the model: the finish result is dropped"
    assert turns[0]["value"] == "Solve the task."
    assert (
        turns[1]["value"]
        == '<think>\nlook first\n</think>\n\n{"name": "move_to", "arguments": {"x": 1}}'
    )
    assert turns[2]["value"] == "moved\n<image>\nthe block moved", (
        "the extension note joins the observation"
    )
    assert json.loads(turns[3]["value"]) == [
        {"name": "move_to", "arguments": {"x": 2}},
        {"name": "move_to", "arguments": {"x": 3}},
    ], "parallel calls are a list; thinking is dropped"
    assert turns[4]["value"] == "ok2\n<image>\nok3"
    assert "99" not in json.dumps(turns), "the abandoned branch is not exported"
    assert sum(t["value"].count("<image>") for t in turns) == len(row["images"])
    assert [(out / p).read_bytes() for p in row["images"]] == [PNG_A, PNG_B]
    assert all(p.startswith(f"images/{row['id']}/") for p in row["images"])
    info = json.loads((out / "dataset_info.json").read_text())
    assert info["pi_embodied_planner"]["file_name"] == "planner.jsonl"
    assert info["pi_embodied_planner"]["tags"]["function_tag"] == "function_call"


def test_failures_only_on_request_and_invalid_episodes_never(tmp_path: Path) -> None:
    runs = tmp_path / "runs"
    write_session(
        runs / "ok", outcome={"success": True}, result_json={"status": "success"}
    )
    write_session(
        runs / "fail", outcome={"success": False}, result_json={"status": "failure"}
    )
    write_session(
        runs / "broken",
        outcome={"success": True, "planner_error": "503"},
        result_json={"status": "planner_error"},
    )
    # No result.json: eval.sh's rule on the robot_result entry (success, or terminated for LIBERO).
    write_session(runs / "raw_ok", outcome={"terminated": True})
    write_session(runs / "raw_fail", outcome={"success": False})
    write_session(runs / "raw_env", outcome={"success": True, "env_error": True})
    write_session(runs / "unfinished", outcome=None)
    summary = export_planner([runs], tmp_path / "a")
    assert summary["episodes"] == 2 and summary["failures"] == 0
    assert summary["skipped"] == {
        "env_error": 1,
        "failure": 2,
        "missing": 1,
        "planner_error": 1,
    }
    summary = export_planner([runs], tmp_path / "b", include_failures=True)
    assert (summary["successes"], summary["failures"]) == (2, 2)
    by_id = {r["id"].split("-")[0]: r for r in rows(tmp_path / "b")}
    assert sorted(by_id) == ["fail", "ok", "raw_fail", "raw_ok"]
    assert by_id["fail"]["reward"] == 0.0 and by_id["raw_ok"]["reward"] == 1.0


def test_a_claimed_success_the_environment_denies_earns_nothing(tmp_path: Path) -> None:
    runs = tmp_path / "runs"
    write_session(runs / "lie", outcome={"success": False, "claimed": "success"})
    assert export_planner([runs], tmp_path / "out")["episodes"] == 0
    export_planner([runs], tmp_path / "all", include_failures=True)
    assert rows(tmp_path / "all")[0]["reward"] == 0.0


def test_openai_sft_format(tmp_path: Path) -> None:
    runs = tmp_path / "runs"
    write_session(runs / "ep", outcome={"success": True}, forced_prompt=False)
    out = tmp_path / "out"
    export_planner([runs], out, fmt="openai")
    [row] = rows(out)
    assert not (out / "dataset_info.json").exists()
    msgs = row["messages"]
    assert msgs[0] == {"role": "system", "content": "You are pi.\n\n<cwd>\n/\n</cwd>"}
    assert row["system_prompt_source"] == "session", (
        "no robot entry: the recorded system message"
    )
    assert [m["role"] for m in msgs] == [
        "system",
        "user",
        "assistant",
        "tool",
        "user",
        "assistant",
        "tool",
        "tool",
        "assistant",
    ]
    assert msgs[2]["tool_calls"][0]["function"] == {
        "name": "move_to",
        "arguments": '{"x": 1}',
    }
    assert msgs[3] == {
        "role": "tool",
        "tool_call_id": "c1",
        "content": "moved\n<image>",
    }
    assert row["tools"][0] == {"type": "function", "function": TOOLS[0]}
    assert [Path(i["image"]).read_bytes() for i in row["images"]] == [PNG_A, PNG_B]
    assert all(Path(i["image"]).is_absolute() for i in row["images"])
    assert row["reward"] == 1.0 and "prompt" not in row and "reward_model" not in row


def test_verl_rl_rows_are_task_prompts_scored_by_the_new_rollout(
    tmp_path: Path,
) -> None:
    runs = tmp_path / "runs"
    write_session(runs / "ok", outcome={"success": True}, prompt_image=True)
    write_session(runs / "fail", outcome={"success": False})
    write_session(runs / "broken", outcome={"success": True, "env_error": True})
    out = tmp_path / "out"
    summary = export_planner([runs], out, fmt="verl-rl")
    assert (summary["episodes"], summary["successes"], summary["failures"]) == (2, 1, 1)
    assert summary["skipped"] == {"env_error": 1}
    by_id = {r["id"].split("-")[0]: r for r in rows(out)}
    ok, fail = by_id["ok"], by_id["fail"]
    assert ok["prompt"] == [
        {"role": "system", "content": "You drive the toy arm."},
        {"role": "user", "content": "Solve the task.\n<image>"},
    ], "the system prompt and the first user turn, not the trajectory"
    assert [Path(i["image"]).read_bytes() for i in ok["images"]] == [PNG_C], (
        "only the prompt's own image"
    )
    assert fail["prompt"][1] == {"role": "user", "content": "Solve the task."}
    assert fail["images"] == []
    env = {"robot": "toy", "task": "1"}
    for r in (ok, fail):
        assert r["reward_model"] == {
            "style": "env_success",
            "ground_truth": {"task": env},
        }
        assert r["agent_name"] == "tool_agent" and r["data_source"] == "pi_embodied/toy"
        assert r["extra_info"]["task"] == env
        assert r["extra_info"]["tools_kwargs"]["move_to"] == {
            "create_kwargs": {"task": env}
        }
        assert "reward" not in r and "success" not in r
    assert ok["extra_info"]["source"]["success"] is True
    assert fail["extra_info"]["source"]["success"] is False
    # The reward hook scores the rollout's own environment verdict, never the recorded one.
    gt, info = ok["reward_model"]["ground_truth"], ok["extra_info"]
    score = lambda result: compute_score(  # noqa: E731
        ok["data_source"], "any text", gt, {**info, "rollout_result": result}
    )
    assert score({"robot": "toy", "task": 1, "success": False}) == 0.0
    assert score({"robot": "toy", "task": 1, "success": True}) == 1.0
    assert score({"robot": "toy", "terminated": True}) == 1.0
    assert score({"robot": "toy", "success": True, "planner_error": "503"}) == 0.0
    with pytest.raises(ValueError, match="rollout_result"):
        compute_score(ok["data_source"], "I succeeded", gt, info)
    with pytest.raises(ValueError, match="task='2'"):
        score({"robot": "toy", "task": "2", "success": True})


def test_every_row_has_one_placeholder_per_image(tmp_path: Path) -> None:
    """Loaders (LLaMA-Factory's mm plugin, VeRL) refuse a row whose counts differ."""
    runs = tmp_path / "runs"
    variants = {
        "plain": {},
        "prompt_image": {"prompt_image": True},
        "literal": {"literal": True},
        "finish_image": {"finish_image": True},
        "cut": {"cut": True},
        "all": {"prompt_image": True, "literal": True, "finish_image": True},
    }
    for name, kw in variants.items():
        write_session(runs / name, outcome={"success": name != "cut"}, **kw)
    cases = [(fmt, keep) for fmt in FORMATS for keep in (None, 0, 1, 2, 5)]
    for fmt, keep in cases:
        out = tmp_path / f"{fmt}-{keep}"
        export_planner([runs], out, fmt=fmt, include_failures=True, keep_images=keep)
        got = rows(out)
        assert len(got) == len(variants), (fmt, keep)
        for row in got:
            content = {"sharegpt": "conversations", "openai": "messages"}.get(
                fmt, "prompt"
            )
            n = placeholders(row[content]) + placeholders(row.get("system", ""))
            assert n == len(row["images"]), (fmt, keep, row["id"])
            # In order: the images are the kept turns' frames, newest last.
            paths = [i if isinstance(i, str) else i["image"] for i in row["images"]]
            assert all((out / p).is_file() for p in paths)
            assert len(set(paths)) == len(paths)
    # The dropped trailing frames are not written or listed; the literal tag is defused.
    [cut] = [r for r in rows(tmp_path / "sharegpt-None") if r["id"].startswith("cut")]
    assert [(tmp_path / "sharegpt-None" / p).read_bytes() for p in cut["images"]] == [
        PNG_A,
        PNG_B,
    ]
    [lit] = [
        r for r in rows(tmp_path / "sharegpt-None") if r["id"].startswith("literal")
    ]
    assert lit["conversations"][2]["value"].startswith(
        "moved &lt;image&gt; tag\n<image>"
    )
    [fin] = [
        r for r in rows(tmp_path / "openai-None") if r["id"].startswith("finish_image")
    ]
    assert len(fin["images"]) == 2, (
        "the finish result's frame goes with the dropped result"
    )


def test_keep_images_stubs_the_older_frames(tmp_path: Path) -> None:
    runs = tmp_path / "runs"
    write_session(runs / "ep", outcome={"success": True})
    out = tmp_path / "out"
    export_planner([runs], out, keep_images=1)
    [row] = rows(out)
    assert [(out / p).read_bytes() for p in row["images"]] == [PNG_B]
    assert (
        row["conversations"][2]["value"]
        == "moved\n[older camera frame omitted]\nthe block moved"
    )


def test_cli_export_planner(tmp_path: Path, capsys) -> None:
    runs = tmp_path / "runs"
    write_session(runs / "ep", outcome={"success": True})
    assert (
        cli.main(["export-planner", str(runs), "--output", str(tmp_path / "out")]) == 0
    )
    summary = json.loads(capsys.readouterr().out)
    assert summary["episodes"] == 1 and summary["format"] == "sharegpt"
