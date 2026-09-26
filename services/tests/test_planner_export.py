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
PNG_D = b"\x89PNG\r\n\x1a\nDDDD"
TOOLS = [
    {
        "name": "move_to",
        "description": "Move.",
        "parameters": {"type": "object", "properties": {"x": {"type": "number"}}},
    },
    {"name": "finish", "description": "End.", "parameters": {"type": "object"}},
]
TASK = {"robot": "toy", "suite": "s", "task": "1", "seed": "7"}


def image(data: bytes) -> dict:
    return {
        "type": "image",
        "data": base64.b64encode(data).decode(),
        "mimeType": "image/png",
    }


def text(t: str) -> dict:
    return {"type": "text", "text": t}


def assistant(content: list, stop: str = "toolUse") -> dict:
    return {"role": "assistant", "content": content, "stopReason": stop, "usage": {}}


def call(i: str, name: str, **args) -> dict:
    return {"type": "toolCall", "id": i, "name": name, "arguments": args}


def result(i: str, name: str, *content, error: bool = False) -> dict:
    return {
        "role": "toolResult",
        "toolCallId": i,
        "toolName": name,
        "content": list(content),
        "isError": error,
    }


class Session:
    """A pi session file being written: header, task, prompt entry, system message, user prompt."""

    def __init__(
        self, *, forced_prompt: bool = True, task: dict | None = None, prompt=None
    ):
        self.entries: list[dict] = [
            {"type": "session", "version": 3, "id": "s", "cwd": "/"}
        ]
        self.last: str | None = None
        self.add({"type": "model_change", "provider": "selfhost", "modelId": "muse"})
        self.add({"type": "custom", "customType": "robot_task", "data": task or TASK})
        if forced_prompt:
            self.add(
                {
                    "type": "custom",
                    "customType": "robot_system_prompt",
                    "data": {"text": "You drive the toy arm."},
                }
            )
        system = {
            "role": "system",
            "content": "",
            "sections": {"preamble": "You are pi.", "cwd": "<cwd>\n/\n</cwd>"},
            "toolsAdded": TOOLS,
        }
        self.msg(system)
        self.msg({"role": "user", "content": prompt or [text("Solve the task.")]})

    def add(self, entry: dict) -> str:
        entry = {"id": f"e{len(self.entries)}", "parentId": self.last, **entry}
        self.entries.append(entry)
        self.last = entry["id"]
        return entry["id"]

    def msg(self, message: dict) -> str:
        return self.add({"type": "message", "message": message})

    def step(
        self, i: str, *shown, name: str = "move_to", said: str = "", error: bool = False
    ) -> str:
        """One reply calling ``name`` and its result showing ``shown``; returns the result's id."""
        self.msg(assistant(([text(said)] if said else []) + [call(i, name, x=i)]))
        return self.msg(result(i, name, *shown, error=error))

    def finish(self, *shown) -> None:
        self.step("fin", text("success"), *shown, name="finish")

    def write(
        self, episode: Path, outcome: dict | None, result_json: dict | None = None
    ) -> Path:
        if outcome is not None:
            self.add(
                {
                    "type": "custom",
                    "customType": "robot_result",
                    "data": {"robot": "toy", **outcome},
                }
            )
        episode.mkdir(parents=True, exist_ok=True)
        path = episode / "2026-09-26T00-00-00-000Z_0000abcd.jsonl"
        path.write_text("".join(json.dumps(e) + "\n" for e in self.entries))
        (episode / "stdout.log").write_text("")
        if result_json is not None:
            (episode / "result.json").write_text(json.dumps(result_json))
        return path


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
    steer: bool = False,
) -> Path:
    """An eval episode: a reply that errored (retried), a move, parallel moves, then finish.
    ``prompt_image``: the user prompt carries an image; ``literal``: a result's text holds a
    literal ``<image>``; ``finish_image``: the finish result carries a frame; ``cut``: the budget
    ended the episode on a result with a frame; ``steer``: a user message follows a result."""
    s = Session(
        forced_prompt=forced_prompt,
        prompt=[text("Solve the task.")] + ([image(PNG_C)] if prompt_image else []),
    )
    s.msg(assistant([text("partial")], stop="error"))
    s.msg(assistant([text("look first"), call("c1", "move_to", x=1)]))
    s.msg(
        result(
            "c1",
            "move_to",
            text("moved <image> tag" if literal else "moved"),
            image(PNG_A),
        )
    )
    if fork:
        parent = s.last
        s.msg(assistant([call("cx", "move_to", x=99)]))
        s.last = parent
    if steer:
        s.msg({"role": "user", "content": [text("go faster")]})
    s.msg(
        assistant(
            [
                {"type": "thinking", "thinking": "secret"},
                call("c2", "move_to", x=2),
                call("c3", "move_to", x=3),
            ]
        )
    )
    s.msg(result("c2", "move_to", text("ok2"), image(PNG_B)))
    s.msg(result("c3", "move_to", text("ok3")))
    if cut:
        s.step("c5", text("budget"), image(PNG_C))
    else:
        s.finish(*([image(PNG_C)] if finish_image else []))
    return s.write(episode, outcome, result_json)


def rows(out: Path) -> list[dict]:
    return [
        json.loads(line) for line in (out / "planner.jsonl").read_text().splitlines()
    ]


def paths(out: Path, row: dict) -> list[bytes]:
    return [
        (out / (i if isinstance(i, str) else i["image"])).read_bytes()
        for i in row["images"]
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
    assert row["task"] == TASK and row["model"] == "selfhost/muse"
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
    assert turns[2]["value"] == "moved\n<image>"
    assert json.loads(turns[3]["value"]) == [
        {"name": "move_to", "arguments": {"x": 2}},
        {"name": "move_to", "arguments": {"x": 3}},
    ], "parallel calls are a list; thinking is dropped"
    assert turns[4]["value"] == "ok2\n<image>\nok3"
    assert "99" not in json.dumps(turns), "the abandoned branch is not exported"
    assert paths(out, row) == [PNG_A, PNG_B]
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


def test_privileged_and_operator_runs_only_on_request(tmp_path: Path) -> None:
    runs = tmp_path / "runs"
    write_session(runs / "plain", outcome={"success": True})
    write_session(runs / "priv", outcome={"success": True, "privileged": True})
    write_session(
        runs / "judged",
        outcome={
            "success": True,
            "operator_verdict": "success",
            "operator_finished": True,
        },
    )
    for fmt in FORMATS:
        summary = export_planner([runs], tmp_path / f"{fmt}-default", fmt=fmt)
        assert summary["skipped"] == {"operator": 1, "privileged": 1}, fmt
        summary = export_planner(
            [runs], tmp_path / f"{fmt}-all", fmt=fmt, include=["privileged", "operator"]
        )
        assert summary["episodes"] == 3, fmt
    with pytest.raises(ValueError, match="include takes"):
        export_planner([runs], tmp_path / "x", include=["everything"])


def test_explore_attempts_before_the_last_reset_are_dropped_unless_asked(
    tmp_path: Path,
) -> None:
    s = Session()
    s.step("a1", text("scene 1"), image(PNG_A))
    s.step("r1", text("Episode restarted; attempt 2"), image(PNG_B), name="reset")
    s.step("a2", text("scene 2"), image(PNG_C))
    s.step("r2", text("refused"), name="reset", error=True)
    s.step("a3", text("scene 3"), image(PNG_D))
    s.finish()
    s.write(tmp_path / "runs" / "ep", {"success": True})
    out = tmp_path / "out"
    export_planner([tmp_path / "runs"], out)
    [row] = rows(out)
    turns = row["conversations"]
    assert (
        turns[0]["value"] == "Solve the task.\nEpisode restarted; attempt 2\n<image>"
    ), "the prompt, then the restored scene; the refused reset does not count"
    shown = json.dumps(turns)
    assert "scene 1" not in shown and "scene 2" in shown
    assert paths(out, row) == [PNG_B, PNG_C, PNG_D]
    assert row["explore_attempts_dropped"] == 1
    export_planner([tmp_path / "runs"], tmp_path / "all", include=["explore_attempts"])
    [whole] = rows(tmp_path / "all")
    assert paths(tmp_path / "all", whole) == [PNG_A, PNG_B, PNG_C, PNG_D]
    export_planner([tmp_path / "runs"], tmp_path / "oa", fmt="openai")
    [oa] = rows(tmp_path / "oa")
    assert [m["role"] for m in oa["messages"][:4]] == [
        "system",
        "user",
        "user",
        "assistant",
    ]


def test_the_context_follows_pis_projection(tmp_path: Path) -> None:
    """Compaction keeps its summary and the entries from firstKeptEntryId; context_edit replaces or drops."""
    s = Session()
    s.step("a1", text("old"), image(PNG_A))
    s.msg(assistant([call("a2", "move_to", x=2)]))
    kept = s.msg(result("a2", "move_to", text("kept"), image(PNG_B)))
    s.msg(assistant([call("a3", "move_to", x=3)]))
    s.msg(result("a3", "move_to", text("secret"), image(PNG_C)))
    s.add(
        {
            "type": "context_edit",
            "targetId": s.last,
            "replacement": {"content": "[redacted]"},
        }
    )
    dropped = s.step("a4", text("gone"), image(PNG_D))
    s.add({"type": "context_edit", "targetId": dropped, "replacement": None})
    s.add(
        {
            "type": "compaction",
            "summary": "moved twice",
            "firstKeptEntryId": kept,
            "tokensBefore": 1,
        }
    )
    s.finish()
    s.write(tmp_path / "runs" / "ep", {"success": True})
    out = tmp_path / "out"
    export_planner([tmp_path / "runs"], out, fmt="openai")
    [row] = rows(out)
    msgs = row["messages"]
    assert row["compacted"] is True
    assert msgs[1]["role"] == "user" and "moved twice" in msgs[1]["content"]
    assert msgs[1]["content"].startswith(
        "The conversation history before this point was compacted"
    )
    assert msgs[2] == {
        "role": "tool",
        "tool_call_id": "a2",
        "content": "kept\n<image>",
    }, "the kept range starts at firstKeptEntryId"
    assert msgs[4] == {"role": "tool", "tool_call_id": "a3", "content": "[redacted]"}
    assert "gone" not in json.dumps(msgs) and "old" not in json.dumps(msgs)
    assert paths(out, row) == [PNG_B]
    # ShareGPT: the summary and the kept result before the first reply form the prompt.
    export_planner([tmp_path / "runs"], tmp_path / "sg")
    [sg] = rows(tmp_path / "sg")
    assert sg["conversations"][0]["from"] == "human"
    assert sg["conversations"][0]["value"].endswith("kept\n<image>")


def test_keep_images_prunes_like_the_robot(tmp_path: Path) -> None:
    """Newest result first, each result's images in order; user images stay; the anchor survives."""
    s = Session(prompt=[text("Solve the task."), image(PNG_D)])
    s.step("a1", text("first"), image(PNG_A), image(PNG_B))
    s.step("a2", text("second"), image(PNG_C), image(PNG_A))
    s.finish()
    s.write(
        tmp_path / "runs" / "ep",
        {"success": True},
        {"status": "success", "anchor_image": True},
    )
    out = tmp_path / "out"
    export_planner([tmp_path / "runs"], out, keep_images=1, image_stub="[stub]")
    [row] = rows(out)
    turns = row["conversations"]
    assert turns[2]["value"] == "first\n<image>\n[stub]", (
        "the run's --anchor-image kept the first frame"
    )
    assert turns[4]["value"] == "second\n<image>\n[stub]", (
        "the newest result's first image"
    )
    assert paths(out, row) == [PNG_D, PNG_A, PNG_C]
    export_planner(
        [tmp_path / "runs"], tmp_path / "noanchor", keep_images=1, anchor_image=False
    )
    [plain] = rows(tmp_path / "noanchor")
    assert (
        plain["conversations"][2]["value"]
        == "first\n[older camera frame omitted]\n[older camera frame omitted]"
    )


def test_steering_is_never_merged_into_an_observation_by_default(
    tmp_path: Path,
) -> None:
    runs = tmp_path / "runs"
    write_session(runs / "ep", outcome={"success": True}, steer=True)
    summary = export_planner([runs], tmp_path / "sg")
    assert summary["episodes"] == 0 and summary["skipped"] == {"steering": 1}
    export_planner([runs], tmp_path / "merged", merge_steering=True)
    assert (
        rows(tmp_path / "merged")[0]["conversations"][2]["value"]
        == "moved\n<image>\ngo faster"
    )
    export_planner([runs], tmp_path / "oa", fmt="openai")
    msgs = rows(tmp_path / "oa")[0]["messages"]
    assert msgs[4] == {"role": "user", "content": "go faster"}, "its own user message"


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
    assert paths(out, row) == [PNG_A, PNG_B]
    assert all(Path(i["image"]).is_absolute() for i in row["images"])
    assert row["reward"] == 1.0 and "prompt" not in row and "reward_model" not in row


def test_verl_rl_rows_are_task_prompts_scored_by_the_new_rollout(
    tmp_path: Path,
) -> None:
    runs = tmp_path / "runs"
    write_session(runs / "ok", outcome={"success": True}, prompt_image=True)
    write_session(runs / "fail", outcome={"success": False})
    write_session(runs / "broken", outcome={"success": True, "env_error": True})
    Session(task={"robot": "toy", "task": "1"}).write(
        runs / "noseed", {"success": True}
    )
    out = tmp_path / "out"
    summary = export_planner([runs], out, fmt="verl-rl")
    assert (summary["episodes"], summary["successes"], summary["failures"]) == (2, 1, 1)
    assert summary["skipped"] == {"env_error": 1, "no_seed": 1}
    by_id = {r["id"].split("-")[0]: r for r in rows(out)}
    ok, fail = by_id["ok"], by_id["fail"]
    assert ok["prompt"] == [
        {"role": "system", "content": "You drive the toy arm."},
        {"role": "user", "content": "Solve the task.\n<image>"},
    ], "the system prompt and the first user turn, not the trajectory"
    assert paths(out, ok) == [PNG_C], "only the prompt's own image"
    assert fail["prompt"][1] == {"role": "user", "content": "Solve the task."}
    assert fail["images"] == []
    env = {"robot": "toy", "seed": 7, "init_state": {"suite": "s", "task": "1"}}
    for r in (ok, fail):
        assert r["reward_model"] == {"style": "env_success", "ground_truth": env}
        assert r["agent_name"] == "tool_agent" and r["data_source"] == "pi_embodied/toy"
        assert r["extra_info"]["seed"] == 7 and r["extra_info"]["task"] == TASK
        assert r["extra_info"]["tools_kwargs"]["move_to"] == {"create_kwargs": env}
        assert "reward" not in r and "success" not in r
    assert ok["extra_info"]["source"]["success"] is True
    assert fail["extra_info"]["source"]["success"] is False
    # The reward hook scores the rollout's own environment verdict, never the recorded one.
    gt, info = ok["reward_model"]["ground_truth"], ok["extra_info"]

    def score(rollout: dict) -> float:
        return compute_score(
            ok["data_source"], "any text", gt, {**info, "rollout_result": rollout}
        )

    assert score({"robot": "toy", "task": 1, "seed": 7, "success": False}) == 0.0
    assert score({"robot": "toy", "task": 1, "seed": 7, "success": True}) == 1.0
    assert score({"robot": "toy", "terminated": True}) == 1.0
    assert score({"robot": "toy", "success": True, "planner_error": "503"}) == 0.0
    with pytest.raises(ValueError, match="rollout_result"):
        compute_score(ok["data_source"], "I succeeded", gt, info)
    with pytest.raises(ValueError, match="seed=8"):
        score({"robot": "toy", "seed": 8, "success": True})


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
    for fmt in FORMATS:
        for keep in (None, 0, 1, 2, 5):
            out = tmp_path / f"{fmt}-{keep}"
            export_planner(
                [runs], out, fmt=fmt, include_failures=True, keep_images=keep
            )
            got = rows(out)
            assert len(got) == len(variants), (fmt, keep)
            for row in got:
                content = {"sharegpt": "conversations", "openai": "messages"}.get(
                    fmt, "prompt"
                )
                n = placeholders(row[content]) + placeholders(row.get("system", ""))
                assert n == len(row["images"]), (fmt, keep, row["id"])
                assert len(set(map(str, row["images"]))) == len(row["images"])
    # The dropped trailing frames are not written or listed; the literal tag is defused.
    out = tmp_path / "sharegpt-None"
    by_id = {r["id"].split("-")[0]: r for r in rows(out)}
    assert paths(out, by_id["cut"]) == [PNG_A, PNG_B]
    assert (
        by_id["literal"]["conversations"][2]["value"]
        == "moved &lt;image&gt; tag\n<image>"
    )
    assert len(by_id["finish_image"]["images"]) == 2, (
        "the finish frame goes with its dropped result"
    )


def test_cli_export_planner(tmp_path: Path, capsys) -> None:
    runs = tmp_path / "runs"
    write_session(runs / "ep", outcome={"success": True})
    write_session(runs / "priv", outcome={"success": True, "privileged": True})
    assert (
        cli.main(["export-planner", str(runs), "--output", str(tmp_path / "out")]) == 0
    )
    summary = json.loads(capsys.readouterr().out)
    assert summary["episodes"] == 1 and summary["format"] == "sharegpt"
    argv = ["export-planner", str(runs), "--output", str(tmp_path / "rl")]
    assert cli.main([*argv, "--format", "verl-rl", "--include-privileged"]) == 0
    assert json.loads(capsys.readouterr().out)["episodes"] == 2
