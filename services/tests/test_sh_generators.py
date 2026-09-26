"""generate_subgoals.py / generate_affordance.py (finetuned/showharness) against a fake
OpenAI-compatible endpoint: the planner's retries, the pre-grasp merge, the written configs,
and the conditioned mvtoken templates the converter formats with them."""

from __future__ import annotations

import importlib.util
import json
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest
from PIL import Image

ROOT = (
    Path(__file__).resolve().parents[1]
    / "pi_embodied_services"
    / "finetuned"
    / "showharness"
)
PREP = ROOT / "train" / "data_preparation"


def _load(name: str):
    sys.path.insert(0, str(PREP))
    spec = importlib.util.spec_from_file_location(name, PREP / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class _Endpoint:
    """Answers each chat completion with the next reply; records the request bodies."""

    def __init__(self, replies: list[str]):
        self.replies = list(replies)
        self.bodies: list[dict] = []
        outer = self

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):  # noqa: N802
                n = int(self.headers["Content-Length"])
                outer.bodies.append(json.loads(self.rfile.read(n)))
                reply = outer.replies.pop(0) if outer.replies else ""
                data = json.dumps(
                    {"choices": [{"message": {"role": "assistant", "content": reply}}]}
                ).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(data)

            def log_message(self, *args):
                pass

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        threading.Thread(target=self.server.serve_forever, daemon=True).start()
        self.url = f"http://127.0.0.1:{self.server.server_address[1]}/v1"

    def close(self):
        self.server.shutdown()


@pytest.fixture
def gumi_run(tmp_path: Path) -> Path:
    """A pi GUMI run: images/agentview/0000.png, images/wrist/0000.png, actions.jsonl."""
    run = tmp_path / "task_3" / "10-11-12"
    for view in ("agentview", "wrist"):
        (run / "images" / view).mkdir(parents=True)
        Image.new("RGB", (8, 8), (200, 10, 10)).save(run / "images" / view / "0000.png")
    (run / "actions.jsonl").write_text(
        json.dumps(
            {
                "step": 0,
                "token": "MV_FWD",
                "agentview": "images/agentview/0000.png",
                "wrist": "images/wrist/0000.png",
            }
        )
        + "\n"
    )
    return run


PLAN = {
    "subgoals": [
        {
            "id": "reach",
            "target": "the banana",
            "affordance": "left end",
            "motion": "APPROACH",
            "description": "move above the banana",
            "completion": "gripper above the banana",
        },
        {
            "id": "grasp",
            "target": "banana",
            "affordance": "left end",
            "motion": "grasp",
            "description": "close on the left end",
            "completion": "banana between the fingers",
        },
        {
            "id": "place",
            "target": "plate",
            "affordance": "center",
            "motion": "PLACE",
            "description": "lower onto the plate",
            "completion": "banana on the plate",
        },
    ]
}


def test_subgoals_retry_merge_and_write(gumi_run: Path):
    gen = _load("generate_subgoals")
    # Guided JSON comes back empty; the free-JSON retry (fenced prose) is parsed.
    ep = _Endpoint(["", "Here:\n```json\n" + json.dumps(PLAN) + "\n```"])
    try:
        rc = gen.main(
            [str(gumi_run.parent), "--task", "put the banana on the plate"]
            + ["--vlm-url", ep.url, "--model", "planner"]
        )
    finally:
        ep.close()
    assert rc == 0
    cfg = json.loads((gumi_run.parent / "task_config.json").read_text())
    # The pure reach stage folds into the grasp stage of the same target.
    assert [s["motion"] for s in cfg["subgoals"]] == ["GRASP", "PLACE"]
    assert cfg["subgoals"][0]["description"] == (
        "move above the banana. Then close on the left end."
    )
    assert cfg["planner"]["route"] == "free_json"
    first, second = ep.bodies
    assert first["structured_outputs"]["json"]["required"] == ["subgoals"]
    assert "structured_outputs" not in second
    assert second["messages"][0]["content"][-1]["text"].endswith("markdown, or prose.")
    assert first["chat_template_kwargs"] == {
        "enable_thinking": False,
        "thinking": False,
    }
    images = [c for c in first["messages"][0]["content"] if c["type"] == "image_url"]
    assert len(images) == 2 and images[0]["image_url"]["url"].startswith(
        "data:image/png;base64,"
    )


def test_subgoals_fall_back_to_one_task_stage(gumi_run: Path, tmp_path: Path):
    gen = _load("generate_subgoals")
    ep = _Endpoint(["no", "no", "no", "no"])
    try:
        rc = gen.main(
            [str(gumi_run), "--task", "tidy up", "--vlm-backend", "llamafactory"]
            + ["--vlm-url", ep.url, "--model", "planner", "--out-dir", str(tmp_path)]
        )
    finally:
        ep.close()
    assert rc == 0
    cfg = json.loads((tmp_path / "task_config.json").read_text())
    assert cfg["subgoals"][0]["id"] == "task_fallback"
    assert cfg["planner"]["route"] == "task_fallback"
    # LLaMA-Factory has no guided decoding: free JSON, then the two text retries.
    assert len(ep.bodies) == 3
    assert all("structured_outputs" not in b for b in ep.bodies)
    assert "chat_template_kwargs" not in ep.bodies[-1]


def test_affordance_writes_the_grasp_hint(gumi_run: Path):
    gen = _load("generate_affordance")
    reply = json.dumps({"target": "banana", "affordance": "left end"})
    ep = _Endpoint([reply])
    try:
        rc = gen.main(
            [str(gumi_run), "--task", "hand over the banana"]
            + ["--vlm-url", ep.url, "--model", "planner"]
        )
    finally:
        ep.close()
    assert rc == 0
    cfg = json.loads((gumi_run / "affordance_config.json").read_text())
    assert cfg == {
        "task": "hand over the banana",
        "target": "banana",
        "affordance": "left end",
    }
    assert ep.bodies[0]["max_tokens"] == 512


def test_conditioned_templates_format_with_the_converter_fields():
    subgoal = (ROOT / "prompts" / "v3" / "mvtoken_generator.txt").read_text()
    text = subgoal.format(
        task="T",
        stage="GRASP",
        target="banana",
        affordance="left end",
        description="close",
        completion="held",
        gripper_state="open",
        recent_moves="none",
        gripper_color="black",
    )
    assert "Stage: GRASP\nTarget: banana\nAffordance: left end\n" in text
    affordance = (
        ROOT / "prompts" / "v3" / "mvtoken_generator_affordance.txt"
    ).read_text()
    text = affordance.format(
        task="T",
        target="banana",
        affordance="left end",
        gripper_state="open",
        recent_moves="none",
    )
    assert "Grasp first: banana, at left end\n" in text
    # The provider renders the same text (packages/embodied/src/finetuned/templates).
    tpl = (
        Path(__file__).resolve().parents[2]
        / "packages/embodied/src/finetuned/templates/v3_mvtoken_generator_subgoal.txt"
    )
    assert tpl.read_text() == subgoal
