"""Golden texts for the TS port of HumanCLAW's planner (packages/embodied/src/robots/humanclaw).

HumanCLAW's own Python (agent/planner.py, prompts/v4.py, verifiers/v3.py, evaluator._history_item)
runs on fixed inputs and scripted model replies; the TS port must reproduce every prompt, every
final action and every history row byte for byte (packages/embodied/test/humanclaw.test.ts).

    HUMANCLAW_SRC=~/workspace/pi-work/refs/HumanCLAW/src python services/tests/humanclaw_golden.py

rewrites packages/embodied/test/fixtures/humanclaw-golden.json; test_humanclaw.py checks that the
committed file is what this produces.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

FIXTURE = (
    Path(__file__).resolve().parents[2]
    / "packages/embodied/test/fixtures/humanclaw-golden.json"
)
DEFAULT_SRC = Path.home() / "workspace/pi-work/refs/HumanCLAW/src"


def humanclaw_src() -> Path | None:
    """HUMANCLAW_SRC, else the reference checkout, else an installed humanclaw_bench."""
    for candidate in (os.environ.get("HUMANCLAW_SRC"), DEFAULT_SRC):
        if candidate and (Path(candidate) / "humanclaw_bench").is_dir():
            return Path(candidate)
    return None


def importable() -> bool:
    src = humanclaw_src()
    if src is not None and str(src) not in sys.path:
        sys.path.insert(0, str(src))
    try:
        import humanclaw_bench.agent.planner  # noqa: F401
        import humanclaw_bench.evaluation.evaluator  # noqa: F401
    except Exception:  # noqa: BLE001
        return False
    return True


#: _chooser_action inputs (planner or verifier JSON) covering every family, the clamps, Python's
#: rounding (round-half-even, "%.2f" of exact binary ties), the name-only recovery path and the
#: odd action_id types a provider sends.
CHOOSER_CASES = [
    {"action_id": 0, "action_name": "Walk<forward><slow>"},
    {"action_id": 0, "action_name": "Walk<forward><normal>"},
    {"action_id": 0, "action_name": "Walk<forward><fast>"},
    {"action_id": 0, "action_name": "Walk<forward><Very Fast>"},
    {"action_id": 0, "action_name": "Walk"},
    {"action_id": "0", "action_name": "Walk<forward><normal>"},
    {"action_id": 1, "action_name": "Stop/Stand"},
    {"action_id": 2, "action_name": "Turn<left><30>"},
    {"action_id": 2, "action_name": "Turn<right><45>"},
    {"action_id": 2, "action_name": "Turn<right><5>"},
    {"action_id": 2, "action_name": "Turn<left><200>"},
    {"action_id": 2, "action_name": "Turn<left><-30>"},
    {"action_id": 2, "action_name": "Turn<left><36.5>"},
    {"action_id": 2, "action_name": "Turn<left><37.5>"},
    {"action_id": 2, "action_name": "Turn<Right><12.25 degrees>"},
    {"action_id": 2, "action_name": "Turn<up><30>"},
    {"action_id": 2, "action_name": "Turn"},
    {"action_id": 2, "action_name": "Turn right 90"},
    {"action_id": 3, "action_name": "Climb upstairs<normal>"},
    {"action_id": 4, "action_name": "Sit down<0.45>"},
    {"action_id": 4, "action_name": "Sit down<0.375>"},
    {"action_id": 4, "action_name": "Sit down<0.625>"},
    {"action_id": 4, "action_name": "Sit down<0.05>"},
    {"action_id": 4, "action_name": "Sit down<1.2>"},
    {"action_id": 4, "action_name": "Sit down<target height>"},
    {"action_id": 5, "action_name": "Step back<0.25>"},
    {"action_id": 5, "action_name": "Step back<0.125>"},
    {"action_id": 5, "action_name": "Step back<0.9>"},
    {"action_id": 5, "action_name": "Step back<0.01>"},
    {"action_id": 5, "action_name": "Step back"},
    {"action_id": 6, "action_name": "Side step<left><0.25>"},
    {"action_id": 6, "action_name": "Side step<right><0.3>"},
    {"action_id": 6, "action_name": "Side step<right><0.875>"},
    {"action_id": 6, "action_name": "Side step<left><0.05>"},
    {"action_id": 6, "action_name": "Side step<left><0.375>"},
    {"action_id": 6, "action_name": "Side step"},
    {"action_id": 7, "action_name": "Walk downstairs<normal>"},
    {"action_id": 9, "action_name": "Walk<forward><fast>"},
    {"action_id": -1, "action_name": "Turn<left><30>"},
    {"action_id": 2.0, "action_name": "Turn<left><30>"},
    {"action_id": 2.7, "action_name": "Turn<left><30>"},
    {"action_id": True, "action_name": "Turn<left><30>"},
    {"action_id": "2.5", "action_name": "Turn<left><30>"},
    {"action_id": " 6 ", "action_name": "Side step<right><0.2>"},
    {"action_id": None, "action_name": "Walk<forward><fast>"},
    {"action_id": None, "action_name": "Walk downstairs<normal>"},
    {"action_id": None, "action_name": "climb down the stairs"},
    {"action_id": None, "action_name": "Stop/Stand"},
    {"action_id": None, "action_name": "Turn<right><60>"},
    {"action_id": None, "action_name": "Climb upstairs<normal>"},
    {"action_id": None, "action_name": "Sit down<0.5>"},
    {"action_id": None, "action_name": "Step back<0.3>"},
    {"action_id": None, "action_name": "back up<0.3>"},
    {"action_id": None, "action_name": "Sidestep<right><0.4>"},
    {"action_id": None, "action_name": "side walk left 0.2"},
    {"action_id": None, "action_name": "dance"},
    {"action_name": "Turn<left><15>"},
    {"action_id": 0},
    {},
]


def plan(
    name: str,
    *,
    action_id: object = 0,
    visible: str = "From left to right, I can see a sofa, about 3 meters from me. The target is not visible. The near-body lane is clear.",
    goal: str = "Scan the room for the bed.",
    info: object = None,
    **extra: object,
) -> str:
    body = {
        "visible_state": visible,
        "mid_level_progress_analysis": "The target is not visible yet; keep scanning.",
        "mid_level_goal": goal,
        "low_level_action_reasoning": f"Choose {name} to make progress.",
        "action_id": action_id,
        "action_name": name,
        "additional_info": {} if info is None else info,
    }
    body.update(extra)
    return json.dumps(body, ensure_ascii=False)


def verdict(
    verdict_: str, final_name: str = "", final_id: object = None, **extra: object
) -> str:
    body = {"verdict": verdict_, "reason": f"{verdict_} for the test", **extra}
    if final_name:
        body["final_action_name"] = final_name
        body["final_action_id"] = final_id
    return json.dumps(body, ensure_ascii=False)


LONG = (
    "From left to right, I can see "
    + "a chair, a lamp and a rug; " * 12
    + "\nThe bed is visible straight ahead."
)
#: Scripted episodes: the instruction, then each decision's model replies in call order (planner
#: attempts, then verifier attempts). A reply is text, or {"raise": message} for a transport error.
EPISODES = [
    {
        "name": "walk_turn_verify_and_history",
        "instruction": "Find the bed and move until your body is touching the target, with zero distance to the target object. Finally, sit on the bed.",
        "decisions": [
            [plan("Turn<left><30>", action_id=2)],
            [plan("Walk<forward><normal>"), verdict("accept")],
            [
                plan("Walk<forward><fast>", visible=LONG, goal="Walk to the bed."),
                verdict(
                    "replace",
                    "Turn<right><45>",
                    2,
                    lane_observation="A wall is right ahead.",
                ),
            ],
            [
                "not json at all",
                "```json\n" + plan("Side step<left><0.25>", action_id=6) + "\n```",
            ],
            [
                plan("Climb upstairs<normal>", action_id=3),
                verdict("replace", "Walk<forward><slow>", 0),
            ],
            [plan("Walk downstairs<normal>", action_id=7)],
            [plan("Step back<0.3>", action_id=5)],
            [
                plan("Turn<right><60>", action_id=2, info={"turn_for_sit": True}),
                verdict("accept"),
            ],
            [plan("Turn<right><60>", action_id=2, info={"turn_for_sit": True})],
            [plan("Sit down<0.45>", action_id=4)],
            [plan("Sit down<0.45>", action_id=4)],
            [
                plan(
                    "Stop/Stand",
                    action_id=1,
                    goal="Stop/Stand",
                    info={"stop_after_sit": True},
                ),
                verdict("accept", clear_sit_failure=False),
            ],
        ],
    },
    {
        "name": "retries_fallback_and_stop_verifier",
        "instruction": "Find the toilet and move until your body is touching the target, with zero distance to the target object. Finally, sit on the toilet.",
        "decisions": [
            [
                {"raise": "APIConnectionError: connection reset"},
                plan("Walk<forward><slow>"),
                verdict("accept"),
            ],
            [
                {"raise": "timeout"},
                "",
                '{"visible_state": ',
                "<think>hmm</think>no json",
                {"raise": "timeout"},
            ],
            [
                plan("Stop/Stand", action_id=1, goal="Stop/Stand"),
                "garbage",
                verdict("replace", "Walk<forward><normal>", 0, goal_completed=False),
            ],
            [
                plan("Stop/Stand", action_id=1, goal="stop"),
                {"raise": "e1"},
                {"raise": "e2"},
                {"raise": "e3"},
                {"raise": "e4"},
                {"raise": "e5"},
            ],
        ],
    },
    {
        "name": "unicode_odd_fields_and_long_history",
        "instruction": "Find the couch and move until your body is touching the target, with zero distance to the target object. Finally, sit on the couch.",
        "decisions": [
            [
                plan(
                    "Turn<left><37.5>",
                    action_id="2",
                    visible="我看到沙发 — the couch is visible.\nLane clear.",
                )
            ],
            [plan("Turn<left><36.5>", action_id=None)],
            [
                plan("Walk<forward><normal>", action_id=0, goal=""),
                verdict("replace", "<final action>", 2),
            ],
            [
                plan("Walk<forward><normal>", action_id=0),
                verdict("replace", "Stop/Stand", 1),
            ],
        ]
        + [
            [plan(f"Turn<right><{10 + i}>", action_id=2, goal=f"Scan {i}")]
            for i in range(10)
        ]
        + [[plan("Side step<right><0.875>", action_id=6, info="not a dict")]],
    },
]


class ScriptedModel:
    """HumanCLAW's VLM interface (``respond(messages) -> str``, ``last_usage``) over a script."""

    def __init__(self, replies: list) -> None:
        self.replies = list(replies)
        self.calls: list[str] = []
        self.last_usage: dict = {}

    def respond(self, messages: list) -> str:
        text = messages[0]["content"][0]["text"]
        self.calls.append(text)
        if not self.replies:
            raise AssertionError("script exhausted")
        reply = self.replies.pop(0)
        self.last_usage = {
            "prompt_tokens": 1000 + len(self.calls),
            "completion_tokens": 50,
        }
        if isinstance(reply, dict):
            raise RuntimeError(reply["raise"])
        return reply


def generate() -> dict:
    if not importable():
        raise RuntimeError("humanclaw_bench is not importable (set HUMANCLAW_SRC)")
    from types import SimpleNamespace

    import humanclaw_bench.agent.planner as planner_module
    from humanclaw_bench.agent.planner import HumanClawBenchPSVPlanSkillPlanner as P
    from humanclaw_bench.agent.skills import skill_to_text
    from humanclaw_bench.evaluation.evaluator import _history_item
    from PIL import Image

    planner_module.time.sleep = lambda _s: None  # retries back off; not in a test
    chooser = []
    for case in CHOOSER_CASES:
        call = P._chooser_action(P, case)
        chooser.append(
            {"input": case, "action": call.to_json(), "text": skill_to_text(call)}
        )

    episodes = []
    for spec in EPISODES:
        replies = [r for d in spec["decisions"] for r in d]
        model = ScriptedModel(replies)
        planner = P(
            model,
            prompt_version="v4",
            verifier_version="v3",
            max_history=10,
            plan_horizon_steps=6,
        )
        planner.reset(SimpleNamespace(instruction=spec["instruction"]))
        history: list = []
        steps = []
        for step, decision_replies in enumerate(spec["decisions"]):
            before = len(model.calls)
            result = planner.act(Image.new("RGB", (4, 4)), history, [])
            used = len(model.calls) - before
            if used != len(decision_replies):
                raise AssertionError(
                    f"{spec['name']} step {step}: {used} calls, script has {len(decision_replies)}"
                )
            item = _history_item(step, result)
            history.append(item)
            steps.append(
                {
                    "prompts": model.calls[before:],
                    "stages": [
                        {"stage": s.stage, "error": s.error, "raw": s.raw}
                        for s in result.stage_outputs
                    ],
                    "action": result.action.to_json(),
                    "action_text": skill_to_text(result.action),
                    "raw_plan": result.raw_plan,
                    "planner_skill": result.planner_skill,
                    "verifier": result.verifier,
                    "history_item": item,
                    "planner_state": {
                        "current_plan": planner.current_plan,
                        "current_step": planner.current_step,
                        "turn_for_sit_sequence_verified": planner.turn_for_sit_sequence_verified,
                    },
                }
            )
        episodes.append(
            {
                "name": spec["name"],
                "instruction": spec["instruction"],
                "replies": [[r for r in d] for d in spec["decisions"]],
                "steps": steps,
            }
        )
    return {
        "source": "github.com/Human-CLAW/HumanCLAW @c4f9351 (agent/planner.py, prompts/v4.py, verifiers/v3.py, evaluation/evaluator.py _history_item)",
        "chooser": chooser,
        "episodes": episodes,
    }


def dumps(value: dict) -> str:
    return json.dumps(value, indent=1, ensure_ascii=False, sort_keys=True) + "\n"


if __name__ == "__main__":
    FIXTURE.write_text(dumps(generate()), encoding="utf-8")
    print(f"wrote {FIXTURE}")
