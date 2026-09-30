"""HumanCLAW env server over HumanCLAW's own evaluator helpers with a fake Habitat env and a fake
motion model: reset / record_decision / step / stop / finish, the metric pass-through, hidden
state kept on the server, the episode selectors, and that the committed golden fixture is what
HumanCLAW's Python produces. Skipped without a HumanCLAW checkout (HUMANCLAW_SRC)."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import humanclaw_golden as golden
import numpy as np
import pytest

pytestmark = pytest.mark.skipif(
    not golden.importable(), reason="humanclaw_bench not importable (set HUMANCLAW_SRC)"
)


def test_golden_fixture_is_current():
    assert golden.FIXTURE.read_text(encoding="utf-8") == golden.dumps(golden.generate())


class FakeEnv:
    fps = 30.0
    ego_camera = SimpleNamespace(resolution=(448, 448))

    def __init__(self, max_steps):
        self.max_episode_steps = max_steps
        self.steps = []
        self.closed = False
        self.agent = SimpleNamespace(
            translation=np.zeros(3),
            rotation=SimpleNamespace(
                vector=SimpleNamespace(x=0.0, y=0.0, z=0.0), scalar=1.0
            ),
        )
        self._n = 0

    def _obs(self):
        return SimpleNamespace(head_rgb=np.full((448, 448, 3), self._n, dtype=np.uint8))

    def reset(self, episode, **_):
        return self._obs()

    def replay_initial_state(self):
        return {}

    def metric_find_observation(self):
        return {"available": True, "target_pixel_count": 150}

    def step(self, action, reasoning=None):
        self.steps.append(action)
        if isinstance(action, dict) and action.get("stop"):
            return self._obs(), 0.0, True, {}
        self._n += 1
        t = 15
        body = {
            "transl": np.zeros((t, 3), np.float32),
            "global_orient": np.zeros((t, 3), np.float32),
            "body_pose": np.zeros((t, 54, 3), np.float32),
        }
        info = {
            "body_state": body,
            "object_states": {},
            "metric_frames": {"fixed_contacts": [[1], [], []]},
        }
        return self._obs(), 0.0, self._n >= self.max_episode_steps, info

    def close(self):
        self.closed = True


class FakeMotion:
    def __init__(self):
        self.calls = []
        self.unloaded = 0

    def reset(self, *args):
        return np.zeros((1, 75), np.float32)

    def generate(self, skill, cond):
        self.calls.append((skill, cond))
        return SimpleNamespace(xb_world_75=np.zeros((15, 75), np.float32))

    def unload(self):
        self.unloaded += 1


class FakeMetrics:
    def __init__(self):
        self.decisions = []
        self.motions = []

    def record_decision(self, *, step, decision, find_observation):
        self.decisions.append(
            (
                step,
                decision.planner_skill.get("visible_state"),
                find_observation,
                decision.action.skill,
            )
        )

    def record_motion(self, *, step, action_skill, info):
        self.motions.append((step, action_skill))

    def finalize(self, *, before, after):
        return {
            "success": {"find_sr": True, "nav_sr_20cm": False, "interact_sr": False},
            "body_scene": {"collision_step_fraction": 0.5, "big_list": [1, 2, 3]},
            "action_quality": {"motion_jerk_m_s3": 12.5},
            "cost": {"decision_steps": len(self.decisions)},
        }


@pytest.fixture
def server(tmp_path, monkeypatch):
    from humanclaw_bench.benchmark.episodes import HCFindNavInteractEpisode
    from humanclaw_bench.config import load_config

    from pi_embodied_services.robots.humanclaw import env_server

    f = env_server.HumanclawEnvFacade(
        profile=load_config("paper_fullval_v1"),
        output_root=tmp_path,
        device="cpu",
        compute_metrics=True,
        save_video=False,
        max_steps=3,
    )
    state = SimpleNamespace(metrics=None, built=[])

    def build(ident):
        state.built.append(ident)
        f._ev.env = FakeEnv(3)
        f._ev.motion = f._ev.motion or FakeMotion()
        f._ev.seed_mode, f._ev.seed_pkl, f._ev.seed_pt = "pt", None, None
        return HCFindNavInteractEpisode(
            name="t",
            task_type="find_nav_interact",
            instruction="Find the bed.",
            scene_id=ident["scene_id"],
            scene_label=ident["scene_id"],
            scene_dataset_config="x",
            episode_id=ident["episode_id"],
            object_category=ident["object_category"],
            object_label="bed",
            init_offset=(0.0, 0.0, 0.0),
            init_yaw=0.0,
            max_steps=3,
            goals=[],
            viewpoint_positions=[],
            goal_objects=[
                {"object_name": "bed_1", "object_id": 4, "center": [1.0, 0.0, 2.0]}
            ],
        )

    def new_metrics(episode, rollout):
        state.metrics = FakeMetrics()
        return state.metrics

    monkeypatch.setattr(f, "_build", build)
    monkeypatch.setattr(f._ev, "_new_metric_recorder", new_metrics)
    return f, state


def test_reset_step_stop_finish_and_metrics(server, tmp_path):
    f, state = server
    obs = f.reset(episode="one")
    assert state.built == [
        {"scene_id": "102343992", "episode_id": "0", "object_category": "bed"}
    ]
    assert (
        obs["ego"].shape == (448, 448, 3)
        and obs["instruction"] == "Find the bed."
        and obs["step"] == 0
    )
    decision = {
        "planner_skill": {"visible_state": "The bed is visible."},
        "action": {
            "skill": "turn",
            "cond": 30.0,
            "action_id": None,
            "action_name": "Turn<left><30>",
        },
        "stages": [
            {
                "stage": "percept_mid_low",
                "prompt": "P",
                "raw": {"a": 1},
                "raw_output": "{}",
                "usage": {},
            }
        ],
    }
    assert f.record_decision(decision)["written"] == ["step000_percept_mid_low.json"]
    out = f.step("turn", 30.0, "Turn<left><30>")
    assert set(out) >= {"ego", "done", "step", "collision"} and "body_state" not in out
    assert out["collision"] == {"collided": False} and out["step"] == 1
    assert f._ev.motion.calls == [("turn", 30.0)]
    step = json.loads((f._dir / "step000_percept_mid_low.json").read_text())
    assert step == {"prompt": "P", "response": {"a": 1}}
    # An operator's step without a recorded decision is still a decision step.
    out = f.step("stand", None, "Stop/Stand")
    assert out["done"] and out["stopped"]
    assert f._ev.env.steps[-1]["stop"] is True
    with pytest.raises(RuntimeError, match="over"):
        f.step("turn", 10.0)
    s = f.finish()
    assert state.metrics.decisions[0] == (
        0,
        "The bed is visible.",
        {"available": True, "target_pixel_count": 150},
        "turn",
    )
    assert [d[3] for d in state.metrics.decisions] == ["turn", "stand"]
    assert state.metrics.motions == [(0, "turn")]
    assert s["metrics"]["find_sr"] is True and s["metrics"]["motion_jerk_m_s3"] == 12.5
    assert "big_list" not in s["metrics"] and s["active_stop"] and s["steps"] == 2
    assert (f._dir / "metrics.json").is_file() and (
        f._dir / "replay_manifest.json"
    ).is_file()
    assert f._ev.env.closed and f._ev.motion.unloaded == 1
    with pytest.raises(RuntimeError, match="no running episode"):
        f.step("turn", 10.0)


def test_step_limit_ends_and_privileged_pose(server):
    f, _ = server
    f.reset(episode="102343992_ep0_bed")
    gt = f.ground_truth_poses()
    assert gt["targets"][0]["center"] == [1.0, 0.0, 2.0] and gt["human_root"][
        "quat_xyzw"
    ] == [0.0, 0.0, 0.0, 1.0]
    for _ in range(3):
        out = f.step("walk_forward", [0.0, 0.2, 0.0], "Walk<forward><slow>")
    assert out["done"] and not out["stopped"]


def test_proprioception_is_opt_in_and_resets_between_episodes(server, monkeypatch):
    f, _ = server
    assert "proprioception" not in f.reset(episode="one")
    assert "proprioception" not in f.skill_turn("left", 30)
    assert f.reset(episode="one", proprioception=True)["proprioception"] == {}
    old_step = f._ev.env.step

    def moved(action, reasoning=None):
        f._ev.env.agent.translation = np.array([0.3, -0.2, 0.4])
        f._ev.env.agent.rotation = SimpleNamespace(
            vector=SimpleNamespace(x=0.0, y=np.sin(np.pi / 8), z=0.0),
            scalar=np.cos(np.pi / 8),
        )
        return old_step(action, reasoning)

    monkeypatch.setattr(f._ev.env, "step", moved)
    out = f.skill_turn("left", 120)
    assert out["proprioception"]["turned_left_deg"] == 45.0
    assert out["proprioception"]["horizontal_displacement_m"] == 0.5
    assert out["proprioception"]["height_from_start_m"] == -0.2
    assert "targets" not in out and "body_state" not in out
    assert "proprioception" not in f.reset(episode="one")


def test_code_api_skills_use_humanclaws_parser(server):
    f, _ = server
    f.reset(episode="val100:0")
    out = f.skill_turn("right", 200)
    assert out["action"]["cond"] == -120.0 and out["action_text"] == "Turn<right><120>"
    names = f._rpc["code.api"]("raw")["available"]
    assert {
        "walk_forward",
        "turn",
        "side_walk",
        "step_back",
        "step_climb_up",
        "step_climb_down",
        "sit",
        "stand",
    } <= set(names)


def test_selectors():
    from pi_embodied_services.robots.humanclaw.env_server import parse_selector

    assert parse_selector("val100:17") == ("val100", 17, None)
    assert parse_selector("one") == ("one", 0, None)
    assert parse_selector("104862384_172226319_ep12_toilet")[2] == {
        "scene_id": "104862384_172226319",
        "episode_id": "12",
        "object_category": "toilet",
    }
    with pytest.raises(ValueError):
        parse_selector("nonsense")


def test_val100_and_fullval_sizes(server):
    f, _ = server
    assert (
        len(f.list_episodes("val100")) == 100
        and len(f.list_episodes("fullval")) == 1218
    )
    assert len(f.list_episodes("one")) == 1


def test_golden_generator_script_path():
    assert (
        golden.FIXTURE.name == "humanclaw-golden.json"
        and Path(golden.FIXTURE).is_file()
    )
