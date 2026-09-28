# Copyright 2026 The HumanCLAW Authors (github.com/Human-CLAW/HumanCLAW @c4f9351).
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
# Modified by pi-embodied: the per-step body of evaluation/evaluator.py run_rollout() and the
# construction of HCFindNavInteractEvaluator.evaluate_main() split into RPC calls, so the VLM
# loop (planner + verifier, or pi's own) lives in the pi robot.

"""RPC server over one HumanClawBench episode at a time (humanoid Find / Nav / Interact in HSSD).

Nothing of the benchmark is reimplemented: the server imports ``humanclaw_bench`` and drives its
objects in the order ``HCFindNavInteractEvaluator.run_rollout`` does, one decision per call:

- ``env.reset(episode, rollout)``: ``load_episode`` -> ``apply_instruction_version`` -> a fresh
  ``HCFindNavInteractEnv`` for the episode's scene -> ``motion.reset()`` + ``env.reset()`` (the
  evaluator's ``_reset_env_for_rollout``), its trajectory recorder and, with ``--metrics``, its
  ``PaperMetricRecorder`` (``record_reset``). Returns the first 448x448 ego image, the instruction,
  ``max_steps`` and the episode's identity.
- ``env.record_decision(decision)``: the planner/verifier record of the decision about to run
  (paper mode: the ``humanclaw-psv`` provider's raw JSON; pi mode: ``act``'s ``target_visible``).
  Writes the evaluator's ``stepNNN_percept_mid_low.json`` / ``stepNNN_verifier.json`` and hands the
  decision with the target-pixel count of the image the model saw to the metric recorder.
- ``env.step(skill, cond, ...)``: Stop/Stand commits the pose (``{"stop": True}``); any other skill
  runs ``motion.generate(skill, cond)`` -> ``trajectory.record_before`` -> ``env.step(xb_world_75)``
  -> ``record_after`` -> ``metric_recorder.record_motion``. Returns only the new ego image, done,
  the step count and whether the step collided (metric mode); body and object states never leave the server.
- ``env.finish()``: ``motion.unload()`` and the evaluator's ``_finalize_rollout_artifacts``
  (``metrics.json``, the two trajectory NPZs, ``replay_manifest.json``, ego/exo MP4 with
  ``--video``); returns the metric summary.

Habitat contexts cannot be shared between threads, so every call runs on the main thread
(``MainThreadServeMixin``). The evaluator closes Habitat at the end of every rollout; so does
``env.finish`` here, and the next ``env.reset`` builds the next episode's simulator. The motion
runner stays loaded between episodes only as the evaluator keeps it (weights are loaded at
``motion.reset`` and released at ``finish``).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import traceback
from pathlib import Path
from typing import Any

import numpy as np

from pi_embodied_services.components.env_facade_base import BaseEnvFacade
from pi_embodied_services.components.manifest import load_manifest, serve_code_api
from pi_embodied_services.utils.rpc.main_thread_serve import MainThreadServeMixin

#: The motion skills of the paper profile (configs/paper_fullval_v1.json ``motion.skills``).
SKILLS = (
    "walk_forward",
    "side_walk",
    "step_back",
    "turn",
    "step_climb_up",
    "step_climb_down",
    "sit",
    "stand",
)
#: Fixed subsets of the release (main.py ``--episodes``).
SUBSETS = ("one", "val100", "fullval")


def episode_specs(bench: Any, profile: Any, subset: str) -> list[dict[str, str]]:
    """The episode identities of ``one`` / ``val100`` / ``fullval`` or a JSON episode list, in the
    release's order (batch.run_batch: canonical shards, then the list's rows)."""
    from humanclaw_bench.batch import _load_episode_subset
    from humanclaw_bench.benchmark.episodes import list_episode_specs
    from humanclaw_bench.paths import resolve_release_path

    benchmark = profile.section("benchmark")
    rows = list_episode_specs(
        resolve_release_path(benchmark["dataset_dir"]), str(benchmark["split"])
    )
    key = subset.lower()
    if key == "one":
        return rows[:1]
    if key == "fullval":
        return rows
    path = "resources/benchmark/val100.json" if key == "val100" else subset
    return _load_episode_subset(path, rows)


def parse_selector(
    episode: Any,
) -> tuple[str | None, int | None, dict[str, str] | None]:
    """``episode`` as (subset, index, identity): ``"val100:17"``, ``"one"``, a
    ``{scene_id, episode_id, object_category}`` object, or ``"<scene>_ep<id>_<category>"``."""
    if isinstance(episode, dict):
        ident = {
            k: str(episode[k]) for k in ("scene_id", "episode_id", "object_category")
        }
        return None, None, ident
    text = str(episode).strip()
    subset, _, index = text.partition(":")
    if subset.lower() in SUBSETS or subset.endswith(".json"):
        return subset, int(index or 0), None
    scene, sep, rest = text.partition("_ep")
    ep, sep2, category = rest.partition("_")
    if not (sep and sep2 and scene and ep and category):
        raise ValueError(
            f"episode {text!r}: want one|val100|fullval[:index], a JSON list[:index], "
            "<scene_id>_ep<episode_id>_<category> or {scene_id, episode_id, object_category}"
        )
    return (
        None,
        None,
        {"scene_id": scene, "episode_id": ep, "object_category": category},
    )


def collision_steps(metrics: Any) -> int:
    """Motion steps the paper's collision tracker has counted (metric mode), else 0."""
    tracker = getattr(metrics, "collision", None)
    return len(getattr(tracker, "collision_steps", ()) or ())


class HumanclawEnvFacade(MainThreadServeMixin, BaseEnvFacade):
    """One HumanClawBench rollout at a time over the release's own evaluator objects."""

    SERVICE_NAME = "humanclaw-env"

    def __init__(
        self,
        *,
        profile: Any,
        output_root: Path,
        device: str,
        compute_metrics: bool,
        save_video: bool,
        scene_dataset_config: str | None = None,
        max_steps: int | None = None,
        agent_asset: str | None = None,
        bench: Any = None,
    ):
        super().__init__()
        self._profile = profile
        self._bench = bench
        self._output_root = Path(output_root)
        self._meta = {
            "profile": getattr(profile, "profile", ""),
            "compute_metrics": bool(compute_metrics),
            "save_video": bool(save_video),
            "device": device,
            "output_root": str(self._output_root),
            "agent_asset": agent_asset,
            "max_steps": max_steps,
        }
        from humanclaw_bench.evaluation.evaluator import HCFindNavInteractEvaluator

        # The evaluator object holds the construction options and the per-rollout helpers
        # (_reset_env_for_rollout, the recorders, _finalize_rollout_artifacts) this server calls.
        self._ev = HCFindNavInteractEvaluator(
            {
                "profile": profile,
                "output_root": str(self._output_root),
                "device": device,
                "compute_metrics": compute_metrics,
                "save_video": save_video,
                "scene_dataset_config": scene_dataset_config,
                "max_steps": max_steps,
                "agent_asset": agent_asset,
            }
        )
        self._episode: Any = None
        self._rollout = 0
        self._dir: Path | None = None
        self._obs: Any = None
        self._traj: Any = None
        self._metrics: Any = None
        self._video: Any = None
        self._step = 0
        self._done = False
        self._stopped = False
        self._decided = False
        self._summary: dict | None = None

    def _register_rpc(self) -> None:
        super()._register_rpc()
        self._rpc["env.record_decision"] = self.record_decision
        self._rpc["env.metric_observation"] = self.metric_observation
        self._rpc["env.finish"] = self.finish
        self._rpc["env.ground_truth_poses"] = self.ground_truth_poses
        self._rpc["env.list_episodes"] = self.list_episodes
        for skill in SKILLS:
            self._rpc[f"env.{skill}"] = getattr(self, f"skill_{skill}")
        self._readonly_methods.update(
            {"env.metric_observation", "env.ground_truth_poses", "env.list_episodes"}
        )
        # code.api and the startup self-check: packages/embodied/src/primitives/manifests/humanclaw.json.
        serve_code_api(self, load_manifest("humanclaw"), lambda c: c == "privileged")

    # ---- construction (HCFindNavInteractEvaluator.evaluate_main, minus the VLM) ----

    def _build(self, ident: dict[str, str]) -> Any:
        from humanclaw_bench.assets import resolve_agent_asset
        from humanclaw_bench.benchmark.episodes import (
            apply_instruction_version,
            load_episode,
        )
        from humanclaw_bench.envs.find_nav_interact_env import HCFindNavInteractEnv
        from humanclaw_bench.motion.runner import MotionSkillRunner
        from humanclaw_bench.paths import resolve_release_path

        ev, profile = self._ev, self._profile
        benchmark = profile.section("benchmark")
        motion_config = profile.section("motion")
        physics = profile.section("physics")
        rendering = dict(profile.data.get("rendering") or {})
        max_steps = int(ev.config.get("max_steps") or benchmark["max_steps"])
        if ev.config.get("agent_asset"):
            agent_urdf, agent_shift = resolve_agent_asset(str(ev.config["agent_asset"]))
        else:
            agent_urdf = resolve_release_path(physics["agent_urdf"])
            agent_shift = resolve_release_path(physics["agent_shift_npy"])
        override = ev.config.get("scene_dataset_config")
        scene_config = (
            Path(override).expanduser().resolve()
            if override
            else resolve_release_path(benchmark["scene_dataset_config"])
        )
        if not scene_config.is_file():
            raise FileNotFoundError(
                f"Prepared HumanClaw HSSD scene config not found: {scene_config}. Run "
                "`humanclaw-bench prepare-hssd --hssd-root /path/to/hssd-hab` first."
            )
        episode = load_episode(
            benchmark_dataset_dir=resolve_release_path(benchmark["dataset_dir"]),
            split=str(benchmark["split"]),
            scene_id=ident["scene_id"],
            scene_dataset_config=scene_config,
            episode_id=ident["episode_id"],
            object_category=ident["object_category"],
            max_steps=max_steps,
        )
        episode = apply_instruction_version(
            episode, str(benchmark["instruction_version"])
        )
        physics_kwargs = {
            k: v
            for k, v in physics.items()
            if k not in {"backend", "agent_urdf", "agent_shift_npy", "physics_config"}
        }
        ev.env = HCFindNavInteractEnv(
            scene_id=episode.scene_id,
            scene_dataset_config=episode.scene_dataset_config,
            half_physics_backend=str(physics["backend"]),
            agent_urdf=agent_urdf,
            agent_shift_npy=agent_shift,
            physics_config=resolve_release_path(physics["physics_config"]),
            max_episode_steps=max_steps,
            lighting=str(rendering.get("lighting", "ambient")),
            ambient_strength=float(rendering.get("ambient_strength", 1.2)),
            ego_resolution=tuple(rendering.get("ego_resolution", [448, 448])),
            third_person_resolution=tuple(
                rendering.get("third_person_resolution", [512, 512])
            ),
            compute_metrics=ev.compute_metrics,
            video_enabled=ev.save_video,
            **physics_kwargs,
        )
        if ev.motion is None:
            ev.motion = MotionSkillRunner(
                skills=tuple(motion_config["skills"]),
                device=str(ev.config.get("device") or motion_config["device"]),
                weights_root=str(motion_config["weights_root"]),
                weights_manifest=str(motion_config["weights_manifest"]),
                verify_weights=bool(motion_config["verify_weights"]),
            )
        ev.seed_mode = str(motion_config["seed_mode"])
        ev.seed_pkl = (
            str(resolve_release_path(motion_config["seed_pkl"]))
            if motion_config.get("seed_pkl")
            else None
        )
        ev.seed_pt = (
            str(resolve_release_path(motion_config["seed_pt"]))
            if motion_config.get("seed_pt")
            else None
        )
        return episode

    # ---- the rollout, one call per evaluator stage ----

    def list_episodes(self, subset: str = "one") -> list[dict[str, str]]:
        """The episode identities of a subset (``one``, ``val100``, ``fullval``, a JSON list)."""
        return episode_specs(self._bench, self._profile, subset)

    def reset(self, episode: Any = "one", rollout: int = 0, **_: Any) -> dict:
        """Start ``episode`` (see :func:`parse_selector`) as rollout ``rollout``; an unfinished
        earlier rollout is closed first (its replay is written, no metrics: it did not end)."""
        from humanclaw_bench.evaluation.evaluator import (
            _clear_generated_rollout_artifacts,
        )

        if self._episode is not None and self._summary is None:
            self._close(succeeded=False)
        subset, index, ident = parse_selector(episode)
        if ident is None:
            rows = self.list_episodes(subset or "one")
            if not 0 <= int(index or 0) < len(rows):
                raise ValueError(
                    f"{subset}: index {index} out of range 0..{len(rows) - 1}"
                )
            ident = rows[int(index or 0)]
        ev = self._ev
        self._episode = self._build(ident)
        self._rollout = int(rollout)
        self._dir = ev.rollout_dir(self._episode, self._rollout)
        self._dir.mkdir(parents=True, exist_ok=True)
        _clear_generated_rollout_artifacts(self._dir)
        self._traj = self._metrics = self._video = None
        self._step, self._done, self._stopped, self._decided = 0, False, False, False
        self._summary = None
        if ev.save_video:
            from humanclaw_bench.evaluation.video import RolloutVideoWriter

            self._video = RolloutVideoWriter(self._dir, float(ev.env.fps))
            ev.env.set_video_frame_sink(self._video.append)
        self._obs = ev._reset_env_for_rollout(self._episode)
        if self._video is not None:
            ev.env.emit_initial_video_frame()
        self._traj = ev._new_trajectory_recorder(self._episode, self._rollout)
        self._metrics = ev._new_metric_recorder(self._episode, self._rollout)
        return self._observation()

    def _observation(self) -> dict:
        ep = self._episode
        return {
            "ego": np.asarray(self._obs.head_rgb, dtype=np.uint8),
            "instruction": ep.instruction,
            "step": self._step,
            "max_steps": int(ep.max_steps),
            "done": self._done,
            "stopped": self._stopped,
            "episode": {
                "scene_id": ep.scene_label,
                "episode_id": ep.episode_id,
                "object_category": ep.object_category,
                "rollout": self._rollout,
                "key": f"{ep.scene_label}_ep{ep.episode_id}_{ep.object_category}",
                "output_dir": str(self._dir),
            },
        }

    def _require(self) -> None:
        if self._episode is None or self._summary is not None:
            raise RuntimeError("no running episode: call env.reset")

    def metric_observation(self) -> dict:
        """``metric_find_observation()`` on the image last returned (metric mode only)."""
        self._require()
        if self._metrics is None:
            return {"available": False, "target_pixel_count": 0, "metrics": False}
        return self._ev.env.metric_find_observation()

    def record_decision(
        self, decision: dict, find_observation: dict | None = None
    ) -> dict:
        """Record the decision about to run: the step JSON the evaluator writes, then the metric
        recorder's ``record_decision`` with the target pixels of the image the model saw (the
        simulator has not moved since that image, so the current semantic render is that one)."""
        from humanclaw_bench.evaluation.evaluator import _write_step_vlm_records

        self._require()
        if self._done:
            raise RuntimeError("the episode is over; call env.finish")
        result = planner_result(decision)
        written = _write_step_vlm_records(self._dir, self._step, result)
        if self._metrics is not None:
            find = (
                find_observation
                if find_observation is not None
                else self._ev.env.metric_find_observation()
            )
            self._metrics.record_decision(
                step=self._step, decision=result, find_observation=find
            )
        self._decided = True
        return {"step": self._step, "written": [p.name for p in written]}

    def step(
        self,
        skill: str,
        cond: Any = None,
        action_name: str | None = None,
        action_id: int | None = None,
        reasoning: dict | None = None,
        **_: Any,
    ) -> dict:
        """Execute one decision's final SkillCall (run_rollout's loop body)."""
        from humanclaw_bench.agent.skills import SkillCall, skill_to_text

        self._require()
        if self._done:
            raise RuntimeError("the episode is over; call env.finish")
        if skill not in SKILLS:
            raise ValueError(f"unknown skill {skill!r} (have {', '.join(SKILLS)})")
        action = SkillCall(
            skill=skill, cond=cond, action_id=action_id, action_name=action_name
        )
        if not self._decided and self._metrics is not None:
            # A decision nobody recorded (an operator's key) still counts as a decision step.
            self.record_decision({"action": action.to_json()})
        ev, step = self._ev, self._step
        if skill == "stand":
            env_action: Any = {
                "stop": True,
                "skill": "stand",
                "action": action.to_json(),
            }
        else:
            generated = ev.motion.generate(skill, cond)
            self._traj.record_before(
                step=step,
                action=action,
                action_text=skill_to_text(action),
                xb_world_75=generated.xb_world_75,
            )
            env_action = generated.xb_world_75
        self._obs, _reward, done, info = ev.env.step(
            env_action, reasoning=reasoning or {}
        )
        body_state = info.get("body_state") if isinstance(info, dict) else None
        if isinstance(body_state, dict):
            self._traj.record_after(
                step=step,
                body_state=body_state,
                object_states=dict(info.get("object_states") or {}),
            )
        before_collisions = collision_steps(self._metrics)
        if self._metrics is not None and skill != "stand":
            self._metrics.record_motion(
                step=step,
                action_skill=skill,
                info=info if isinstance(info, dict) else {},
            )
        self._step += 1
        self._decided = False
        self._stopped = skill == "stand"
        self._done = bool(done) or self._step >= int(self._episode.max_steps)
        out = self._observation()
        out["action"] = action.to_json()
        out["action_text"] = skill_to_text(action)
        # Only the paper's verdict for this step (fixed-geometry contact above the floor), metric mode.
        out["collision"] = (
            {"collided": collision_steps(self._metrics) > before_collisions}
            if self._metrics is not None
            else {}
        )
        return out

    def _skill(self, action_name: str) -> dict:
        """A code.api skill: HumanCLAW's own action-name parser (planner._chooser_action) picks
        the SkillCall, as for a planner reply."""
        from humanclaw_bench.agent.planner import HumanClawBenchPSVPlanSkillPlanner as P

        call = P._chooser_action(P, {"action_name": action_name})
        return self.step(call.skill, call.cond, call.action_name, call.action_id)

    def skill_walk_forward(self, speed: str = "slow") -> dict:
        return self._skill(f"Walk<forward><{speed}>")

    def skill_turn(self, direction: str, degree: float) -> dict:
        return self._skill(f"Turn<{direction}><{degree}>")

    def skill_side_walk(self, direction: str, distance: float) -> dict:
        return self._skill(f"Side step<{direction}><{distance}>")

    def skill_step_back(self, distance: float) -> dict:
        return self._skill(f"Step back<{distance}>")

    def skill_step_climb_up(self) -> dict:
        return self._skill("Climb upstairs<normal>")

    def skill_step_climb_down(self) -> dict:
        return self._skill("Walk downstairs<normal>")

    def skill_sit(self, height: float) -> dict:
        return self._skill(f"Sit down<{height}>")

    def skill_stand(self) -> dict:
        return self._skill("Stop/Stand")

    def _close(self, *, succeeded: bool) -> None:
        ev = self._ev
        try:
            unload = getattr(ev.motion, "unload", None)
            if callable(unload):
                unload()
        finally:
            ev._finalize_rollout_artifacts(
                output_dir=self._dir,
                trajectory=self._traj,
                metric_recorder=self._metrics,
                video_writer=self._video,
                rollout_succeeded=succeeded,
            )

    def finish(self) -> dict:
        """End the rollout: unload the motion model, write metrics/replay/videos, close Habitat.
        Returns the paper-metric summary of this episode (metric mode) and the artifacts."""
        self._require()
        self._close(succeeded=True)
        metrics_path = self._dir / "metrics.json"
        metrics = (
            json.loads(metrics_path.read_text(encoding="utf-8"))
            if metrics_path.is_file()
            else None
        )
        self._summary = {
            "episode": self._observation()["episode"],
            "steps": self._step,
            "active_stop": self._stopped,
            "metrics": metric_summary(metrics) if metrics else None,
            "metrics_path": str(metrics_path) if metrics else None,
            "videos": (
                [str(self._dir / "ego.mp4"), str(self._dir / "exo.mp4")]
                if self._ev.save_video
                else []
            ),
        }
        return self._summary

    # ---- privileged, cameras, meta ----

    def ground_truth_poses(self, names: list | None = None) -> dict:
        """--privileged only (the TS robot refuses it otherwise): the target instances' centres
        and the humanoid's root pose, Habitat world frame (y up)."""
        del names
        self._require()
        env = self._ev.env
        agent = env.agent
        q = agent.rotation
        return {
            "frame": "habitat_world_y_up",
            "targets": [
                {
                    "object_name": o.get("object_name"),
                    "object_id": o.get("object_id"),
                    "center": o.get("center"),
                }
                for o in self._episode.goal_objects
            ],
            "human_root": {
                "pos": [
                    float(v) for v in np.asarray(agent.translation).reshape(-1)[:3]
                ],
                "quat_xyzw": [
                    float(q.vector.x),
                    float(q.vector.y),
                    float(q.vector.z),
                    float(q.scalar),
                ],
            },
        }

    def get_camera_meta(self, camera_name: str = "ego", **_: Any) -> dict:
        self._require()
        env = self._ev.env
        cam = getattr(env, "ego_camera", None)
        res = list(getattr(cam, "resolution", (448, 448)))
        return {
            "camera_name": camera_name,
            "resolution": res,
            "fps": float(env.fps),
            "frames_per_step": 15,
        }

    def render_camera(self, camera_name: str = "ego", **_: Any) -> dict:
        self._require()
        return {"rgb": np.asarray(self._obs.head_rgb, dtype=np.uint8)}

    def get_task_language(self) -> str:
        self._require()
        return self._episode.instruction

    def chunk_step(self, *args: Any, **kwargs: Any):
        raise NotImplementedError("HumanCLAW steps one skill per decision (env.step)")

    def get_env_meta(self) -> dict:
        return dict(self._meta)

    def close(self) -> None:
        if self._episode is not None and self._summary is None:
            try:
                self._close(succeeded=False)
            except Exception:  # noqa: BLE001
                traceback.print_exc()


def planner_result(decision: dict) -> Any:
    """A ``PlannerResult`` from the TS decision record: raw planner JSON, verifier record, the final
    action and each VLM stage (prompt, parsed response, raw text, error, usage)."""
    from humanclaw_bench.agent.skills import SkillCall
    from humanclaw_bench.agent.types import PlannerResult, PSVStageOutput

    a = dict(decision.get("action") or {})
    action = SkillCall(
        skill=str(a.get("skill") or "stand"),
        cond=a.get("cond"),
        action_id=a.get("action_id"),
        action_name=a.get("action_name"),
    )
    stages = [
        PSVStageOutput(
            stage=str(s.get("stage")),
            raw=dict(s.get("raw") or {}),
            raw_output=str(s.get("raw_output") or ""),
            prompt=str(s.get("prompt") or ""),
            error=s.get("error") or None,
            usage=dict(s.get("usage") or {}),
        )
        for s in decision.get("stages") or []
    ]
    return PlannerResult(
        raw_plan=dict(decision.get("raw_plan") or {}),
        action=action,
        planner_skill=dict(decision.get("planner_skill") or {}),
        verifier=dict(decision.get("verifier") or {}),
        stage_outputs=stages,
    )


def metric_summary(metrics: dict) -> dict:
    """The per-episode paper metrics result.json keeps, read from the evaluator's own
    metrics.json: every success variant, the collision and disturbance scalars, Motion Jerk
    and the cost row (METRICS.md)."""
    success = dict(metrics.get("success") or {})
    body = dict(metrics.get("body_scene") or {})
    keep_body = (
        "collision_step_fraction",
        "collision_steps",
        "motion_steps",
        "by_body_group_step_fraction",
        "initial_penetration_detected",
        "affected_dynamic_object_count",
        "affected_object_path_length_sum_m",
    )
    return {
        **success,
        **{k: body[k] for k in keep_body if k in body},
        "motion_jerk_m_s3": (metrics.get("action_quality") or {}).get(
            "motion_jerk_m_s3"
        ),
        "cost": dict(metrics.get("cost") or {}),
    }


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="HumanClawBench env server")
    p.add_argument("--transport", choices=["http"], default="http")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=0)
    p.add_argument("--profile", default="paper_fullval_v1")
    p.add_argument(
        "--output-root",
        required=True,
        help="rollout artifacts (the evaluator's layout)",
    )
    p.add_argument(
        "--cuda-device",
        default=None,
        help="physical GPU for Habitat and the motion model (sets CUDA_VISIBLE_DEVICES, as the "
        "release's batch dispatcher does; Habitat's CUDA build picks the matching EGL device)",
    )
    p.add_argument("--scene-dataset-config", default=None)
    p.add_argument(
        "--max-steps", type=int, default=None, help="override (smoke tests only)"
    )
    p.add_argument(
        "--metrics", action="store_true", help="the paper metrics (metrics.json)"
    )
    p.add_argument("--video", action="store_true", help="ego.mp4 + exo.mp4 per rollout")
    p.add_argument(
        "--agent-asset", default=None, help="hand-merged (paper) unless given"
    )
    p.add_argument("--parent-watch", action="store_true")
    args = p.parse_args(argv)

    if args.cuda_device is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.cuda_device)
    # The release's dispatcher bounds BLAS pools per rollout process (batch._child_process_env).
    for name in (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        os.environ.setdefault(name, "1")
    try:
        from humanclaw_bench.config import load_config

        profile = load_config(args.profile)
        facade = HumanclawEnvFacade(
            profile=profile,
            output_root=Path(args.output_root).expanduser().resolve(),
            device="cuda",
            compute_metrics=args.metrics,
            save_video=args.video,
            scene_dataset_config=args.scene_dataset_config,
            max_steps=args.max_steps,
            agent_asset=args.agent_asset,
        )
    except Exception:  # noqa: BLE001
        traceback.print_exc()
        return 1
    try:
        facade.serve(
            transport=args.transport,
            host=args.host,
            port=args.port,
            parent_watch=args.parent_watch,
        )
    finally:
        facade.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
