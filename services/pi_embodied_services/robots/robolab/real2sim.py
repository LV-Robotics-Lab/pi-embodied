# Copyright 2026 The Show-Harness Authors.
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
# Modified by pi-embodied: the RoboLab generators of github.com/showlab/Show-Harness @137d571
# (scripts/trajectory/real2sim/robolab/{tasks,oracle,record_demos,follow_tokenize}.py and
# backends/robolab.py) driving this package's env server facade in-process; the frames go
# through the finetuned provider's own camera transform (finetuned/transform.ts, robot robolab)
# instead of core.record.images; no preview video, no fingertip calibration sweep or
# short-finger asset builder (the USD ships in ./assets).

"""Real2sim training data on RoboLab: Scheme A (oracle) and Scheme D (record + follow).

    PY="python -m pi_embodied_services.robots.robolab.real2sim"
    $PY oracle --task RubiksCubeTask --episodes 2 --out <stage>/RubiksCubeTask
    $PY record --task RubiksCubeTask --episodes 2 --tracks <stage>/tracks/RubiksCubeTask
    $PY follow --tracks <stage>/tracks/RubiksCubeTask --out <stage>/RubiksCubeTask_follow

Run with the RoboLab venv (``ROBOLAB_ROOT`` set, as for ./env_server.py); Isaac Sim starts once
per process and one process generates many episodes. The plan is read off the task's own
``subtasks`` (``sim.task_targets``): any single-object pick-and-place task works, others raise
:class:`UnsupportedTask`. Every reset re-samples the object and its container within
+-``--randomize-xy`` (8 cm), so episodes differ; the episode clock is raised to 300 s (the atomic
executor spends ~12 control steps per token; hitting the task's own limit resets the scene
mid-carry and reads exactly like a drop). RoboLab's success DoneTerm is lifted out of the
termination manager during generation (it would freeze the env at RELEASE, so the retreat would
be recorded over a frozen scene) and evaluated on demand with the same function and params.

``oracle``: approach, align, descend, GRASP (bounded empty-grasp retries), lift to one carry
height, carry the OBJECT (not the flange) over the target, descend to the drop height, RELEASE,
>= 2 MV_UP that really move. Kept only if RoboLab's predicate holds, the last token is MV_UP and
<= 5% of the move tokens stalled. ``record``: a continuous servo demo of the same plan, its TCP
path and gripper events to ``<tracks>/track_epNNN.json`` (with the layout it saw). ``follow``:
reset the track's seed (checked against that layout), chase its RDP corners with 2 cm tokens,
re-aim the placement from live state, RELEASE, lift 6 cm, settle.

Output: ``<out>/rollout_NNN`` in the teleop layout, frames as the finetuned provider serves the
RoboLab env server's views at inference (agentview letterboxed to 256; wrist turned fingertips-
up by the server, cropped to 4:3 and letterboxed), what ``train.sh --from-rollouts`` takes.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Optional

import numpy as np

from pi_embodied_services.finetuned.atomic import (
    GRASP,
    RELEASE,
    AtomicExec,
    DemoRecorder,
    EpisodeComplete,
    RolloutWriter,
    TokenBudgetExceeded,
    TokenEpisode,
    rdp,
    stalled_token_count,
)

# -- task geometry (robolab/tasks.py) ----------------------------------------------------------

#: Panda-hand flange (``panda_hand``) -> short-finger tip, measured by Show-Harness on its 35.9 mm
#: bracket (calibrate_fingertip.py on RubiksCubeTask: held from 103.7 to 156.4 mm above the
#: centroid); 0.130 leaves the 46 mm blocks ~20 mm of margin on both sides.
FLANGE_TO_FINGERTIP_M = 0.130
#: Carry height above the grasp / the target rim (bowls and bins are 6-12 cm tall).
CARRY_CLEARANCE_M = 0.12
#: Object underside above the drop surface when the fingers open (one token).
DROP_MARGIN_M = 0.02
APPROACH_CLEARANCE_M = 0.10
#: Closed on something: a held object reports its own width; empty reads ~0.
GRASPED_WIDTH_M = 0.005
#: Control steps before a gripper width means anything (the binary command travels slowly).
GRIPPER_SETTLE_STEPS = 40
#: Commanded closed but this open: the object slipped out.
DROPPED_WIDTH_M = 0.075
#: Generation clock and layout jitter (backends/robolab.py).
GEN_EPISODE_LENGTH_S = 300.0
RANDOMIZE_XY_M = 0.08
#: follow_tokenize.py: settle budget (the static predicate needs ~40 steps), post-release lift,
#: and how far a reset may put an object from where the demo saw it.
SETTLE_STEPS = 90
RELEASE_RETREAT_M = 0.06
LAYOUT_TOL_M = 0.02
MIN_TOKEN_TRAVEL_M = 0.006


class UnsupportedTask(RuntimeError):
    """The task's subtask spec is not a single-object pick-and-place."""


def resolve_plan(targets: dict) -> dict:
    """``{"object", "target", "target_kind": container|surface}`` from ``sim.task_targets``."""
    objects = list(targets.get("objects") or [])
    if not objects:
        raise UnsupportedTask(f"no manipulable object in the subtasks ({targets!r})")
    if len(objects) > 1:
        raise UnsupportedTask(f"{len(objects)} objects ({objects}); single-object only")
    for kind in ("container", "surface"):
        if targets.get(kind):
            return {"object": objects[0], "target": targets[kind], "target_kind": kind}
    raise UnsupportedTask(
        f"object {objects[0]!r} but no container/surface to place it on"
    )


def _extent(backend, name: str) -> Optional[tuple[float, float]]:
    try:
        return backend.object_extent(name)
    except Exception:  # noqa: BLE001 -- prims without queryable geometry
        return None


def grasp_tcp(backend, obj: str, z_jitter: float = 0.0, rng=None) -> np.ndarray:
    """Flange pose putting the fingertips around the object's centre."""
    c = backend.object_centroid(obj)
    z = float(c[2]) + FLANGE_TO_FINGERTIP_M
    if z_jitter and rng is not None:
        z += float(rng.uniform(-z_jitter, z_jitter))
    return np.array([c[0], c[1], z], dtype=np.float64)


def approach_tcp(backend, obj: str) -> np.ndarray:
    return grasp_tcp(backend, obj) + [0.0, 0.0, APPROACH_CLEARANCE_M]


def carry_z(
    backend, obj: str, target: str, grasp_flange_z: Optional[float] = None
) -> float:
    """Clear of the pick site and the target rim. Pass ``grasp_flange_z`` once the object is
    held: the held object's centroid follows the hand, and each recomputation would ratchet up."""
    ext = _extent(backend, target)
    top = ext[1] if ext is not None else float(backend.object_centroid(target)[2])
    pick = grasp_flange_z if grasp_flange_z is not None else grasp_tcp(backend, obj)[2]
    return max(top + FLANGE_TO_FINGERTIP_M, float(pick)) + CARRY_CLEARANCE_M


def place_tcp(
    backend, obj: str, target: str, drop_margin_m: float = DROP_MARGIN_M
) -> np.ndarray:
    """Flange pose to open at: the carried OBJECT over the target centre, its underside
    ``drop_margin_m`` above the target's top (all from live geometry)."""
    tc, oc, tcp = (
        backend.object_centroid(target),
        backend.object_centroid(obj),
        backend.tcp_pos(),
    )
    xy = tc[:2] - (oc[:2] - tcp[:2])
    text = _extent(backend, target)
    top = text[1] if text is not None else float(tc[2])
    oext = _extent(backend, obj)
    hang = float(tcp[2] - oext[0]) if oext is not None else FLANGE_TO_FINGERTIP_M
    return np.array([xy[0], xy[1], top + drop_margin_m + hang], dtype=np.float64)


# -- the backend over the env server facade (backends/robolab.py) ------------------------------


class FacadeBackend:
    """The atomic core's backend over a RobolabEnvFacade, run in-process."""

    def __init__(self, facade: Any, handle: Any) -> None:
        self.f = facade
        self.env = handle.env
        self.targets = dict(handle.targets)
        self.task_description = handle.instruction
        self.env_id = handle.env_name
        self._suspended: list = []

    def reset(self, seed: int) -> None:
        """Seed the global RNG the reset events draw from, then the facade's reset."""
        np.random.seed(int(seed))
        try:
            self.env.seed(int(seed))
        except Exception:  # noqa: BLE001 -- no env.seed() on this Isaac Lab
            pass
        self.f.reset()

    def tcp_pos(self) -> np.ndarray:
        from pi_embodied_services.robots.robolab import sim

        return sim.rl_tcp(self.env).astype(np.float64)

    def tcp_pose7(self) -> list[float]:
        from pi_embodied_services.robots.robolab import sim

        return [float(v) for v in (*sim.rl_tcp(self.env), *sim.rl_ee_quat(self.env))]

    def gripper_width(self) -> float:
        from pi_embodied_services.robots.robolab import sim

        return float(sim.rl_gripper_width(self.env))

    def ee_tilt_deg(self) -> float:
        from pi_embodied_services.robots.robolab import sim

        return float(sim.ee_tilt_deg(sim.rl_ee_quat(self.env)))

    def apply_delta(self, delta_m, grip_cmd: float, max_cmd_m: float) -> None:
        from pi_embodied_services.robots.robolab.env_server import CLOSE, OPEN

        self.f._gripper = CLOSE if float(grip_cmd) < 0 else OPEN
        self.f._control(
            np.clip(np.asarray(delta_m, dtype=float), -max_cmd_m, max_cmd_m), 1
        )

    def grab_frames(self):
        """The env server's views (the finetuned transform runs over the finished rollout)."""
        views = self.f._images()
        return views["agentview"], views["wrist"]

    def frozen(self) -> bool:
        return bool(self.f._terminated or self.f._truncated)

    def object_centroid(self, name: str) -> np.ndarray:
        from pi_embodied_services.robots.robolab import sim

        return sim.rl_centroid(self.env, name)

    def object_extent(self, name: str) -> tuple[float, float]:
        from pi_embodied_services.robots.robolab import sim

        return sim.rl_extent(self.env, name)

    def success(self) -> bool:
        if self._suspended:
            for _name, cfg in self._suspended:
                try:
                    value = cfg.func(self.env, **cfg.params)
                except Exception:  # noqa: BLE001 -- a predicate that cannot run is not success
                    return False
                from pi_embodied_services.robots.robolab import sim

                if bool(np.asarray(sim.to_np(value)).reshape(-1)[0]):
                    return True
            return False
        from pi_embodied_services.robots.robolab import sim

        return bool(self.f._terminated or sim.rl_success(self.env))

    def suspend_task_termination(self) -> list[str]:
        """Lift every DoneTerm but ``time_out`` out of the termination manager (generation only:
        RoboLab freezes the env the instant its success predicate fires, i.e. at RELEASE)."""
        manager = getattr(self.env, "termination_manager", None)
        if self._suspended or manager is None:
            return [n for n, _c in self._suspended]
        keep_names, keep_cfgs = [], []
        for name, cfg in zip(list(manager._term_names), list(manager._term_cfgs)):
            if bool(getattr(cfg, "time_out", False)):
                keep_names.append(name)
                keep_cfgs.append(cfg)
            else:
                self._suspended.append((name, cfg))
        manager._term_names, manager._term_cfgs = keep_names, keep_cfgs
        gone = {id(c) for _n, c in self._suspended}
        manager._class_term_cfgs = [
            c for c in getattr(manager, "_class_term_cfgs", []) if id(c) not in gone
        ]
        return [n for n, _c in self._suspended]


def executor(backend, step_m: float) -> AtomicExec:
    # max_cmd_m == step_m keeps the IK target within what the relative IK tracks cleanly.
    return AtomicExec(
        backend,
        step_m=step_m,
        max_cmd_m=step_m,
        max_ctrl_steps=24,
        gripper_steps=GRIPPER_SETTLE_STEPS,
    )


# -- Scheme A: the oracle (robolab/oracle.py) ---------------------------------------------------


class OracleEpisode(TokenEpisode):
    def __init__(
        self, backend, writer, rng, plan, step_m, max_tokens, retreat_tokens=2
    ):
        super().__init__(backend, writer, executor(backend, step_m), max_tokens, rng)
        self.plan = plan
        self.retreat_tokens = int(retreat_tokens)
        self.retreat_travel_m: list[float] = []

    def _grasp(self) -> bool:
        for _ in range(3):
            self.emit(GRASP)
            if self.backend.gripper_width() > GRASPED_WIDTH_M:
                return True
            self.emit(RELEASE)
            self.emit("MV_DOWN")
        return False

    def run(self) -> dict:
        b, obj, target = self.backend, self.plan["object"], self.plan["target"]
        # one dominant-axis pursuit to the approach pose (descends first from a high start)
        self.chase(approach_tcp(b, obj), tol=0.012)
        self.align_xy(grasp_tcp(b, obj))
        self.go_z(grasp_tcp(b, obj, z_jitter=0.004, rng=self.rng)[2])
        self.align_xy(grasp_tcp(b, obj))
        grasp_z = float(b.tcp_pos()[2])
        if not self._grasp():
            return {"success": False, "reason": "grasp_failed"}
        self.go_z(
            carry_z(b, obj, target, grasp_z)
        )  # pure vertical, then pure horizontal
        if b.gripper_width() < GRASPED_WIDTH_M:
            return {"success": False, "reason": "dropped_on_lift"}
        place = place_tcp(b, obj, target)
        self.align_xy(place)
        if b.gripper_width() < GRASPED_WIDTH_M:
            return {"success": False, "reason": "dropped_in_transit"}
        self.go_z(place[2])
        self.emit(RELEASE)
        if not self._retreat():
            return {"success": False, "reason": "retreat_blocked"}
        return {"success": bool(self.settle(20) or b.success())}

    def _retreat(self) -> bool:
        """>= ``retreat_tokens`` MV_UP that really lift: the episode must not end on RELEASE
        (its last frame also becomes the synthesized DONE)."""
        for _ in range(self.retreat_tokens):
            before = float(self.backend.tcp_pos()[2])
            try:
                self.emit("MV_UP", "move")
            except EpisodeComplete:
                return False
            travel = float(self.backend.tcp_pos()[2]) - before
            self.retreat_travel_m.append(travel)
            if travel < 0.3 * self.exec.step_m:
                return False
        return True


def oracle(args, backend) -> tuple[int, int]:
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    kept = tried = 0
    while kept < args.episodes and tried < args.episodes * 4:
        seed = args.seed0 + tried
        tried += 1
        rng = np.random.default_rng(seed)
        backend.reset(seed)
        plan = resolve_plan(backend.targets)
        rollout = out / f"rollout_{kept:03d}"
        writer = RolloutWriter(rollout)
        ep = OracleEpisode(backend, writer, rng, plan, args.step_m, args.max_tokens)
        try:
            result = ep.run()
        except EpisodeComplete:
            result = {"success": bool(backend.success()), "reason": ""}
        except TokenBudgetExceeded as exc:
            result = {"success": False, "reason": str(exc)}
        ok = bool(result.get("success"))
        last = writer.tokens[-1] if writer.tokens else None
        if ok and last != "MV_UP":
            ok, result = False, {"reason": f"bad_ending:{last}"}
        stalled = stalled_token_count(rollout, MIN_TOKEN_TRAVEL_M)
        if ok and writer.step and stalled / writer.step > args.max_stalled_frac:
            ok, result = False, {"reason": f"stalled:{stalled}/{writer.step}"}
        meta = {
            "source": f"oracle_{args.task}",
            "method": "privileged_oracle",
            "retreat_travel_m": [round(t, 5) for t in ep.retreat_travel_m],
            "last_token": last,
            "stalled_tokens": stalled,
            "plan": plan,
            "seed": seed,
            "success": ok,
            "reason": result.get("reason", ""),
        }
        kept += finish(args, backend, writer, meta, ok)
    return kept, tried


# -- Scheme D (robolab/record_demos.py, follow_tokenize.py) -------------------------------------


def layout_of(backend, plan: dict) -> dict:
    out = {}
    for key in ("object", "target"):
        try:
            out[plan[key]] = [
                round(float(v), 5) for v in backend.object_centroid(plan[key])
            ]
        except Exception:  # noqa: BLE001
            pass
    return out


def scripted_demo(rec: DemoRecorder, backend, plan: dict, rng) -> bool:
    obj, target = plan["object"], plan["target"]
    hover = 0.05 + float(rng.uniform(0, 0.02))
    grasp = grasp_tcp(backend, obj)
    rec.servo_to(grasp + [0, 0, hover])
    rec.servo_to(grasp)
    grasp_z = float(rec._tcp()[2])
    rec.set_gripper(close=True, steps=GRIPPER_SETTLE_STEPS)
    if backend.gripper_width() < GRASPED_WIDTH_M:
        return False
    travel_z = carry_z(
        backend, obj, target, grasp_z
    )  # ONE carry height for the transport
    rec.servo_to(np.array([grasp[0], grasp[1], travel_z]))
    place = place_tcp(backend, obj, target)
    rec.servo_to(np.array([place[0], place[1], travel_z]))
    rec.servo_to(place)
    rec.set_gripper(close=False)
    rec.servo_to(rec._tcp() + [0, 0, 0.08])
    return rec.hold_until_success(20)


def record(args, backend) -> tuple[int, int]:
    tracks = Path(args.tracks)
    tracks.mkdir(parents=True, exist_ok=True)
    kept = tried = 0
    layouts = set()
    while kept < args.episodes and tried < args.episodes * 3:
        seed = args.seed0 + tried
        tried += 1
        rng = np.random.default_rng(seed)
        backend.reset(seed)
        plan = resolve_plan(backend.targets)
        layout = layout_of(backend, plan)
        rec = DemoRecorder(backend)
        rec.capture()
        if not scripted_demo(rec, backend, plan, rng):
            print(f"[record:{args.task}] seed={seed} demo failed; skip", flush=True)
            continue
        track = {
            "seed": seed,
            "sim": "robolab",
            "env_id": backend.env_id,
            "task": backend.task_description,
            "task_key": args.task,
            "plan": plan,
            "layout": layout,
            **rec.track(),
        }
        name = f"track_ep{kept:03d}.json"
        (tracks / name).write_text(json.dumps(track))
        layouts.add(tuple(round(v, 4) for xyz in layout.values() for v in xyz))
        print(
            f"[record:{args.task}] seed={seed} {len(rec.tcp)} ctrl steps -> {name}",
            flush=True,
        )
        kept += 1
    if kept > 1 and len(layouts) == 1:
        print(
            f"[record:{args.task}] WARNING: all {kept} tracks share ONE layout",
            flush=True,
        )
    return kept, tried


def _moves(seg: np.ndarray, step_m: float) -> bool:
    """Did the demo travel in this segment (not a parked hold)?"""
    return len(seg) >= 2 and bool(
        np.max(np.linalg.norm(seg - seg[0], axis=1)) > 0.5 * step_m
    )


def follow_track(args, backend, track: dict, writer: RolloutWriter) -> dict:
    tcp = np.asarray(track["tcp"], dtype=np.float64)
    events = [(int(i), str(t)) for i, t in track["events"]]
    bounds = [0] + [i for i, _ in events] + [len(tcp) - 1]
    plan = track["plan"]
    ep = TokenEpisode(backend, writer, executor(backend, args.step_m), args.max_tokens)

    def dropped() -> bool:
        return ep.exec.gripper_closed and backend.gripper_width() > DROPPED_WIDTH_M

    settled = None
    try:
        for si in range(len(bounds) - 1):
            a, b = bounds[si], bounds[si + 1]
            if b > a and _moves(tcp[a : b + 1], args.step_m):
                seg = tcp[a : b + 1]
                corners = rdp(seg)
                released = si > 0 and events[si - 1][1] == RELEASE
                for k, ci in enumerate(corners[1:], 1):
                    last = si == len(bounds) - 2 and k == len(corners) - 1
                    tight = (last or si < len(events)) and k == len(corners) - 1
                    goal = seg[ci]
                    if released:
                        # after RELEASE only the demo's height is followed: its x/y were
                        # above the DEMO's placement, not the follower's re-aimed one
                        t = backend.tcp_pos()
                        goal = np.array([t[0], t[1], max(goal[2], t[2])])
                    ep.chase(goal, tol=0.008 if tight else 0.012)
                    if dropped():
                        return {"success": False, "reason": "dropped_in_transit"}
            if si < len(events):
                tok = events[si][1]
                if tok == RELEASE:
                    place = place_tcp(backend, plan["object"], plan["target"])
                    ep.chase(
                        np.array([place[0], place[1], backend.tcp_pos()[2]]), tol=0.008
                    )
                    place = place_tcp(backend, plan["object"], plan["target"])
                    t = backend.tcp_pos()
                    ep.chase(np.array([t[0], t[1], place[2]]), tol=0.008)
                ep.emit(tok, "grasp" if tok == GRASP else "release")
                if tok == RELEASE:
                    # straight up: require_gripper_detached needs the fingers off the object
                    ep.go_z(backend.tcp_pos()[2] + RELEASE_RETREAT_M)
        settled = ep.settle_steps(SETTLE_STEPS)
        if settled is None and _moves(tcp[bounds[-2] :], args.step_m):
            # One corrective round, VERTICAL only: the demo parked above ITS release point,
            # and chasing that pose in x/y after the follower released elsewhere drags the
            # open hand sideways over the placed object (the episode then ends on MV_FWD).
            t = backend.tcp_pos()
            ep.chase(np.array([t[0], t[1], max(tcp[-1][2], t[2])]), tol=0.011)
            settled = ep.settle_steps(SETTLE_STEPS)
        success = settled is not None
    except EpisodeComplete:
        success, settled = backend.success(), 0
    except TokenBudgetExceeded as exc:
        return {"success": False, "reason": str(exc)}
    geom = {}
    try:
        oc = backend.object_centroid(plan["object"])
        tc = backend.object_centroid(plan["target"])
        geom = {
            "final_object_to_target_xy_m": round(
                float(np.linalg.norm(oc[:2] - tc[:2])), 4
            ),
            "final_ee_tilt_deg": round(backend.ee_tilt_deg(), 2),
        }
    except Exception:  # noqa: BLE001 -- diagnostics only
        pass
    return {
        "success": bool(success),
        "reason": "",
        "settle_steps_used": settled,
        **geom,
    }


def follow(args, backend) -> tuple[int, int]:
    files = sorted(Path(args.tracks).glob("track_ep*.json"))
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    kept = 0
    for tf in files:
        track = json.loads(tf.read_text())
        backend.reset(int(track["seed"]))
        drift = [
            float(np.linalg.norm(backend.object_centroid(n) - np.asarray(xyz)))
            for n, xyz in (track.get("layout") or {}).items()
        ]
        if drift and max(drift) > LAYOUT_TOL_M:
            print(
                f"[follow] {tf.name} SKIPPED: layout drift {max(drift) * 100:.1f} cm",
                flush=True,
            )
            continue
        writer = RolloutWriter(out / f"rollout_{kept:03d}")
        result = follow_track(args, backend, track, writer)
        last = writer.tokens[-1] if writer.tokens else None
        if result["success"] and last != "MV_UP":  # the oracle's ending rule
            result = {**result, "success": False, "reason": f"bad_ending:{last}"}
        meta = {
            "source": f"follow_{args.task}",
            "method": "closed_loop_follower",
            "track": tf.name,
            "plan": track["plan"],
            "seed": track["seed"],
            "last_token": writer.tokens[-1] if writer.tokens else None,
            **result,
        }
        kept += finish(args, backend, writer, meta, bool(result["success"]))
    return kept, len(files)


# -- output ------------------------------------------------------------------------------------


def finish(args, backend, writer: RolloutWriter, meta: dict, ok: bool) -> int:
    """Close the rollout; keep it (frames through the provider's transform) or drop it."""
    from pi_embodied_services.finetuned.lerobot_to_rollouts import transform_frames

    views = None
    if ok or args.keep_failures:
        jobs = [
            {"src": str(p), "dst": str(p), "camera": cam}
            for cam in ("agentview", "wrist")
            for p in sorted((writer.dir / cam).glob("*.png"))
        ]
        views = transform_frames(jobs, None, "robolab", "", args.node) if jobs else None
    writer.close(
        {
            **meta,
            "sim": "robolab",
            "env_id": backend.env_id,
            "task": args.task,
            "task_text": backend.task_description,
            "step_m": args.step_m,
            "views": views,
            "success": ok,
        }
    )
    print(
        f"[{args.cmd}:{args.task}] seed={meta['seed']} {writer.step} tokens "
        f"success={ok} {meta.get('reason') or ''}",
        flush=True,
    )
    if ok or args.keep_failures:
        return 1
    shutil.rmtree(writer.dir, ignore_errors=True)
    return 0


def build(args):
    """Isaac Sim + the task (randomised layout, generation clock) + the env server facade."""
    from pi_embodied_services.robots.robolab import sim
    from pi_embodied_services.robots.robolab.env_server import (
        COMMAND_GAIN,
        GRIPPER_HOLD_STEPS,
        STEP_M,
        STEPS_PER_DECISION,
        RobolabEnvFacade,
        fresh_dir,
    )

    os.environ.pop(
        "CUDA_VISIBLE_DEVICES", None
    )  # Vulkan ignores it; --cuda-device pins
    os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
    os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")
    device = f"cuda:{args.cuda_device}"
    logs = Path(
        os.environ.get("PI_EMBODIED_LOGS", Path.home() / ".cache" / "pi-embodied")
    )
    out = fresh_dir(
        logs
        / "robolab-real2sim"
        / f"{args.task}-{time.strftime('%Y%m%d-%H%M%S')}-{os.getpid()}"
    )
    os.environ["ROBOLAB_PANDA_USD"] = str(
        sim.localize_panda_usd(out, args.isaac_assets)
    )
    app = sim.launch_isaac(device=device)
    handle = sim.make_task(
        args.task,
        device=device,
        seed=args.seed0,
        output_dir=out,
        episode_length_s=GEN_EPISODE_LENGTH_S,
        randomize_xy_m=args.randomize_xy or None,
    )
    meta = {
        "task": args.task,
        "seed": args.seed0,
        "robot": "franka",
        "agentview_camera": "front_cam",
        "wrist_camera": "wrist_cam",
        "step_m": STEP_M,
        "command_gain": COMMAND_GAIN,
        "steps_per_decision": STEPS_PER_DECISION,
        "gripper_hold_steps": GRIPPER_HOLD_STEPS,
        "settle_steps": 8,
        "output_dir": str(out),
    }
    return RobolabEnvFacade(app=app, handle=handle, meta=meta), handle


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    parsers = {
        "oracle": sub.add_parser(
            "oracle", help="Scheme A: privileged oracle in action units"
        ),
        "record": sub.add_parser(
            "record", help="Scheme D step 1: continuous demos -> tracks"
        ),
        "follow": sub.add_parser(
            "follow", help="Scheme D step 2: tracks -> action units"
        ),
    }
    for name, p in parsers.items():
        if name != "follow":
            p.add_argument(
                "--task", required=True, help="RoboLab task class, e.g. RubiksCubeTask"
            )
            p.add_argument("--episodes", type=int, default=10)
            p.add_argument(
                "--seed0", type=int, default=1000 if name == "oracle" else 2000
            )
        p.add_argument("--step-m", type=float, default=0.02)
        p.add_argument(
            "--max-tokens", type=int, default=160 if name == "oracle" else 200
        )
        p.add_argument(
            "--randomize-xy",
            type=float,
            default=RANDOMIZE_XY_M,
            help="0 = authored layout",
        )
        p.add_argument("--cuda-device", type=int, default=0)
        p.add_argument("--isaac-assets", default=os.environ.get("ROBOLAB_ISAAC_ASSETS"))
        p.add_argument(
            "--node", default=shutil.which("node") or "node", help="for transform.ts"
        )
        p.add_argument("--keep-failures", action="store_true")
    parsers["oracle"].add_argument("--out", required=True)
    parsers["oracle"].add_argument("--max-stalled-frac", type=float, default=0.05)
    parsers["record"].add_argument("--tracks", required=True)
    parsers["follow"].add_argument("--tracks", required=True)
    parsers["follow"].add_argument("--out", required=True)
    args = ap.parse_args(argv)
    if args.cmd == "follow":
        files = sorted(Path(args.tracks).glob("track_ep*.json"))
        if not files:
            raise SystemExit(f"no tracks under {args.tracks}")
        first = json.loads(files[0].read_text())
        args.task, args.seed0 = first["task_key"], int(first["seed"])

    facade = None
    try:
        facade, handle = build(args)
        backend = FacadeBackend(facade, handle)
        resolve_plan(backend.targets)
        print(
            f"[{args.cmd}] {args.task}: {backend.task_description!r} plan={resolve_plan(backend.targets)}"
        )
        print(
            f"[{args.cmd}] termination suspended: {backend.suspend_task_termination()}"
        )
        kept, tried = {"oracle": oracle, "record": record, "follow": follow}[args.cmd](
            args, backend
        )
        print(f"[{args.cmd}:{args.task}] kept {kept}/{tried}", flush=True)
        code = 0 if kept else 2
    except UnsupportedTask as exc:
        print(f"[{args.cmd}] {exc}", flush=True)
        code = 2
    except BaseException:
        # SimulationApp.close() can swallow the traceback and exit 0: print, leave hard.
        traceback.print_exc()
        code = 1
    sys.stdout.flush()
    sys.stderr.flush()
    if facade is not None:
        os._exit(code)  # Kit's shutdown can hang or eat the exit code
    return code


if __name__ == "__main__":
    raise SystemExit(main())
