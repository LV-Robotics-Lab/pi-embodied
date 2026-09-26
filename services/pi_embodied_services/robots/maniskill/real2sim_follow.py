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
# Modified by pi-embodied: Scheme D of github.com/showlab/Show-Harness @137d571
# (scripts/trajectory/real2sim/maniskill/record_demos.py + follow_tokenize.py, their ManiSkill
# backend), driving the env server's facade in-process like ./real2sim.py (Scheme A); the scene
# is reproduced by the facade's own seeded reset (scenes.reset) and checked against the layout
# the track recorded; no preview video.

"""Real2sim training data, Scheme D: a continuous demo re-walked in action units.

    PY=-m pi_embodied_services.robots.maniskill.real2sim_follow
    python $PY record --env-id BlockPAP-v1 --episodes 4 --seed0 20000 --tracks <stage>/tracks
    python $PY follow --tracks <stage>/tracks --out <stage>/blockpap_follow

``record`` drives each episode with a smooth privileged servo (continuous, multi-axis, like a
motion planner or a human demo: grasp the carried block, lift, carry, seat it on the target,
open, lift) and writes the successful ones' TCP paths and gripper events to
``<tracks>/track_epNNN.json``. No frame is recorded.

``follow`` resets each track's seed (the facade's reset reproduces the layout; a track whose
layout drifts > 2 cm is skipped), reduces the path to RDP corners between gripper events and
chases each corner with dominant-axis 2 cm tokens, recording the frame before every token
(``finetuned/atomic.py``); before RELEASE the carried block is steered onto the target from the
live state. Every frame is a state the discrete controller reached, so frame and label agree;
episodes the env does not score successful are dropped. The frames are the env server's
(``--scene`` as ./real2sim.py), and the output is the teleop layout ``train.sh --from-rollouts``
and ``finetuned/make_dataset.py`` read: ``<out>/rollout_NNN``.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any

import numpy as np

from pi_embodied_services.finetuned.atomic import (
    GRASP,
    RELEASE,
    AtomicExec,
    DemoRecorder,
    RolloutWriter,
    TokenBudgetExceeded,
    TokenEpisode,
    rdp,
)
from pi_embodied_services.robots.maniskill.scenes import SCENES

#: Height the carried block's centre ends at, given the target's position (tasks.TASKS drop_z).
DROP_Z = {
    "BlockPAP-v1": lambda env, t: (
        t[2] + float(env.COASTER_THICKNESS) + float(env.BLOCK_HALF_SIZE[2]) + 0.004
    ),
    "BlockStack-v1": lambda env, t: t[2] + 2 * float(env.BLOCK_HALF_SIZE[2]),
}
#: A layout further than this from the track's (one token) means the reset did not reproduce it.
LAYOUT_TOL_M = 0.02


class FacadeBackend:
    """The atomic core's backend over a ManiskillEnvFacade (ManiSkillBackend of Show-Harness)."""

    delta_bound_m = 0.1  # pd_ee_delta_pos: metres at action 1.0

    def __init__(self, facade: Any, env_id: str) -> None:
        self.f = facade
        self.env = facade._env.unwrapped
        self.env_id = env_id

    def reset(self, seed: int) -> dict:
        self.f.reset(seed=int(seed))
        return dict(self.f.get_env_meta().get("layout") or {})

    def tcp_pos(self) -> np.ndarray:
        return np.asarray(self.f._state()["tcp_pos"], dtype=np.float64)

    def tcp_pose7(self) -> list[float]:
        s = self.f._state()
        return [float(v) for v in (*s["tcp_pos"], *s["tcp_quat_wxyz"])]

    def gripper_width(self) -> float:
        return float(self.f._state()["gripper_width"])

    def apply_delta(self, delta_m, grip_cmd: float, max_cmd_m: float) -> None:
        b = self.delta_bound_m
        a = np.clip(np.asarray(delta_m) / b, -max_cmd_m / b, max_cmd_m / b)
        obs, *_ = self.f._step(np.array([a[0], a[1], a[2], float(grip_cmd)]))
        self.f._obs = obs

    def grab_frames(self):
        views = self.f._pack(self.f._obs)
        return views["agentview"], views["wrist"]

    def success(self) -> bool:
        return bool(self.env.evaluate()["success"].reshape(-1)[0])

    def actor_pos(self, name: str) -> np.ndarray:
        return (
            getattr(self.env, name).pose.p.reshape(-1, 3)[0].cpu().numpy().astype(float)
        )


def layout_drift(backend: FacadeBackend, layout: dict) -> float:
    """Largest distance between where the track saw an actor and where it is now."""
    return max(
        (
            float(np.linalg.norm(backend.actor_pos(name) - np.asarray(pose["p"])))
            for name, pose in (layout or {}).items()
        ),
        default=0.0,
    )


def scripted_demo(rec: DemoRecorder, backend: FacadeBackend, hover: float) -> bool:
    """record_demos._demo_pick_and_place: steer by the CARRIED block, not the TCP."""
    rig, env = SCENES[backend.env_id], backend.env
    obj, tgt = backend.actor_pos(rig.carried), backend.actor_pos(rig.target)
    rec.servo_to(obj + [0, 0, hover])
    rec.servo_to(obj + [0, 0, 0.002])
    rec.set_gripper(close=True)
    rec.servo_to(obj + [0, 0, 0.12])
    off = backend.actor_pos(rig.carried)[:2] - rec._tcp()[:2]
    tx, ty = tgt[0] - off[0], tgt[1] - off[1]
    rec.servo_to(np.array([tx, ty, tgt[2] + 0.12]))
    above = rec._tcp()[2] - backend.actor_pos(rig.carried)[2]
    rec.servo_to(np.array([tx, ty, DROP_Z[backend.env_id](env, tgt) + above]))
    rec.set_gripper(close=False)
    rec.servo_to(rec._tcp() + [0, 0, 0.08])
    return rec.hold_until_success(14)


def retarget_place(ep: TokenEpisode, backend: FacadeBackend) -> None:
    """Before RELEASE: steer the carried block onto the target, drop height from its live
    underside (the follower's grasp offset differs from the demo's)."""
    rig = SCENES[backend.env_id]
    target = backend.actor_pos(rig.target)
    obj, tcp = backend.actor_pos(rig.carried), backend.tcp_pos()
    off = obj[:2] - tcp[:2]
    ep.chase(np.array([target[0] - off[0], target[1] - off[1], tcp[2]]), tol=0.006)
    obj, tcp = backend.actor_pos(rig.carried), backend.tcp_pos()
    drop = DROP_Z[backend.env_id](backend.env, target) + (tcp[2] - obj[2])
    ep.chase(np.array([tcp[0], tcp[1], drop]), tol=0.006)


def follow_track(backend, track: dict, out_dir: Path, step_m: float) -> dict:
    tcp = np.asarray(track["tcp"], dtype=np.float64)
    events = [(int(i), str(t)) for i, t in track["events"]]
    bounds = [0] + [i for i, _ in events] + [len(tcp) - 1]
    writer = RolloutWriter(out_dir)
    ep = TokenEpisode(
        backend, writer, AtomicExec(backend, step_m=step_m), max_tokens=160
    )
    try:
        for si in range(len(bounds) - 1):
            a, b = bounds[si], bounds[si + 1]
            if b > a:
                seg = tcp[a : b + 1]
                corners = rdp(seg)
                for k, ci in enumerate(corners[1:], 1):
                    last = si == len(bounds) - 2 and k == len(corners) - 1
                    tight = (last or si < len(events)) and k == len(corners) - 1
                    ep.chase(seg[ci], tol=0.008 if tight else 0.012)
            if si < len(events):
                tok = events[si][1]
                if tok == RELEASE:
                    retarget_place(ep, backend)
                ep.emit(tok, "grasp" if tok == GRASP else "release")
        success = ep.settle(16)
        if not success and len(tcp) > 1:
            ep.chase(tcp[-1], tol=0.011)  # one corrective round
            success = ep.settle(16)
    except TokenBudgetExceeded as exc:
        return {"success": False, "reason": str(exc), "writer": writer}
    return {"success": bool(success), "reason": "", "writer": writer}


def parse_scene(scene: str) -> dict | None:
    return dict(kv.split("=", 1) for kv in scene.split(",") if kv) or None


def make_facade(env_id: str, seed: int, scene: dict | None):
    from pi_embodied_services.robots.maniskill.env_server import ManiskillEnvFacade

    return ManiskillEnvFacade(env_id=env_id, seed=seed, scene=scene)


def record(args) -> int:
    facade = make_facade(args.env_id, args.seed0, parse_scene(args.scene))
    backend = FacadeBackend(facade, args.env_id)
    task = facade.get_task_language()
    tracks = Path(args.tracks)
    tracks.mkdir(parents=True, exist_ok=True)
    kept = tried = 0
    try:
        while kept < args.episodes and tried < args.episodes * 3:
            seed = args.seed0 + tried
            tried += 1
            layout = backend.reset(seed)
            rec = DemoRecorder(backend)
            rec.capture()
            hover = 0.05 + float(np.random.default_rng(seed).uniform(0, 0.02))
            if not scripted_demo(rec, backend, hover):
                print(
                    f"[record:{args.env_id}] seed={seed} demo failed; skip", flush=True
                )
                continue
            name = f"track_ep{kept:03d}.json"
            track = {
                "seed": seed,
                "sim": "maniskill",
                "env_id": args.env_id,
                "task": task,
                "scene": facade.get_env_meta()["scene"],
                "layout": layout,
                **rec.track(),
            }
            (tracks / name).write_text(json.dumps(track))
            print(
                f"[record:{args.env_id}] seed={seed} {len(rec.tcp)} ctrl steps -> {name}",
                flush=True,
            )
            kept += 1
    finally:
        facade.close()
    print(f"[record:{args.env_id}] kept {kept}/{tried} tracks -> {tracks}")
    return 0 if kept else 2


def follow(args) -> int:
    files = sorted(Path(args.tracks).glob("track_ep*.json"))
    if not files:
        raise SystemExit(f"no tracks under {args.tracks}")
    first = json.loads(files[0].read_text())
    env_id = first["env_id"]
    scene = parse_scene(args.scene) if args.scene else first.get("scene")
    facade = make_facade(env_id, int(first["seed"]), scene)
    backend = FacadeBackend(facade, env_id)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    kept = 0
    try:
        for tf in files:
            track = json.loads(tf.read_text())
            backend.reset(int(track["seed"]))
            drift = layout_drift(backend, track.get("layout") or {})
            if drift > LAYOUT_TOL_M:
                print(
                    f"[follow] {tf.name} SKIPPED: layout drift {drift * 100:.1f} cm",
                    flush=True,
                )
                continue
            out_dir = out / f"rollout_{kept:03d}"
            result = follow_track(backend, track, out_dir, args.step_m)
            writer = result.pop("writer")
            writer.close(
                {
                    "source": f"follow_{env_id}",
                    "method": "closed_loop_follower",
                    "sim": "maniskill",
                    "env_id": env_id,
                    "seed": track["seed"],
                    "track": tf.name,
                    "scene": facade.get_env_meta()["scene"],
                    "step_m": args.step_m,
                    "task": track["task"],
                    "task_text": track["task"],
                    **result,
                }
            )
            print(
                f"[follow:{env_id}] {tf.name} steps={writer.step} "
                f"success={result['success']} {result['reason']}",
                flush=True,
            )
            if result["success"] or args.keep_failures:
                kept += 1
            else:
                shutil.rmtree(out_dir, ignore_errors=True)
    finally:
        facade.close()
    print(f"[follow:{env_id}] kept {kept}/{len(files)} -> {out}")
    return 0 if kept else 2


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("record", help="continuous demos -> TCP tracks")
    r.add_argument("--env-id", choices=sorted(DROP_Z), default="BlockPAP-v1")
    r.add_argument("--episodes", type=int, default=8)
    r.add_argument("--seed0", type=int, default=20000)
    r.add_argument("--tracks", required=True)
    r.add_argument("--scene", default="", help="rig options key=value,... (env_server)")
    f = sub.add_parser("follow", help="TCP tracks -> rollouts in action units")
    f.add_argument("--tracks", required=True)
    f.add_argument("--out", required=True)
    f.add_argument("--step-m", type=float, default=0.02)
    f.add_argument(
        "--scene", default="", help="default: the scene the tracks were recorded in"
    )
    f.add_argument("--keep-failures", action="store_true")
    args = ap.parse_args(argv)
    return record(args) if args.cmd == "record" else follow(args)


if __name__ == "__main__":
    raise SystemExit(main())
