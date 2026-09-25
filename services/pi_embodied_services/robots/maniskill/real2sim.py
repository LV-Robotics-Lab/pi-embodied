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
# Modified by pi-embodied: the Scheme A generator of github.com/showlab/Show-Harness @137d571
# (scripts/trajectory/real2sim/maniskill/oracle.py) with the parts of atomic_tokenizer.py it
# uses (AtomicExec, TokenEpisode.emit/align_xy/go_z, manhattan_tokens, RolloutWriter), driving
# the env server's facade in-process instead of their backend layer.

"""Real2sim training data: a privileged oracle that plays each episode in action units.

    python -m pi_embodied_services.robots.maniskill.real2sim --env-id BlockPAP-v1 \\
        --episodes 20 --seed0 30000 --out /root/autodl-tmp/data/real2sim/blockpap_oracle

Every episode resets like the deployment env server (``scenes.reset``: the training
episodes' reset, ``layout: wide``), then a state machine reads the object poses and walks
the task on the 2 cm lattice: Manhattan runs over the object, down, GRASP (bounded empty-
grasp retries), up, over the target, down, RELEASE, two MV_UP. Each unit is recorded
BEFORE it runs -- the frame the policy would see, exactly as the env server renders it for
deployment (both views letterboxed to 256, the wrist turned and cropped) -- then executed
closed-loop to exactly ``step_m`` (Show-Harness's AtomicExec). Episodes the env does not
score as successful are dropped unless ``--keep-failures``.

Output, one directory per kept episode, in the Show-Harness teleop layout that
``train/data_preparation/rollouts_to_alpaca.py`` reads (the layout of Show-Harness-Data and
of ``finetuned/prepare.ts``'s GUMI conversion): ``<out>/rollout_NNN/{agentview,wrist}/NNNN.png``,
``actions.jsonl`` ({step, token, kind, gripper_closed, ee_pose, gripper_width, agentview,
wrist, time}) and ``metadata.json`` (task_text, env, seed, layout, success, token counts).
"""

from __future__ import annotations

import argparse
import json
import shutil
import time
from pathlib import Path
from typing import Any, Optional

import numpy as np

from pi_embodied_services.robots.maniskill.scenes import SCENES

MOVE_DIRS = {
    "MV_FWD": (0, 1),
    "MV_BACK": (0, -1),
    "MV_LEFT": (1, -1),
    "MV_RIGHT": (1, 1),
    "MV_UP": (2, 1),
    "MV_DOWN": (2, -1),
}
_AXIS_TOKENS = {v: k for k, v in MOVE_DIRS.items()}
GRASP, RELEASE = "GRASP", "RELEASE"
#: Fingers stopped on the block instead of closing on air (both rigs hold 4 cm cubes).
GRASPED_WIDTH_M = 0.012


def _const(env, name: str) -> Any:
    return getattr(env, name)


#: oracle.SPECS: per-task geometry that reproduces their ms_0717 oracle datasets.
SPECS: dict[str, dict[str, Any]] = {
    "BlockPAP-v1": {
        "max_tokens": 140,
        "grasp_z_jitter": 0.003,
        "carry_offset": False,
        "carry_z": lambda env, obj, tgt: (
            max(
                obj[2] + 0.03,
                tgt[2]
                + float(_const(env, "COASTER_THICKNESS"))
                + float(_const(env, "BLOCK_HALF_SIZE")[2])
                + 0.03,
            )
            + 0.06
        ),
        # tasks.TASKS["blockpap"]["drop_z"]: block seated on the coaster, +4 mm.
        "place_tcp_z": lambda env, tgt, tcp, obj: (
            tgt[2]
            + float(_const(env, "COASTER_THICKNESS"))
            + float(_const(env, "BLOCK_HALF_SIZE")[2])
            + 0.004
        ),
        "final_hold": 10,
    },
    "BlockStack-v1": {
        "max_tokens": 160,
        "grasp_z_jitter": 0.0,
        "carry_offset": True,
        "carry_z": lambda env, obj, tgt: (
            tgt[2] + 4 * float(_const(env, "BLOCK_HALF_SIZE")[2]) + 0.04
        ),
        "place_tcp_z": lambda env, tgt, tcp, obj: (
            tgt[2] + 2 * float(_const(env, "BLOCK_HALF_SIZE")[2]) + (tcp[2] - obj[2])
        ),
        "final_hold": 12,
    },
}


def manhattan_tokens(
    delta, step_m: float, min_residual_m: float, axis_order: Optional[list[int]] = None
) -> list[str]:
    """Whole axes in ``axis_order`` (default largest first), round(|d|/step) repeats each;
    a residual >= min_residual_m gets one token."""
    delta = np.asarray(delta, dtype=np.float64)
    order = axis_order if axis_order is not None else list(np.argsort(-np.abs(delta)))
    out: list[str] = []
    for axis in order:
        d = float(delta[axis])
        n = int(round(abs(d) / step_m))
        if n == 0 and abs(d) >= min_residual_m:
            n = 1
        out.extend([_AXIS_TOKENS[(int(axis), 1 if d >= 0 else -1)]] * n)
    return out


class Writer:
    """atomic_tokenizer.RolloutWriter: frames (already in their final form) + actions.jsonl."""

    def __init__(self, rollout_dir: Path) -> None:
        self.dir = Path(rollout_dir)
        for v in ("agentview", "wrist"):
            (self.dir / v).mkdir(parents=True, exist_ok=True)
        self._jsonl = (self.dir / "actions.jsonl").open("w", encoding="utf-8")
        self.step = 0
        self.tokens: list[str] = []

    def add(self, token, kind, obs, gripper_closed: bool) -> None:
        from PIL import Image

        name = f"{self.step:04d}.png"
        for v in ("agentview", "wrist"):
            Image.fromarray(obs[v]).save(self.dir / v / name)
        pose = [*obs["tcp_pos"].tolist(), *obs["tcp_quat_wxyz"].tolist()]
        rec = {
            "step": self.step,
            "token": token,
            "kind": kind,
            "gripper_closed": bool(gripper_closed),
            "ee_pose": [round(float(v), 5) for v in pose],
            "gripper_width": round(float(obs["gripper_width"]), 5),
            "agentview": f"agentview/{name}",
            "wrist": f"wrist/{name}",
            "time": round(time.time(), 3),
        }
        self._jsonl.write(json.dumps(rec) + "\n")
        self._jsonl.flush()
        self.tokens.append(token)
        self.step += 1

    def close(self, metadata: dict) -> None:
        self._jsonl.close()
        counts: dict[str, int] = {}
        for t in self.tokens:
            counts[t] = counts.get(t, 0) + 1
        metadata = {**metadata, "num_steps": self.step, "token_counts": counts}
        (self.dir / "metadata.json").write_text(
            json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8"
        )


class OracleEpisode:
    """AtomicExec + TokenEpisode + OracleEpisode over the env server facade."""

    def __init__(self, facade, writer: Writer, rng, env_id: str, step_m: float):
        self.f = facade
        self.env = facade._env.unwrapped
        self.w = writer
        self.rng = rng
        self.rig = SCENES[env_id]
        self.spec = SPECS[env_id]
        self.step_m = float(step_m)
        self.grip_cmd = 1.0  # +1 open / -1 close
        self.obs = facade._pack(facade._obs)

    # -- AtomicExec: tol 1 mm, <= 2 cm per control step, 16 steps; gripper 10 steps ---
    def _tcp(self) -> np.ndarray:
        return self.f._state()["tcp_pos"].astype(np.float64)

    def _ctrl(self, delta_m) -> None:
        bound = 0.1
        a = np.clip(np.asarray(delta_m) / bound, -0.02 / bound, 0.02 / bound)
        obs, *_ = self.f._step(np.array([a[0], a[1], a[2], self.grip_cmd]))
        self.f._obs = obs

    def _move(self, token: str) -> None:
        axis, sign = MOVE_DIRS[token]
        target = self._tcp()
        target[axis] += sign * self.step_m
        for _ in range(16):
            err = target - self._tcp()
            if np.linalg.norm(err) < 0.001:
                break
            self._ctrl(err)

    def _quiesce(self) -> None:
        prev = self._tcp()
        for _ in range(40):
            self._ctrl(np.zeros(3))
            cur = self._tcp()
            if float(np.linalg.norm(cur - prev)) < 0.0003:
                return
            prev = cur

    def _gripper(self, cmd: float) -> None:
        self._quiesce()  # never act on the gripper while the hand still travels
        self.grip_cmd = cmd
        for _ in range(10):
            self._ctrl(np.zeros(3))

    def hold(self, n: int) -> None:
        for _ in range(n):
            self._ctrl(np.zeros(3))

    # -- TokenEpisode ---------------------------------------------------------
    def emit(self, token: str) -> None:
        """Record (frame BEFORE the unit, unit), then execute it."""
        if self.w.step >= self.spec["max_tokens"]:
            raise RuntimeError("token budget exceeded")
        kind = "move" if token in MOVE_DIRS else token.lower()
        self.w.add(token, kind, self.f._pack(self.f._obs), self.grip_cmd < 0)
        if token in MOVE_DIRS:
            self._move(token)
        else:
            self._gripper(-1.0 if token == GRASP else 1.0)

    def _tol(self, tol: float) -> float:
        return max(float(tol), self.step_m * 0.55)

    def align_xy(self, target, tol: float = 0.011, attempts: int = 4) -> None:
        tol = self._tol(tol)
        target_xy = np.asarray(target, dtype=np.float64).reshape(-1)[:2]
        for _ in range(attempts):
            err = target_xy - self._tcp()[:2]
            if np.all(np.abs(err) < tol):
                return
            order = [0, 1] if self.rng.random() < 0.5 else [1, 0]
            toks = manhattan_tokens(
                [err[0], err[1], 0.0], self.step_m, tol, order + [2]
            )
            if not toks:
                return
            for t in toks:
                self.emit(t)

    def go_z(self, target_z: float, tol: float = 0.011, attempts: int = 3) -> None:
        tol = self._tol(tol)
        for _ in range(attempts):
            dz = float(target_z) - self._tcp()[2]
            if abs(dz) < tol:
                return
            toks = manhattan_tokens([0.0, 0.0, dz], self.step_m, tol)
            if not toks:
                return
            for t in toks:
                self.emit(t)

    # -- the oracle -------------------------------------------------------------
    def _pos(self, name: str) -> np.ndarray:
        return getattr(self.env, name).pose.p.reshape(-1, 3)[0].cpu().numpy()

    def _width(self) -> float:
        return float(self.f._state()["gripper_width"])

    def success(self) -> bool:
        return bool(self.env.evaluate()["success"].reshape(-1)[0])

    def run(self) -> dict:
        carried, target = self.rig.carried, self.rig.target
        obj0, tgt0 = self._pos(carried), self._pos(target)
        jit = self.spec["grasp_z_jitter"]
        grasp_z = obj0[2] + (float(self.rng.uniform(-jit, jit)) if jit else 0.0)
        self.align_xy(obj0)
        self.go_z(grasp_z)
        self.align_xy(self._pos(carried))
        for _ in range(3):  # GRASP with bounded empty-grasp retries
            self.emit(GRASP)
            if self._width() > GRASPED_WIDTH_M:
                break
            self.emit(RELEASE)
            self.emit("MV_DOWN")
        else:
            return {"success": False, "reason": "grasp_failed"}
        self.go_z(self.spec["carry_z"](self.env, obj0, tgt0))
        if self._width() < GRASPED_WIDTH_M:
            return {"success": False, "reason": "dropped_on_lift"}
        target_xy = tgt0[:2]
        if self.spec["carry_offset"]:
            target_xy = target_xy - (self._pos(carried)[:2] - self._tcp()[:2])
        self.align_xy(target_xy)
        self.go_z(
            self.spec["place_tcp_z"](self.env, tgt0, self._tcp(), self._pos(carried))
        )
        self.emit(RELEASE)
        self.emit("MV_UP")
        self.emit("MV_UP")
        self.hold(self.spec["final_hold"])
        return {"success": self.success()}


def main() -> int:
    from pi_embodied_services.robots.maniskill.env_server import ManiskillEnvFacade

    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--env-id", choices=sorted(SPECS), default="BlockPAP-v1")
    ap.add_argument("--episodes", type=int, default=10)
    ap.add_argument("--seed0", type=int, default=30000)
    ap.add_argument("--out", required=True)
    ap.add_argument("--step-m", type=float, default=0.02)
    ap.add_argument(
        "--scene", default="", help="rig options key=value,... (env_server)"
    )
    ap.add_argument("--keep-failures", action="store_true")
    args = ap.parse_args()

    scene = dict(kv.split("=", 1) for kv in args.scene.split(",") if kv) or None
    facade = ManiskillEnvFacade(env_id=args.env_id, seed=args.seed0, scene=scene)
    task = facade.get_task_language()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    kept = tried = 0
    try:
        while kept < args.episodes and tried < args.episodes * 4:
            seed = args.seed0 + tried
            tried += 1
            facade.reset(seed=seed)
            rollout = out / f"rollout_{kept:03d}"
            writer = Writer(rollout)
            ep = OracleEpisode(
                facade, writer, np.random.default_rng(seed), args.env_id, args.step_m
            )
            try:
                result = ep.run()
            except RuntimeError as exc:  # token budget exceeded
                result = {"success": False, "reason": str(exc)}
            ok = bool(result.get("success"))
            writer.close(
                {
                    "source": f"oracle_{args.env_id}",
                    "method": "privileged_oracle",
                    "env_id": args.env_id,
                    "seed": seed,
                    "scene": facade.get_env_meta()["scene"],
                    "layout_sampled": facade.get_env_meta().get("layout"),
                    "step_m": args.step_m,
                    "task": task,
                    "task_text": task,
                    "success": ok,
                    "reason": result.get("reason", ""),
                }
            )
            print(
                f"[real2sim:{args.env_id}] seed={seed} steps={writer.step} "
                f"success={ok} {result.get('reason', '')}",
                flush=True,
            )
            if ok or args.keep_failures:
                kept += 1
            else:
                shutil.rmtree(rollout, ignore_errors=True)
    finally:
        facade.close()
    print(f"[real2sim:{args.env_id}] kept {kept}/{tried} episodes -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
