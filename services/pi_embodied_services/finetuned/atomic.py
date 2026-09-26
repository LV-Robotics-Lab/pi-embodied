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
# Modified by pi-embodied: scripts/trajectory/real2sim/atomic_tokenizer.py of
# github.com/showlab/Show-Harness @137d571 (AtomicExec, DemoRecorder, gripper_events,
# RolloutWriter, TokenEpisode, manhattan_tokens, rdp), with the backend a duck-typed object
# (the generators wrap an env server facade in-process) and frames written as the backend
# hands them over (the camera transform is the backend's job).

"""The sim-agnostic core of the real2sim generators: action units executed, not labelled.

One ``MV_*`` token is one ``step_m`` (2 cm) end-effector displacement, executed closed-loop
(:class:`AtomicExec`) and recorded frame-BEFORE-token (:class:`TokenEpisode.emit`), so every
training frame sits on the lattice the deployed policy walks. Contract (Show-Harness's
real-Franka convention): ``MV_FWD``=+X, ``MV_BACK``=-X, ``MV_LEFT``=-Y, ``MV_RIGHT``=+Y,
``MV_UP``=+Z, ``MV_DOWN``=-Z in the robot base frame; ``gripper_closed`` in a record is the
state before the token.

A backend is any object with ``tcp_pos() -> (3,)``, ``tcp_pose7() -> [x,y,z,qw,qx,qy,qz]``,
``gripper_width() -> m``, ``apply_delta(delta_m, grip_cmd, max_cmd_m)`` (ONE control step,
``grip_cmd`` +1 open / -1 close), ``grab_frames() -> (agentview, wrist)`` uint8 HWC in their
final form, ``success() -> bool`` and optionally ``frozen() -> bool`` (the sim stopped
responding: RoboLab freezes a terminated env).

Output (:class:`RolloutWriter`): ``rollout_NNN/{agentview,wrist}/NNNN.png``, ``actions.jsonl``
and ``metadata.json`` -- the Show-Harness teleop layout that the vendored
``showharness/train/data_preparation/rollouts_to_alpaca.py`` reads and ``train.sh
--from-rollouts`` takes.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Optional, Sequence

import numpy as np

MOVE_DIRS: dict[str, np.ndarray] = {
    "MV_FWD": np.array([1.0, 0.0, 0.0]),
    "MV_BACK": np.array([-1.0, 0.0, 0.0]),
    "MV_LEFT": np.array([0.0, -1.0, 0.0]),
    "MV_RIGHT": np.array([0.0, 1.0, 0.0]),
    "MV_UP": np.array([0.0, 0.0, 1.0]),
    "MV_DOWN": np.array([0.0, 0.0, -1.0]),
}
GRASP, RELEASE = "GRASP", "RELEASE"
OPPOSITE = {
    "MV_FWD": "MV_BACK",
    "MV_BACK": "MV_FWD",
    "MV_LEFT": "MV_RIGHT",
    "MV_RIGHT": "MV_LEFT",
    "MV_UP": "MV_DOWN",
    "MV_DOWN": "MV_UP",
}
_AXIS_TOKENS = {
    (0, 1): "MV_FWD",
    (0, -1): "MV_BACK",
    (1, -1): "MV_LEFT",
    (1, 1): "MV_RIGHT",
    (2, 1): "MV_UP",
    (2, -1): "MV_DOWN",
}
#: token -> (axis, sign)
TOKEN_AXIS = {tok: (ax, sg) for (ax, sg), tok in _AXIS_TOKENS.items()}
#: Continuous-demo servo cap per control step: under AtomicExec's, so the path is smooth.
SERVO_MAX_CMD_M = 0.015


def token_for_axis(axis: int, sign: float) -> str:
    return _AXIS_TOKENS[(int(axis), 1 if sign >= 0 else -1)]


def token_kind(token: str) -> str:
    if token in MOVE_DIRS:
        return "move"
    if token in (GRASP, RELEASE):
        return token.lower()
    raise ValueError(f"unknown token {token!r}")


class EpisodeComplete(RuntimeError):
    """The sim stopped responding: the episode is over (score it), not failed."""


class TokenBudgetExceeded(RuntimeError):
    """An episode ran past ``max_tokens``."""


class AtomicExec:
    """Execute tokens as ~``step_m`` displacements, closed-loop.

    ``move`` commands the remaining error each control step (capped at ``max_cmd_m``) until
    within ``tol_m`` or ``max_ctrl_steps`` run out. Gripper tokens first bring the arm to rest
    (:meth:`quiesce`: a release hands the object the hand's leftover velocity, a grasp drags
    it), then hold the command ``gripper_steps``.
    """

    def __init__(
        self,
        backend: Any,
        step_m: float = 0.02,
        tol_m: float = 0.001,
        max_cmd_m: float = 0.02,
        max_ctrl_steps: int = 16,
        gripper_steps: int = 10,
        quiesce_tol_m: float = 0.0003,
        quiesce_max_steps: int = 40,
    ) -> None:
        self.backend = backend
        self.step_m = float(step_m)
        self.tol_m = float(tol_m)
        self.max_cmd_m = float(max_cmd_m)
        self.max_ctrl_steps = int(max_ctrl_steps)
        self.gripper_steps = int(gripper_steps)
        self.quiesce_tol_m = float(quiesce_tol_m)
        self.quiesce_max_steps = int(quiesce_max_steps)
        self.grip_cmd = 1.0  # +1 open / -1 close

    @property
    def gripper_closed(self) -> bool:
        return self.grip_cmd < 0

    def _step(self, delta_m) -> None:
        self.backend.apply_delta(
            np.asarray(delta_m, dtype=np.float64), self.grip_cmd, self.max_cmd_m
        )

    def move(self, token: str) -> float:
        start = self.backend.tcp_pos()
        target = start + MOVE_DIRS[token] * self.step_m
        for _ in range(self.max_ctrl_steps):
            err = target - self.backend.tcp_pos()
            if np.linalg.norm(err) < self.tol_m:
                break
            self._step(err)
        return float(np.linalg.norm(self.backend.tcp_pos() - start))

    def quiesce(self) -> int:
        prev = self.backend.tcp_pos()
        for i in range(self.quiesce_max_steps):
            self._step(np.zeros(3))
            cur = self.backend.tcp_pos()
            if float(np.linalg.norm(cur - prev)) < self.quiesce_tol_m:
                return i + 1
            prev = cur
        return self.quiesce_max_steps

    def _gripper(self, cmd: float) -> float:
        self.quiesce()
        self.grip_cmd = cmd
        for _ in range(self.gripper_steps):
            self._step(np.zeros(3))
        return self.backend.gripper_width()

    def grasp(self) -> float:
        return self._gripper(-1.0)

    def release(self) -> float:
        return self._gripper(1.0)

    def hold(self, n: int = 1) -> None:
        for _ in range(n):
            self._step(np.zeros(3))


class DemoRecorder:
    """Scheme D input: a continuous servo, every control step's TCP captured.

    Nothing recorded here becomes a training frame; the follower keeps only the path's
    shape (RDP corners) and where the gripper events happened.
    """

    def __init__(self, backend: Any, max_cmd_m: float = SERVO_MAX_CMD_M) -> None:
        self.backend = backend
        self.max_cmd_m = float(max_cmd_m)
        self.grip_cmd = 1.0
        self.tcp: list[np.ndarray] = []
        self.grip_cmds: list[float] = []

    def _tcp(self) -> np.ndarray:
        return self.backend.tcp_pos()

    def capture(self) -> None:
        self.tcp.append(self._tcp())
        self.grip_cmds.append(self.grip_cmd)

    def step(self, delta_m) -> None:
        self.backend.apply_delta(
            np.asarray(delta_m, dtype=np.float64), self.grip_cmd, self.max_cmd_m
        )
        self.capture()

    def servo_to(self, target, tol: float = 0.004, budget: int = 120) -> None:
        for _ in range(budget):
            err = np.asarray(target, dtype=np.float64) - self._tcp()
            if np.linalg.norm(err) < tol:
                return
            self.step(err)

    def set_gripper(self, close: bool, steps: int = 10) -> None:
        self.grip_cmd = -1.0 if close else 1.0
        for _ in range(steps):
            self.step(np.zeros(3))

    def hold_until_success(self, steps: int) -> bool:
        for _ in range(steps):
            self.step(np.zeros(3))
            if self.backend.success():
                return True
        return self.backend.success()

    def track(self) -> dict:
        """The recorded path as track-file fields."""
        return {
            "tcp": [[round(float(v), 5) for v in q] for q in self.tcp],
            "grip_cmds": list(self.grip_cmds),
            "events": gripper_events(self.grip_cmds),
        }


def gripper_events(grip_cmds: Sequence[float]) -> list[list]:
    """``[index, GRASP|RELEASE]`` for every gripper transition of a command track."""
    events, prev = [], 1.0
    for i, g in enumerate(grip_cmds):
        if g < 0 <= prev:
            events.append([i, GRASP])
        elif g >= 0 > prev:
            events.append([i, RELEASE])
        prev = g
    return events


class RolloutWriter:
    """Frames + ``actions.jsonl`` + ``metadata.json`` of one rollout (teleop layout)."""

    def __init__(self, rollout_dir: Path) -> None:
        self.dir = Path(rollout_dir)
        for view in ("agentview", "wrist"):
            (self.dir / view).mkdir(parents=True, exist_ok=True)
        self._jsonl = (self.dir / "actions.jsonl").open("w", encoding="utf-8")
        self.step = 0
        self.tokens: list[str] = []

    def add_step(self, token, kind, agentview, wrist, gripper_closed, ee_pose, width):
        from PIL import Image

        name = f"{self.step:04d}.png"
        Image.fromarray(np.asarray(agentview)).save(self.dir / "agentview" / name)
        Image.fromarray(np.asarray(wrist)).save(self.dir / "wrist" / name)
        rec = {
            "step": self.step,
            "token": token,
            "kind": kind,
            "gripper_closed": bool(gripper_closed),
            "ee_pose": [round(float(v), 5) for v in ee_pose],
            "gripper_width": round(float(width), 5),
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


class TokenEpisode:
    """Record-then-execute loop plus the planners every generator shares.

    Tolerances are clamped to >= 0.55 ``step_m``: tighter than half a step makes 2 cm
    ping-pong around the target geometrically inevitable.
    """

    def __init__(
        self,
        backend: Any,
        writer: RolloutWriter,
        executor: AtomicExec,
        max_tokens: int = 160,
        rng: Optional[np.random.Generator] = None,
    ) -> None:
        self.backend = backend
        self.writer = writer
        self.exec = executor
        self.max_tokens = int(max_tokens)
        self.rng = rng if rng is not None else np.random.default_rng(0)

    def emit(self, token: str, kind: Optional[str] = None) -> None:
        """Record (frame BEFORE the token, token), then execute it."""
        if self.writer.step >= self.max_tokens:
            raise TokenBudgetExceeded("token budget exceeded")
        frozen = getattr(self.backend, "frozen", None)
        if callable(frozen) and frozen():
            # the frame would repeat the last one under a move that cannot happen
            raise EpisodeComplete("environment terminated; no further tokens are real")
        kind = kind or token_kind(token)
        agentview, wrist = self.backend.grab_frames()
        self.writer.add_step(
            token,
            kind,
            agentview,
            wrist,
            self.exec.gripper_closed,
            self.backend.tcp_pose7(),
            self.backend.gripper_width(),
        )
        if kind == "move":
            self.exec.move(token)
        elif token == GRASP:
            self.exec.grasp()
        elif token == RELEASE:
            self.exec.release()

    def _tol(self, tol: float) -> float:
        return max(float(tol), self.exec.step_m * 0.55)

    def align_xy(self, target, tol: float = 0.011, attempts: int = 4) -> None:
        """Manhattan XY runs, axis order coin-flipped per attempt, re-planned on drift."""
        tol = self._tol(tol)
        target_xy = np.asarray(target, dtype=np.float64).reshape(-1)[:2]
        for _ in range(attempts):
            err = target_xy - self.backend.tcp_pos()[:2]
            if np.all(np.abs(err) < tol):
                return
            order = [0, 1] if self.rng.random() < 0.5 else [1, 0]
            toks = manhattan_tokens(
                [err[0], err[1], 0.0], self.exec.step_m, tol, order + [2]
            )
            if not toks:
                return
            for t in toks:
                self.emit(t, "move")

    def go_z(self, target_z: float, tol: float = 0.011, attempts: int = 3) -> None:
        tol = self._tol(tol)
        for _ in range(attempts):
            dz = float(target_z) - self.backend.tcp_pos()[2]
            if abs(dz) < tol:
                return
            toks = manhattan_tokens([0.0, 0.0, dz], self.exec.step_m, tol)
            if not toks:
                return
            for t in toks:
                self.emit(t, "move")

    def chase(
        self,
        waypoint,
        tol: float,
        budget: int = 60,
        stall_frac: float = 0.3,
        stall_limit: int = 3,
    ) -> None:
        """Dominant-axis pursuit: commit to an axis until done; an immediate opposite token
        means the lattice cannot get closer (stop); ``stall_limit`` tokens in a row moving
        under ``stall_frac`` of a step mean the arm is blocked (stop)."""
        tol = self._tol(tol)
        prev: Optional[str] = None
        lock: Optional[int] = None
        stalled = 0
        min_progress = stall_frac * self.exec.step_m
        for _ in range(budget):
            err = np.asarray(waypoint, dtype=np.float64) - self.backend.tcp_pos()
            if lock is not None and abs(err[lock]) >= tol:
                axis = lock
            else:
                axis = int(np.argmax(np.abs(err)))
                if abs(err[axis]) < tol:
                    return
                lock = axis
            token = token_for_axis(axis, err[axis])
            if prev is not None and OPPOSITE.get(token) == prev:
                return
            before = self.backend.tcp_pos()
            self.emit(token, "move")
            if float(np.linalg.norm(self.backend.tcp_pos() - before)) < min_progress:
                stalled += 1
                if stalled >= stall_limit:
                    return
            else:
                stalled = 0
            prev = token

    def settle(self, n: int) -> bool:
        return self.settle_steps(n) is not None

    def settle_steps(self, n: int) -> Optional[int]:
        """Hold still up to ``n`` control steps; how many it took success to fire, or None."""
        for i in range(int(n)):
            self.exec.hold(1)
            if self.backend.success():
                return i + 1
        return None


def manhattan_tokens(
    delta, step_m: float = 0.02, min_residual_m: float = 0.01, axis_order=None
) -> list[str]:
    """Whole axes in ``axis_order`` (default largest first), round(|d|/step) repeats each;
    a residual >= ``min_residual_m`` gets one token."""
    delta = np.asarray(delta, dtype=np.float64)
    order = axis_order if axis_order is not None else list(np.argsort(-np.abs(delta)))
    out: list[str] = []
    for axis in order:
        d = float(delta[axis])
        n = int(round(abs(d) / step_m))
        if n == 0 and abs(d) >= min_residual_m:
            n = 1
        out.extend([token_for_axis(int(axis), d)] * n)
    return out


def rdp(points, eps: float = 0.008) -> list[int]:
    """Ramer-Douglas-Peucker on a 3D polyline: the kept indices."""
    pts = np.asarray(points, dtype=np.float64)
    if len(pts) < 3:
        return list(range(len(pts)))
    keep = np.zeros(len(pts), dtype=bool)
    keep[0] = keep[-1] = True
    stack = [(0, len(pts) - 1)]
    while stack:
        a, b = stack.pop()
        if b <= a + 1:
            continue
        seg = pts[b] - pts[a]
        seg_len = np.linalg.norm(seg)
        if seg_len < 1e-9:
            d = np.linalg.norm(pts[a + 1 : b] - pts[a], axis=1)
        else:
            t = np.clip(((pts[a + 1 : b] - pts[a]) @ seg) / seg_len**2, 0, 1)
            d = np.linalg.norm(pts[a + 1 : b] - (pts[a] + t[:, None] * seg), axis=1)
        imax = int(np.argmax(d))
        if d[imax] > eps:
            m = a + 1 + imax
            keep[m] = True
            stack += [(a, m), (m, b)]
    return [int(i) for i in np.where(keep)[0]]


def stalled_token_count(rollout_dir: Path, min_travel_m: float = 0.006) -> int:
    """Recorded move tokens whose frame-to-frame travel is under ``min_travel_m``."""
    path = Path(rollout_dir) / "actions.jsonl"
    if not path.exists():
        return 0
    recs = [json.loads(line) for line in path.open() if line.strip()]
    return sum(
        1
        for cur, nxt in zip(recs, recs[1:])
        if cur.get("kind") == "move"
        and float(np.linalg.norm(np.subtract(cur["ee_pose"][:3], nxt["ee_pose"][:3])))
        < min_travel_m
    )
