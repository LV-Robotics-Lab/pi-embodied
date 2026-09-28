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

"""pi's limits and code mode (``code.run``, code_exec.py ``CodeRunMixin``) on a real-robot env server.

What every real-robot server needs:

- pi's per-call limits, enforced by the server's motion methods for every caller (pi's tools, a
  program, a manual call). pi passes its flags at spawn (:func:`add_limit_arguments`:
  ``--max-move``, ``--max-rotate`` / ``--max-yaw``, ``--workspace-xy``, ``--z-floor``); the server
  applies them on top of its own config limits and reports what it enforces as
  ``motion_limits`` in ``env.get_env_meta`` (pi refuses an attached server whose limits are looser
  than its flags). A robot's motion methods call :meth:`RealCodeMode._limit`.
- ``--code`` (:func:`add_code_argument`): only then does the server serve ``code.run`` and
  require its RPC token (pi reads it from the listening line, or takes ``URL#token=HEX``); without
  it ``code.api`` still answers (pi records the API an episode ran with) but no program runs.
- The primitive manifest (packages/embodied/src/primitives/manifests/<robot>.json): ``code.api``,
  the programs' whitelist and the startup self-check come from it
  (:meth:`RealCodeMode._install_real`).
- The run's report: its motion count, the last wrapped ``states`` a motion returned (pi's state
  cache) and a bounded video (one frame of the camera pi's video follows after each motion;
  :data:`CODE_MAX_FRAMES` full frames, every other one dropped when full). pi records the run as
  its next state step.

The robot supplies ``move_m`` (its motions' translation, for the run's budget), ``check``
(program-only refusals), its limit defaults (``_LIMIT_DEFAULTS``) and
:meth:`RealCodeMode._video_frame`.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable
from typing import Any

import numpy as np

from pi_embodied_services.utils.code_exec import CodeRunMixin

#: Full camera frames one run hands back (halved, every other one kept, when full).
CODE_MAX_FRAMES = 32

#: pi's limits: name -> (flag, help). ``max_*`` are per call and positive; ``z_floor_m`` is the
#: lowest TCP z (m); ``workspace_xy`` the TCP box ``[xmin, xmax, ymin, ymax]`` (m).
LIMIT_FLAGS: dict[str, tuple[str, str]] = {
    "max_move_m": ("--max-move", "largest translation per motion call, m"),
    "max_rotate_rad": ("--max-rotate", "largest rotation per motion call, rad"),
    "max_yaw_rad": ("--max-yaw", "largest yaw per motion call, rad"),
    "z_floor_m": ("--z-floor", "lowest TCP z, m (a move below it is refused)"),
    "workspace_xy": (
        "--workspace-xy",
        "TCP x/y box, m: xmin,xmax,ymin,ymax (a move out of it is refused)",
    ),
}


def add_code_argument(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--code",
        action="store_true",
        help="Serve code.run (pi's run_code with --code-real) and require the RPC token",
    )


def _box(text: str) -> list[float] | None:
    text = text.strip()
    if not text:
        return None
    return [float(v) for v in text.split(",")]


def add_limit_arguments(
    parser: argparse.ArgumentParser, defaults: dict[str, Any]
) -> None:
    """``--max-move`` etc. for the limits named in ``defaults`` (name -> default; None: off).
    pi passes its flags here at spawn (packages/embodied/src/primitives/motion.ts limitArgs)."""
    for name, default in defaults.items():
        flag, text = LIMIT_FLAGS[name]
        parser.add_argument(
            flag,
            dest=f"limit_{name}",
            type=_box if name == "workspace_xy" else float,
            default=default,
            help=f"{text} (default {default})",
        )


def limits_from_args(args: argparse.Namespace, names: Any) -> dict[str, Any]:
    """The limits :func:`add_limit_arguments` parsed, by name."""
    return {name: getattr(args, f"limit_{name}") for name in names}


def check_limits(limits: dict[str, Any]) -> dict[str, Any]:
    """Validated limits: finite numbers, ``max_*`` positive, the box ordered; None is off."""
    out: dict[str, Any] = {}
    for k, v in limits.items():
        if k not in LIMIT_FLAGS:
            raise ValueError(f"unknown limit {k!r} (have {sorted(LIMIT_FLAGS)})")
        if v is None:
            out[k] = None
            continue
        a = np.asarray(v, dtype=np.float64)
        if not np.all(np.isfinite(a)) or (k.startswith("max_") and not np.all(a > 0)):
            raise ValueError(f"{LIMIT_FLAGS[k][0]} must be finite (positive for max_*)")
        if k == "workspace_xy" and not (
            a.shape == (4,) and a[0] < a[1] and a[2] < a[3]
        ):
            raise ValueError("--workspace-xy must be xmin,xmax,ymin,ymax")
        out[k] = a.tolist() if a.ndim else float(a)
    return out


def vec3(value: Any, name: str) -> np.ndarray:
    """Three finite numbers, else ValueError naming ``name``."""
    a = np.asarray(value, dtype=np.float64).reshape(-1)
    if a.shape != (3,) or not np.all(np.isfinite(a)):
        raise ValueError(f"{name} must be 3 finite numbers")
    return a


def outside(p: np.ndarray, floor: float | None, box: Any) -> float:
    """How far ``p`` (xyz) lies below ``floor`` / outside the x/y ``box`` (0 inside), m."""
    d = max(0.0, floor - p[2]) if floor is not None else 0.0
    if box is not None:
        d += max(0.0, box[0] - p[0], p[0] - box[1])
        d += max(0.0, box[2] - p[1], p[1] - box[3])
    return d


def workspace_refusal(
    start: Any, target: Any, floor: float | None, box: Any, where: str = ""
) -> str | None:
    """Why a move from ``start`` to ``target`` leaves pi's workspace, else None: refused when it
    ends outside, unless it moves back toward the box."""
    start, target = np.asarray(start, float)[:3], np.asarray(target, float)[:3]
    if outside(target, floor, box) > 1e-6 and outside(target, floor, box) >= (
        outside(start, floor, box) - 1e-6
    ):
        span = f"x {box[0]}..{box[1]}, y {box[2]}..{box[3]}, " if box else ""
        low = f"z >= {floor} m" if floor is not None else "no floor"
        return (
            f"the move ends at {np.round(target, 3).tolist()}{where}, outside the workspace "
            f"({span}{low}; --workspace-xy / --z-floor)"
        )
    return None


def check_translation(
    delta: Any, limit: float | None, name: str = "delta_xyz"
) -> float:
    """The translation's length; ValueError beyond ``limit`` (m per call)."""
    norm = float(np.linalg.norm(vec3(delta, name)))
    if limit is not None and not norm <= limit + 1e-9:
        raise ValueError(
            f"{name} moves {norm:.4f} m; the limit is {limit} m per call (--max-move). "
            "Split the motion into smaller calls."
        )
    return norm


def check_rotation(rpy: Any, limit: float | None, name: str = "delta_rpy") -> float:
    """The rotation's size (norm of the rpy vector); ValueError beyond ``limit`` (rad per call)."""
    norm = float(np.linalg.norm(vec3(rpy, name)))
    if limit is not None and not norm <= limit + 1e-9:
        raise ValueError(
            f"{name} rotates {norm:.4f} rad; the limit is {limit} rad per call "
            "(--max-rotate). Split the rotation into smaller calls."
        )
    return norm


class RealCodeMode(CodeRunMixin):
    """pi's limits and ``code.run`` for a real-robot facade (see the module doc). The facade
    calls :meth:`_enable_code` with its ``--code`` and :meth:`_set_limits` with pi's limits before
    its base ``__init__`` registers the RPC, and :meth:`_install_real` at the end of
    ``_register_rpc``."""

    #: Without --code the server serves no code.run and needs no token.
    REQUIRE_TOKEN = False
    #: pi's limits this robot takes, name -> default (None: off). The server's own config limits
    #: apply as well; these are pi's flags.
    _LIMIT_DEFAULTS: dict[str, Any] = {}
    _code_on = False
    _limits: dict[str, Any] | None = None

    def _enable_code(self, on: bool) -> None:
        self._code_on = bool(on)
        self.REQUIRE_TOKEN = self._code_on

    def _set_limits(self, limits: dict[str, Any] | None) -> None:
        """pi's limits (its flags at spawn); names left out keep their defaults."""
        unknown = sorted(set(limits or {}) - set(self._LIMIT_DEFAULTS))
        if unknown:
            raise ValueError(f"{type(self).__name__} takes no limit(s) {unknown}")
        self._limits = check_limits({**self._LIMIT_DEFAULTS, **(limits or {})})

    def _limit(self, name: str) -> Any:
        """One of pi's limits (None: off)."""
        if self._limits is None:
            self._set_limits(None)
        return (self._limits or {}).get(name)

    def motion_limits(self) -> dict[str, Any]:
        """pi's limits as this server enforces them (``env.get_env_meta().motion_limits``)."""
        return {k: self._limit(k) for k in self._LIMIT_DEFAULTS}

    def _install_real(self, robot: str, have: Callable[[str], bool]) -> None:
        """``code.api`` from the robot's manifest (with the startup self-check), ``code.run``
        only with --code, and ``motion_limits`` in ``env.get_env_meta``."""
        meta = self._rpc["env.get_env_meta"]

        def get_env_meta(*args: Any, **kwargs: Any) -> Any:
            out = meta(*args, **kwargs)
            if isinstance(out, dict):
                out = {**out, "motion_limits": self.motion_limits()}
            return out

        self._rpc["env.get_env_meta"] = get_env_meta
        self._manifest_code_run(
            robot,
            have=have,
            move_m=self._code_move_m,
            after=self._code_after,
            check=self._code_check,
            reply=self._code_reply,
            begin=self._begin_run,
            finish=self._finish_run,
        )
        if not self._code_on:
            self._rpc.pop("code.run", None)
            self._rpc.pop("code.helpers", None)

    def _install_code_run(self, api: Any, **hooks: Any) -> Any:
        # Without --code only code.api is served (built by _manifest_ready with the self-check).
        if not self._code_on:
            return None
        self._begin_run()
        return super()._install_code_run(api, **hooks)

    # ---- the run ----

    def _begin_run(self) -> None:
        self._run_motions = 0
        self._run_states: Any = None
        self._run_frames: list[np.ndarray] = []

    def _finish_run(self) -> dict:
        return {
            "motions": self._run_motions,
            "states": self._run_states,
            "frames": list(self._run_frames),
        }

    def _code_reply(self, method: str, out: Any) -> Any:
        # A motion's wrapped states refresh pi's state cache; the program gets the result as is
        # (a real robot has no simulator ground truth to hide).
        if isinstance(out, dict) and out.get("states") is not None:
            self._run_states = out["states"]
        return out

    def _code_after(self, _primitive: Any) -> None:
        self._run_motions += 1
        try:
            obs = self._rpc["env.get_observation"]()
        except Exception:
            return
        frame = self._video_frame(obs if isinstance(obs, dict) else {})
        if frame is None:
            return
        if len(self._run_frames) >= CODE_MAX_FRAMES:
            self._run_frames = self._run_frames[::2]
        self._run_frames.append(np.asarray(frame))

    def _video_frame(self, obs: dict) -> np.ndarray | None:
        """The frame of the camera pi's episode video follows (None: no per-motion video)."""
        return None

    def _code_move_m(self, method: str, kwargs: dict) -> float:
        """A program call's translation, m (the run's move budget)."""
        raise NotImplementedError

    def _code_check(self, method: str, kwargs: dict) -> None:
        """Program-only refusals (pi's limits are the motion methods' own)."""


__all__ = [
    "CODE_MAX_FRAMES",
    "LIMIT_FLAGS",
    "RealCodeMode",
    "add_code_argument",
    "add_limit_arguments",
    "check_limits",
    "check_rotation",
    "check_translation",
    "limits_from_args",
    "outside",
    "vec3",
    "workspace_refusal",
]
