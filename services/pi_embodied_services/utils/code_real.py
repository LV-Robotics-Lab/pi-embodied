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

"""Code mode (``code.run``, code_exec.py ``CodeRunMixin``) on a real-robot env server.

What every real-robot server needs on top of the mixin:

- ``--code`` (:func:`add_code_argument`): only then does the server serve ``code.run`` and
  require its RPC token (pi reads it from the listening line, or takes ``URL#token=HEX``); without
  it nothing of code mode is registered and the server answers as it always did.
- pi's per-call limits. A program calls the registry's primitives, which are the facade methods
  pi's tools call, so the server's own limits stay in the path. pi's tool checks (e.g. a
  ``--max-move`` tighter than the server's cap) run in pi, before an env call, where a program's
  calls never pass: pi hands them over with ``code.set_limits`` when code mode starts, and the
  robot's ``check`` refuses a program motion beyond them. Until they are set a program cannot
  move. A program never reaches ``code.set_limits`` (not in the registry; business calls are
  refused while a program runs).
- The run's report: its motion count, the last wrapped ``states`` a motion returned (pi's state
  cache) and a bounded video (one frame of the camera pi's video follows after each motion;
  :data:`CODE_MAX_FRAMES` full frames, every other one dropped when full). pi records the run as
  its next state step.

The robot supplies ``move_m`` / ``check`` (its motions' translation and refusals), the names of
its limits (``_CODE_LIMITS``) and :meth:`RealCodeMode._video_frame`.
"""

from __future__ import annotations

import argparse
from typing import Any

import numpy as np

from pi_embodied_services.utils.code_exec import CodeRunMixin

#: Full camera frames one run hands back (halved, every other one kept, when full).
CODE_MAX_FRAMES = 32


def add_code_argument(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--code",
        action="store_true",
        help="Serve code.run (pi's run_code with --code-real) and require the RPC token",
    )


def vec3(value: Any, name: str) -> np.ndarray:
    """Three finite numbers, else ValueError naming ``name``."""
    a = np.asarray(value, dtype=np.float64).reshape(-1)
    if a.shape != (3,) or not np.all(np.isfinite(a)):
        raise ValueError(f"{name} must be 3 finite numbers")
    return a


class RealCodeMode(CodeRunMixin):
    """``code.run`` for a real-robot facade (see the module doc). The facade calls
    :meth:`_enable_code` with its ``--code`` before its base ``__init__`` registers the RPC, and
    :meth:`_install_real_code_run` at the end of ``_register_rpc``."""

    #: Without --code the server serves no code.run and needs no token.
    REQUIRE_TOKEN = False
    #: The limits ``code.set_limits`` takes: name -> required. ``max_*`` must be positive.
    _CODE_LIMITS: dict[str, bool] = {}
    _code_on = False
    _code_limits: dict[str, Any] | None = None

    def _enable_code(self, on: bool) -> None:
        self._code_on = bool(on)
        self.REQUIRE_TOKEN = self._code_on

    def _install_real_code_run(self, api: Any) -> None:
        if not self._code_on:
            return
        self._code_limits = None
        self._begin_run()
        self._rpc["code.set_limits"] = self.set_code_limits
        self._install_code_run(
            api,
            move_m=self._code_move_m,
            after=self._code_after,
            check=self._code_check,
            reply=self._code_reply,
            begin=self._begin_run,
            finish=self._finish_run,
        )

    def set_code_limits(self, **limits: Any) -> dict[str, Any]:
        """pi's per-call limits for a program's motions (its tools' own checks)."""
        unknown = sorted(set(limits) - set(self._CODE_LIMITS))
        missing = sorted(
            k for k, req in self._CODE_LIMITS.items() if req and limits.get(k) is None
        )
        if unknown or missing:
            raise ValueError(f"code.set_limits: unknown {unknown}, missing {missing}")
        out: dict[str, Any] = {}
        for k, v in limits.items():
            if v is None:
                out[k] = None
                continue
            a = np.asarray(v, dtype=np.float64)
            if not np.all(np.isfinite(a)) or (
                k.startswith("max_") and not np.all(a > 0)
            ):
                raise ValueError(
                    f"code.set_limits: {k} must be finite (positive for max_*)"
                )
            out[k] = a.tolist()
        self._code_limits = out
        return dict(out)

    def _limit(self, name: str) -> Any:
        """One of pi's limits; a program motion is refused until pi set them."""
        if self._code_limits is None:
            raise ValueError(
                "code mode motions need pi's per-call limits (code.set_limits); none were set"
            )
        return self._code_limits.get(name)

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
        raise NotImplementedError

    def _code_check(self, method: str, kwargs: dict) -> None:
        raise NotImplementedError


__all__ = ["CODE_MAX_FRAMES", "RealCodeMode", "add_code_argument", "vec3"]
