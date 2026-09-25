# Copyright 2026 The RPent Authors.
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
# Modified by pi-embodied: import paths rewritten to pi_embodied_services.

"""Unified VLA backend base class."""

from __future__ import annotations

import random
from collections.abc import Iterator
from contextlib import contextmanager

import numpy as np

from pi_embodied_services.utils.rpc import RpcFacade
from pi_embodied_services.utils.rpc.rpc_facade import DEFAULT_SESSION_TIMEOUT_S


def inference_seed(options: dict | None) -> int | None:
    """The optional per-inference ``seed`` in ``options``: a non-negative int below 2**32, or None."""
    seed = (options or {}).get("seed")
    if seed is None:
        return None
    if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed < 2**32:
        raise ValueError(f"seed must be an int in [0, 2**32), got {seed!r}")
    return seed


@contextmanager
def seeded(seed: int | None) -> Iterator[None]:
    """Make the enclosed inference a function of ``seed``; a no-op when ``seed`` is None.

    Seeds every RNG a VLA sampler draws from (torch CPU and every visible CUDA
    device, numpy's and Python's global generators) and restores their previous
    states on exit, so unseeded calls keep their own streams. Callers hold the
    facade's call lock, so no other inference draws from these generators meanwhile.
    Bitwise-equal outputs also need deterministic kernels; the servers leave
    cuDNN/cuBLAS defaults alone (identical repeats measured on the RTX 5090).
    """
    if seed is None:
        yield
        return
    try:
        import torch
    except ImportError:  # CPU-only test environments
        torch = None
    py_state, np_state = random.getstate(), np.random.get_state()
    devices = (
        list(range(torch.cuda.device_count()))
        if torch is not None and torch.cuda.is_available()
        else []
    )
    try:
        random.seed(seed)
        np.random.seed(seed)
        if torch is None:
            yield
            return
        with torch.random.fork_rng(devices=devices):
            torch.manual_seed(seed)  # also seeds every CUDA device
            yield
    finally:
        random.setstate(py_state)
        np.random.set_state(np_state)


class BaseVLAFacade(RpcFacade):
    """Unified VLA backend base class.

    Methods subclasses must implement:
        ``predict`` — the subclass performs the actual inference. An optional
        ``options["seed"]`` makes that one inference deterministic: run the
        sampler under ``seeded(inference_seed(options))``.
        ``__init__`` —  the subclass loads the model itself.

    RPC routing:
        ``_dispatch`` uses a registration dict (``self._rpc``) instead of an
        ``if method == "predict"`` chain. Subclasses register their own methods
        in ``_register_rpc``.

    Session-isolation model (backend-specific, not in the base class):
        For session-aware VLA models, the subclass may implement a session
        isolation model. See the robocasa RLDX VLA implementation for reference.
        Implement ``_on_session_drop`` and ``reset_session``; optionally
        customize ``session_timeout_s`` and ``session_sweep_s`` to periodically
        evict expired sessions.
    """

    def __init__(
        self,
        *,
        enable_sessions: bool = False,
        session_timeout_s: float = DEFAULT_SESSION_TIMEOUT_S,
    ):
        super().__init__(
            enable_sessions=enable_sessions, session_timeout_s=session_timeout_s
        )
        self._register_rpc()

    # ---- framework ----
    def _register_rpc(self):
        self._rpc["vla.predict"] = self.predict

    # ---- abstract methods (subclasses must override) ----
    def predict(self, *args, **kwargs):
        raise NotImplementedError
