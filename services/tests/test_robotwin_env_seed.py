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

"""The RoboTwin env worker's reset seeding and deterministic torch mode (no simulator)."""

from __future__ import annotations

import os
import random

import numpy as np
import pytest

pytest.importorskip("robotwin")
torch = pytest.importorskip("torch")

from pi_embodied_services.robots.robotwin import env_server  # noqa: E402


def draws():
    return random.random(), float(np.random.standard_normal()), float(torch.rand(1))


def test_seed_globals_makes_reset_draws_repeat():
    env_server._seed_globals(100000)
    first = draws()
    draws()
    env_server._seed_globals(100000)
    assert draws() == first
    env_server._seed_globals(100001)
    assert draws() != first


def test_deterministic_cuda_sets_torch_and_cublas(monkeypatch):
    monkeypatch.delenv("CUBLAS_WORKSPACE_CONFIG", raising=False)
    try:
        env_server._deterministic_cuda()
        assert torch.are_deterministic_algorithms_enabled()
        assert torch.is_deterministic_algorithms_warn_only_enabled()
        assert os.environ["CUBLAS_WORKSPACE_CONFIG"] == ":4096:8"
    finally:
        torch.use_deterministic_algorithms(False)


def test_torch_lbfgs_step_patches_curobo_once():
    env_server._torch_lbfgs_step()
    patched = env_server.LBFGSOpt.__init__
    env_server._torch_lbfgs_step()
    assert env_server.LBFGSOpt.__init__ is patched
    assert patched.__name__ == "torch_step_init"
