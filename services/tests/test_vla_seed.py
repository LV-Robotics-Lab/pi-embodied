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

"""Per-inference VLA seeding with mock samplers (no model, no GPU)."""

from __future__ import annotations

import random

import numpy as np
import pytest

from pi_embodied_services.components.vla_facade_base import (
    BaseVLAFacade,
    inference_seed,
    seeded,
)
from pi_embodied_services.robots.robocasa.vla_server import RoboCasaVLAFacade
from pi_embodied_services.robots.robotwin.vla_server import LingBotVLAFacade

try:
    import torch
except ImportError:
    torch = None


def draw() -> list[float]:
    """A sampler that draws from every global RNG a VLA may use."""
    out = [random.random(), float(np.random.standard_normal())]
    if torch is not None:
        out.append(float(torch.randn(1)))
        if torch.cuda.is_available():
            out.append(float(torch.randn(1, device="cuda")))
    return out


class MockVLA(BaseVLAFacade):
    SERVICE_NAME = "mock-vla"

    def predict(self, obs, options=None):
        with seeded(inference_seed(options)):
            return draw()


def call(facade, *args):
    return facade._serve_dispatch("vla.predict", args, {})


def test_same_seed_same_actions_different_seed_differs():
    vla = MockVLA()
    a = call(vla, {}, {"mode": "eval", "seed": 7})
    call(vla, {}, None)  # unseeded calls in between do not matter
    assert call(vla, {}, {"mode": "eval", "seed": 7}) == a
    assert call(vla, {}, {"seed": 8}) != a


def test_seeded_call_leaves_unseeded_streams_untouched():
    random.seed(1)
    np.random.seed(1)
    if torch is not None:
        torch.manual_seed(1)
    expected = draw()
    random.seed(1)
    np.random.seed(1)
    if torch is not None:
        torch.manual_seed(1)
    call(MockVLA(), {}, {"seed": 123})
    assert draw() == expected


@pytest.mark.parametrize("bad", [-1, 2**32, 1.5, "3", True])
def test_bad_seed_is_rejected(bad):
    with pytest.raises(ValueError):
        inference_seed({"seed": bad})


def test_absent_seed_is_none():
    assert inference_seed(None) is None
    assert inference_seed({"mode": "eval"}) is None


class MockLingBot:
    def __init__(self):
        self.seen: list[dict] = []

    def infer(self, obs):
        self.seen.append(dict(obs))
        return {"action": draw()}


def test_lingbot_seed_key_is_consumed_and_applied():
    policy = MockLingBot()
    vla = LingBotVLAFacade(policy)
    a = vla.infer({"task": "t", "seed": 5})
    assert policy.seen[-1] == {"task": "t"}  # LingBot never sees the seed
    assert vla.infer({"task": "t", "seed": 5}) == a
    assert vla.infer({"task": "t", "seed": 6}) != a
    vla.infer({"task": "t"})
    assert policy.seen[-1] == {"task": "t"}


class MockRLDX:
    def __init__(self):
        self.options: list[dict] = []

    def get_action(self, obs, options):
        self.options.append(options)
        return {"action": draw()}, {}


def test_rldx_seed_is_consumed_and_applied():
    vla = RoboCasaVLAFacade.__new__(RoboCasaVLAFacade)
    vla.policy = MockRLDX()
    a = vla.predict({}, {"seed": 9, "reset_memory": [True]}, session_id="s")
    assert vla.policy.options[-1] == {"reset_memory": [True], "session_ids": ["s"]}
    assert vla.predict({}, {"seed": 9}, session_id="s") == a
    assert vla.predict({}, {"seed": 10}, session_id="s") != a
