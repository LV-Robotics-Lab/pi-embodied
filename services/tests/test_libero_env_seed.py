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

"""The LIBERO worker env's seed() also seeds the worker's global RNGs (no simulator)."""

from __future__ import annotations

import random

import numpy as np

from pi_embodied_services.robots.libero.env_server import _seeding_globals


class StoresSeed:
    """Like robosuite 1.5 under LIBERO's wrapper: seed() only stores the value."""

    def seed(self, value):
        self.stored = value

    def reset(self):
        return random.random(), float(np.random.standard_normal())


def test_seed_before_reset_makes_reset_draws_repeat():
    env = _seeding_globals(StoresSeed)()
    env.seed(1)
    first = env.reset()
    env.reset()  # draws move on without a new seed
    env.seed(1)
    assert env.reset() == first
    assert env.stored == 1
    env.seed(2)
    assert env.reset() != first
