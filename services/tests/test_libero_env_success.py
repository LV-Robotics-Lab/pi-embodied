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

"""With ignore_terminations the LIBERO facade reports success from RLinf's latched
success_once (no simulator)."""

from __future__ import annotations

import numpy as np

from pi_embodied_services.robots.libero.env_server import (
    LiberoEnvFacade,
    build_env_cfg,
)


class IgnoresTerminations:
    """Like RLinf's LiberoEnv with ignore_terminations: terminations are always zero, and
    episode.success_once latches the first success (here at the second step)."""

    def __init__(self):
        self.steps = 0

    def _info(self):
        return {"episode": {"success_once": np.array([self.steps >= 2])}}

    def step(self, action):
        self.steps += 1
        zeros = np.zeros(1, dtype=bool)
        return {"x": np.zeros((1, 1))}, np.zeros(1), zeros, zeros, self._info()

    def chunk_step(self, actions):
        infos, obs = [], []
        for _ in range(actions.shape[1]):
            self.steps += 1
            infos.append(self._info())
            obs.append({"x": np.zeros((1, 1))})
        n = actions.shape[1]
        zeros = np.zeros((1, n), dtype=bool)
        return obs, np.zeros((1, n)), zeros, zeros, infos


def test_the_env_keeps_stepping_after_success():
    assert build_env_cfg().ignore_terminations is True


def test_success_is_latched_from_success_once():
    env = LiberoEnvFacade(IgnoresTerminations(), meta={})
    assert not env.step(np.zeros(7))[2]
    assert env.step(np.zeros(7))[2]
    # Stepping on after success (release and lift) keeps reporting it.
    assert env.step(np.zeros(7))[2]
    chunk = LiberoEnvFacade(IgnoresTerminations(), meta={})
    assert chunk.chunk_step(np.zeros((3, 7)))[2].tolist() == [False, True, True]
