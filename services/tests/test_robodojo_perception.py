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

"""RoboDojo's render_camera depth (the perception primitives' view), without Isaac."""

import numpy as np
import pytest

env_server = pytest.importorskip("pi_embodied_services.robots.robodojo.env_server")


def test_the_head_renders_its_same_step_depth_and_the_wrists_rgb_alone():
    f = object.__new__(env_server.RobodojoEnvFacade)
    color = np.zeros((4, 6, 4), np.uint8)
    f._obs = {
        "vision": {
            "cam_head": {"color": color, "depth": np.full((4, 6, 1), 0.9, np.float32)},
            "cam_left_wrist": {"color": color},
            "cam_right_wrist": {"color": color},
        }
    }
    rgb, depth = f.render_camera("head", depth=True)
    assert rgb.shape == (4, 6, 3) and depth.shape == (4, 6)
    assert np.all(depth == np.float32(0.9))
    assert f.render_camera("left_wrist", depth=True).shape == (4, 6, 3)
    assert f.render_camera("head").shape == (4, 6, 3)
