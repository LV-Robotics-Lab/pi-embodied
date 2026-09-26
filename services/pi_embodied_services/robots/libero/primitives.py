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

"""The LIBERO primitive registry (``code.api``, ../../components/code_api.py).

The server holds the raw simulator surface; LIBERO's pose-level motions (IK servo, scripted
grasps) and its perception run agent-side, so the high tier here is the scene read-outs.
"""

from __future__ import annotations

from pi_embodied_services.components.code_api import (
    GROUND_TRUTH,
    TASK_LANGUAGE,
    Param,
    Primitive,
)

LIBERO_PRIMITIVES = (
    TASK_LANGUAGE,
    Primitive(
        "render_camera",
        "env.render_camera",
        "One camera's RGB (and depth) frame.",
        {
            "camera_name": Param(
                "string", "'agentview' or 'robot0_eye_in_hand'", False
            ),
            "height": Param("integer", "pixels (default 1024)", False),
            "width": Param("integer", "pixels (default 1024)", False),
            "depth": Param("boolean", "also return the depth map", False),
        },
    ),
    Primitive(
        "get_camera_meta",
        "env.get_camera_meta",
        "Camera intrinsics and extrinsics at a resolution.",
        {
            "camera_name": Param("string", "camera (default agentview)", False),
            "height": Param("integer", "pixels (default 256)", False),
            "width": Param("integer", "pixels (default 256)", False),
        },
    ),
    Primitive(
        "raw_obs",
        "env.raw_obs",
        "The simulator's current observation dict (proprioception and images).",
        tiers=("low",),
    ),
    Primitive(
        "step",
        "env.step",
        "One OSC action [dx, dy, dz, drx, dry, drz, gripper] (gripper -1 open, +1 close).",
        {"action": Param("array", "7 floats")},
        mutating=True,
        tiers=("low",),
    ),
    Primitive(
        "chunk_step",
        "env.chunk_step",
        "Run a chunk of OSC actions [N, 7] in one call.",
        {
            "actions": Param("array", "N x 7 floats"),
            "return_all_frames": Param("boolean", "one observation per action", False),
        },
        mutating=True,
        tiers=("low",),
    ),
    GROUND_TRUTH,
)
