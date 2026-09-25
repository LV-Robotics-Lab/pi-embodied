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

"""Ground-truth object poses for the simulators' ``env.ground_truth_poses`` (pi's
``--privileged``, CaP-X's S1 tier). Simulation only: a real robot has no such call.

A pose is ``{"pos": [x, y, z], "quat_xyzw": [x, y, z, w]}`` in the simulator's world
frame, metres, rounded to 1e-5 (the repo's xyzw convention; MuJoCo and SAPIEN store wxyz).
"""

from __future__ import annotations

import numpy as np


def pose(pos, quat_wxyz) -> dict:
    """A world pose from a position and a wxyz quaternion."""
    w, x, y, z = np.asarray(quat_wxyz, dtype=np.float64).reshape(4)
    return {
        "pos": [round(float(v), 5) for v in np.asarray(pos).reshape(3)],
        "quat_xyzw": [round(float(v), 5) for v in (x, y, z, w)],
    }


def mujoco_body_poses(sim, body_ids: dict[str, int]) -> dict[str, dict]:
    """World poses of the named MuJoCo bodies (robosuite ``sim``: ``data.body_xpos``/``body_xquat``)."""
    return {
        name: pose(sim.data.body_xpos[i], sim.data.body_xquat[i])
        for name, i in body_ids.items()
    }


def respond(poses: dict[str, dict], names=None) -> dict:
    """The ``env.ground_truth_poses`` result: ``names`` (all when None or empty) of ``poses``.
    An unknown name raises, listing the scene's objects."""
    if not names:
        names = sorted(poses)
    unknown = [n for n in names if n not in poses]
    if unknown:
        raise ValueError(f"unknown objects {unknown}; the scene has {sorted(poses)}")
    return {"frame": "world", "poses": {n: poses[n] for n in dict.fromkeys(names)}}
