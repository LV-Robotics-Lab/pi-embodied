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

"""The GraspNet-1Billion server (components/graspnet1b_server.py) without torch: the
GraspGroup-row post-processing (collision check, NMS, on-object filter), the conversion to the
GraspNet frame, and its wiring as the ``graspnet1b`` backend of the env servers' planner."""

from __future__ import annotations

import argparse

import numpy as np
import pytest

from pi_embodied_services.components import graspnet1b_server as S
from pi_embodied_services.utils import grasp as G


def _row(score, center, R=np.eye(3), width=0.04, height=0.02, depth=0.02):
    return np.concatenate(
        [[score, width, height, depth], np.asarray(R).reshape(9), center, [-1]]
    )


def _rot_z(deg: float) -> np.ndarray:
    a = np.deg2rad(deg)
    return np.array(
        [[np.cos(a), -np.sin(a), 0], [np.sin(a), np.cos(a), 0], [0, 0, 1.0]]
    )


def test_rows_become_graspnet_frame_candidates_of_the_graspnet1b_backend():
    R = _rot_z(30)
    [c] = G.anygrasp_candidates(
        S.grasps_from_array(_row(0.7, [0.1, 0.2, 0.5], R, width=0.06)),
        source_model="graspnet1b",
    )
    assert c["grasp_frame"] == "graspnet" and c["source_model"] == "graspnet1b"
    assert np.allclose(c["rotation_matrix"], R)
    assert c["translation_xyz"] == [0.1, 0.2, 0.5] and c["width"] == 0.06
    assert np.allclose(
        np.subtract(c["contact_points_xyz"][1], c["contact_points_xyz"][0]),
        0.06 * R[:, 1],
    )
    # AnyGrasp keeps its own name.
    [a] = G.anygrasp_candidates(S.grasps_from_array(_row(0.7, [0, 0, 0.5])))
    assert a["source_model"] == "anygrasp"


def test_nms_keeps_the_best_of_near_duplicates_and_distinct_rotations():
    rows = np.stack(
        [
            _row(0.5, [0.0, 0.0, 0.5]),
            _row(0.9, [0.01, 0.0, 0.5]),  # 1 cm from the first, same rotation
            _row(0.8, [0.0, 0.0, 0.5], _rot_z(90)),  # same place, rotated 90 deg
            _row(0.7, [0.2, 0.0, 0.5]),  # 20 cm away
        ]
    )
    kept = S.grasp_nms(rows)
    assert kept[:, 0].tolist() == [0.9, 0.8, 0.7]


def test_voxel_down_sample_is_the_centroid_per_voxel():
    pts = np.array([[0.001, 0, 0], [0.003, 0, 0], [0.5, 0.5, 0.5]])
    out = S.voxel_down_sample(pts, 0.01)
    assert sorted(out[:, 0].round(4).tolist()) == [0.002, 0.5]


def test_collision_check_rejects_fingers_through_the_table():
    # A table: the plane z = 0.6 (camera frame), 1 mm grid over 20 cm x 20 cm.
    g = np.arange(-0.1, 0.1, 0.002)
    xx, yy = np.meshgrid(g, g)
    table = np.stack([xx.ravel(), yy.ravel(), np.full(xx.size, 0.6)], axis=1)
    # Approach +z (camera forward) toward the table; columns approach, closing, normal.
    down = np.column_stack([[0, 0, 1.0], [1, 0, 0], [0, 1, 0]])
    free = _row(0.9, [0.0, 0.0, 0.45], down)  # fingers end at 0.47 m: 13 cm above
    through = _row(0.9, [0.0, 0.0, 0.59], down)  # fingers reach 0.61 m: into the table
    hit = S.collision_mask(np.stack([free, through]), table)
    assert hit.tolist() == [False, True]


def _scene(h=48, w=48):
    K = np.array([[60.0, 0, w / 2], [0, 60.0, h / 2], [0, 0, 1]])
    depth = np.full((h, w), 0.9, np.float32)
    depth[18:30, 18:30] = 0.8
    mask = np.zeros((h, w), bool)
    mask[18:30, 18:30] = True
    return depth, K, mask


class _NoModel(S.GraspNet1BFacade):
    def __init__(self, rows, **kw):
        super().__init__("gsnet", "/nonexistent", "/nonexistent.tar", load=False, **kw)
        self.rows = rows
        self.clouds = []

    def _forward(self, cloud):
        self.clouds.append(cloud)
        return self.rows


def test_plan_keeps_grasps_on_the_object_and_answers_in_the_graspnet_frame():
    depth, K, mask = _scene()
    on = _row(0.6, [0.0, 0.0, 0.8])
    off = _row(0.95, [0.12, 0.12, 0.9])  # on the table, far from the block
    f = _NoModel(np.stack([on, off]), collision_thresh=0.0)
    out = f.plan(depth, K.tolist(), mask, max_candidates=5)
    assert out["backend"] == "graspnet1b" and out["grasp_frame"] == "graspnet"
    assert out["variant"] == "gsnet"
    assert [c["score"] for c in out["candidates"]] == [0.6]
    assert out["metadata"]["on_object"] == 1 and out["metadata"]["model_grasps"] == 2
    assert f.clouds[0].shape == (S.NUM_POINTS["gsnet"], 3)
    # Only the scene around the object goes to the model: the box grown by the margin.
    obj_x = (np.arange(18, 30) - 24) / 60.0 * 0.8
    assert np.abs(f.clouds[0][:, 0]).max() <= np.abs(obj_x).max() + 0.15 + 1e-6


def test_plan_refuses_an_unknown_variant_and_sparse_masks():
    with pytest.raises(ValueError, match="--model"):
        S.GraspNet1BFacade("resnet", "/x", "/y", load=False)
    depth, K, _ = _scene()
    tiny = np.zeros_like(depth, bool)
    tiny[20, 20] = True
    with pytest.raises(ValueError, match="too sparse"):
        _NoModel(np.zeros((0, 17))).plan(depth, K, tiny)


def test_env_servers_take_graspnet1b_as_a_backend():
    p = argparse.ArgumentParser()
    G.add_grasp_arguments(p)
    urls = G.urls_from_args(p.parse_args(["--graspnet1b", "http://127.0.0.1:8124"]))
    assert urls["graspnet1b"] == "http://127.0.0.1:8124"
    assert "graspnet1b" in G.BACKENDS and "graspnet1b" in G.GRASP_URL_KEYS
    view = {
        "rgb": np.zeros((4, 4, 3), np.uint8),
        "depth": np.ones((4, 4), np.float32),
        "intrinsic_K": np.eye(3),
        "extrinsic_cam2world": np.eye(4),
    }
    planner = G.GraspPlanner.from_args(lambda cam: view, cameras=["agentview"], **urls)
    assert planner is not None
    assert planner.capabilities()["grasp_backends"] == ["graspnet1b"]
