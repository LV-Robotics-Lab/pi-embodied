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

"""The BEHAVIOR server's perception (segment, point, back_project: once composed in pi's tools,
now facade methods the tools and programs share) and its joint-space primitives
(move_to_joints, move_along_trajectory), against the mock OmniGibson of test_behavior_env.py."""

from __future__ import annotations

import base64
import io

import numpy as np
import pytest
from PIL import Image
from test_behavior_env import FakeSim

from pi_embodied_services.robots.behavior import env_server, sim


def png(a: np.ndarray) -> str:
    buf = io.BytesIO()
    Image.fromarray(a).save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


class Client:
    def __init__(self, answer):
        self.answer = answer
        self.calls: list[tuple[str, dict]] = []

    def call(self, method, args=(), kwargs=None, *, timeout_s=None):
        self.calls.append((method, kwargs))
        return self.answer


def facade():
    """A 4x4 head camera 1.2 m up looking down (OpenGL: along -z), fx = fy = 2, c = (2, 2):
    pixel (r, c) at depth d is world [0.1 + (c - 2) d / 2, -(r - 2) d / 2, 1.2 - d]. Depth 1 m
    but 0.5 m in row 0 and no hit at (3, 3)."""
    fake = FakeSim()
    depth = np.ones((4, 4), np.float32)
    depth[0, :] = 0.5
    depth[3, 3] = np.inf
    fake.frame["robot_r:zed_link:Camera:0"]["depth_linear"] = depth
    f = fake.facade()
    f.reset()
    return fake, f


def test_project_follows_the_opengl_camera():
    _fake, f = facade()
    _rgb, xyz = f._world_map("head")
    assert xyz[2, 2] == pytest.approx([0.1, 0.0, 0.2])
    assert xyz[0, 3] == pytest.approx([0.35, 0.5, 0.7])
    assert np.isnan(xyz[3, 3]).all()


def test_back_project_pixel_and_region_modes():
    _fake, f = facade()
    r = f.back_project(2, 2)
    assert r["world_xyz"] == pytest.approx([0.1, 0.0, 0.2]) and r["pixel"] == [2, 2]
    region = f.back_project(row_range=[1, 4], col_range=[0, 4])
    assert region["mode"] == "region" and region["n_valid"] == 11
    assert region["center_xyz"][2] == pytest.approx(0.2)
    with pytest.raises(ValueError, match="out of bounds"):
        f.back_project(9, 0)
    with pytest.raises(ValueError, match="both row_range and col_range"):
        f.back_project(row_range=[0, 2])
    with pytest.raises(ValueError, match="camera"):
        f.back_project(1, 1, camera="agentview")


def test_segment_projects_the_mask_and_the_program_gets_no_picture():
    fake, f = facade()
    with pytest.raises(RuntimeError, match="--sam3"):
        f.segment("radio")
    f._sam3_url = "http://sam3"
    sam3 = f._clients["segment"] = Client(
        {
            "found": True,
            "score": 0.91,
            "box": [0, 0, 4, 4],
            "mask_png_base64": png(np.full((4, 4), 255, np.uint8)),
        }
    )
    with pytest.raises(ValueError, match="exactly one"):
        f.segment()
    r = f.segment("radio", min_score=0.3)
    assert sam3.calls[0][0] == "sam3.segment"
    assert (
        sam3.calls[0][1]["text_prompt"] == "radio"
        and sam3.calls[0][1]["min_score"] == 0.3
    )
    assert r["found"] and r["n_pixels"] == 16 and r["n_valid"] == 15
    assert r["mask"].shape == (4, 4) and len(r["top_xyz"]) == 3
    assert r["world_xyz"][2] == pytest.approx(0.2) and "overlay_png_base64" in r
    assert "overlay_png_base64" not in f._code_reply("env.segment", r)
    sam3.answer = {"found": False, "reason": "nothing above 0.3"}
    miss = f.segment(point=[1, 1])
    assert miss == {
        "found": False,
        "camera": "head",
        "error": "nothing above 0.3",
        "fallback": "Pick pixels in the image and use back_project.",
    }


def test_point_asks_molmo_and_back_projects_a_window():
    _fake, f = facade()
    f._molmo_url = "http://molmo"
    molmo = f._clients["point"] = Client(
        {"point_xy": [2.2, 1.9], "answer": "the radio"}
    )
    r = f.point("the radio", camera="left_wrist")
    assert (
        molmo.calls[0][0] == "molmo.ground"
        and molmo.calls[0][1]["query"] == "the radio"
    )
    assert r["pixel"] == [2, 2] and r["found"] and r["answer"] == "the radio"
    assert len(r["world_xyz"]) == 3 and "overlay_png_base64" in r
    molmo.answer = {"point_xy": None, "answer": "no radio"}
    assert f.point("x")["found"] is False


def joint_facade(rate: float = 0.1):
    """The fake with 7-joint arms whose position controllers close ``rate`` rad per step."""
    fake, f = facade()
    r = fake.robot
    r.arm_control_idx = {"left": np.arange(10, 17), "right": np.arange(17, 24)}
    r.joint_lower_limits = np.full(28, -2.0)
    r.joint_upper_limits = np.full(28, 2.0)
    q = np.zeros(28)
    r.get_joint_positions = lambda: q
    step = fake.env.step

    def joint_step(action):
        if action[0] == "joints":
            q[:] = q + np.clip(action[1] - q, -rate, rate)
            return step(("settle", None, 0))
        return step(action)

    fake.env.step = joint_step
    return fake, f, q


@pytest.fixture(autouse=True)
def joint_actions(monkeypatch):
    monkeypatch.setattr(
        sim, "joint_target_action", lambda robot, q: ("joints", np.array(q), 0)
    )


def test_move_to_joints_servos_the_arm_within_its_limits():
    fake, f, q = joint_facade()
    target = [0.3, -0.2, 0.0, 0.1, 0.0, 0.0, 5.0]  # the last beyond its 2 rad limit
    r = f.move_to_joints(target, "left")
    assert r["ok"] and r["reached"] and r["primitive"] == "move_to_joints"
    assert q[10:17] == pytest.approx([0.3, -0.2, 0.0, 0.1, 0.0, 0.0, 2.0])
    assert r["steps"] == 20 and fake.steps == r["env_steps"] and "head" in r
    assert q[17:24] == pytest.approx(np.zeros(7)), "the other arm holds"
    short = f.move_to_joints([0.0] * 7, "left", max_steps=2)
    assert (
        short["ok"]
        and not short["reached"]
        and short["joints_left_rad"] == pytest.approx(1.8)
    )
    with pytest.raises(ValueError, match="7 finite angles"):
        f.move_to_joints([0.0] * 6, "left")
    with pytest.raises(ValueError, match="arm"):
        f.move_to_joints([0.0] * 7, "head")


def test_move_along_trajectory_runs_its_waypoints_and_stops_on_stop():
    fake, f, q = joint_facade()
    traj = [[0.1 * k] * 7 for k in (1, 2, 3)]
    r = f.move_along_trajectory(traj, "right")
    assert r["ok"] and r["waypoints"] == 3 and q[17:24] == pytest.approx([0.3] * 7)
    fake.stop_after = fake.steps + 1
    r = f.move_along_trajectory([[0.0] * 7, [0.5] * 7], "right")
    assert r.get("cancelled") and r["waypoints"] == 0
    with pytest.raises(ValueError, match="N <= 100"):
        f.move_along_trajectory([[0.0] * 7] * 101, "right")


def test_a_program_moves_joints_under_the_run_cap():
    fake, f, _q = joint_facade()
    out = f._rpc["code.run"](
        "r = move_to_joints([0.4, 0, 0, 0, 0, 0, 0], 'left')\n"
        "t = move_along_trajectory([[0.4, 0.2, 0, 0, 0, 0, 0]], 'left')\n"
        "RESULT = [r['reached'], sorted(t)]\n",
        timeout_s=30,
        tier="low",
    )
    assert out["status"] == "ran", out
    reached, keys = out["result"]
    assert reached is True and "head" not in keys and "privileged" not in keys
    assert [c.get("move_m") for c in out["calls"]] == pytest.approx([0.4, 0.2])
    capped = f._rpc["code.run"](
        "move_to_joints([2.0, 0, 0, 0, 0, 0, 0], 'right')\n",
        timeout_s=30,
        tier="low",
        max_move_m=1.0,
    )
    assert capped["status"] == "error" and capped["limit"] == "max_move_m", capped
    assert env_server.MAX_WAYPOINTS == 100
