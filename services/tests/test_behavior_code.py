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

"""Code mode on the BEHAVIOR server (``code.run`` over its registry) against the mock OmniGibson
of test_behavior_env.py: a program reaches the tools' own facade methods, on the thread that
serves every RPC (Kit is not thread-safe), and never receives images or simulator-only state."""

from __future__ import annotations

import threading
import time

import numpy as np
import pytest
from test_behavior_env import FakeSim

from pi_embodied_services.robots.behavior import env_server, sim

#: A fake R1Pro action layout: base x, y, rz; a 4-joint trunk; two 7-joint arms; two grippers.
LAYOUT = {
    "base": range(0, 3),
    "trunk": range(3, 7),
    "arm_left": range(7, 14),
    "arm_right": range(14, 21),
    "gripper_left": range(21, 22),
    "gripper_right": range(22, 23),
}


def facade(fake: FakeSim | None = None):
    fake = fake or FakeSim()
    r = fake.robot
    r.controller_action_idx = {k: np.arange(v.start, v.stop) for k, v in LAYOUT.items()}
    r.action_dim = 23
    r.trunk_control_idx = np.arange(6, 10)
    r.arm_control_idx = {"left": np.arange(10, 17), "right": np.arange(17, 24)}
    f = fake.facade()
    f.reset()
    return fake, f


def test_the_server_serves_code_run_with_a_token_and_exclusivity():
    _fake, f = facade()
    assert env_server.BehaviorEnvFacade.REQUIRE_TOKEN and f._rpc_token
    assert "code.run" in f._rpc and "code.helpers" in f._rpc
    with pytest.raises(PermissionError):
        f._admit("env.state", None)
    f._code._active = True
    with pytest.raises(RuntimeError, match="run_code program is running"):
        f._admit("env.state", f._rpc_token)


def test_a_run_reaches_the_primitives_and_hands_back_no_images_or_privileged_state():
    fake, f = facade()
    fake.success_at = fake.steps + 5
    out = f._rpc["code.run"](
        "n = navigate_to_pose(1.0, 0.5, 0.0)\n"
        "g = grasp_object('left', [1.0, 0.5, 0.45])\n"
        "RESULT = [sorted(n), sorted(g), g['ok'], g['gripper_width']]\n",
        timeout_s=30,
    )
    assert out["status"] == "ran", out
    nav, grasp, ok, width = out["result"]
    for keys in (nav, grasp):
        for hidden in ("head", "head_depth", "left_wrist", "right_wrist_depth"):
            assert hidden not in keys
        assert "privileged" not in keys and "base_pos" in keys and "success" in keys
    assert ok is True and width == pytest.approx(0.02)
    assert [k for k, *_ in fake.calls][1:3] == ["navigate", "release"]
    # The run: its control steps, BDDL success latched mid-run, the tools' observation shape
    # (images, depth and the privileged block the TS side shows under --privileged).
    assert out["steps"] == fake.steps - 3 and out["success"] is True
    assert out["obs"]["head"].shape == (4, 4, 3)
    assert out["obs"]["privileged"]["in_hand"]["left"] == "radio_89"
    assert len(out["frames"]) == 2 and out["frames"][0].shape == (4, 4, 3)
    # navigate: the base's drive; grasp: to the pre-grasp, down and back up.
    nav_m = np.hypot(1.0, 0.5)
    grasp_m = np.linalg.norm(np.array([1.0, 0.5, 0.55]) - [0.3, 0.2, 0.9]) + 0.2
    assert [c.get("move_m") for c in out["calls"]] == pytest.approx(
        [nav_m, grasp_m], abs=1e-4
    )


def test_the_low_tier_reads_and_raw_steps_carry_no_object_state():
    fake, f = facade()
    a = np.zeros(23)
    a[:2] = [0.3, 0.4]  # base: 0.5 m in its frame
    b = a.copy()
    b[7] = 0.2  # left arm: one joint 0.2 rad from the current 0
    out = f._rpc["code.run"](
        f"a = {a.tolist()}\nb = {b.tolist()}\n"
        "s = step(a)\nc = chunk_step([a, b])\nst = state()\nro = raw_obs()\n"
        "RESULT = [sorted(s), sorted(s['state']), sorted(c), sorted(st), sorted(ro)]\n",
        timeout_s=30,
        tier="low",
    )
    assert out["status"] == "ran", out
    step, step_state, chunk, state, raw = out["result"]
    assert step == ["reward", "state", "success", "terminated", "truncated"]
    assert {"state", "terminated", "truncated", "success"} <= set(chunk)
    for keys in (step_state, state):
        assert "privileged" not in keys and "head" not in keys and "eef" in keys
    assert raw == ["joint_names", "joint_positions", "proprio"]
    assert [c.get("move_m") for c in out["calls"][:2]] == pytest.approx(
        [0.5, 0.5 + 0.5 + 0.2]
    )
    assert out["steps"] == 3 and len(out["frames"]) == 2
    out = f._rpc["code.run"](
        f"chunk_step([{a.tolist()}] * {env_server.CODE_MAX_CHUNK + 1})\n",
        timeout_s=30,
        tier="low",
    )
    assert out["status"] == "error" and "at most" in out["error"], out


def test_a_programs_primitives_run_on_the_thread_that_serves_every_rpc():
    """``serve`` executes business calls on its own (main) thread: ``code.run`` arrives there
    through the transport's proxy and the program's primitive calls step the env from there too,
    never from the transport thread or another one."""
    fake, f = facade()
    stepped: set[int] = set()
    step = fake.env.step

    def recording_step(action):
        stepped.add(threading.get_ident())
        return step(action)

    fake.env.step = recording_step
    main = threading.get_ident()
    got: dict = {}

    def client():
        while getattr(f, "_main_thread_queue", None) is None:
            time.sleep(0.01)
        try:
            got["run"] = f._dispatch_main_thread(
                "code.run",
                (),
                {"code": "open_gripper('left')\nclose_gripper('right')\n"},
                token=f._rpc_token,
            )
        finally:
            f._dispatch_main_thread("shutdown", (), {}, token=f._rpc_token)

    worker = threading.Thread(target=client, daemon=True)
    worker.start()
    f.serve(transport="http", host="127.0.0.1", port=0)
    worker.join(10)
    assert got["run"]["status"] == "ran", got
    assert got["run"]["steps"] == 2 * fake.n
    assert stepped == {main}


def test_action_move_m_bounds_the_base_and_the_joint_targets():
    fake, _f = facade()
    a = np.zeros(23)
    assert sim.action_move_m(fake.robot, [a]) == 0.0
    a[14] = -0.3  # right arm
    a[3] = 0.1  # trunk
    assert sim.action_move_m(fake.robot, [a, a]) == pytest.approx(0.4)
    with pytest.raises(ValueError):
        sim.action_move_m(fake.robot, [np.zeros(5)])
