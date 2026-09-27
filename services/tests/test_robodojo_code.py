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

"""Code mode on the RoboDojo server (``code.run`` over its registry), without Isaac Sim: the
facade on test_robodojo's fake EvalEnv (RoboDojo's action-dict contract)."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
from test_robodojo import FakeEvalEnv

from pi_embodied_services.robots.robodojo import env_server, sim
from pi_embodied_services.robots.robodojo.env_server import (
    CODE_MAX_CHUNK,
    RobodojoEnvFacade,
)


@pytest.fixture
def facade(monkeypatch):
    monkeypatch.setattr(
        sim, "unstable_error", lambda: type("UnStableError", (Exception,), {})
    )
    f = RobodojoEnvFacade(
        app=SimpleNamespace(close=lambda: None),
        env=FakeEvalEnv(),
        meta={"task": "stack_bowls", "seed": 0, "layouts": 25, "eval_seed": 0},
    )
    f.stop_requested = lambda: False
    f.reset()
    return f


def keys(value) -> set[str]:
    if isinstance(value, dict):
        return set(value) | {k for v in value.values() for k in keys(v)}
    if isinstance(value, (list, tuple)):
        return {k for v in value for k in keys(v)}
    return set()


def test_the_server_serves_code_run_with_a_token(facade):
    assert RobodojoEnvFacade.REQUIRE_TOKEN and facade._rpc_token
    assert "code.run" in facade._rpc and "code.helpers" in facade._rpc
    # Kit is not thread-safe: the primitives run on the server's own thread.
    assert facade._code._primitive_thread is False


def test_a_run_reports_steps_success_obs_and_frames_and_hides_images_and_score(
    facade,
):
    facade._env.solve_when = lambda rm: rm.q["left_arm"][2] > 0.25
    out = facade._rpc["code.run"](
        "r = move_delta('left', [0, 0, 0.1])\nRESULT = [r, state()]\n",
        timeout_s=30,
    )
    assert out["status"] == "ran", out
    seen = keys(out["result"])
    assert not seen & set(env_server.CODE_HIDDEN), seen & set(env_server.CODE_HIDDEN)
    r, st = out["result"]
    assert r["arm"] == "left" and r["success"] is True
    assert st["success"] is True and st["ended"] is True
    assert out["steps"] == 5 and out["success"] is True and out["ended"] is True
    assert out["obs"]["head"].shape == (48, 64, 3), "the tools' Obs, with images"
    assert len(out["frames"]) >= 1, "the motion's head frames go to the run's video"


def test_native_steps_reply_without_images_and_count_toward_the_move_cap(facade):
    q = list(facade._q["left"])
    q[2] += 0.1  # 0.1 rad on one joint
    out = facade._rpc["code.run"](
        f"RESULT = step({{'left_arm_joint_state': {q}, 'left_ee_joint_state': [0.5]}})\n",
        timeout_s=30,
        tier="low",
    )
    assert out["status"] == "ran", out
    assert set(out["result"]) == {"state", "terminated", "truncated", "info"}
    assert (
        "head" not in out["result"]["state"] and "score" not in out["result"]["state"]
    )
    assert out["move_m"] == pytest.approx(0.1 * env_server.CODE_M_PER_RAD)
    too_many = [{"right_ee_joint_state": [1.0]}] * (CODE_MAX_CHUNK + 1)
    out = facade._rpc["code.run"](f"chunk_step({too_many})\n", timeout_s=30, tier="low")
    assert out["status"] == "error" and "at most" in str(out["error"]) + str(
        out["traceback"]
    )


def test_move_estimates(facade):
    ee = facade._ee_pose("right")[:3]
    assert facade._code_move_m(
        "env.move_to", {"arm": "right", "xyz": (ee + [0, 0, 0.2]).tolist()}
    ) == pytest.approx(0.2)
    assert facade._code_move_m(
        "env.move_delta", {"arm": "left", "delta_xyz": [0.03, 0.04, 0]}
    ) == pytest.approx(0.05)
    assert facade._code_move_m("env.rotate_delta", {"arm": "left", "yaw": 0.5}) == 0
    pose = np.r_[ee + [0.1, 0, 0], [1.0, 0, 0, 0]].tolist()
    assert facade._code_move_m(
        "env.chunk_step", {"actions": [{"right_ee_pose": pose}]}
    ) == pytest.approx(0.1)
