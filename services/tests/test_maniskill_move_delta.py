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

"""``env.move_delta``: the waypoint split and the servo legs pi's move_delta tool used to run
from TypeScript (one ``env.servo`` per ~2 cm leg), now one server method for the tool, the units
and programs, without a simulator (tests/test_maniskill_code.py's fake env)."""

from __future__ import annotations

import json

import pytest

from pi_embodied_services.robots.maniskill import env_server as ms
from tests.test_maniskill_code import facade

#: The legs the TypeScript ``phases`` (robots/maniskill/index.ts before the move to the server)
#: produced for these inputs, printed by node: every target float must come out bit-identical.
TS_PHASES = json.loads(
    r"""[{"start":[0,0,0.2],"delta":[0.05,0,-0.03],"changed":false,"gripper_steps":6,"phases":[[[0.016666666666666666,0,0.19],2,8],[[0.03333333333333333,0,0.18000000000000002],2,8],[[0.05000000000000001,0,0.17],2,8]]},{"start":[0.1,0,0.1],"delta":[0,0,0.04],"changed":true,"gripper_steps":6,"phases":[[[0.1,0,0.1],6,6],[[0.1,0,0.12000000000000001],2,8],[[0.1,0,0.14],2,8]]},{"start":[-0.4,0,0.08],"delta":[0,0,-0.04],"changed":true,"gripper_steps":10,"phases":[[[-0.4,0,0.08],10,10],[[-0.4,0,0.06],2,8],[[-0.4,0,0.04],2,8]]},{"start":[0.1234,-0.0567,0.0891],"delta":[0.1,-0.07,0.13],"changed":false,"gripper_steps":6,"phases":[[[0.1345111111111111,-0.06447777777777777,0.10354444444444444],2,8],[[0.1456222222222222,-0.07225555555555556,0.1179888888888889],2,8],[[0.15673333333333334,-0.08003333333333333,0.13243333333333335],2,8],[[0.16784444444444444,-0.08781111111111112,0.14687777777777777],2,8],[[0.17895555555555553,-0.09558888888888889,0.16132222222222223],2,8],[[0.19006666666666666,-0.10336666666666666,0.17576666666666668],2,8],[[0.2011777777777778,-0.11114444444444445,0.1902111111111111],2,8],[[0.2122888888888889,-0.11892222222222223,0.20465555555555556],2,8],[[0.2234,-0.1267,0.21910000000000002],2,8]]},{"start":[0,0.12,0.18],"delta":[0,0.02,-0.02],"changed":true,"gripper_steps":6,"phases":[[[0,0.12,0.18],6,6],[[0,0.13,0.16999999999999998],2,8],[[0,0.13999999999999999,0.16],2,8]]},{"start":[0.3,0.1,0.05],"delta":[0,0,0],"changed":false,"gripper_steps":0,"phases":[[[0.3,0.1,0.05],2,2]]},{"start":[0.3,0.1,0.05],"delta":[0,0,0],"changed":true,"gripper_steps":4,"phases":[[[0.3,0.1,0.05],4,4]]},{"start":[0.2,0.2,0.2],"delta":[0.02,0,0],"changed":false,"gripper_steps":6,"phases":[[[0.22,0.2,0.2],2,8]]},{"start":[0.2,0.2,0.2],"delta":[0.06,0,0],"changed":false,"gripper_steps":0,"phases":[[[0.22,0.2,0.2],2,8],[[0.24000000000000002,0.2,0.2],2,8],[[0.26,0.2,0.2],2,8]]}]"""
)


def test_the_server_split_equals_the_old_typescript_split_bit_for_bit():
    for case in TS_PHASES:
        got = ms.phases(
            case["start"], case["delta"], case["changed"], case["gripper_steps"]
        )
        assert [[t, lo, hi] for t, lo, hi in got] == case["phases"], case


def test_the_old_typescript_waypoint_cases_hold():
    w = ms.waypoints([0, 0, 0.2], [0.05, 0, -0.03])
    assert len(w) == 3  # |delta| 5.8 cm -> 3 decisions
    assert w[2] == pytest.approx([0.05, 0, 0.17]) and w[0] == pytest.approx(
        [0.05 / 3, 0, 0.19]
    )
    assert ms.waypoints([0, 0, 0.2], [0, 0, 0]) == [[0, 0, 0.2]]
    # Each 2 cm unit is one decision on every arm.
    assert len(ms.waypoints([0, 0, 0.2], [0, 0, ms.STEP_M])) == 1
    assert ms.GAIN == 0.026 / 0.02
    assert {k: r.gripper_steps for k, r in ms.ROBOTS.items()} == {
        "panda": 6,
        "xarm6_robotiq": 6,
        "widowxai": 10,
        "panda_stick": 0,
        "panda_pair": 6,
        "widowx250s": 4,
    }


def recording(f):
    """Record every servo leg move_delta runs: (target, gripper, min_steps, max_steps, gain, tol)."""
    legs = []
    servo = f.servo

    def spy(target, gripper, **kw):
        legs.append(
            (
                list(target),
                gripper,
                kw["min_steps"],
                kw["max_steps"],
                kw["gain"],
                kw["tol_m"],
            )
        )
        return servo(target, gripper, **kw)

    f.servo = spy
    return legs


def test_move_delta_runs_the_old_tool_legs_with_the_tool_defaults():
    f = facade()
    legs = recording(f)
    start = [float(v) for v in f._env.tcp]
    out = f.move_delta([0, 0, 0.04], gripper="close")
    expect = ms.phases(start, [0.0, 0.0, 0.04], True, 6)
    assert [(t, lo, hi) for t, _g, lo, hi, _k, _tol in legs] == expect
    assert all(
        g == -1.0 and k == ms.GAIN and tol == 0.002 for _t, g, _lo, _hi, k, tol in legs
    )
    r = out["result"]
    assert r["commanded_m"] == [0, 0, 0.04] and r["gripper"] == "close"
    assert r["moved_m"][2] == pytest.approx(0.04, abs=0.003)
    assert r["env_steps"] == f._steps == sum(1 for x in out["frames"] if "action" in x)
    assert out["obs"] is out["frames"][-1]
    # The same command again holds no gripper leg; a move-less call without a change holds one decision.
    legs.clear()
    f.move_delta([0, 0, 0], gripper="close")
    assert [(lo, hi) for _t, _g, lo, hi, _k, _tol in legs] == [(2, 2)]
    legs.clear()
    f.move_delta([0, 0, 0], gripper="open")
    assert [(lo, hi, g) for _t, g, lo, hi, _k, _tol in legs] == [(6, 6, 1.0)]


def test_the_gripper_hold_is_the_robots_own():
    f = facade()
    f._robot = ms.ROBOTS["widowxai"]
    legs = recording(f)
    f.move_delta([0, 0, -0.04], gripper="close")
    assert [(lo, hi, g) for _t, g, lo, hi, _k, _tol in legs] == [
        (10, 10, -1.0),
        (2, 8, -1.0),
        (2, 8, -1.0),
    ]


def test_move_delta_refuses_what_the_tool_refused():
    f = facade()
    with pytest.raises(
        ValueError, match="the limit is 0.2 m per call. Split the motion"
    ):
        f.move_delta([0.3, 0, 0])
    with pytest.raises(ValueError, match="one arm; leave .arm. out"):
        f.move_delta([0, 0, 0.02], arm="left")
    f._robot = ms.ROBOTS["panda_stick"]
    with pytest.raises(ValueError, match="has no gripper"):
        f.move_delta([0, 0, -0.02], gripper="close")
    f._robot = ms.ROBOTS["panda_pair"]
    with pytest.raises(ValueError, match="arm must be one of left, right"):
        f.move_delta([0, 0, -0.02])
    assert f._steps == 0


def test_success_latches_and_the_next_call_refuses_without_stepping():
    f = facade()
    legs = recording(f)
    # The fake env succeeds above z = 0.2: the first 2 cm leg crosses it, the rest never run.
    f._env.tcp[2] = 0.195
    out = f.move_delta([0, 0, 0.1])
    assert len(legs) == 1 and f._success_once
    assert out["result"]["env_steps"] == f._steps
    steps = f._steps
    again = f.move_delta([0, 0, -0.02])
    assert again["result"] == {"error": "the task is already solved; call finish"}
    assert again["frames"] == [] and f._steps == steps and len(legs) == 1


def test_stop_cancels_between_steps_and_ends_the_call():
    f = facade()
    legs = recording(f)
    f.stop_requested = lambda: True
    out = f.move_delta([0, 0, 0.06])
    assert len(legs) == 1, "a cancelled leg ends the call"
    assert out["result"]["cancelled"] is True and out["result"]["env_steps"] == 0
    assert f._steps == 0


def test_a_spent_step_budget_ends_the_move_and_says_to_finish(monkeypatch):
    f = facade()
    legs = recording(f)
    monkeypatch.setitem(ms.DOT_LIMIT, "PickCube-v1", 5)
    f._env.unwrapped.draw_step = 5
    out = f.move_delta([0.06, 0, 0])
    assert len(legs) == 1
    assert "step limit is reached" in out["result"]["step_limit"]
