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

"""The robosuite facade's success judgement, limits and code.api, without a simulator."""

from __future__ import annotations

import numpy as np
import pytest

from pi_embodied_services.robots.robosuite import tasks
from pi_embodied_services.robots.robosuite.env_server import (
    ARM_DIM,
    CLOSE,
    OPEN,
    RobosuiteEnvFacade,
)


class FakeSim:
    """Just enough of robosuite's ``sim`` for the success checks: body positions by id."""

    def __init__(self, positions: dict[int, list[float]]):
        self.data = type("Data", (), {})()
        n = max(positions) + 1
        self.data.body_xpos = np.zeros((n, 3))
        for i, p in positions.items():
            self.data.body_xpos[i] = p


class FakeEnv:
    def __init__(self, *, success: bool, sim=None, action_dim=7):
        self._ok = success
        self.sim = sim
        self.action_dim = action_dim
        self.table_offset = np.array([0.0, 0.0, 0.8])
        self.cubeA_body_id, self.cubeB_body_id = 1, 2

    def _check_success(self):
        return self._ok


def facade(task: str, env: FakeEnv) -> RobosuiteEnvFacade:
    """A facade over ``env`` for ``task``, bypassing robosuite (``__init__`` builds the sim)."""
    f = object.__new__(RobosuiteEnvFacade)
    f._task_name = task
    f._task = tasks.TASKS[task]
    f._env = env
    f._grip = {arm: OPEN for arm in f._task.arms}
    f._steps = 0
    f._success_step = None
    return f


# ---- success per task ------------------------------------------------------------------


@pytest.mark.parametrize(
    "task",
    ["Lift", "Stack", "Wipe", "NutAssemblySquare", "TwoArmLift", "TwoArmHandover"],
)
def test_success_is_robosuites_check_success(task):
    assert facade(task, FakeEnv(success=True))._success() is True
    assert facade(task, FakeEnv(success=False))._success() is False


def test_restack_success_needs_the_stack_check_and_a_cube_near_the_table():
    table = 0.8
    stacked_on_table = FakeSim({1: [0, 0, table + 0.02], 2: [0, 0, table + 0.06]})
    carried_pile = FakeSim({1: [0, 0, table + 0.12], 2: [0, 0, table + 0.16]})
    assert (
        facade("Restack", FakeEnv(success=True, sim=stacked_on_table))._success()
        is True
    )
    # robosuite says stacked while the arm holds the whole pile in the air: not a success.
    assert (
        facade("Restack", FakeEnv(success=True, sim=carried_pile))._success() is False
    )
    # Without robosuite's check nothing counts, whatever the heights.
    assert (
        facade("Restack", FakeEnv(success=False, sim=stacked_on_table))._success()
        is False
    )
    # Exactly one cube above the threshold is fine (the other rests on the table).
    one_up = FakeSim({1: [0, 0, table + 0.02], 2: [0, 0, table + 0.2]})
    assert facade("Restack", FakeEnv(success=True, sim=one_up))._success() is True


def test_restack_rule_threshold():
    assert tasks.restack_success(True, (0.03, 0.03))
    assert tasks.restack_success(True, (0.041, 0.04))
    assert not tasks.restack_success(True, (0.041, 0.041))
    assert not tasks.restack_success(False, (0.0, 0.0))


def test_success_is_latched_at_its_first_step():
    assert tasks.latch(None, False, 3) is None
    assert tasks.latch(None, True, 4) == 4
    assert tasks.latch(4, False, 9) == 4
    assert tasks.latch(4, True, 9) == 4


def test_restack_scene_is_two_4cm_cubes_with_the_green_on_top():
    t = tasks.TASKS["Restack"]
    assert t.env == "Stack" and t.restack
    assert tasks.RESTACK_CUBE_HALF == 0.02 and tasks.RESTACK_OFF_TABLE_M == 0.04


# ---- limits ----------------------------------------------------------------------------


def test_check_move_refuses_long_moves_the_box_and_the_floor():
    kw = dict(max_move_m=0.3, box=(-0.4, 0.4, -0.4, 0.4), z_floor=0.805, z_ceiling=1.4)
    tasks.check_move((0, 0, 1.0), (0.1, 0.1, 0.9), **kw)
    with pytest.raises(ValueError, match="0.4.* m; the limit is 0.3"):
        tasks.check_move((0, 0, 1.0), (0.4, 0, 1.0), **kw)
    with pytest.raises(ValueError, match="outside the workspace"):
        tasks.check_move((0.3, 0, 1.0), (0.45, 0, 1.0), **kw)
    with pytest.raises(ValueError, match="below the floor"):
        tasks.check_move((0, 0, 0.9), (0, 0, 0.8), **kw)
    with pytest.raises(ValueError, match="above the ceiling"):
        tasks.check_move((0, 0, 1.3), (0, 0, 1.5), **kw)
    # NaN never passes.
    with pytest.raises(ValueError):
        tasks.check_move((0, 0, 1.0), (float("nan"), 0, 1.0), **kw)


def test_two_arm_actions_hold_the_other_arm_and_each_gripper_command():
    f = facade("TwoArmLift", FakeEnv(success=False, action_dim=14))
    f._grip["robot1"] = CLOSE
    a = f._action({0: np.ones(ARM_DIM)})
    assert a.shape == (14,)
    assert a[:6].tolist() == [1] * 6 and a[6] == OPEN
    assert a[7:13].tolist() == [0] * 6 and a[13] == CLOSE


def test_wipe_has_no_gripper_in_its_action():
    f = facade("Wipe", FakeEnv(success=False, action_dim=6))
    assert f._action({}).shape == (6,)
    with pytest.raises(ValueError, match="no fingers"):
        f._set_grip("robot0", "close")


def test_arm_parameter_is_required_on_two_arms_and_refused_when_unknown():
    two = facade("TwoArmHandover", FakeEnv(success=False, action_dim=14))
    assert two._arm_index("robot1") == 1
    with pytest.raises(ValueError, match="two arms"):
        two._arm_index(None)
    one = facade("Lift", FakeEnv(success=False))
    assert one._arm_index(None) == 0
    with pytest.raises(ValueError, match="unknown arm"):
        one._arm_index("robot1")


# ---- the primitive registry (code.api) -----------------------------------------------------


def test_code_api_tiers_follow_capx_and_the_privileged_ground_truth():
    """CaP-X's tiers on this server: high (S2) = perception + pose-level motion, low (S3) = raw
    observation + relative moves, privileged (S1) = high + the simulator's poses."""
    from pi_embodied_services.components.code_api import CodeApi
    from pi_embodied_services.robots.robosuite.primitives import ROBOSUITE_PRIMITIVES

    f = facade("Lift", FakeEnv(success=False))
    f._rpc, f._readonly_methods = {}, set()
    f._grasp = None
    RobosuiteEnvFacade._register_rpc(f)
    api = f.code_api
    assert isinstance(api, CodeApi) and f._rpc["code.api"]("high")["primitives"]

    def names(tier):
        return [p.name for p in api.primitives(tier)]

    assert names("high") == [
        "get_task_language",
        "get_state",
        "get_observation",
        "segment",
        "back_project",
        "preview_reach",
        "move_to",
        "set_gripper",
    ]
    assert names("low") == [
        "get_task_language",
        "get_state",
        "get_observation",
        "move_delta",
        "set_gripper",
        "raw_obs",
        "render_camera",
        "get_camera_meta",
        "step",
    ]
    assert names("privileged") == names("high") + ["ground_truth_poses"]
    assert "solve_ik" not in names("low"), "OSC_POSE servo, not CaP-X's joint-space IK"
    # Every motion primitive takes `arm` (the two-arm tasks), and only the movers are mutating.
    by_name = {p.name: p for p in ROBOSUITE_PRIMITIVES}
    for n in ("move_to", "move_delta", "set_gripper", "preview_reach"):
        assert "arm" in by_name[n].params, n
    assert {p.name for p in ROBOSUITE_PRIMITIVES if p.mutating} == {
        "move_to",
        "move_delta",
        "set_gripper",
        "step",
    }
    method, kw = api.resolve("move_to", {"target_xyz": [0, 0, 1], "arm": "robot0"})
    assert method == "env.move_to" and kw["arm"] == "robot0"
    with pytest.raises(ValueError, match="unknown parameter"):
        api.resolve("move_to", {"target_xyz": [0, 0, 1], "joints": [0] * 7})
