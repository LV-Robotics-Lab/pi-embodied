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
    CODE_MAX_TRAJECTORY,
    HOME_STEP_RAD,
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
    facade._manifest_ready()
    # Kit is not thread-safe: the primitives run on the server's own thread.
    assert facade._code._primitive_thread is False


def test_a_run_reports_steps_success_obs_and_frames_and_hides_images_and_score(
    facade,
):
    facade._env.solve_when = lambda rm: rm.q["left_arm"][2] > 0.25
    out = facade._rpc["code.run"](
        "r = move_delta('left', [0, 0, 0.1])\nRESULT = [r, state()]\n",
        timeout_s=30,
        tier="low",
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
        tier="raw",
    )
    assert out["status"] == "ran", out
    assert set(out["result"]) == {"state", "terminated", "truncated", "info"}
    assert (
        "head" not in out["result"]["state"] and "score" not in out["result"]["state"]
    )
    assert out["move_m"] == pytest.approx(0.1 * env_server.CODE_M_PER_RAD)
    too_many = [{"right_ee_joint_state": [1.0]}] * (CODE_MAX_CHUNK + 1)
    out = facade._rpc["code.run"](f"chunk_step({too_many})\n", timeout_s=30, tier="raw")
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


def test_the_server_checks_itself_against_its_manifest(facade):
    """A facade method the manifest does not declare (nor lists as internal) stops the server;
    so does a declared one it does not serve."""
    from pi_embodied_services.components.manifest import ManifestError

    facade._manifest_ready()
    assert facade.manifest["robot"] == "robodojo"
    assert "env.locate" not in facade._rpc, "the tool locate is back_project"
    facade._manifest_done = False
    facade._rpc["env.secret_teleport"] = lambda: None
    with pytest.raises(ManifestError, match="env.secret_teleport"):
        facade._manifest_ready()
    del facade._rpc["env.secret_teleport"]
    del facade._rpc["env.traj_plan"]
    with pytest.raises(ManifestError, match="traj_plan"):
        facade._manifest_ready()


def test_the_tiers(facade):
    low = facade._rpc["code.api"]("low")["available"]
    for name in ("solve_ik", "move_to_joints", "traj_plan", "move_along_trajectory"):
        assert name in low, name
    assert "back_project" in low and "step" not in low
    assert set(facade._rpc["code.api"]("raw")["available"]) == {
        "get_obs",
        "render_camera",
        "step",
        "chunk_step",
    }
    docs = {p["name"]: p["doc"] for p in facade._code.api("low")}
    assert all("Example:" in d for n, d in docs.items() if n != "get_task_language")


# -- CaP-X's joint-space primitives -------------------------------------------------------


def test_solve_ik_answers_the_joints_and_moves_nothing(facade):
    rm = facade._env.robot_manager
    before = rm.q["left_arm"].copy()
    q = facade.solve_ik([-0.3, -0.25, 0.9], [1.0, 0, 0, 0], "left")
    assert len(q) == 6 and q[:3] == pytest.approx([0.0, 0.2, 0.135])
    assert np.array_equal(rm.q["left_arm"], before) and facade._env.actions == []
    with pytest.raises(ValueError, match="no IK solution"):
        facade.solve_ik([-0.3, 0.6, 0.9], [1.0, 0, 0, 0], "left")
    with pytest.raises(ValueError, match="arm"):
        facade.solve_ik([-0.3, -0.25, 0.9], [1.0, 0, 0, 0], "middle")


def test_move_to_joints_interpolates_and_the_other_arm_holds(facade):
    rm = facade._env.robot_manager
    right = rm.q["right_arm"].copy()
    target = [0.0, 0.3, 0.3, 0.0, 0.0, 0.0]  # 0.1 rad on joint 2
    r = facade.move_to_joints(target, "left")
    n = int(np.ceil(0.1 / HOME_STEP_RAD - 1e-9))
    assert r["control_steps"] == n and r["executed"] == 1
    assert r["final_joint_error_rad"] == pytest.approx(0.0)
    assert rm.q["left_arm"] == pytest.approx(target)
    assert np.array_equal(rm.q["right_arm"], right)
    steps = [np.asarray(a["left_arm_joint_state"]) for a in facade._env.actions]
    assert (
        max(np.max(np.abs(b - a)) for a, b in zip(steps, steps[1:]))
        <= HOME_STEP_RAD + 1e-9
    )
    with pytest.raises(ValueError, match="6 finite"):
        facade.move_to_joints([0.0] * 5, "left")
    with pytest.raises(ValueError, match="6 finite"):
        facade.move_to_joints([0.0, float("nan"), 0, 0, 0, 0], "left")


def test_joint_motions_stop_on_a_stop_and_refuse_after_the_end(facade):
    calls = iter([False, True])
    facade.stop_requested = lambda: next(calls, True)
    r = facade.move_to_joints([0.0, 0.3, 0.5, 0.0, 0.0, 0.0], "left")
    assert r["cancelled"] is True and r["control_steps"] == 1 and r["executed"] == 0
    facade.stop_requested = lambda: False
    facade._env.end_flag[0] = True
    n = len(facade._env.actions)
    r = facade.move_along_trajectory([[0.0, 0.3, 0.2, 0.0, 0.0, 0.0]], "left")
    assert r["error"] == "the episode is over" and len(facade._env.actions) == n


def test_traj_plan_starts_at_the_current_pose_and_moves_nothing(facade):
    here = facade._ee_pose("left")
    start = list(here[3:]) + list(here[:3])
    end = list(here[3:]) + list(here[:3] + [0.0, 0.0, 0.05])
    traj = facade.traj_plan(start, end, "left")
    assert len(traj) == 5 and all(len(q) == 6 for q in traj)
    assert traj[-1][:3] == pytest.approx([0.0, 0.3, 0.25])
    assert facade._env.actions == []
    far = list(here[3:]) + list(here[:3] + [0.05, 0.0, 0.0])
    with pytest.raises(ValueError, match="current pose"):
        facade.traj_plan(far, end, "left")
    too_far = list(here[3:]) + list(here[:3] + [0.0, 0.0, 2.0])
    with pytest.raises(ValueError, match="at most"):
        facade.traj_plan(start, too_far, "left")


def test_traj_plan_uses_robodojos_curobo_planner_when_there_is_one(facade):
    rm = facade._env.robot_manager
    for r in rm.robot_list:
        r.robot_name = r.arm_name
        r.entity_origin_pose = [0.0] * 7
    seen = {}

    class Planner:
        def plan_path(self, curr_joint_pos, target_ee_pose, real_robot_pose):
            seen["args"] = (list(curr_joint_pos), list(target_ee_pose))
            n = 250
            pos = np.linspace(curr_joint_pos, np.r_[0.0, 0.3, 0.3, 0, 0, 0], n)
            return {
                "status": "Success",
                "position": pos,
                "velocity": np.zeros_like(pos),
            }

    rm.planner = {"left_arm": Planner()}
    here = facade._ee_pose("left")
    start = list(here[3:]) + list(here[:3])
    traj = facade.traj_plan(start, [1.0, 0, 0, 0, -0.3, -0.15, 1.065], "left")
    assert len(traj) == CODE_MAX_TRAJECTORY and traj[-1][:3] == pytest.approx(
        [0, 0.3, 0.3]
    )
    assert seen["args"][1] == pytest.approx([-0.3, -0.15, 1.065, 1.0, 0, 0, 0])
    rm.planner = {
        "left_arm": SimpleNamespace(plan_path=lambda *a, **k: {"status": "Fail"})
    }
    with pytest.raises(ValueError, match="no path"):
        facade.traj_plan(start, [1.0, 0, 0, 0, -0.3, -0.15, 1.065], "left")


def test_move_along_trajectory_runs_the_waypoints_in_order(facade):
    rm = facade._env.robot_manager
    traj = [[0.0, 0.3, 0.2 + 0.02 * k, 0.0, 0.0, 0.0] for k in range(1, 6)]
    traj.append([0.0, 0.3, 0.4, 0.0, 0.0, 0.0])  # 0.1 rad on: interpolated in two steps
    r = facade.move_along_trajectory(traj, "left", gripper=0.0)
    assert r["executed"] == 6 and r["waypoints"] == 6
    assert r["control_steps"] == env_server.GRIPPER_STEPS + 5 + 2
    assert rm.q["left_arm"] == pytest.approx(traj[-1])
    assert rm.grip["left_arm"] == pytest.approx(0.0)
    with pytest.raises(ValueError, match="1..100"):
        facade.move_along_trajectory([], "left")


def test_a_program_plans_and_follows_a_trajectory_under_the_caps(facade):
    out = facade._rpc["code.run"](
        "st = state()['arms']['left']\n"
        "start = list(st['eef_quat_wxyz']) + list(st['eef_pos'])\n"
        "end = start[:4] + [start[4], start[5], start[6] + 0.05]\n"
        "traj = traj_plan(start, end, 'left')\n"
        "r = move_along_trajectory(traj, 'left')\n"
        "q = solve_ik(end[4:], end[:4], 'left')\n"
        "RESULT = [len(traj), r['executed'], sorted(r), q]\n",
        timeout_s=30,
        tier="low",
    )
    assert out["status"] == "ran", out
    n, executed, keys, q = out["result"]
    assert n == executed == 5 and "head" not in keys and "final_joint_error_rad" in keys
    assert out["calls"][-2]["move_m"] == pytest.approx(0.05 * env_server.CODE_M_PER_RAD)
    too_long = [[0.0, 0.3, 0.2, 0.0, 0.0, 0.0]] * (CODE_MAX_TRAJECTORY + 1)
    out = facade._rpc["code.run"](
        f"move_along_trajectory({too_long}, 'left')\n", timeout_s=30, tier="low"
    )
    assert out["status"] == "error" and "at most" in str(out["error"]), out
    out = facade._rpc["code.run"](
        "move_to_joints([0.0, 0.3, 1.2, 0, 0, 0], 'left')\n",
        timeout_s=30,
        tier="low",
        max_move_m=0.3,
    )
    assert out["status"] == "error" and out["limit"] == "max_move_m", out
