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

"""Code mode on the Genesis server (``code.run`` over its manifest), without a simulator: the
facade's own motion methods (move_delta, set_gripper, step, chunk_step, state) run over a fake
servo, so a program reaches exactly what the tools reach."""

from __future__ import annotations

import numpy as np
import pytest

from pi_embodied_services.robots.genesis import env_server as g
from pi_embodied_services.utils.rpc import RpcFacade


def facade(ready: bool = True) -> g.GenesisEnvFacade:
    """A registered Genesis facade whose servo moves the TCP to its target in 5 control steps;
    the cube rises with the TCP once the gripper closed on it."""
    f = object.__new__(g.GenesisEnvFacade)
    RpcFacade.__init__(f)
    f._task, f._seed, f._view_size = "cube_pick", 0, 4
    f._steps, f._success, f._hold, f._gripper_open = 0, False, 0, True
    f._offset = np.zeros(3)
    f._rule, f._sam3, f._reach, f._grasp = (
        "grasp",
        g.sam3_segment.Sam3(None),
        None,
        None,
    )
    f._meta = {"instruction": g.TASK_TEXT[("cube_pick", "grasp")]}
    f._cmd_tcp, f._record = None, None
    f._run_start, f._run_frames = 0, []
    f._hand_yaw = lambda: 0.0  # the hand points down at yaw 0 here
    f.pos = np.array([0.5, 0.0, 0.2])
    f.cube_z = g.CUBE_SIZE_M / 2
    f.target = f.pos.copy()

    def servo(target, max_steps=g.SERVO["max_steps"]):
        f.target = np.asarray(target, dtype=np.float64)
        for _ in range(5):
            f._step()
        return 5, False

    def step():
        if not f._gripper_open and f.pos[2] < 0.05:
            f.grasped = True
        if getattr(f, "grasped", False):
            f.cube_z += f.target[2] - f.pos[2]
        f.pos = f.target.copy()
        f._steps += 1
        f._success = f._success or g.lifted(f.cube_z)

    f._tcp = lambda: f.pos.copy()
    f._servo = servo
    f._step = step
    f._grip = lambda: (step(), (1, False))[1]
    f._command_arm = lambda target: setattr(f, "target", np.asarray(target))
    f._command_gripper = lambda: None
    f._width = lambda: 0.04 if f._gripper_open else 0.03
    f._render = lambda name, depth=False: np.full(
        (4, 4, 3), 1 if name == "agentview" else 2, np.uint8
    )
    f._state = lambda: {
        "tcp_pos": f.pos.astype(np.float32),
        "gripper_width": f._width(),
        "gripper_command": "open" if f._gripper_open else "close",
        "success": bool(f._success),
        "is_grasped": getattr(f, "grasped", False),
        "lift_m": round(max(0.0, f.cube_z - g.CUBE_SIZE_M / 2), 4),
        "env_steps": f._steps,
    }
    g.GenesisEnvFacade._register_rpc(f)
    if ready:
        f._manifest_ready()
    return f


def test_the_server_serves_code_run_with_a_token_and_exclusivity():
    f = facade()
    assert g.GenesisEnvFacade.REQUIRE_TOKEN and f._rpc_token
    assert "code.run" in f._rpc and "code.helpers" in f._rpc
    with pytest.raises(PermissionError):
        f._admit("env.state", None)
    f._code._active = True
    with pytest.raises(RuntimeError, match="run_code program is running"):
        f._admit("env.state", f._rpc_token)


def test_a_run_moves_through_the_tools_methods_and_reports_steps_success_obs_frames():
    f = facade()
    out = f._rpc["code.run"](
        "move_delta([0, 0, -0.18], gripper='open')\n"
        "set_gripper(True)\n"
        "r = move_delta([0, 0, 0.15])\n"
        "RESULT = sorted(r)\n",
        timeout_s=30,
        tier="low",
    )
    assert out["status"] == "ran", out
    assert out["success"] is True and out["steps"] == f._steps > 0
    for hidden in ("agentview", "wrist", "lift_m", "frames"):
        assert hidden not in out["result"], hidden
    assert {"tcp_pos", "moved_m", "is_grasped", "success"} <= set(out["result"])
    # The new observation in the shape the tools return it (images and the full state).
    assert out["obs"]["agentview"].shape == (4, 4, 3) and "lift_m" in out["obs"]
    assert out["obs"]["env_steps"] == f._steps
    # One run-video frame per primitive call: the two views side by side.
    assert len(out["frames"]) == 3 and out["frames"][0].shape == (4, 8, 3)
    assert [c.get("move_m", 0) for c in out["calls"]] == pytest.approx(
        [0.18, 0.0, 0.15]
    )


def test_the_low_tier_step_and_state_carry_no_object_state():
    f = facade()
    out = f._rpc["code.run"](
        "a = step([0, 0, 0.01, 1])\n"
        "c = chunk_step([[0, 0, 0.01, 1]] * 3)\n"
        "RESULT = [sorted(a), sorted(a['state']), sorted(c)]\n",
        timeout_s=30,
        tier="raw",
    )
    assert out["status"] == "ran", out
    s = f._rpc["code.run"]("RESULT = sorted(state())\n", timeout_s=30, tier="low")
    out["result"].append(s["result"])
    step, step_state, chunk, state = out["result"]
    assert step == ["reward", "state", "success", "truncated"]
    assert "chunk" not in chunk and "state" in chunk and "success" in chunk
    for keys in (step_state, state):
        assert "lift_m" not in keys and "agentview" not in keys and "tcp_pos" in keys
    assert out["calls"][1].get("move_m") == pytest.approx(0.03)
    assert f._steps == 4 and len(out["frames"]) == 2


def test_oversized_chunks_and_moves_past_the_cap_are_refused_before_they_run():
    f = facade()
    out = f._rpc["code.run"](
        f"chunk_step([[0, 0, 0, 1]] * {g.CODE_MAX_CHUNK})\n",
        timeout_s=30,
        tier="raw",
    )
    assert out["status"] == "ran", out
    out = f._rpc["code.run"](
        f"chunk_step([[0, 0, 0, 1]] * {g.CODE_MAX_CHUNK + 1})\n",
        timeout_s=30,
        tier="raw",
    )
    assert out["status"] == "error" and "200" in out["error"], out
    out = f._rpc["code.run"](
        "move_delta([0, 0, 0.2])\nmove_delta([0, 0, 0.2])\n",
        timeout_s=30,
        tier="low",
        max_move_m=0.3,
    )
    assert out["status"] == "error" and out["limit"] == "max_move_m", out
    assert f.pos[2] == pytest.approx(0.4), "only the first move ran"


def test_the_run_video_is_bounded_and_s4_drops_the_examples():
    f = facade()
    for _ in range(g.CODE_MAX_FRAMES + 5):
        f._keep_frame({"agentview": np.zeros((1, 1, 3)), "wrist": np.zeros((1, 1, 3))})
    assert len(f._run_frames) <= g.CODE_MAX_FRAMES
    low = {p["name"]: p["doc"] for p in f._code.api("low")}
    raw = {p["name"]: p["doc"] for p in f._code.api("raw")}
    assert all("Example:" in low[n] for n in ("state", "move_delta", "set_gripper"))
    assert all("Example:" in raw[n] for n in ("step", "chunk_step"))
    assert all("Example:" not in p["doc"] for p in f._code.api("low-noexamples"))
    # The joint-space primitives (Genesis's own IK: no --ik needed); segment needs --sam3.
    api = f._rpc["code.api"]("low")["available"]
    assert {"solve_ik", "move_to_joints", "traj_plan", "move_along_trajectory"} <= set(
        api
    )
    assert "segment" not in api and "execute_grasp" not in api


def test_the_server_checks_itself_against_its_manifest():
    """A facade method the manifest does not declare (nor lists as internal) stops the server."""
    from pi_embodied_services.components.manifest import ManifestError

    facade()
    g2 = facade(ready=False)
    g2._rpc["env.secret_teleport"] = lambda: None
    with pytest.raises(ManifestError, match="env.secret_teleport"):
        g2._manifest_ready()
    g3 = facade(ready=False)
    del g3._rpc["env.back_project"]
    with pytest.raises(ManifestError, match="back_project"):
        g3._manifest_ready()


def test_with_sam3_the_server_serves_segment():
    """pi passes --sam3 by default: the manifest's segment then needs env.segment served."""
    f = facade(ready=False)
    from pi_embodied_services.components.manifest import ManifestError

    f._sam3 = g.sam3_segment.Sam3("http://127.0.0.1:1")
    assert "env.segment" in f._rpc
    # The detection ids come from install_perception in main(); only they may be missing here.
    with pytest.raises(ManifestError) as e:
        f._manifest_ready()
    assert "env.detect" in str(e.value) and "env.segment" not in str(e.value)


def test_both_success_rules():
    """grasp (default): OpenETA's both-finger contact, fingers not fully open, EEF within 8 cm;
    lift: the cube's bottom 8 cm up."""
    assert g.upstream_grasped((True, True), [0.02, 0.02], 0.03)
    assert not g.upstream_grasped((True, False), [0.02, 0.02], 0.03)
    assert not g.upstream_grasped((True, True), [0.04, 0.02], 0.03), "fully open"
    assert not g.upstream_grasped((True, True), [0.02, 0.02], 0.08), "too far"
    assert g.finger_contacts({"link_a": [3, 11], "link_b": [12, 2]}, (11, 12)) == (
        True,
        True,
    )
    assert g.finger_contacts({"link_a": [3], "link_b": [12]}, (11, 12)) == (False, True)
    assert g.lifted(g.CUBE_SIZE_M / 2 + 0.08) and not g.lifted(0.05)
    assert g.SUCCESS_RULES == ("grasp", "lift")
    assert g.TASK_TEXT[("cube_pick", "lift")].endswith("and lift it.")
    assert "lift" not in g.TASK_TEXT[("cube_pick", "grasp")]


def test_the_success_latch_follows_the_rule_and_its_hold():
    """_step latches success after 3 grasp steps (grasp) or 5 lifted steps (lift)."""
    for rule, need in (("grasp", g.GRASP_HOLD_STEPS), ("lift", g.SUCCESS_HOLD_STEPS)):
        f = object.__new__(g.GenesisEnvFacade)
        f._rule, f._record, f._hold, f._success, f._steps = rule, None, 0, False, 0
        f._scene = type("S", (), {"step": lambda self: None})()
        f._success_instant = lambda: True
        for _ in range(need - 1):
            g.GenesisEnvFacade._step(f)
        assert not f._success, rule
        g.GenesisEnvFacade._step(f)
        assert f._success, rule


def test_a_chain_runs_a_planned_grasp_as_bounded_move_deltas():
    f = facade()
    calls = []

    def resolve_grasp(grasp_id, standoff=0.0):
        return {"approach": [0, 0, -1]}

    def claim_waypoints(grasp_id, standoff=0.1, lift=0.1):
        calls.append(("claim", grasp_id))
        return {
            "kind": "grasp",
            "waypoints": {
                "pre_grasp": [0.5, 0, 0.5],
                "grasp": [0.5, 0, 0.12],
                "lift": [0.5, 0, 0.3],
            },
            "steps": [
                {"to": "pre_grasp", "gripper": -1},
                {"to": "grasp", "gripper": -1},
                {"gripper": 1},
                {"to": "lift", "gripper": 1},
            ],
            "eef_yaw": 0.0,
            "expired_ids": ["g1"],
        }

    f._grasp = object()
    f._rpc["env.resolve_grasp"] = resolve_grasp
    f._rpc["env.claim_waypoints"] = claim_waypoints
    f.move_delta = lambda d, gripper=None, return_frames=False, record=False: (
        calls.append(("move", list(d), gripper)),
        setattr(f, "pos", f.pos + np.asarray(d)),
        {"control_steps": 5, "frames": ["x"]},
    )[2]
    f.set_gripper = lambda close, return_frames=False, record=False: (
        calls.append(("grip", close)),
        {"control_steps": 1},
    )[1]
    out = f.execute_grasp("g1")
    moves = [c for c in calls if c[0] == "move"]
    assert all(np.linalg.norm(m[1]) <= g.MAX_MOVE_M + 1e-9 for m in moves)
    assert [c[0] for c in calls].count("grip") == 1 and ("grip", True) in calls
    assert out["legs"][-1]["to"] == "lift" and "stalled" not in out
    assert out["control_steps"] == 5 * len(moves) + 1 and out["frames"] == ["x"] * len(
        moves
    )
    assert f.pos[2] == pytest.approx(0.3)
    # A tilted candidate is refused before anything moves.
    f._rpc["env.resolve_grasp"] = lambda grasp_id, standoff=0.0: {"approach": [1, 0, 0]}
    n = len(calls)
    out = f.execute_grasp("g2")
    assert out["refused"] and len(calls) == n
