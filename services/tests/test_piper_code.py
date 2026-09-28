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

"""pi's limits and code mode on the Piper server (``--max-move`` / ``--max-yaw``, ``--code``,
utils/code_real.py; the primitives of manifests/piper.json) against the mocked ROS arm of
test_piper.py: pi's tools and programs call the same guarded methods under the same limits."""

from __future__ import annotations

import numpy as np
import pytest
from test_piper import HOME, FakeArm, FakeCamera, controller, dual_facade

from pi_embodied_services.components.manifest import ManifestError
from pi_embodied_services.robots.piper.env_server import PiperEnvFacade


def cfg(**extra):
    return {
        "arm": "left",
        "cameras": {"front": "/f", "wrist": "/w", "image_size": 32},
        "calibration": {
            "z_floor_m": 0.0,
            "begin_joints": HOME.tolist(),
            "arm_id": "usb-A",
        },
        "limits": {"speed_mps": 0.3, "reset_time_s": 0.1},
        "gripper": {"settle_s": 0.0},
        "motion": {"settle_steps": 1, "settle_dt_s": 0.0},
        "smooth": {"dt_s": 0.001},
        **extra,
    }


def facade(code=True, limits=None) -> tuple[PiperEnvFacade, FakeArm]:
    arm = FakeArm()
    f = PiperEnvFacade(
        cfg(),
        arm,
        {"front": FakeCamera(), "wrist": FakeCamera()},
        code=code,
        limits={"max_move_m": 0.03, "max_yaw_rad": 0.1} if limits is None else limits,
    )
    return f, arm


def run(f, program, **kw):
    return f._rpc["code.run"](program, timeout_s=30, **{"tier": "low", **kw})


def test_code_mode_needs_code_and_then_the_token():
    plain, _ = facade(code=False)
    assert "code.run" not in plain._rpc and plain._rpc_token is None
    # Without --code the API an episode ran with is still answered.
    assert "step" in plain._rpc["code.api"]("raw")["available"]
    f, _ = facade()
    assert f._rpc_token and "code.run" in f._rpc and "code.set_limits" not in f._rpc
    with pytest.raises(PermissionError):
        f._serve_dispatch("env.get_robot_state", (), {})


def test_the_server_checks_itself_against_its_manifest():
    f, _ = facade()
    f._manifest_ready()  # one arm: passes
    left, right = FakeArm(), FakeArm()
    dual_facade(left, right)._manifest_ready()  # two arms: passes
    g, _ = facade()
    g._rpc["env.secret_teleport"] = lambda: None
    with pytest.raises(ManifestError, match="env.secret_teleport"):
        g._manifest_ready()
    h, _ = facade()
    del h._rpc["env.rotate_yaw"]
    with pytest.raises(ManifestError, match="rotate_yaw"):
        h._manifest_ready()


def test_the_manifest_tiers_and_capabilities():
    f, _ = facade()
    api = f._rpc["code.api"]
    assert set(api("high")["available"]) == {"open_gripper", "close_gripper"}
    assert api("raw")["available"] == ["step"]
    low = set(api("low")["available"])
    assert {"move_delta", "rotate_yaw", "solve_ik", "move_to_joints"} <= low
    # halt_arm needs two arms; move_joints and step_pair are internal (no program reaches them).
    assert not {"halt_arm", "move_joints", "step_pair"} & low
    left, right = FakeArm(), FakeArm()
    assert "halt_arm" in dual_facade(left, right)._rpc["code.api"]("low")["available"]
    meta = f._rpc["env.get_env_meta"]()
    assert meta["motion_limits"] == {"max_move_m": 0.03, "max_yaw_rad": 0.1}


def test_a_program_steps_the_arm_and_the_run_reports_it():
    f, _ = facade()
    z0 = f._arm_state("left")["eef_pos"][2]
    out = run(f, "r = move_delta([0, 0, 0.02])\nRESULT = r['ok']\n")
    assert out["status"] == "ran" and out["result"] is True, out
    assert f._arm_state("left")["eef_pos"][2] == pytest.approx(z0 + 0.02, abs=2e-3)
    assert out["motions"] == 1 and out["calls"][0]["move_m"] == pytest.approx(0.02)
    assert len(out["frames"]) == 1 and out["frames"][0].shape == (32, 32, 3)


def test_pi_limits_hold_for_programs_and_tools_alike():
    f, arm = facade()
    q0 = np.array(arm.q)
    for program, tier, why in [
        ("step([0.04, 0, 0])", "raw", "limit is 0.03 m per call"),
        ("move_delta([0.04, 0, 0])", "low", "limit is 0.03 m per call"),
        ("step(yaw=0.15)", "raw", "limit of 0.1 rad"),
        ("rotate_yaw(0.15)", "low", "limit of 0.1 rad"),
        ("move_joints('rest')", "low", "move_joints"),
        ("halt_arm(reason='done')", "low", "halt_arm"),
    ]:
        out = run(f, program, tier=tier)
        assert out["status"] == "error" and why in out["error"], (program, out)
    np.testing.assert_allclose(arm.q, q0)
    # pi's tools call the same methods: refused the same way, nothing commanded.
    for method, kwargs, why in [
        ("env.move_delta", {"delta_xyz": [0.04, 0, 0]}, "limit is 0.03 m"),
        ("env.rotate_yaw", {"yaw": 0.15}, "limit of 0.1 rad"),
        ("env.step", {"delta_xyz": [0, 0.035, 0]}, "limit is 0.03 m"),
    ]:
        with pytest.raises(ValueError, match=why):
            f._rpc[method](**kwargs)
    np.testing.assert_allclose(arm.q, q0)
    out = run(f, "for _ in range(4):\n    move_delta([0.02, 0, 0])\n", max_move_m=0.05)
    assert out["status"] == "error" and out["limit"] == "max_move_m", out
    assert out["motions"] == 2


def test_the_config_cap_applies_under_a_looser_pi_limit():
    f, arm = facade(limits={"max_move_m": 0.2, "max_yaw_rad": 0.2})
    with pytest.raises(ValueError, match="0.05 m per call"):
        f._rpc["env.move_delta"](delta_xyz=[0.08, 0, 0])


def test_the_gripper_methods_wrap_the_step():
    f, arm = facade()
    out = f._rpc["env.close_gripper"]()
    assert out["gripper_command"] == "close"
    out = f._rpc["env.open_gripper"]()
    assert out["gripper_command"] == "open" and out["gripper_closed"] is False


def test_dual_limits_refuse_a_paired_step_whole():
    left, right = FakeArm(), FakeArm()
    f = dual_facade(left, right)
    f._set_limits({"max_move_m": 0.03})
    ql, qr = np.array(left.q), np.array(right.q)
    with pytest.raises(ValueError, match="0.03 m per call"):
        f._rpc["env.step_pair"](
            steps=[
                {"arm": "left", "delta_xyz": [0, 0, 0.01]},
                {"arm": "right", "delta_xyz": [0, 0, 0.04]},
            ]
        )
    np.testing.assert_allclose(left.q, ql)
    np.testing.assert_allclose(right.q, qr)
    assert not f._halted


def test_solve_ik_and_a_bounded_move_to_joints():
    f, arm = facade()
    st = f._arm_state("left")
    target = [st["eef_pos"][0], st["eef_pos"][1], st["eef_pos"][2] + 0.02]
    out = run(
        f,
        f"r = solve_ik({target})\n"
        "assert r['reachable'], r\n"
        "m = move_to_joints(r['joints'])\n"
        "RESULT = m['ok']\n",
    )
    assert out["status"] == "ran" and out["result"] is True, out
    assert f._arm_state("left")["eef_pos"][2] == pytest.approx(target[2], abs=2e-3)
    assert out["calls"][-1]["move_m"] == pytest.approx(0.02, abs=2e-3)
    # solve_ik with an orientation (wxyz) returns joints that FK back onto the pose.
    q = f._arm_state("left")
    w = [q["eef_pose"][6], *q["eef_pose"][3:6]]
    r = f._rpc["env.solve_ik"](position=q["eef_pos"], quaternion_wxyz=w)
    assert r["reachable"] and r["max_joint_change_rad"] < 1e-2, r


def test_move_to_joints_is_refused_beyond_its_bounds_before_any_motion():
    f, arm = facade()
    q0 = np.array(arm.q)
    for dq, why in [
        ([0.5, 0, 0, 0, 0, 0], "limit is 0.35 rad per call"),
        ([0, 0.2, 0, 0, 0, 0], "limit is 0.03 m per call"),
        ([0, 0, 0, 0, 0, 0.3], "rad per call"),
    ]:
        out = run(f, f"move_to_joints({(q0 + dq).tolist()})")
        assert out["status"] == "error" and why in out["error"], (dq, out)
        with pytest.raises(ValueError, match=why):
            f._rpc["env.move_to_joints"](joints=(q0 + dq).tolist())
    np.testing.assert_allclose(arm.q, q0)
    assert arm.streamed == 0


def test_move_to_joints_keeps_the_floor_and_stops():
    f, arm = facade(limits={"max_move_m": 0.2, "max_yaw_rad": 0.2})
    c = f._controllers["left"]
    c.limits.z_floor_m = float(arm.pose[2]) - 0.005
    q0 = np.array(arm.q)
    # A joint change that lowers the gripper more than 5 mm crosses the floor.
    down = f._rpc["env.solve_ik"](position=[*arm.pose[:2], arm.pose[2] - 0.02])
    assert down["reachable"]
    with pytest.raises(ValueError, match="Z floor"):
        f._rpc["env.move_to_joints"](joints=down["joints"])
    np.testing.assert_allclose(arm.q, q0)
    # A stop between waypoints holds the arm where it is.
    arm2 = FakeArm()
    c = controller(arm2, stop=lambda: arm2.streamed >= 2)
    out = c.move_to_joints_bounded(down["joints"], max_move_m=0.2, max_rotate_rad=0.2)
    assert out.get("cancelled") and not out["ok"], out
    assert not np.allclose(arm2.q, down["joints"])
