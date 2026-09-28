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

"""The UR5e server's manifest (code.api, the startup self-check), pi's limits held by its motion
methods for every caller (a tool's RPC call and a program alike), code mode (``--code``,
utils/code_real.py) and the bounded joint-space pair (solve_ik / move_to_joints), against the mock
arm, gripper and cameras of test_ur5e.py. 未上机验证."""

from __future__ import annotations

import numpy as np
import pytest
from scipy.spatial.transform import Rotation
from test_ur5e import call, cfg, facade

from pi_embodied_services.components.cameras.mock import MOCK_ENV
from pi_embodied_services.components.manifest import ManifestError
from pi_embodied_services.robots.ur5e.mock import DOWN, MockUrArm

#: pi's limits the tests spawn the server with (tighter than the config's 0.08 m / 0.2 rad).
PI = {"max_move_m": 0.05, "max_rotate_rad": 0.1}


@pytest.fixture(autouse=True)
def _mock_env(monkeypatch):
    monkeypatch.setenv(MOCK_ENV, "1")


def coded(**kw):
    return facade(code=True, limits=PI, **kw)


def run(f, program, **kw):
    kw.setdefault("tier", "low")
    return f._rpc["code.run"](program, timeout_s=30, **kw)


def tcp(f) -> np.ndarray:
    return np.asarray(f.controller.state()["tcp_pose"][:3])


# -- a toy kinematic model: joints 0..2 move the TCP along x/y/z (SCALE m per rad), joint 5
# turns it about z; the mock arm reports fk(q) as its pose. --------------------------------

Q0 = np.array([1.571, -1.571, 1.571, -1.571, -1.571, 0.0])
SCALE = 0.3


def kin(z0=0.30):
    base = np.array([0.45, 0.0, z0])

    def fk(q):
        d = np.asarray(q, float) - Q0
        rot = Rotation.from_euler("z", d[5]) * Rotation.from_rotvec(DOWN)
        return [*(base + SCALE * d[:3]), *rot.as_rotvec()]

    def ik(pose, qnear):
        q = np.asarray(qnear, float).copy()
        q[:3] = Q0[:3] + (np.asarray(pose[:3]) - base) / SCALE
        yaw = (
            Rotation.from_rotvec(pose[3:]) * Rotation.from_rotvec(DOWN).inv()
        ).as_euler("xyz")
        q[5] = Q0[5] + yaw[2]
        return q

    return MockUrArm(fk(Q0), joints=Q0, fk=fk, ik=ik)


# -- the manifest --------------------------------------------------------------------------


def test_the_startup_self_check_passes_and_refuses_an_undeclared_method():
    facade()._manifest_ready()
    coded()._manifest_ready()
    f = facade()
    f._rpc["env.nudge"] = lambda: None
    with pytest.raises(ManifestError, match="env.nudge"):
        f._manifest_ready()
    g = facade()
    del g._rpc["env.close_gripper"]
    with pytest.raises(ManifestError, match="close_gripper"):
        g._manifest_ready()


def test_the_manifest_tiers():
    api = facade()._rpc["code.api"]
    assert set(api("high")["available"]) == {"open_gripper", "close_gripper"}
    assert api("raw")["available"] == []  # no raw step on this robot
    low = set(api("low")["available"])
    assert low == {
        "get_robot_state",
        "get_observation",
        "get_camera_meta",
        "move_delta",
        "move_pose",
        "rotate_delta",
        "set_gripper",
        "solve_ik",
        "move_to_joints",
    }
    # Without SAM3 / UniDepth the perception primitives are left out.
    assert not {"detect", "enhance_depth"} & low


def test_the_server_reports_the_limits_it_enforces():
    meta = coded()._rpc["env.get_env_meta"]()
    assert meta["motion_limits"] == PI
    assert call(facade(), "env.get_env_meta")["motion_limits"] == {
        "max_move_m": 0.08,
        "max_rotate_rad": 0.2,
    }
    with pytest.raises(ValueError, match="max_yaw_rad"):
        facade(limits={"max_yaw_rad": 0.1})
    with pytest.raises(ValueError, match="--max-move"):
        facade(limits={"max_move_m": -1})


# -- pi's limits for every caller -----------------------------------------------------------


REFUSED = [
    ("move_delta", ([0.06, 0, 0],), {}, "limit is 0.05 m per call"),
    ("move_pose", ([0.45, 0.06, 0.30],), {}, "limit is 0.05 m per call"),
    ("rotate_delta", ([0, 0, 0.15],), {}, "limit is 0.1 rad per call"),
    (
        "move_pose",
        ([0.45, 0.0, 0.30],),
        {
            "rpy": (Rotation.from_euler("z", 0.15) * Rotation.from_rotvec(DOWN))
            .as_euler("xyz")
            .tolist()
        },
        "limit is 0.1 rad per call",
    ),
]


@pytest.mark.parametrize("name,args,kwargs,why", REFUSED)
def test_a_tool_call_beyond_pis_limits_is_refused_before_the_arm_moves(
    name, args, kwargs, why
):
    arm = MockUrArm((0.45, 0.0, 0.30, *DOWN))
    f = facade(arm, limits=PI)
    with pytest.raises(ValueError, match=why):
        call(f, f"env.{name}", *args, **kwargs)
    assert arm.moves == [] and f.controller.commands == 0
    # Within them it runs.
    assert call(f, "env.move_delta", [0.04, 0, 0])["ok"]


@pytest.mark.parametrize("name,args,kwargs,why", REFUSED)
def test_a_program_meets_the_same_limits(name, args, kwargs, why):
    f = coded()
    start = tcp(f)
    arglist = ", ".join(
        [*(repr(list(a)) for a in args), *(f"{k}={v!r}" for k, v in kwargs.items())]
    )
    out = run(f, f"{name}({arglist})")
    assert out["status"] == "error" and why in out["error"], out
    np.testing.assert_allclose(tcp(f), start, atol=1e-9)


def test_the_configs_caps_still_apply_under_looser_pi_limits():
    f = facade(limits={"max_move_m": 0.15, "max_rotate_rad": 0.4})
    with pytest.raises(ValueError, match="limit is 0.08 m per call"):
        call(f, "env.move_delta", [0.1, 0, 0])
    with pytest.raises(ValueError, match="limit is 0.2 rad per call"):
        call(f, "env.rotate_delta", [0, 0, 0.3])


# -- code mode ------------------------------------------------------------------------------


def test_code_mode_needs_code_and_then_the_token():
    plain = facade()
    assert "code.run" not in plain._rpc and plain._rpc_token is None
    assert "code.set_limits" not in plain._rpc
    f = coded()
    assert f._rpc_token and "code.run" in f._rpc and "code.set_limits" not in f._rpc
    with pytest.raises(PermissionError):
        f._serve_dispatch("env.get_robot_state", (), {})


def test_a_program_moves_through_the_facade_and_the_run_reports_it():
    f = coded()
    start = tcp(f)
    out = run(
        f,
        "move_delta([0, 0, 0.03])\n"
        "p = get_robot_state()['raw_base_state']['tcp_pose']\n"
        "move_pose([p[0] + 0.02, p[1], p[2]])\n"
        "set_gripper(False)\n"
        "set_gripper(True)\n",
    )
    assert out["status"] == "ran", out
    np.testing.assert_allclose(tcp(f), start + [0.02, 0, 0.03], atol=1e-3)
    assert out["motions"] == 4 and len(out["frames"]) == 4
    moves = [
        c.get("move_m", 0.0) for c in out["calls"] if c["name"] != "get_robot_state"
    ]
    assert moves == pytest.approx([0.03, 0.02, 0.0, 0.0], abs=1e-3)


def test_the_high_tier_opens_and_closes_the_gripper():
    f = coded()
    out = run(f, "close_gripper()\nopen_gripper()\n", tier="high")
    assert out["status"] == "ran" and out["motions"] == 2, out
    assert f.controller.commanded_open is True


def test_the_floor_and_the_run_cap_hold_in_a_program():
    f = coded()
    # The server's floor (z 0.14) still applies under pi's limits.
    out = run(f, "for _ in range(5):\n    move_delta([0, 0, -0.04])\n")
    assert out["status"] == "error" and tcp(f)[2] >= 0.14 - 1e-9, out
    f = coded()
    out = run(f, "for _ in range(4):\n    move_delta([0, 0, 0.04])\n", max_move_m=0.1)
    assert out["status"] == "error" and out["limit"] == "max_move_m", out
    assert out["motions"] == 2


# -- the bounded joint-space pair ------------------------------------------------------------


def test_solve_ik_is_read_only_and_checked_with_forward_kinematics():
    arm = kin()
    f = facade(arm, limits=PI)
    r = call(f, "env.solve_ik", [0.45, 0.0, 0.28])
    assert r["reachable"], r
    np.testing.assert_allclose(r["joints"][2], Q0[2] - 0.02 / SCALE, atol=1e-9)
    assert r["max_joint_change_rad"] == pytest.approx(0.02 / SCALE)
    assert r["max_joint_step_rad"] == 0.3
    assert arm.joint_moves == [] and arm.moves == []
    # An orientation (wxyz) is honoured.
    rot = Rotation.from_euler("z", 0.05) * Rotation.from_rotvec(DOWN)
    x, y, z, w = rot.as_quat()
    r = call(f, "env.solve_ik", [0.45, 0.0, 0.30], quaternion_wxyz=[w, x, y, z])
    assert r["reachable"] and r["joints"][5] == pytest.approx(0.05, abs=1e-6)
    # An IK answer that forward kinematics does not confirm is no solution.
    arm.ik = lambda pose, qnear: np.asarray(qnear) + 0.01
    r = call(f, "env.solve_ik", [0.45, 0.0, 0.28])
    assert not r["reachable"] and r["joints"] is None and "do not reach" in r["reason"]
    arm.ik = None
    assert not call(f, "env.solve_ik", [0.45, 0.0, 0.28])["reachable"]


def test_move_to_joints_moves_within_its_bounds():
    arm = kin()
    f = coded(arm=arm)
    out = run(
        f,
        "p = get_robot_state()['raw_base_state']['tcp_pose']\n"
        "r = solve_ik([p[0], p[1], p[2] - 0.02])\n"
        "m = move_to_joints(r['joints'])\n"
        "print(m['ok'])\n",
    )
    assert out["status"] == "ran" and "True" in out["stdout"], out
    np.testing.assert_allclose(tcp(f), [0.45, 0.0, 0.28], atol=1e-9)
    assert len(arm.joint_moves) == 1 and out["motions"] == 1
    moves = [
        c.get("move_m", 0.0) for c in out["calls"] if c["name"] == "move_to_joints"
    ]
    assert moves == pytest.approx([0.02])


@pytest.mark.parametrize(
    "dq,z0,why",
    [
        ([0, 0, 0, 0, 0.35, 0], 0.30, "limit is 0.3 rad per call"),  # one joint too far
        (
            [0.2, 0, 0, 0, 0, 0],
            0.30,
            "limit is 0.05 m per call",
        ),  # TCP 6 cm (--max-move)
        (
            [0, 0, 0, 0, 0, 0.15],
            0.30,
            "limit is 0.1 rad per call",
        ),  # TCP turn (--max-rotate)
        ([0, 0, -0.15, 0, 0, 0], 0.18, "outside the workspace"),  # through the floor
    ],
)
def test_move_to_joints_is_refused_whole(dq, z0, why):
    arm = kin(z0)
    f = facade(arm, limits=PI)
    with pytest.raises(ValueError, match=why):
        call(f, "env.move_to_joints", (Q0 + np.asarray(dq)).tolist())
    assert arm.joint_moves == [] and f.controller.commands == 0
    # A program is refused the same way.
    g = coded(arm=kin(z0))
    out = run(g, f"move_to_joints({(Q0 + np.asarray(dq)).tolist()!r})")
    assert out["status"] == "error" and why in out["error"], out


def test_move_to_joints_stops_on_stop():
    arm = kin()
    arm.step_joint = 0.01
    f = facade(arm, limits=PI)
    polls = {"n": 0}
    real = arm.async_status

    def status():
        polls["n"] += 1
        if polls["n"] == 3:
            f.request_stop()
        return real()

    arm.async_status = status
    r = call(f, "env.move_to_joints", (Q0 + [0, 0, -0.1, 0, 0, 0]).tolist())
    assert r.get("cancelled") and not r["ok"] and arm.stops, r


def test_the_joint_move_limit_comes_from_the_config():
    f = facade(kin(), config=cfg(limits={"max_joint_step_rad": 0.1}), limits=PI)
    with pytest.raises(ValueError, match="limit is 0.1 rad per call"):
        call(f, "env.move_to_joints", (Q0 + [0, 0, -0.12, 0, 0, 0]).tolist())
    assert call(f, "env.get_env_meta")["limits"]["max_joint_step_rad"] == 0.1
