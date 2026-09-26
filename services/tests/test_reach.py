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

"""The reach preview helper (utils/reach.py) against a fake ik service, and its wiring into
the LIBERO and Franka env servers (fake envs, no simulator or arm)."""

from __future__ import annotations

import argparse

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from pi_embodied_services.robots.libero.env_server import LiberoEnvFacade
from pi_embodied_services.utils import reach
from pi_embodied_services.utils.rpc import RpcError

DOWN = [1.0, 0.0, 0.0, 0.0]


class FakeIk:
    """An ik client: reachable inside 0.8 m of the base; can be told to fail."""

    def __init__(self, fail: Exception | None = None):
        self.calls: list[dict] = []
        self.fail = fail

    def call(self, method, args=(), kwargs=None, *, timeout_s=None):
        if self.fail is not None:
            raise self.fail
        if method == "ik.robots":
            return {"backend": "fake", "robots": {}}
        assert method == "ik.solve"
        self.calls.append(kwargs)
        pos = np.asarray(kwargs["target_pose"]["pos"])
        ok = float(np.linalg.norm(pos)) < 0.8
        return {
            "q": [0.1] * 7 if ok else [0.0] * 7,
            "ok": ok,
            "error": None if ok else "unreachable: misses by 300.0 mm",
            "position_err": 0.001 if ok else 0.3,
            "orientation_err": 0.01,
        }


def preview(client, **kwargs):
    return reach.ReachPreview("http://ik", "panda", client=client)


def test_reachable_and_unreachable_answers_with_the_seed_forwarded():
    ik = FakeIk()
    rp = preview(ik)
    out = rp.preview([0.0] * 7, [0.4, 0.0, 0.3], DOWN)
    assert out["status"] == "reachable" and out["reachable"] is True
    assert out["q"] == [0.1] * 7 and out["target"]["frame"] == "base"
    assert "1.0 mm" in out["message"]
    assert ik.calls[0]["robot"] == "panda" and ik.calls[0]["seed_q"] == [0.0] * 7
    far = rp.preview([0.0] * 7, [2.0, 0.0, 0.3], DOWN)
    assert far["status"] == "unreachable" and far["reachable"] is False
    assert far["message"].startswith("unreachable: misses by 300.0 mm")
    assert far["path_checked"] is False


def test_a_down_service_is_unknown_not_approval():
    for exc in (RpcError("ik.solve", "HTTP request failed: refused"), OSError("down")):
        out = preview(FakeIk(fail=exc)).preview([0.0] * 7, [0.4, 0.0, 0.3], DOWN)
        assert out["status"] == "unknown" and out["reachable"] is None
        assert "IK service unavailable" in out["message"]
    # Only "unreachable" refuses; unknown passes (with a warning).
    reach.require_reachable(out, "move_to")
    with pytest.raises(ValueError, match="move_to refused: target \\[2.0, 0.0, 0.3\\]"):
        reach.require_reachable(
            preview(FakeIk()).preview([0.0] * 7, [2.0, 0.0, 0.3], DOWN), "move_to"
        )


def test_world_targets_are_converted_into_the_base_frame():
    ik = FakeIk()
    # Base at world (-0.5, 0, 0.9), turned 90 deg about z.
    base = {
        "pos": [-0.5, 0.0, 0.9],
        "quat_xyzw": Rotation.from_euler("z", np.pi / 2).as_quat().tolist(),
    }
    out = preview(ik).preview([0.0] * 7, [-0.5, 0.4, 1.2], DOWN, base_pose=base)
    sent = ik.calls[0]["target_pose"]
    assert np.allclose(sent["pos"], [0.4, 0.0, 0.3], atol=1e-9)  # world +y is base +x
    assert out["target"] == {
        "frame": "world",
        "pos": [-0.5, 0.4, 1.2],
        "quat_xyzw": DOWN,
        "base_pose": base,
    }
    assert out["status"] == "reachable"
    # The same point 2 m further out in the world is unreachable in the base frame too.
    far = preview(ik).preview([0.0] * 7, [-0.5, 2.4, 1.2], DOWN, base_pose=base)
    assert far["status"] == "unreachable"


def test_pose_in_base_and_delta_target():
    pos, quat = reach.pose_in_base([1, 0, 0], [0, 0, 0, 1], [1, 0, 0], [0, 0, 0, 1])
    assert np.allclose(pos, 0) and np.allclose(quat, [0, 0, 0, 1])
    tcp = [0.5, 0.0, 0.3, *DOWN]
    pos, quat = reach.delta_target(tcp, delta_xyz=[0.1, -0.1, 0.05])
    assert np.allclose(pos, [0.6, -0.1, 0.35]) and np.allclose(quat, DOWN)
    pos, quat = reach.delta_target(tcp, delta_rpy=[0.0, 0.0, np.pi / 2])
    assert np.allclose(pos, tcp[:3])
    expected = Rotation.from_euler("z", np.pi / 2) * Rotation.from_quat(DOWN)
    assert np.isclose(
        (Rotation.from_quat(quat) * expected.inv()).magnitude(), 0.0, atol=1e-9
    )
    with pytest.raises(ValueError):
        reach.delta_target(tcp, delta_xyz=[0.1, 0.1])


def test_argparse_wiring_is_optional():
    parser = argparse.ArgumentParser()
    reach.add_ik_argument(parser)
    assert reach.reach_from_args(parser.parse_args([]), "panda") is None
    rp = reach.reach_from_args(
        parser.parse_args(["--ik", "http://127.0.0.1:18400"]), "panda"
    )
    assert isinstance(rp, reach.ReachPreview) and rp.robot == "panda"
    assert reach.no_service()["status"] == "unknown"


# ---------------------------------------------------------------------------
# LIBERO env server wiring (fake env: raw obs and a worker answering robot_base_pose)


class _Worker:
    def env_call(self, name, target):
        assert (name, target) == ("robot_base_pose", "self")
        return {"pos": [-0.5, 0.0, 0.9], "quat_xyzw": [0.0, 0.0, 0.0, 1.0]}


class _Env:
    class env:  # noqa: N801 - LiberoEnv.env.workers
        workers = [_Worker()]

    current_raw_obs = [
        {
            "robot0_joint_pos": np.zeros(7),
            "robot0_eef_pos": np.array([-0.1, 0.0, 1.2]),
            "robot0_eef_quat": np.array(DOWN),
            "robot0_gripper_qpos": np.array([0.04, -0.04]),
        }
    ]

    def __init__(self):
        self.steps = 0

    def step(self, action):
        self.steps += 1
        zeros = np.zeros(1, dtype=bool)
        return (
            {"main_images": np.zeros((1, 2, 2, 3), dtype=np.uint8)},
            np.zeros(1),
            zeros,
            zeros,
            {"episode": {"success_once": np.array([False])}},
        )


def test_libero_preview_reach_uses_the_base_frame_and_declares_the_primitive():
    ik = FakeIk()
    facade = LiberoEnvFacade(
        _Env(),
        meta={},
        ik_reach=reach.ReachPreview("http://ik", "panda_libero", client=ik),
    )
    out = facade.preview_reach([-0.1, 0.0, 1.2])
    assert out["status"] == "reachable"
    # World (-0.1, 0, 1.2) is base (0.4, 0, 0.3): within reach; the current quat was kept.
    assert np.allclose(ik.calls[0]["target_pose"]["pos"], [0.4, 0.0, 0.3])
    assert np.allclose(ik.calls[0]["target_pose"]["quat_xyzw"], DOWN)
    assert ik.calls[0]["robot"] == "panda_libero"
    assert ik.calls[0]["seed_q"] == [0.0] * 7
    far = facade._dispatch("env.preview_reach", ([1.5, 0.0, 1.2],), {})
    assert far["status"] == "unreachable" and far["target"]["frame"] == "world"
    with pytest.raises(ValueError, match="move_to refused"):
        reach.require_reachable(far, "move_to")
    assert facade._env.steps == 0  # the sim was never stepped
    # The primitive registry (code.api) declares it for code-as-policy callers.
    names = [p["name"] for p in facade._dispatch("code.api", (), {})["primitives"]]
    assert "preview_reach" in names


def test_libero_without_ik_answers_unknown():
    facade = LiberoEnvFacade(_Env(), meta={})
    out = facade._dispatch("env.preview_reach", ([1.5, 0.0, 1.2],), {})
    assert out["status"] == "unknown" and out["reachable"] is None
    reach.require_reachable(out, "move_to")  # unknown does not refuse
    assert facade._env.steps == 0


# ---------------------------------------------------------------------------
# Franka env server wiring (fake backend with a TCP 0.5 m ahead)


class _FrankaBackend:
    def __init__(self):
        self.moves: list = []

    def get_env_meta(self):
        return {}

    def reset(self):
        return {}

    def get_robot_state(self):
        return {
            "raw_base_state": {
                "tcp_pose": np.array([0.5, 0.0, 0.3, *DOWN]),
                "arm_joint_position": np.zeros(7),
            }
        }

    def get_observation(self):
        return {}

    def get_camera_meta(self):
        return {}

    def move_delta(self, delta_xyz):
        self.moves.append(("move", list(delta_xyz)))
        return {"ok": True}

    def rotate_delta(self, delta_rpy):
        self.moves.append(("rotate", list(delta_rpy)))
        return {"ok": True}

    def set_gripper(self, *, open):
        return {"ok": True}

    def chunk_step(self, actions, *, return_all_frames=False):
        return {}


def test_franka_move_delta_is_checked_against_its_end_pose():
    # The franka server imports its runtime_config (omegaconf, from the franka extra).
    pytest.importorskip("omegaconf")
    from pi_embodied_services.robots.franka.env_server import FrankaEnvFacade

    ik = FakeIk()
    backend = _FrankaBackend()
    facade = FrankaEnvFacade(
        backend, ik_reach=reach.ReachPreview("http://ik", "panda", client=ik)
    )
    assert facade._dispatch("env.move_delta", (), {"delta_xyz": [0.1, 0.0, 0.0]}) == {
        "ok": True
    }
    assert np.allclose(ik.calls[-1]["target_pose"]["pos"], [0.6, 0.0, 0.3])
    with pytest.raises(ValueError, match="env.move_delta refused"):
        facade._dispatch("env.move_delta", ([0.5, 0.0, 0.0],), {})
    assert facade._dispatch("env.rotate_delta", (), {"delta_rpy": [0.0, 0.0, 0.5]}) == {
        "ok": True
    }
    assert backend.moves == [("move", [0.1, 0.0, 0.0]), ("rotate", [0.0, 0.0, 0.5])]
    out = facade._dispatch("env.preview_reach", ([0.6, 0.0, 0.3],), {})
    assert out["status"] == "reachable" and out["target"]["frame"] == "base"
    plain = FrankaEnvFacade(_FrankaBackend())
    assert (
        plain._dispatch("env.preview_reach", ([0.6, 0.0, 0.3],), {})["status"]
        == "unknown"
    )
    assert plain._dispatch("env.move_delta", ([0.5, 0.0, 0.0],), {}) == {"ok": True}
