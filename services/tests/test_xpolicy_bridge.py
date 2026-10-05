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

"""The XPolicyLab bridge against a fake WsModelClient (no XPolicyLab checkout, no network)."""

from __future__ import annotations

import enum
import threading

import numpy as np
import pytest

from pi_embodied_services.components.xpolicy_bridge import (
    XPolicyBridge,
    action_dims,
    normalize_chunk,
)
from pi_embodied_services.utils.rpc.http_rpc import HttpRpcClient


class Code(str, enum.Enum):
    TIMEOUT = "timeout"
    CALL_FAILED = "call_failed"


class WsError(Exception):
    def __init__(self, code: Code, message: str) -> None:
        super().__init__(message)
        self.code = code


class ServerRestartedError(ConnectionError):
    pass


class FakeClient:
    """WsModelClient's surface: call(func_name, obs) -> payload["result"]; the pinned instance id."""

    def __init__(self, script: dict, **kwargs) -> None:
        self.kwargs = kwargs
        self.calls: list[tuple[str, object]] = []
        self.script = script
        self.closed = False

        class Inner:
            _server_instance_id = "srv-1"

        self._client = Inner()

    def call(self, func_name=None, obs=None):
        self.calls.append((func_name, obs))
        out = self.script.get(func_name)
        if isinstance(out, BaseException):
            raise out
        return out

    def close(self) -> None:
        self.closed = True


def bridge(script: dict) -> tuple[XPolicyBridge, list[FakeClient]]:
    made: list[FakeClient] = []

    def factory(**kwargs):
        made.append(FakeClient(script, **kwargs))
        return made[-1]

    return XPolicyBridge(client_factory=factory), made


def test_action_dims_follow_the_packaged_env_cfg():
    assert action_dims("aloha_agilex") == {
        "robot": "aloha_agilex",
        "arm_dim": [6, 6],
        "ee_dim": [1, 1],
    }
    # XPolicyLab's own table (utils/robot/_robot_info.json @d6332bf): Piper and Franka are two-armed.
    assert action_dims("piper")["arm_dim"] == [6, 6]
    assert action_dims("franka") == {
        "robot": "franka",
        "arm_dim": [7, 7],
        "ee_dim": [1, 1],
    }
    # arx_x5.yml is RoboDojo's own (robot dual_x5: the two 6-DoF arms of upstream's arx_x5).
    assert action_dims("arx_x5") == {
        "robot": "dual_x5",
        "arm_dim": [6, 6],
        "ee_dim": [1, 1],
    }
    with pytest.raises(ValueError, match="unknown env_cfg_type"):
        action_dims("nope")


def test_connect_passes_trial_ids_and_reports_the_server_instance():
    b, made = bridge({})
    info = b.connect(
        "ws://127.0.0.1:19000",
        trial_id="t1",
        action_case_id="case",
        request_timeout_s=180,
        max_connect_seconds=60,
    )
    assert info["server_instance_id"] == "srv-1"
    assert made[0].kwargs == {
        "url": "ws://127.0.0.1:19000",
        "evaluation_id": "pi-embodied",
        "trial_id": "t1",
        "action_case_id": "case",
        "request_timeout_s": 180,
        "max_connect_seconds": 60,
    }
    # Audit 92245e3 CM-6: the bridge no longer reports a precision read from pi's shell.
    assert "precision" not in info
    # A new connect closes the previous trial's client.
    b.connect("ws://127.0.0.1:19000", trial_id="t2")
    assert made[0].closed and not made[1].closed


def test_get_action_returns_a_plain_chunk_of_action_dicts():
    # Decoded msgpack arrays are read-only; tuples arrive as lists.
    row = np.arange(6, dtype=np.float64)
    row.setflags(write=False)
    chunk = [
        {
            "left_arm_joint_state": row,
            "left_ee_joint_state": np.float32(0.5),
            "action_type": "joint",
        },
        {"left_arm_joint_state": [1, 2, 3, 4, 5, 6], "left_ee_joint_state": [1.0]},
    ]
    b, made = bridge({"get_action": chunk, "update_obs": None})
    b.connect("ws://x", trial_id="t")
    obs = {
        "instruction": "beat the block",
        "vision": {"cam_head": {"color": np.zeros((2, 2, 3), np.uint8)}},
        "state": {"left_arm_joint_state": np.zeros(6, np.float32)},
    }
    assert b.update_obs(obs)["result"] is None
    assert made[0].calls[0] == ("update_obs", obs)
    out = b.get_action()
    assert out["actions"] == [
        {
            "left_arm_joint_state": [0.0, 1.0, 2.0, 3.0, 4.0, 5.0],
            "left_ee_joint_state": 0.5,
            "action_type": "joint",
        },
        {"left_arm_joint_state": [1, 2, 3, 4, 5, 6], "left_ee_joint_state": [1.0]},
    ]
    assert made[0].calls[1] == ("get_action", None)


def test_normalize_chunk_accepts_the_shapes_robotwin_accepts():
    assert normalize_chunk(None) == []
    assert normalize_chunk({"actions": np.ones((2, 3))}) == [[1.0] * 3, [1.0] * 3]
    assert normalize_chunk(np.zeros(3)) == [[0.0, 0.0, 0.0]]
    assert normalize_chunk([0.5, 1.5]) == [[0.5, 1.5]]
    assert normalize_chunk({"left_ee_pose": (1, 2)}) == [{"left_ee_pose": [1, 2]}]


def test_a_timeout_or_restart_ends_the_trial_until_reconnect():
    b, _ = bridge(
        {"get_action": WsError(Code.TIMEOUT, "timeout waiting for call_result")}
    )
    b.connect("ws://x", trial_id="t")
    with pytest.raises(RuntimeError, match=r"get_action: WsError \(timeout\)"):
        b.get_action()
    with pytest.raises(RuntimeError, match="trial is over"):
        b.reset()
    b.connect("ws://x", trial_id="t2")
    assert b.reset()["result"] is None

    b, _ = bridge({"update_obs": ServerRestartedError("policy server restarted")})
    b.connect("ws://x", trial_id="t")
    with pytest.raises(RuntimeError, match="ServerRestartedError"):
        b.update_obs({"state": {}})
    with pytest.raises(RuntimeError, match="trial is over"):
        b.get_action()


def test_a_model_error_does_not_end_the_trial():
    b, _ = bridge(
        {"update_obs": WsError(Code.CALL_FAILED, "bad obs"), "get_action": []}
    )
    b.connect("ws://x", trial_id="t")
    with pytest.raises(RuntimeError, match=r"WsError \(call_failed\): bad obs"):
        b.update_obs({})
    assert b.get_action()["actions"] == []


def test_calls_before_connect_fail():
    b, _ = bridge({})
    with pytest.raises(RuntimeError, match="xpolicy.connect first"):
        b.get_action()


def test_over_http_arrays_arrive_as_numpy():
    b, made = bridge({"update_obs": None, "get_action": [{"ee_pose": np.ones(7)}]})
    server = b._bind_and_announce("http", "127.0.0.1", 0, b._serve_dispatch)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        host, port = server.server_address
        client = HttpRpcClient(f"http://{host}:{port}")
        client.call("xpolicy.connect", kwargs={"url": "ws://x", "trial_id": "t"})
        obs = {"state": {"ee_pose": np.zeros(7, np.float32)}}
        client.call("xpolicy.update_obs", kwargs={"obs": obs})
        sent = made[0].calls[0][1]["state"]["ee_pose"]
        assert isinstance(sent, np.ndarray) and sent.dtype == np.float32
        assert client.call("xpolicy.get_action")["actions"] == [{"ee_pose": [1.0] * 7}]
    finally:
        server.shutdown()
        server.server_close()
