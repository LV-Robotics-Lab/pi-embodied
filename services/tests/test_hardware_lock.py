"""The single-machine hardware lock (utils/hardware_lock.py) and its use in the real env servers."""

from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

import pytest
import yaml

from pi_embodied_services.utils import hardware_lock
from pi_embodied_services.utils.hardware_lock import (
    RobotBusyError,
    acquire,
    config_arm_ids,
)

SERVICES = Path(__file__).resolve().parents[1]


def test_second_server_for_the_same_arm_is_refused_naming_the_holder(tmp_path):
    with acquire(["franka:172.16.0.2"], directory=str(tmp_path), holder="franka-env"):
        with pytest.raises(
            RobotBusyError, match=r"arm franka:172\.16\.0\.2 .*franka-env"
        ):
            acquire(["franka:172.16.0.2"], directory=str(tmp_path))
        # A different arm is free.
        acquire(["ur5e:192.168.1.10"], directory=str(tmp_path)).release()
    # Released: the arm can be taken again.
    acquire(["franka:172.16.0.2"], directory=str(tmp_path)).release()


def test_a_dual_server_takes_both_arms_or_none(tmp_path):
    with acquire(["franka:10.0.0.2"], directory=str(tmp_path)):
        with pytest.raises(RobotBusyError, match="10.0.0.2"):
            acquire(["franka:10.0.0.1", "franka:10.0.0.2"], directory=str(tmp_path))
        # The refused dual server released the arm it had already locked.
        acquire(["franka:10.0.0.1"], directory=str(tmp_path)).release()


def test_a_crashed_holder_releases_its_arm(tmp_path):
    code = (
        "import sys, time; from pi_embodied_services.utils.hardware_lock import acquire; "
        f"acquire(['piper:left'], directory={str(tmp_path)!r}, holder='child'); "
        "print('held', flush=True); time.sleep(60)"
    )
    child = subprocess.Popen(
        [sys.executable, "-c", code],
        cwd=SERVICES,
        env={"PYTHONPATH": str(SERVICES), "PATH": "/usr/bin:/bin"},
        stdout=subprocess.PIPE,
        text=True,
    )
    try:
        assert child.stdout is not None and child.stdout.readline().strip() == "held"
        with pytest.raises(RobotBusyError, match=r"pid=\d+ arm=piper:left child"):
            acquire(["piper:left"], directory=str(tmp_path))
    finally:
        child.kill()
        child.wait()
    deadline = time.time() + 5
    while True:
        try:
            acquire(["piper:left"], directory=str(tmp_path)).release()
            break
        except RobotBusyError:
            assert time.time() < deadline
            time.sleep(0.05)


def test_arm_ids_come_from_the_config_addresses():
    dual = {
        "robot": {"arms": {"left": {"ip": "10.0.0.1"}, "right": {"ip": "10.0.0.2"}}}
    }
    assert config_arm_ids("franka", dual) == ["franka:10.0.0.1", "franka:10.0.0.2"]
    same = {
        "robot": {"arms": {"left": {"ip": "10.0.0.1"}, "right": {"ip": "10.0.0.1"}}}
    }
    assert config_arm_ids("franka", same) == ["franka:10.0.0.1"]
    rlinf = {
        "cluster": {
            "node_groups": [
                {
                    "hardware": {
                        "configs": [{"robot_ip": "1.2.3.4", "node_ip": "9.9.9.9"}]
                    }
                }
            ]
        },
        "env": {"left_robot_ip": "1.2.3.5"},
    }
    assert config_arm_ids("franka", rlinf, keys=r"(\w+_)?robot_ip") == [
        "franka:1.2.3.4",
        "franka:1.2.3.5",
    ]
    assert config_arm_ids(
        "ur5e", {"ip": "192.168.1.10", "gripper": {"port": 63352}}
    ) == ["ur5e:192.168.1.10"]


def test_lock_dir_comes_from_the_flag_then_the_environment(monkeypatch, tmp_path):
    monkeypatch.setenv(hardware_lock.LOCK_DIR_ENV, str(tmp_path / "env"))
    assert hardware_lock.lock_dir() == tmp_path / "env"
    assert hardware_lock.lock_dir(str(tmp_path / "flag")) == tmp_path / "flag"
    monkeypatch.delenv(hardware_lock.LOCK_DIR_ENV)
    assert str(hardware_lock.lock_dir()) == hardware_lock.DEFAULT_LOCK_DIR


def test_ur5e_server_refuses_to_start_on_a_held_arm_before_touching_hardware(
    tmp_path, monkeypatch, capsys
):
    from pi_embodied_services.robots.ur5e import env_server

    ur5e_tests = pytest.importorskip("test_ur5e")
    path = tmp_path / "ur5e.yaml"
    path.write_text(yaml.safe_dump(ur5e_tests.cfg()))
    ip = ur5e_tests.cfg()["robot"]["ip"]

    def build(*_a, **_k):
        raise AssertionError("hardware touched")

    monkeypatch.setattr(env_server, "build", build)
    with acquire([f"ur5e:{ip}"], directory=str(tmp_path / "locks")):
        with pytest.raises(SystemExit) as exc:
            env_server.main(
                ["--robot-config", str(path), "--lock-dir", str(tmp_path / "locks")]
            )
    assert exc.value.code == 3
    assert f"robot busy: arm ur5e:{ip}" in capsys.readouterr().err
