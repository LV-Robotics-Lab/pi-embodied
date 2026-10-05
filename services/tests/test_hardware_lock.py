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
    # The config names the controller serial (calibration.arm_id): that is the arm's lock id.
    arm = hardware_lock.hardware_ids("ur5e", ur5e_tests.cfg())[0]

    def build(*_a, **_k):
        raise AssertionError("hardware touched")

    monkeypatch.setattr(env_server, "build", build)
    (tmp_path / "locks").mkdir(mode=0o755)
    with acquire([arm], directory=str(tmp_path / "locks")):
        with pytest.raises(SystemExit) as exc:
            env_server.main(
                [
                    "--robot-config",
                    str(path),
                    "--lock-dir",
                    str(tmp_path / "locks"),
                    # pi's limits at spawn parse before the lock is taken.
                    "--max-move",
                    "0.05",
                    "--max-rotate",
                    "0.1",
                ]
            )
    assert exc.value.code == 3
    assert f"robot busy: arm {arm}" in capsys.readouterr().err


def test_a_planted_symlink_and_an_unsafe_lock_dir_are_refused(tmp_path):
    root = tmp_path / "locks"
    root.mkdir(mode=0o755)
    target = tmp_path / "victim"
    target.write_text("keep")
    (root / "franka_1.2.3.4.lock").symlink_to(target)
    with pytest.raises(RuntimeError, match="symlink"):
        acquire(["franka:1.2.3.4"], directory=str(root))
    assert target.read_text() == "keep"
    open_dir = tmp_path / "open"
    open_dir.mkdir()
    open_dir.chmod(0o777)
    with pytest.raises(RuntimeError, match="world-writable"):
        acquire(["franka:1.2.3.4"], directory=str(open_dir))
    link = tmp_path / "link"
    link.symlink_to(root)
    with pytest.raises(RuntimeError, match="not a plain directory"):
        acquire(["x:1"], directory=str(link))


def test_a_lock_file_this_user_cannot_open_reads_as_in_use(tmp_path, monkeypatch):
    real_open = hardware_lock.os.open

    def denied(path, *a, **k):
        if str(path).endswith("ur5e_10.0.0.9.lock"):
            raise PermissionError(13, "Permission denied")
        return real_open(path, *a, **k)

    monkeypatch.setattr(hardware_lock.os, "open", denied)
    with pytest.raises(RobotBusyError, match="in use|another user"):
        acquire(["ur5e:10.0.0.9"], directory=str(tmp_path))


def test_serials_come_first_and_the_address_locks_stay_alongside_them():
    rlinf_yaml = {"robot": {"ip": "172.16.0.2", "serial": "295341-1325480"}}
    polymetis_yaml = {"robot": {"nuc_ip": "192.168.1.100", "serial": "295341-1325480"}}
    # Both Franka backends lock the serial (one id for one arm) and keep their own address lock.
    assert hardware_lock.hardware_ids("franka", rlinf_yaml, ["franka:172.16.0.2"]) == [
        "franka:295341-1325480",
        "franka:172.16.0.2",
    ]
    assert hardware_lock.hardware_ids(
        "franka", polymetis_yaml, ["franka-polymetis:192.168.1.100"]
    ) == ["franka:295341-1325480", "franka-polymetis:192.168.1.100"]
    # Without a serial: the addresses given, else the config's.
    assert hardware_lock.hardware_ids("ur5e", {"robot": {"ip": "10.0.0.9"}}) == [
        "ur5e:10.0.0.9"
    ]
    # UR5e's calibration.arm_id is its controller serial; the address is locked too.
    assert hardware_lock.hardware_ids(
        "ur5e", {"robot": {"ip": "10.0.0.9"}, "calibration": {"arm_id": "2023300001"}}
    ) == ["ur5e:2023300001", "ur5e:10.0.0.9"]


def test_a_server_configured_by_serial_and_one_by_address_alone_collide(tmp_path):
    # The same arm at 172.16.0.2: one config names its serial, the other only its address. Both
    # must hold the address lock, or they would lock disjoint ids and drive the arm together.
    by_serial = hardware_lock.hardware_ids(
        "franka", {"robot": {"ip": "172.16.0.2", "serial": "arm123"}}
    )
    by_address = hardware_lock.hardware_ids("franka", {"robot": {"ip": "172.16.0.2"}})
    assert by_serial == ["franka:arm123", "franka:172.16.0.2"]
    assert by_address == ["franka:172.16.0.2"]
    with acquire(by_serial, directory=str(tmp_path), holder="franka-env"):
        with pytest.raises(
            RobotBusyError, match=r"arm franka:172\.16\.0\.2 .*franka-env"
        ):
            acquire(by_address, directory=str(tmp_path))
    # And the other way round: the address holder refuses the serial-configured server.
    with acquire(by_address, directory=str(tmp_path), holder="franka-env"):
        with pytest.raises(
            RobotBusyError, match=r"arm franka:172\.16\.0\.2 .*franka-env"
        ):
            acquire(by_serial, directory=str(tmp_path))


def test_a_dual_arm_config_naming_one_serial_keeps_both_address_locks():
    dual = {
        "robot": {
            "arms": {
                "left": {"ip": "10.0.0.1", "serial": "left-serial"},
                "right": {"ip": "10.0.0.2"},
            }
        }
    }
    assert hardware_lock.hardware_ids("franka", dual) == [
        "franka:left-serial",
        "franka:10.0.0.1",
        "franka:10.0.0.2",
    ]
    # The RLinf server passes the addresses it resolved itself: the same rule.
    assert hardware_lock.hardware_ids(
        "franka", dual, ["franka:10.0.0.1", "franka:10.0.0.2"]
    ) == ["franka:left-serial", "franka:10.0.0.1", "franka:10.0.0.2"]


def test_camera_devices_are_locked_too():
    cfg = {
        "robot": {"ip": "10.0.0.9"},
        "cameras": {
            "width": 640,
            "devices": {
                "wrist": {"type": "realsense", "serial": "141722070657"},
                "front": {"type": "webcam", "device": "/dev/video2"},
                "any": {"type": "realsense", "serial": ""},
                "net": {"type": "rtsp", "device": "rtsp://cam/1"},
            },
        },
    }
    assert hardware_lock.hardware_ids("ur5e", cfg) == [
        "ur5e:10.0.0.9",
        "camera:141722070657",
        "camera:/dev/video2",
    ]
    dual = {"cameras": {"observation": {"base": [{"serial": "311322304048"}]}}}
    assert hardware_lock.camera_ids(dual) == ["camera:311322304048"]
