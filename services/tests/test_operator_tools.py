"""Operator tools with fake RPC clients: dual-Franka manual calls, calibration capture, GUMI tools."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
import pytest
import yaml
from PIL import Image

from pi_embodied_services.flywheel import gumi_tools
from pi_embodied_services.robots.dual_franka import manual_call
from pi_embodied_services.robots.franka import capture as franka_capture
from pi_embodied_services.robots.piper import capture_z_floor as piper_capture
from pi_embodied_services.utils.yaml_edit import set_value, write_value


class FakeClient:
    def __init__(self, replies: dict):
        self.replies = replies
        self.calls: list[tuple[str, dict]] = []

    def call(self, method, args=(), kwargs=None, *, timeout_s=None):
        self.calls.append((method, kwargs or {}))
        reply = self.replies[method]
        return reply(kwargs or {}) if callable(reply) else reply


def dual_state(right_tcp=(0.5, 0.0, 0.2)):
    return {
        "left_arm": {"tcp_pose": [0.5, 0.3, 0.2, 0, 0, 0, 1]},
        "right_arm": {"tcp_pose": [*right_tcp, 0, 0, 0, 1]},
    }


def out_json(capsys):
    return json.loads(capsys.readouterr().out.strip().splitlines()[-1])


# -- dual Franka manual calls ------------------------------------------------------------


def test_manual_call_dry_run_by_default_and_checked(capsys):
    client = FakeClient({"env.get_robot_state": dual_state()})
    argv = [
        "--env",
        "http://x:1",
        "--z-floor",
        "0.14",
        "move_delta",
        "--arm",
        "right",
        "--delta",
        "0",
        "0",
        "0.02",
    ]
    assert manual_call.main(argv, client) == 0
    out = out_json(capsys)
    assert out["dry_run"] is True and out["method"] == "env.move_delta"
    assert out["kwargs"] == {"arm": "right", "delta_xyz": [0.0, 0.0, 0.02]}
    assert [m for m, _ in client.calls] == [
        "env.get_robot_state"
    ]  # only read the state

    client.replies["env.move_delta"] = {"ok": True}
    assert manual_call.main(["--execute", *argv], client) == 0
    method, kwargs = client.calls[-1]
    assert method == "env.move_delta" and kwargs["delta_xyz"].dtype == np.float32


@pytest.mark.parametrize(
    "argv, reason",
    [
        (
            [
                "--z-floor",
                "0.14",
                "move_delta",
                "--arm",
                "right",
                "--delta",
                "0.2",
                "0",
                "0",
            ],
            "limit is 0.1 m",
        ),
        (
            [
                "--z-floor",
                "0.19",
                "move_delta",
                "--arm",
                "right",
                "--delta",
                "0",
                "0",
                "-0.02",
            ],
            "outside the right_base",
        ),
        (
            ["move_delta", "--arm", "right", "--delta", "0", "0", "0.01"],
            "--z-floor must be",
        ),
        (
            [
                "--workspace-xy",
                "0,1,0",
                "--z-floor",
                "0.1",
                "move_delta",
                "--arm",
                "left",
                "--delta",
                "0",
                "0",
                "0",
            ],
            "--workspace-xy",
        ),
        (
            ["rotate_delta", "--arm", "left", "--rpy", "0", "0", "0.8"],
            "limit is 0.5 rad",
        ),
    ],
)
def test_manual_call_refuses_like_the_pi_tools(argv, reason, capsys):
    client = FakeClient({"env.get_robot_state": dual_state()})
    assert manual_call.main(["--env", "http://x:1", "--execute", *argv], client) == 2
    out = out_json(capsys)
    assert reason in out["refused"]
    assert not any(
        m.startswith("env.move") or m.startswith("env.rotate") for m, _ in client.calls
    )


def test_manual_call_moving_back_toward_the_floor_is_allowed_and_reads_run(capsys):
    client = FakeClient(
        {
            "env.get_robot_state": dual_state(right_tcp=(0.5, 0.0, 0.1)),
            "env.get_env_meta": {"n": 1},
        }
    )
    argv = [
        "--env",
        "u",
        "--z-floor",
        "0.14",
        "move_delta",
        "--arm",
        "right",
        "--delta",
        "0",
        "0",
        "0.02",
    ]
    assert manual_call.main(argv, client) == 0
    assert manual_call.main(["--env", "u", "get_env_meta"], client) == 0
    assert out_json(capsys)["result"] == {"n": 1}
    assert (
        manual_call.main(["--env", "u", "set_gripper", "--arm", "left", "open"], client)
        == 0
    )
    assert out_json(capsys)["kwargs"] == {"arm": "left", "open": True}


# -- calibration capture ----------------------------------------------------------------


def test_yaml_edit_keeps_comments_and_appends_missing_keys(tmp_path):
    text = "calibration:   # capture\n  z_floor_m: null   # rest the gripper\n  arm_id: null\nlimits:\n  x: 1\n"
    new = set_value(text, ["calibration", "z_floor_m"], 0.1234)
    assert "z_floor_m: 0.1234  # rest the gripper" in new and "# capture" in new
    assert yaml.safe_load(new)["limits"] == {"x": 1}
    assert yaml.safe_load(set_value("a: 1\n", ["z_floors", "drawer"], 0.2)) == {
        "a": 1,
        "z_floors": {"drawer": 0.2},
    }
    f = tmp_path / "r.yaml"
    f.write_text("workspace:\n  target: [1, 2]\n")
    write_value(f, ["workspace", "reset_ee_pose"], [0.5, 0.0, 0.3])
    assert yaml.safe_load(f.read_text())["workspace"] == {
        "target": [1, 2],
        "reset_ee_pose": [0.5, 0.0, 0.3],
    }


def test_franka_capture_z_floor_and_pose(tmp_path, capsys):
    raw = {
        "tcp_pose": [0.6, 0.05, 0.1412345, 1.0, 0.0, 0.0, 0.0],
        "arm_joint_position": [0.1] * 7,
    }
    client = FakeClient({"env.get_robot_state": {"raw_base_state": raw}})
    floors = tmp_path / "floors.yaml"
    assert (
        franka_capture.main(
            ["--env", "u", "z-floor", "--name", "drawer", "--write", str(floors)],
            client,
        )
        == 0
    )
    out = out_json(capsys)
    assert out["z_floor"] == 0.14123 and out["flag"] == "--z-floor=0.14123"
    assert yaml.safe_load(floors.read_text()) == {"z_floors": {"drawer": 0.14123}}

    cfg = tmp_path / "rig.yaml"
    shutil.copy(Path(franka_capture.__file__).parent / "config" / "example.yaml", cfg)
    assert franka_capture.main(["--env", "u", "pose", "--write", str(cfg)], client) == 0
    pose = out_json(capsys)["reset_ee_pose"]
    assert pose[:3] == [0.6, 0.05, 0.141235] and abs(abs(pose[3]) - np.pi) < 1e-6
    loaded = yaml.safe_load(cfg.read_text())
    assert loaded["workspace"]["reset_ee_pose"] == pose
    assert loaded["robot"]["ip"] == "172.16.0.2"
    # Only read calls were made: the arm never moves.
    assert {m for m, _ in client.calls} == {"env.get_robot_state"}


def test_piper_capture_writes_that_arms_floor(tmp_path, capsys):
    client = FakeClient(
        {
            "env.get_robot_state": lambda kw: {
                "eef_pos": [0.2, 0.0, 0.0921 if kw.get("arm") == "right" else 0.5]
            }
        }
    )
    cfg = tmp_path / "dual.yaml"
    shutil.copy(
        Path(piper_capture.__file__).parent / "config" / "dual_example.yaml", cfg
    )
    assert (
        piper_capture.main(
            ["--env", "u", "--arm", "right", "--write", str(cfg)], client
        )
        == 0
    )
    assert out_json(capsys)["z_floor_m"] == 0.0921
    arms = yaml.safe_load(cfg.read_text())["arms"]
    assert (
        arms["right"]["calibration"]["z_floor_m"] == 0.0921
        and arms["left"]["calibration"]["z_floor_m"] is None
    )

    single = tmp_path / "one.yaml"
    shutil.copy(Path(piper_capture.__file__).parent / "config" / "example.yaml", single)
    assert (
        piper_capture.main(
            ["--env", "u", "--write", str(single)],
            FakeClient({"env.get_robot_state": {"eef_pos": [0, 0, 0.1]}}),
        )
        == 0
    )
    assert yaml.safe_load(single.read_text())["calibration"]["z_floor_m"] == 0.1
    two = FakeClient(
        {"env.get_robot_state": {"arms": {"left": {"eef_pos": [0, 0, 1]}}}}
    )
    assert piper_capture.main(["--env", "u"], two) == 1
    assert "pass --arm" in out_json(capsys)["error"]


# -- GUMI tools -------------------------------------------------------------------------


def gumi_run(root: Path, dual=False) -> Path:
    run = root / "0926" / "task_0" / "12-00-00"
    views = (
        ["agentview", "wrist_left", "wrist_right"] if dual else ["agentview", "wrist"]
    )
    for v in views:
        (run / "images" / v).mkdir(parents=True)
        for i in range(3):
            Image.fromarray(np.full((31, 41, 3), 40 * i, np.uint8)).save(
                run / "images" / v / f"{i:04d}.png"
            )
    rows = []
    for i, ts in enumerate([100.0, 101.5, 104.0]):
        if dual:
            rows.append(
                {
                    "i": i,
                    "left": {"act": "MV_FWD", "grip": "OPEN", "src": "human"},
                    "right": {"act": "STILL", "grip": "OPEN", "src": "human"},
                    "ts": ts,
                }
            )
        else:
            rows.append(
                {
                    "i": i,
                    "stage": "-",
                    "act": "MV_UP",
                    "grip": "OPEN",
                    "src": "agent" if i else "human",
                    "ts": ts,
                }
            )
    (run / "steps.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    return run


def test_rebuild_video_frames_and_headers(tmp_path):
    run = gumi_run(tmp_path)
    frames = gumi_tools.build_frames(
        "annotated", *gumi_tools.run_dirs_of(run / "images" / "agentview")
    )
    assert len(frames) == 3 and frames[0].shape == (
        30 + 34,
        82,
        3,
    )  # agent|wrist side by side, header, even
    assert gumi_tools.header(gumi_tools.records(run)[1]) == "#1 agent MV_UP OPEN"
    dual = gumi_run(tmp_path / "d", dual=True)
    side = gumi_tools.build_frames("side", *gumi_tools.run_dirs_of(dual))
    assert side[0].shape == (30, 122, 3)
    assert (
        gumi_tools.header(gumi_tools.records(dual)[0])
        == "#0 human L:MV_FWD OPEN  R:STILL OPEN"
    )


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="no ffmpeg on PATH")
def test_rebuild_video_writes_an_mp4(tmp_path, capsys):
    run = gumi_run(tmp_path)
    assert (
        gumi_tools.main(["rebuild-video", str(run), "--view", "side", "--fps", "5"])
        == 0
    )
    out = out_json(capsys)
    assert out["frames"] == 3 and Path(out["output"]).stat().st_size > 0


def test_step_timing_per_source(tmp_path, capsys):
    gumi_run(tmp_path)
    assert gumi_tools.main(["step-timing", str(tmp_path)]) == 0
    text = capsys.readouterr().out
    assert "human  period ms  mean    1500" in text
    assert "agent  period ms  mean    2500" in text
    assert gumi_tools.main(["step-timing", str(tmp_path / "none")]) == 1
