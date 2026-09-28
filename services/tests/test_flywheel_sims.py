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

"""The Flywheel rules of Metaworld, Genesis, Robosuite and ManiSkill (raw episodes as their TS
recorders write them, validated and exported), and UR5e's rules for its session data."""

from __future__ import annotations

import json

import numpy as np
import pytest

from pi_embodied_services.flywheel import cli
from pi_embodied_services.flywheel.episode import EpisodeWriter, validate_episode
from pi_embodied_services.flywheel.export import features
from pi_embodied_services.flywheel.specs import ROBOTS, spec
from pi_embodied_services.robots.ur5e import flywheel as ur5e

#: Each sim's spaces: the raw path, the episode metadata and its state / action widths.
CASES = [
    ("metaworld", None, ["reach-v3"], {"task": "reach-v3"}, 4, 4),
    ("genesis", None, ["cube_pick"], {"task": "cube_pick"}, 8, 4),
    ("robosuite", "one_arm", ["Lift"], {"task": "Lift", "space": "one_arm"}, 9, 7),
    (
        "robosuite",
        "two_arm",
        ["TwoArmLift"],
        {"task": "TwoArmLift", "space": "two_arm"},
        18,
        14,
    ),
    ("robosuite", "wipe", ["Wipe"], {"task": "Wipe", "space": "wipe"}, 7, 6),
    *(
        (
            "maniskill",
            arm,
            [arm, "PickCube-v1", "default"],
            {"maniskill_robot": arm, "env_id": "PickCube-v1", "scene": ""},
            8,
            4,
        )
        for arm in ("panda", "xarm6_robotiq", "widowxai")
    ),
]


def obs(s: dict, step: int) -> dict:
    """An observation of ``s``'s arrays: 6x8 images where the spec leaves the size open."""
    out = {}
    for key, field in s["arrays"].items():
        if key == "actions":
            continue
        shape = field["shape"] or (6, 8, 3)
        out[key] = (
            np.full(shape, step, np.uint8)
            if key in s["image_fields"]
            else np.arange(shape[0], dtype=np.float32) + step
        )
    return out


def record(root, robot, space, path, metadata, steps=3, solved_at=1):
    """A raw episode as the robot's recorder writes it: one scripted transition per control
    step, ``terminated`` from the step's success."""
    s = spec(robot, space)
    writer = EpisodeWriter(
        root / "raw" / robot / "/".join(path) / "seed_000",
        metadata={
            **metadata,
            "robot": robot,
            "seed": 0,
            "task_language": "do the task",
        },
        spec=s,
        initial_observation=obs(s, 0),
    )
    width = s["arrays"]["actions"]["shape"][0]
    writer.begin_primitive("move_delta")
    for i in range(steps):
        writer.add_transition(
            np.full(width, 0.1 * i, np.float32),
            obs(s, i + 1),
            float(i >= solved_at),
            i >= solved_at,
            False,
        )
    return writer.finalize()


def test_the_sims_are_registered_and_ur5e_is_not():
    assert {"metaworld", "genesis", "robosuite", "maniskill"} <= set(ROBOTS)
    assert "ur5e" not in ROBOTS
    with pytest.raises(ValueError, match="no Flywheel spec for 'ur5e'"):
        spec("ur5e")


@pytest.mark.parametrize(
    "robot,space,path,metadata,state,action",
    CASES,
    ids=[f"{c[0]}-{c[1]}" for c in CASES],
)
def test_each_space_names_its_dimensions_and_validates_its_episodes(
    tmp_path, robot, space, path, metadata, state, action
):
    s = spec(robot, space)
    assert (s["arrays"]["states"]["shape"], s["arrays"]["actions"]["shape"]) == (
        (state,),
        (action,),
    )
    assert len(s["state_names"]) == state and len(s["action_names"]) == action
    assert set(s["cameras"]) == set(s["image_fields"])
    resolved = {
        **s,
        "arrays": {
            k: {**f, "shape": f["shape"] or (6, 8, 3)} for k, f in s["arrays"].items()
        },
    }
    assert features(resolved)["action"]["names"] == s["action_names"]
    meta = validate_episode(record(tmp_path, robot, space, path, metadata), spec=s)
    # The training prefix ends at the first control step the env judged a success.
    assert (meta["step_count"], meta["training_step_count"]) == (3, 2)
    assert all(meta[k] == v for k, v in metadata.items())


def test_robosuite_and_maniskill_spaces_differ_where_the_embodiments_do(tmp_path):
    two = record(
        tmp_path,
        "robosuite",
        "two_arm",
        ["TwoArmLift"],
        {"task": "TwoArmLift", "space": "two_arm"},
    )
    with pytest.raises(ValueError, match="invalid states shape"):
        validate_episode(two, spec=spec("robosuite", "one_arm"))
    assert spec("robosuite") is spec("robosuite", "one_arm")
    # The WidowX AI has no wrist camera; the Panda's episodes have one it would not export.
    assert spec("maniskill", "widowxai")["cameras"] == {"agentview_images": "agentview"}
    assert set(spec("maniskill", "panda")["cameras"].values()) == {
        "agentview",
        "wrist",
    }
    assert spec("maniskill", "xarm6_robotiq")["robot_type"] == "xarm6_robotiq"
    # The bridge twins step at 5 Hz, the tabletop scenes at 20 Hz.
    assert spec("maniskill", "widowx250s")["fps"] == 5
    assert spec("maniskill", "panda")["fps"] == 20
    widow = record(
        tmp_path,
        "maniskill",
        "widowxai",
        ["widowxai", "PickCube-v1", "default"],
        {"maniskill_robot": "widowxai", "env_id": "PickCube-v1", "scene": ""},
    )
    with pytest.raises(ValueError, match="has no wrist_images"):
        validate_episode(widow, spec=spec("maniskill", "panda"))
    with pytest.raises(ValueError, match="has no 'arm' space"):
        spec("maniskill", "arm")


def test_export_takes_the_arm_space_and_refuses_mixed_arms(tmp_path, capsys):
    pytest.importorskip("lerobot.datasets.lerobot_dataset")
    meta = {"maniskill_robot": "widowxai", "env_id": "PickCube-v1", "scene": ""}
    path = ["widowxai", "PickCube-v1", "default"]
    record(tmp_path, "maniskill", "widowxai", path, meta)
    argv = ["export-lerobot", "--data-root", str(tmp_path), "--robot", "maniskill"]
    select = ["--select", "widowxai/PickCube-v1", "--space", "widowxai"]
    assert cli.main([*argv, *select, "--dataset-id", "d1"]) == 0
    out = json.loads(capsys.readouterr().out)
    assert (out["episode_count"], out["frame_count"]) == (1, 2)
    assert out["maniskill_robot"] == "widowxai"
    root = tmp_path / "datasets/lerobot-widowxai/maniskill/widowxai/PickCube-v1/d1"
    info = json.loads((root / "meta/info.json").read_text())
    assert info["robot_type"] == "widowxai"
    assert "observation.images.agentview" in info["features"]
    assert "observation.images.wrist" not in info["features"]
    # An episode of another scene in the selection is refused, not mixed in.
    record(tmp_path, "maniskill", "widowxai", path, {**meta, "scene": "x"})
    with pytest.raises(ValueError, match="another maniskill_robot/env_id/scene"):
        cli.main([*argv, *select, "--dataset-id", "d2"])


def _step(tool=None, pose=(0.4, 0.0, 0.3, 1.0, 0.0, 0.0, 0.0), width=0.08, **kw):
    """A states.jsonl line; the default TCP points straight down (xyzw [1, 0, 0, 0])."""
    base = {
        "tcp_pose": list(pose),
        "gripper_position": [width],
        "gripper_open": True,
        "gripper_commanded_open": kw.pop("opened", True),
    }
    blob = {"step_idx": 0, "state": {"raw_base_state": base, "backend": "ur_rtde"}}
    if tool:
        blob["command"] = {"action": tool, **kw.pop("params", {})}
    if kw.get("result"):
        blob["result"] = kw["result"]
    return blob


def test_ur5e_rules_describe_its_session_steps():
    s = ur5e.SPEC
    assert len(s["state_names"]) == s["arrays"]["states"]["shape"][0] == 8
    assert len(s["action_names"]) == s["arrays"]["actions"]["shape"][0] == 7
    before = _step("reset")
    assert ur5e.state_of(before).tolist() == pytest.approx(
        [0.4, 0, 0.3, 1, 0, 0, 0, 0.08]
    )
    with pytest.raises(ValueError, match="no gripper width"):
        ur5e.state_of({"state": {"raw_base_state": {"tcp_pose": [0] * 7}}})
    # The reset, a failed call and a read are no transitions.
    assert ur5e.action_of(before, before) is None
    failed = _step(
        "move_delta", params={"delta_xyz": [0, 0, 0.5]}, result={"error": "limit"}
    )
    assert ur5e.action_of(failed, before) is None
    assert ur5e.action_of(_step("view_env_state"), before) is None
    move = _step("move_delta", params={"delta_xyz": [0.01, 0, -0.02]})
    assert ur5e.action_of(move, before).tolist() == pytest.approx(
        [0.01, 0, -0.02, 0, 0, 0, -1]
    )
    # A pose target is the delta from the pose before; its orientation the base-frame turn.
    pose = _step("move_pose", params={"xyz": [0.5, 0.1, 0.3], "rpy": [np.pi, 0, 0.2]})
    a = ur5e.action_of(pose, before)
    assert a[:3].tolist() == pytest.approx([0.1, 0.1, 0])
    assert a[3:6].tolist() == pytest.approx([0, 0, 0.2], abs=1e-6)
    turn = _step("rotate_delta", params={"delta_rpy": [0, 0, -0.3]})
    assert ur5e.action_of(turn, before)[3:6].tolist() == pytest.approx([0, 0, -0.3])
    close = _step("gripper", params={"action": "close"}, opened=False)
    assert ur5e.action_of(close, before).tolist() == pytest.approx([0] * 6 + [1])
    unit = _step(
        "act", params={"move": {"delta": [0, 0.02, 0], "yaw": 0.1}}, opened=False
    )
    assert ur5e.action_of(unit, before).tolist() == pytest.approx(
        [0, 0.02, 0, 0, 0, 0.1, 1]
    )


def test_ur5e_spec_validates_an_episode_built_from_its_steps(tmp_path):
    s = ur5e.SPEC
    steps = [
        _step("reset"),
        _step("move_delta", params={"delta_xyz": [0, 0, -0.05]}),
        _step("gripper", params={"action": "close"}, opened=False),
    ]

    def observation(blob, k):
        images = {key: np.full((4, 6, 3), k, np.uint8) for key in s["image_fields"]}
        return {**images, "states": ur5e.state_of(blob)}

    writer = EpisodeWriter(
        tmp_path / "raw" / "ur5e" / "block_bowl",
        metadata={"arm_id": "1", "task": "block_bowl", "task_language": "bowl"},
        spec=s,
        initial_observation=observation(steps[0], 0),
    )
    for k in (1, 2):
        writer.add_transition(
            ur5e.action_of(steps[k], steps[k - 1]),
            observation(steps[k], k),
            float(k == 2),
            k == 2,
            False,
        )
    meta = validate_episode(writer.finalize(), spec=s)
    assert (meta["step_count"], meta["training_step_count"]) == (2, 2)


def test_every_maniskill_robot_has_a_space_with_its_action_width():
    """One space per ``--robot`` of the env server's table, each with the width of the flat
    action its servo reports (the pair: both arms, left then right)."""
    from pi_embodied_services.robots.maniskill import env_server as ms

    assert set(spec("maniskill", r)["robot_type"] for r in ms.ROBOTS) == set(ms.ROBOTS)
    widths = {r: spec("maniskill", r)["arrays"]["actions"]["shape"] for r in ms.ROBOTS}
    assert widths == {
        "panda": (4,),
        "xarm6_robotiq": (4,),
        "widowxai": (4,),
        "panda_stick": (3,),
        "panda_pair": (8,),
        "widowx250s": (7,),
    }
    pair = spec("maniskill", "panda_pair")
    assert pair["arrays"]["states"]["shape"] == (16,)
    assert pair["state_names"][8] == "right_tcp_x" and pair["cameras"] == {
        "agentview_images": "agentview"
    }


def test_an_arm_space_refuses_another_arms_episodes_of_the_same_shape(tmp_path):
    """The Panda's and the xArm6's episodes have the same arrays; exporting one under the
    other's space would label its columns wrongly, so the space's arm must match."""
    from pi_embodied_services.flywheel.export import _successful_episodes

    meta = {"maniskill_robot": "xarm6_robotiq", "env_id": "PickCube-v1", "scene": ""}
    xarm = record(
        tmp_path,
        "maniskill",
        "xarm6_robotiq",
        ["xarm6_robotiq", "PickCube-v1", "default"],
        meta,
    )
    panda = spec("maniskill", "panda")
    assert validate_episode(xarm, spec=panda)["step_count"] > 0
    with pytest.raises(
        ValueError, match="maniskill_robot='xarm6_robotiq'; this space is for 'panda'"
    ):
        _successful_episodes([xarm], spec=panda, group=tuple(panda["group"]))
    own = spec("maniskill", "xarm6_robotiq")
    assert len(_successful_episodes([xarm], spec=own, group=tuple(own["group"]))) == 1
    for robot in ("panda_stick", "panda_pair", "widowx250s"):
        assert spec("maniskill", robot)["metadata"] == {"maniskill_robot": robot}
