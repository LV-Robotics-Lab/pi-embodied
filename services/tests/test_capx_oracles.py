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

"""CaP-X's ported human oracles (packages/embodied/src/robots/<robot>/oracle/, ``--code-oracle``): each
program runs as ``code.run`` would run it (its prelude, then the file) against fake primitives
that check every call against the robot's registry for the oracle's tier, the way
``CodeApi.resolve`` does on the server. No simulator: the fakes keep a tiny world (TCPs, a few
objects, an overhead RGB-D camera) so the programs run to completion."""

from __future__ import annotations

import math
import re
from pathlib import Path

import numpy as np
import pytest

from pi_embodied_services.components.code_api import CodeApi
from pi_embodied_services.components.manifest import code_primitives, load_manifest
from pi_embodied_services.robots.robosuite import tasks

SRC = Path(__file__).resolve().parents[2] / "packages" / "embodied" / "src" / "robots"
MARK = "# ---- CaP-X's program, verbatim ----\n"


def load(robot: str, name: str) -> tuple[dict[str, str], str]:
    """The oracle's header and the code --code-oracle runs (code/index.ts loadOracle)."""
    path = SRC / robot / "oracle" / name
    text = path.read_text()
    header: dict[str, str] = {}
    for line in text.split("\n"):
        if not line.startswith("#"):
            break
        m = re.match(r"^#\s*([\w-]+):\s*(.*?)\s*$", line)
        if m:
            header[m.group(1)] = m.group(2)
    prelude = (
        (path.parent / header["prelude"]).read_text() if "prelude" in header else ""
    )
    return header, f"{prelude}\n{text}" if prelude else text


def oracles(robot: str) -> list[str]:
    return sorted(
        p.name
        for p in (SRC / robot / "oracle").glob("*.py")
        if not p.name.startswith("capx_")
    )


# ---- a fake world ---------------------------------------------------------------------------

K = np.array([[500.0, 0.0, 256.0], [0.0, 500.0, 256.0], [0.0, 0.0, 1.0]])
CAM_Z = 1.8


def camera(center_xy) -> np.ndarray:
    """An overhead camera looking straight down (image right = world +x, image down = -y)."""
    T = np.eye(4)
    T[:3, :3] = np.diag([1.0, -1.0, -1.0])
    T[:3, 3] = [center_xy[0], center_xy[1], CAM_Z]
    return T


def quat_xyzw_of_yaw(yaw: float) -> list[float]:
    return [0.0, 0.0, math.sin(yaw / 2), math.cos(yaw / 2)]


class Fake:
    """Recording fakes of one robot's primitives in one tier."""

    def __init__(
        self, primitives, tier, *, arms, objects, table_z, cam_xy, libero=False, task=""
    ):
        self.primitives = list(primitives)
        self.api = CodeApi(self.primitives, {p.method: None for p in self.primitives})
        self.tier = tier
        self.names = [p.name for p in self.api.primitives(tier)]
        self.params = {p.name: list(p.params) for p in self.primitives}
        self.calls: list[tuple[str, dict]] = []
        self.eef = {a: np.array(v, dtype=np.float64) for a, v in arms.items()}
        self.home = {a: v.copy() for a, v in self.eef.items()}
        self.yaw = 0.0
        self.objects = {k: np.asarray(v, dtype=np.float64) for k, v in objects.items()}
        self.table_z = table_z
        self.cam = camera(cam_xy)
        self.libero = libero
        self.task = task
        self.poses: list[list[float]] = []

    # The program's globals: a stub per primitive of the tier (utils/code_exec.py _stub).
    def globals(self) -> dict:
        g = {"__name__": "__main__", "RESULT": None, "np": np, "math": math}
        for name in self.names:
            g[name] = self._stub(name)
        return g

    def _stub(self, name):
        def call(*args, **kwargs):
            kw = dict(zip(self.params[name], args))
            kw.update(kwargs)
            self.api.resolve(
                name, kw, self.tier
            )  # undeclared names / parameters raise here
            self.calls.append((name, kw))
            return getattr(self, name)(**kw)

        return call

    # -- state
    def get_state(self):
        if self.libero:
            return {
                "eef_pos": self.eef["robot0"].tolist(),
                "yaw": self.yaw,
                "eef_quat_xyzw": [1, 0, 0, 0],
            }
        out = {
            "table_z": self.table_z,
            "home_eef_pos": {a: v.tolist() for a, v in self.home.items()},
        }
        for a, v in self.eef.items():
            out[f"{a}_eef_pos"] = v.tolist()
            out[f"{a}_eef_quat"] = [1.0, 0.0, 0.0, 0.0]
        return out

    def get_task_language(self):
        return "a task"

    # -- motion
    def move_to(
        self, target_xyz=None, xyz=None, arm=None, quat_xyzw=None, max_steps=100, **_
    ):
        target = np.asarray(
            target_xyz if target_xyz is not None else xyz, dtype=np.float64
        )
        arm = arm or "robot0"
        step = float(np.linalg.norm(target - self.eef[arm]))
        assert step <= 0.3 + 1e-9, f"a move_to leg of {step:.3f} m"
        if quat_xyzw is not None:
            assert abs(np.linalg.norm(quat_xyzw) - 1) < 1e-6
        self.eef[arm] = target
        key = "eef_pos" if self.libero else "final_eef_pos"
        return {key: target.tolist(), "final_dist_m": 0.0}

    def rotate_wrist(self, target_yaw=None, delta_yaw=None, **_):
        self.yaw = float(target_yaw if target_yaw is not None else self.yaw + delta_yaw)
        return {"yaw": self.yaw, "eef_pos": self.eef["robot0"].tolist()}

    def set_gripper(self, close, arm=None, steps=15):
        return {
            "gripper": "close" if close else "open",
            "gripper_width": 0.0 if close else 0.08,
        }

    # -- CaP-X's high tier (the server's semantic functions; privileged: from the simulator)
    def _privileged(self):
        return "privileged" in (self.tier or "")

    def get_object_pose(self, object_name, return_bbox_extent=False):
        if self._privileged():
            sim, ext = tasks.OBJECT_NAMES[self.task][object_name]
            pos = self.objects[sim]
            return [
                pos.tolist(),
                [1.0, 0.0, 0.0, 0.0],
                list(ext) if return_bbox_extent else None,
            ]
        pos = self._object_of(object_name)
        return [
            pos.tolist(),
            [1.0, 0.0, 0.0, 0.0],
            [0.05, 0.05, 0.05] if return_bbox_extent else None,
        ]

    def sample_grasp_pose(self, object_name, arm=None):
        pos, _, _ = self.get_object_pose(object_name)
        return [pos, [0.0, 0.0, 1.0, 0.0]]

    def goto_pose(self, position, quaternion_wxyz, z_approach=0.0, arm=None):
        target = np.asarray(position, dtype=np.float64)
        if z_approach:
            self.poses.append((target + [0.0, 0.0, float(z_approach)]).tolist())
        self.poses.append(target.tolist())
        self.eef[arm or "robot0"] = target
        return {"final_eef_pos": target.tolist()}

    def home_pose(self, arm=None):
        return self.goto_pose(self.home[arm or "robot0"], [0, 0, 1, 0], arm=arm)

    def open_gripper(self, arm=None):
        return self.set_gripper(False, arm=arm)

    def close_gripper(self, arm=None):
        return self.set_gripper(True, arm=arm)

    # -- privileged
    def ground_truth_poses(self, names=None):
        poses = {
            k: {"pos": v.tolist(), "quat_xyzw": [0.0, 0.0, 0.0, 1.0]}
            for k, v in self.objects.items()
        }
        if names is not None:
            poses = {k: poses[k] for k in names}
        return {"frame": "world", "poses": poses}

    # -- perception: each object is a 5 cm square on the overhead camera's image
    def _pixel(self, p):
        x, y, z = self.cam[:3, :3].T @ (np.asarray(p) - self.cam[:3, 3])
        return int(round(K[1, 2] + K[1, 1] * y / z)), int(
            round(K[0, 2] + K[0, 0] * x / z)
        )

    def _object_of(self, prompt):
        for key, pos in self.objects.items():
            if any(w in prompt.lower() for w in key.split("|")):
                return pos
        return None

    def _depth(self):
        d = np.full((512, 512), CAM_Z - self.table_z, dtype=np.float32)
        for pos in self.objects.values():
            r, c = self._pixel(pos)
            d[max(r - 12, 0) : r + 13, max(c - 12, 0) : c + 13] = (
                CAM_Z - pos[2] + np.random.default_rng(0).uniform(-0.002, 0.002, (1,))
            )
        return d

    def segment(self, prompt, camera="agentview", min_score=0.2):
        pos = self._object_of(prompt)
        if pos is None:
            return {"found": False}
        r, c = self._pixel(pos)
        mask = np.zeros((512, 512), dtype=bool)
        mask[r - 12 : r + 13, c - 10 : c + 11] = True
        return {
            "found": True,
            "score": 0.9,
            "box": [c - 10, r - 12, c + 10, r + 12],
            "mask": mask,
            "centroid_rowcol": [r, c],
            "world_xyz": list(pos),
        }

    def get_observation(self):
        view = {
            "rgb": np.zeros((512, 512, 3), np.uint8),
            "depth": self._depth(),
            "intrinsic_K": K,
            "extrinsic_cam2world": self.cam,
        }
        wrist = dict(view, rgb=np.ones((512, 512, 3), np.uint8))
        return {"agentview": view, "wrist": wrist, **self.get_state()}

    def back_project(self, row, col, camera="agentview"):
        raise AssertionError("not used by the oracles")

    def plan_grasp(self, object=None, **_):
        pos = self._object_of(object)
        return {
            "active": "g0",
            "candidates": [
                {
                    "id": "g0",
                    "eef_position": list(pos),
                    "eef_quat_xyzw": [1.0, 0.0, 0.0, 0.0],
                    "eef_yaw": 0.3,
                }
            ],
        }

    def sequence(self):
        """The call names (get_state left out) with consecutive repeats collapsed."""
        out: list[str] = []
        for name, kw in self.calls:
            if name == "get_state":
                continue
            tag = name
            if name == "set_gripper":
                tag = ("close" if kw["close"] else "open") + (
                    f"@{kw['arm']}" if "arm" in kw else ""
                )
            elif name == "move_to" and "arm" in kw:
                tag = f"move_to@{kw['arm']}"
            if not out or out[-1] != tag:
                out.append(tag)
        return out


# ---- per-robot worlds ------------------------------------------------------------------------

ROBOSUITE_WORLDS = {
    "Lift": dict(
        arms={"robot0": [-0.09, 0.0, 1.01]},
        objects={"cube|red": [0.0, -0.02, 0.83]},
        table_z=0.8,
        cam_xy=[0.0, 0.0],
    ),
    "Stack": dict(
        arms={"robot0": [-0.09, 0.0, 1.01]},
        objects={"cubeA|red": [-0.05, 0.02, 0.83], "cubeB|green": [0.02, 0.05, 0.835]},
        table_z=0.8,
        cam_xy=[0.0, 0.0],
    ),
    "Restack": dict(
        arms={"robot0": [-0.09, 0.0, 1.01]},
        objects={"cubeA|red": [0.0, 0.0, 0.82], "cubeB|green": [0.0, 0.0, 0.86]},
        table_z=0.8,
        cam_xy=[0.0, 0.0],
    ),
    "Wipe": dict(
        arms={"robot0": [0.04, 0.03, 1.08]},
        objects={
            f"contact{i}|spill": [0.2 + 0.02 * (i % 5), 0.1 + 0.02 * (i // 5), 0.9]
            for i in range(10)
        },
        table_z=0.9,
        cam_xy=[0.2, 0.1],
    ),
    "NutAssemblySquare": dict(
        arms={"robot0": [-0.12, 0.0, 1.0]},
        objects={"SquareNut|nut": [-0.11, 0.12, 0.83], "peg1|block": [0.23, 0.1, 0.85]},
        table_z=0.82,
        cam_xy=[0.0, 0.1],
    ),
    "TwoArmLift": dict(
        arms={"robot0": [-0.02, -0.11, 1.0], "robot1": [-0.01, 0.09, 1.02]},
        objects={
            "pot_handle0|green": [0.0, -0.16, 0.9],
            "pot_handle1|blue": [0.0, 0.16, 0.9],
            "pot|pot": [0.0, 0.0, 0.87],
        },
        table_z=0.8,
        cam_xy=[0.0, 0.0],
    ),
    "TwoArmHandover": dict(
        arms={"robot0": [-0.01, -0.36, 1.0], "robot1": [-0.01, 0.35, 1.01]},
        objects={
            "hammer_handle|hammer": [0.03, -0.45, 0.82],
            "hammer|zzz": [0.03, -0.47, 0.82],
        },
        table_z=0.8,
        cam_xy=[0.0, -0.3],
    ),
}


def robosuite_fake(task: str, tier: str, grasp: bool = False) -> Fake:
    """The robosuite manifest's primitives (every requirement met, the grasp server optional)."""
    prims = code_primitives(
        load_manifest("robosuite"), lambda c: c not in ("grasp", "place") or grasp
    )
    world = {**ROBOSUITE_WORLDS[task]}
    world["objects"] = dict(world["objects"])
    return Fake(prims, tier, task=task, **world)


def simplify(fake: Fake) -> None:
    """Ground-truth names are the keys before '|'."""
    fake.objects = {
        k.split("|")[0] if "privileged" in fake.tier else k: v
        for k, v in fake.objects.items()
    }


def run(robot: str, name: str, fake: Fake) -> dict:
    header, code = load(robot, name)
    g = fake.globals()
    exec(compile(code, name, "exec"), g, g)
    return g


# ---- tests ------------------------------------------------------------------------------------


def test_every_oracle_has_a_valid_header_and_a_self_contained_prelude():
    rs, lb = oracles("robosuite"), oracles("libero")
    assert len(rs) == 14 and len(lb) == 3
    for robot, names in (("robosuite", rs), ("libero", lb)):
        for name in names:
            header, code = load(robot, name)
            assert header["tier"].removesuffix("+privileged") in (
                "high",
                "low",
                "low-noexamples",
                "privileged",
            ), name
            assert MARK in code, name
            compile(code, name, "exec")
            if "prelude" not in header:
                # The CaP-X functions are the server's high / privileged tier.
                assert header["tier"] in ("high", "privileged"), name
                continue
            prelude = (SRC / robot / "oracle" / header["prelude"]).read_text()
            if "privileged" not in header["tier"]:
                assert "ground_truth_poses" not in prelude, (
                    name,
                    "an S2/S3 prelude reads ground truth",
                )


@pytest.mark.parametrize("name", oracles("robosuite"))
def test_robosuite_oracle_runs_on_its_tier(name):
    header, _ = load("robosuite", name)
    fake = robosuite_fake(header["task"], header["tier"])
    simplify(fake)
    run("robosuite", name, fake)
    assert any(n in ("move_to", "goto_pose") for n, _ in fake.calls)
    if "privileged" not in header["tier"]:
        assert not any(n == "ground_truth_poses" for n, _ in fake.calls)


def test_lift_privileged_sequence():
    """CaP-X's lift program on the server's privileged tier: its calls are the primitives."""
    fake = robosuite_fake("Lift", "privileged")
    simplify(fake)
    run("robosuite", "lift_privileged.py", fake)
    assert fake.sequence() == [
        "sample_grasp_pose",
        "open_gripper",
        "goto_pose",
        "close_gripper",
        "goto_pose",
    ]
    np.testing.assert_allclose(fake.eef["robot0"], [0.0, -0.02, 0.93], atol=1e-9)


def test_lift_s2_calls_the_high_tier_functions():
    fake = robosuite_fake("Lift", "high")
    run("robosuite", "lift.py", fake)
    assert fake.sequence()[:2] == ["sample_grasp_pose", "open_gripper"]
    np.testing.assert_allclose(fake.eef["robot0"], [0.0, -0.02, 0.93], atol=1e-9)
    assert any(np.allclose(p, [0.0, -0.02, 0.93]) for p in fake.poses), (
        "approach from 0.1 m above"
    )


def test_stack_privileged_places_red_on_green():
    fake = robosuite_fake("Stack", "privileged")
    simplify(fake)
    run("robosuite", "stack_privileged.py", fake)
    seq = fake.sequence()
    assert seq[-3:] == ["goto_pose", "open_gripper", "goto_pose"]
    # Retract point: 0.1 m above the place point on green (green z + 0.05).
    np.testing.assert_allclose(
        fake.eef["robot0"], [0.02, 0.05, 0.835 + 0.05 + 0.1], atol=1e-9
    )
    # The place leg approaches from 0.1 m above.
    assert any(np.allclose(p, [0.02, 0.05, 0.985]) for p in fake.poses)


def test_two_arm_lift_privileged_takes_capx_bounding_box_branch_and_lifts_together():
    fake = robosuite_fake("TwoArmLift", "low+privileged")
    simplify(fake)
    g = run("robosuite", "two_arm_lift_privileged.py", fake)
    assert (
        "get_handle0_pose" not in g
    )  # as CaP-X's functions(): the program's first branch
    seq = fake.sequence()
    assert seq.index("close@robot0") < seq.index("close@robot1")
    tail = seq[seq.index("close@robot1") + 1 :]
    assert tail[:4] == [
        "move_to@robot0",
        "move_to@robot1",
        "move_to@robot0",
        "move_to@robot1",
    ]
    # Both handles lifted 0.2 m (world z; robot0's frame is only turned about z).
    np.testing.assert_allclose(fake.eef["robot0"], [0.0, -0.16, 1.1], atol=1e-9)
    np.testing.assert_allclose(fake.eef["robot1"], [0.0, 0.16, 1.1], atol=1e-9)


def test_handover_privileged_uses_robot0_frame_constants():
    fake = robosuite_fake("TwoArmHandover", "low+privileged")
    simplify(fake)
    run("robosuite", "two_arm_handover_privileged.py", fake)
    # handover_pos = (0.81, 0, 0.10) in robot0's frame: world (0, -0.81 + 0.81, 1.022) etc.
    # arm0 retracts to handover + (-0.1, 0, 0.06): world (0, -0.1, 1.082) after the +90 deg turn.
    np.testing.assert_allclose(fake.eef["robot0"], [0.0, -0.1, 0.922 + 0.16], atol=1e-6)
    np.testing.assert_allclose(fake.eef["robot1"], [0.0, 0.1, 0.922 + 0.09], atol=1e-6)


class LiberoFake(Fake):
    """The LIBERO manifest's primitives: set_gripper takes the gripper command (-1 / +1), the
    CaP-X functions take use_multiview, and the privileged ones look names up as CaP-X does."""

    def set_gripper(self, gripper=-1, steps=5, **_):
        return super().set_gripper(float(gripper) > 0)

    def get_object_pose(self, object_name, use_multiview=True):
        if self._privileged():
            pos = self.objects[f"{object_name.split()[0]}_1"]
            return [pos.tolist(), [1.0, 0.0, 0.0, 0.0]]
        return [self._object_of(object_name).tolist(), [1.0, 0.0, 0.0, 0.0]]

    def sample_grasp_pose(self, object_name, use_multiview=True):
        pos, _ = self.get_object_pose(object_name)
        return [pos, [0.0, 1.0, 0.0, 0.0]]

    def sequence(self):
        out: list[str] = []
        for name, kw in self.calls:
            if name == "get_state":
                continue
            tag = {"open_gripper": "open", "close_gripper": "close"}.get(name, name)
            if name == "set_gripper":
                tag = "close" if float(kw.get("gripper", -1)) > 0 else "open"
            if not out or out[-1] != tag:
                out.append(tag)
        return out


@pytest.mark.parametrize("name", oracles("libero"))
def test_libero_oracle_runs_on_its_tier(name):
    header, _ = load("libero", name)
    assert header["suite"] == "libero_object_swap" and header["task"] == "7"
    tier = header["tier"]
    objects = {
        "milk_1|milk": [0.05, 0.1, 0.95],
        "basket_1|basket": [-0.05, -0.15, 0.93],
    }
    if "privileged" in tier:
        objects = {k.split("|")[0]: v for k, v in objects.items()}
    prims = code_primitives(load_manifest("libero"), lambda c: c != "grasp")
    fake = LiberoFake(
        prims,
        tier,
        arms={"robot0": [-0.2, 0.0, 1.15]},
        objects=objects,
        table_z=0.9,
        cam_xy=[0.0, 0.0],
        libero=True,
    )
    run("libero", name, fake)
    seq = fake.sequence()
    assert "close" in seq and seq[-1] in ("open", "move_to", "goto_pose")
    reached = [kw["xyz"] for n, kw in fake.calls if n == "move_to"] + [
        kw["position"] for n, kw in fake.calls if n == "goto_pose"
    ]
    assert any(
        np.allclose(np.asarray(p)[:2], [0.05, 0.1], atol=0.01) for p in reached
    )  # the milk
    basket_xy = np.array([-0.05, -0.15])
    assert (
        np.linalg.norm(fake.eef["robot0"][:2] - basket_xy) < 0.01
    )  # released over the basket
    if tier in ("high", "privileged"):
        # CaP-X's functions are the server's: the program calls them directly.
        assert {n for n, _ in fake.calls} <= {
            "open_gripper",
            "close_gripper",
            "sample_grasp_pose",
            "get_object_pose",
            "goto_pose",
        }


def test_verbatim_blocks_match_capx_when_the_reference_is_present():
    ref = Path.home() / "workspace" / "pi-work" / "refs" / "cap-x"
    if not (ref / "capx").is_dir():
        pytest.skip("CaP-X reference repo not present")
    import ast

    import yaml

    for robot in ("robosuite", "libero"):
        for name in oracles(robot):
            header, code = load(robot, name)
            got = code.split(MARK, 1)[1]
            src = header["program"]
            if src.startswith("capx/"):
                path, var = src.split(" ")[:2]
                tree = ast.parse((ref / path).read_text())
                want = next(
                    n.value.value
                    for n in tree.body
                    if isinstance(n, ast.Assign)
                    and any(getattr(t, "id", None) == var for t in n.targets)
                )
            else:
                cfg = header["capx"].split(" @")[0]
                want = yaml.safe_load((ref / cfg).read_text())["env"]["cfg"][
                    "oracle_code"
                ]
            assert got == want.lstrip("\n"), name
