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

"""``env.ground_truth_poses`` (pi's --privileged) on LIBERO, RoboCasa and ManiSkill, with
mock simulators: names come from the sim's own object list, poses are world-frame xyzw."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from pi_embodied_services.robots.libero import env_server as libero
from pi_embodied_services.robots.maniskill import env_server as maniskill
from pi_embodied_services.robots.robocasa.env_server import RoboCasaEnvFacade
from pi_embodied_services.utils import ground_truth

#: wxyz of a 90 deg turn about z, and the xyzw the result carries.
WXYZ = [0.70711, 0.0, 0.0, 0.70711]
XYZW = [0.0, 0.0, 0.70711, 0.70711]


def _sim(bodies: dict[str, int]):
    """A robosuite-like sim whose body i sits at (i, 2i, 3i) turned 90 deg about z."""
    n = max(bodies.values()) + 1
    xpos = np.array([[i, 2 * i, 3 * i] for i in range(n)], dtype=np.float64)
    xquat = np.tile(WXYZ, (n, 1))

    def body_name2id(name):
        if name not in bodies:
            raise ValueError(f'No "body" with name {name} exists.')
        return bodies[name]

    return SimpleNamespace(
        data=SimpleNamespace(body_xpos=xpos, body_xquat=xquat),
        model=SimpleNamespace(body_name2id=body_name2id),
    )


def test_respond_selects_names_and_refuses_unknown_ones():
    poses = {
        "b": ground_truth.pose([1, 2, 3], WXYZ),
        "a": ground_truth.pose([0, 0, 0], [1, 0, 0, 0]),
    }
    assert list(ground_truth.respond(poses)["poses"]) == ["a", "b"]
    assert ground_truth.respond(poses, [])["frame"] == "world"
    assert ground_truth.respond(poses, ["b"])["poses"] == {
        "b": {"pos": [1.0, 2.0, 3.0], "quat_xyzw": XYZW}
    }
    with pytest.raises(
        ValueError, match=r"unknown objects \['c'\]; the scene has \['a', 'b'\]"
    ):
        ground_truth.respond(poses, ["b", "c"])


class _Worker:
    """RLinf's libero worker: ``env_call(method, target="self")`` calls the env the wrapped
    ``env_fn`` built, as the worker process would."""

    def __init__(self, env):
        self.env = env
        self.calls = 0

    def env_call(self, method, args=None, kwargs=None, target="robosuite"):
        assert target == "self"
        self.calls += 1
        return getattr(self.env, method)(*(args or []), **(kwargs or {}))


def test_libero_reads_obj_body_id_in_the_worker():
    """The worker env (OffScreenRenderEnv -> BDDL domain) answers with every body in LIBERO's
    ``obj_body_id``, movable objects and fixtures alike."""
    domain = SimpleNamespace(
        sim=_sim({"akita_black_bowl_1": 3, "wooden_cabinet_1": 5}),
        obj_body_id={"akita_black_bowl_1": 3, "wooden_cabinet_1": 5},
    )
    env = libero._exposing_poses(lambda: SimpleNamespace(env=domain))()
    worker = _Worker(env)
    facade = object.__new__(libero.LiberoEnvFacade)
    facade._env = SimpleNamespace(env=SimpleNamespace(workers=[worker]))
    facade._env_idx = 0
    out = facade.ground_truth_poses()
    assert out == {
        "frame": "world",
        "poses": {
            "akita_black_bowl_1": {"pos": [3.0, 6.0, 9.0], "quat_xyzw": XYZW},
            "wooden_cabinet_1": {"pos": [5.0, 10.0, 15.0], "quat_xyzw": XYZW},
        },
    }
    assert list(facade.ground_truth_poses(["wooden_cabinet_1"])["poses"]) == [
        "wooden_cabinet_1"
    ]
    # An unknown name is refused by the facade: the worker only ever returns the full list
    # (an exception in the worker loop would kill the env).
    with pytest.raises(ValueError, match="plate"):
        facade.ground_truth_poses(["plate"])
    assert worker.calls == 3


def test_libero_registers_the_method():
    facade = libero.LiberoEnvFacade(SimpleNamespace(), meta={})
    assert facade._rpc["env.ground_truth_poses"] == facade.ground_truth_poses


def test_robocasa_lists_objects_then_fixtures():
    facade = object.__new__(RoboCasaEnvFacade)
    facade.env = SimpleNamespace(
        sim=_sim({"obj_main": 1, "sink_main_group_main": 2}),
        obj_body_id={"obj": 1},
        fixtures={
            "sink_main_group": SimpleNamespace(root_body="sink_main_group_main"),
            # A fixture without a body of its own is left out.
            "wall": SimpleNamespace(root_body="wall_missing"),
        },
    )
    out = facade.ground_truth_poses()
    assert list(out["poses"]) == ["obj", "sink_main_group"]
    assert out["poses"]["sink_main_group"] == {
        "pos": [2.0, 4.0, 6.0],
        "quat_xyzw": XYZW,
    }
    with pytest.raises(ValueError, match="wall"):
        facade.ground_truth_poses(["wall"])


def test_maniskill_lists_actors_and_articulations_but_not_the_robot():
    def body(p):
        return SimpleNamespace(
            pose=SimpleNamespace(p=np.array([p]), q=np.array([WXYZ]))
        )

    robot = body([0.0, 0.0, 0.0])
    env = SimpleNamespace(
        scene=SimpleNamespace(
            actors={"cube": body([0.1, 0.2, 0.02]), "goal_site": body([0.0, 0.1, 0.3])},
            articulations={"panda_wristcam": robot, "drawer": body([0.5, 0.0, 0.0])},
        ),
        agent=SimpleNamespace(robot=robot),
    )
    facade = object.__new__(maniskill.ManiskillEnvFacade)
    facade._env = SimpleNamespace(unwrapped=env)
    out = facade.ground_truth_poses()
    assert list(out["poses"]) == ["cube", "drawer", "goal_site"]
    assert out["poses"]["cube"] == {"pos": [0.1, 0.2, 0.02], "quat_xyzw": XYZW}
    with pytest.raises(ValueError, match="panda_wristcam"):
        facade.ground_truth_poses(["panda_wristcam"])
