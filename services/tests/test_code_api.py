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

"""The primitive registry (components/code_api.py) and its first user, the Polymetis Franka."""

from __future__ import annotations

import pytest

from pi_embodied_services.components.code_api import CodeApi, Param, Primitive
from pi_embodied_services.robots.franka.primitives import (
    DUAL_FRANKA_PRIMITIVES,
    FRANKA_PRIMITIVES,
)

RPC = {"env.state": object(), "env.move": object(), "env.truth": object()}


def api() -> CodeApi:
    return CodeApi(
        [
            Primitive("state", "env.state", "The state."),
            Primitive(
                "move",
                "env.move",
                "Move.",
                {"d": Param("vec3"), "slow": Param("boolean", required=False)},
                True,
            ),
            Primitive("truth", "env.truth", "Ground truth.", tiers=("privileged",)),
            Primitive("frames", "env.state", "Raw frames.", tiers=("low",)),
        ],
        RPC,
    )


def test_tiers_list_their_primitives_and_privileged_extends_high():
    a = api()
    names = lambda tier: [p.name for p in a.primitives(tier)]  # noqa: E731
    assert names(None) == ["state", "move", "frames"]
    assert names("high") == ["state", "move"]
    assert names("low") == ["state", "move", "frames"]
    assert names("privileged") == ["state", "move", "truth"]
    with pytest.raises(ValueError, match="tier must be one of"):
        a.primitives("root")


def test_describe_carries_a_digest_that_changes_with_the_api():
    d = api().describe("high")
    assert [p["name"] for p in d["primitives"]] == ["state", "move"]
    assert d["primitives"][1]["mutating"] is True
    assert d["primitives"][1]["params"]["slow"] == {
        "type": "boolean",
        "description": "",
        "required": False,
    }
    assert d["digest"] == api().describe("high")["digest"]
    assert d["digest"] != api().describe("low")["digest"]


def test_resolve_allows_only_declared_primitives_and_parameters():
    a = api()
    assert a.resolve("move", {"d": [0, 0, 0.01]}) == ("env.move", {"d": [0, 0, 0.01]})
    with pytest.raises(ValueError, match="'truth' is not a primitive"):
        a.resolve("truth", {})
    assert a.resolve("truth", {}, tier="privileged") == ("env.truth", {})
    with pytest.raises(ValueError, match="'frames' is not a primitive"):
        a.resolve("frames", {}, tier="high")
    with pytest.raises(ValueError, match="unknown parameter"):
        a.resolve("move", {"d": [0, 0, 0], "fast": True})
    with pytest.raises(ValueError, match="missing parameter"):
        a.resolve("move", {})


@pytest.mark.parametrize(
    "bad, match",
    [
        (Primitive("state", "env.nope", "x"), "not registered"),
        (Primitive("not a name", "env.state", "x"), "identifier"),
        (Primitive("x", "env.state", "x", tiers=()), "tiers"),
        (
            Primitive("x", "env.state", "x", tiers=("high", "privileged")),
            "privileged primitive",
        ),
        (Primitive("x", "env.state", "x", {"p": Param("tensor")}), "bad parameter"),
    ],
)
def test_declarations_are_validated(bad, match):
    with pytest.raises(ValueError, match=match):
        CodeApi([bad], RPC)
    with pytest.raises(ValueError, match="declared twice"):
        CodeApi(
            [Primitive("s", "env.state", "x"), Primitive("s", "env.state", "y")], RPC
        )


def test_franka_primitives_run_through_the_limited_facade_methods():
    # Every declared primitive is one of the facade's motion/state methods, which apply the limits.
    methods = {p.method for p in FRANKA_PRIMITIVES}
    assert methods == {
        "env.get_robot_state",
        "env.get_observation",
        "env.get_camera_meta",
        "env.move_delta",
        "env.rotate_delta",
        "env.set_gripper",
    }
    assert {p.name for p in FRANKA_PRIMITIVES if p.mutating} == {
        "move_delta",
        "rotate_delta",
        "set_gripper",
    }
    for prims in (FRANKA_PRIMITIVES, DUAL_FRANKA_PRIMITIVES):
        assert not any("privileged" in p.tiers for p in prims), (
            "a real robot has no ground truth"
        )
    CodeApi(FRANKA_PRIMITIVES, {m: object() for m in methods})
    # Two arms: every motion names its arm.
    dual = {p.name: p for p in DUAL_FRANKA_PRIMITIVES}
    assert all(
        "arm" in dual[n].params for n in ("move_delta", "rotate_delta", "set_gripper")
    )
    CodeApi(
        DUAL_FRANKA_PRIMITIVES, {p.method: object() for p in DUAL_FRANKA_PRIMITIVES}
    )
