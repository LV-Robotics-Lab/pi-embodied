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

"""The primitive registry (components/code_api.py): CaP-X's tiers over a server's RPC methods,
the whitelist every program call goes through."""

from __future__ import annotations

import pytest

from pi_embodied_services.components.code_api import CodeApi, Param, Primitive

RPC = {
    "env.state": object(),
    "env.move": object(),
    "env.truth": object(),
    "env.goto": object(),
}


def api() -> CodeApi:
    return CodeApi(
        [
            Primitive("state", "env.state", "The state.\n\nExample:\n    state()"),
            Primitive(
                "move",
                "env.move",
                "Move.",
                {
                    "d": Param("vec3"),
                    "slow": Param("boolean", required=False),
                    "steps": Param("integer", required=False, minimum=1, maximum=100),
                    "gripper": Param("enum", required=False, values=("open", "close")),
                },
                True,
            ),
            Primitive("goto", "env.goto", "Semantic goto.", tiers=("high",)),
            Primitive("pose", "env.state", "Perceived pose.", tiers=("high",)),
            Primitive("pose", "env.truth", "Ground-truth pose.", tiers=("privileged",)),
            Primitive("frames", "env.state", "Raw frames.", tiers=("raw",)),
        ],
        RPC,
    )


def test_tiers_are_capx_levels_and_privileged_replaces_a_high_primitive():
    a = api()
    names = lambda tier: [(p.name, p.method) for p in a.primitives(tier)]  # noqa: E731
    assert [n for n, _ in names("low")] == ["state", "move"]
    assert [n for n, _ in names("high")] == ["goto", "pose"]
    assert [n for n, _ in names("raw")] == ["frames"]
    assert names("privileged") == [("goto", "env.goto"), ("pose", "env.truth")]
    assert names("low+privileged") == [
        ("state", "env.state"),
        ("move", "env.move"),
        ("pose", "env.truth"),
    ]
    assert names("low-noexamples") == names("low")
    with pytest.raises(ValueError, match="tier must be one of"):
        a.primitives("root")


def test_the_s4_tier_drops_the_examples():
    a = api()
    assert "Example:" in a.describe("low")["primitives"][0]["doc"]
    assert "Example:" not in a.describe("low-noexamples")["primitives"][0]["doc"]
    assert a.describe("low")["digest"] != a.describe("low-noexamples")["digest"]


def test_resolve_allows_only_declared_primitives_parameters_and_values():
    a = api()
    assert a.resolve("move", {"d": [0, 0, 0.01]}, "low") == (
        "env.move",
        {"d": [0, 0, 0.01]},
    )
    with pytest.raises(ValueError, match="'goto' is not a primitive"):
        a.resolve("goto", {}, "low")
    assert a.resolve("pose", {}, "privileged") == ("env.truth", {})
    assert a.resolve("pose", {}, "high") == ("env.state", {})
    with pytest.raises(ValueError, match="unknown parameter"):
        a.resolve("move", {"d": [0, 0, 0], "fast": True}, "low")
    with pytest.raises(ValueError, match="missing parameter"):
        a.resolve("move", {}, "low")
    with pytest.raises(ValueError, match="<= 100"):
        a.resolve("move", {"d": [0, 0, 0], "steps": 500}, "low")
    with pytest.raises(ValueError, match="one of"):
        a.resolve("move", {"d": [0, 0, 0], "gripper": "half"}, "low")


@pytest.mark.parametrize(
    "bad, match",
    [
        (Primitive("state", "env.nope", "x"), "not registered"),
        (Primitive("not a name", "env.state", "x"), "identifier"),
        (Primitive("x", "env.state", "x", tiers=()), "exactly one tier"),
        (Primitive("x", "env.state", "x", tiers=("high", "low")), "exactly one tier"),
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
