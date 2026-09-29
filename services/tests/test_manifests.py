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

"""The primitive manifests (packages/embodied/src/primitives/manifests): every file loads, every
entry has one tier and a valid schema, the shared entries resolve, and the loader refuses what the
schema forbids. Each robot's own test runs its server's startup self-check against its manifest."""

from __future__ import annotations

import json

import pytest

from pi_embodied_services.components import manifest as M

ROBOTS = sorted(p.stem for p in M.manifest_dir().glob("*.json"))


def test_the_manifests_are_found_from_the_repo():
    assert M.manifest_dir().is_dir()
    assert "robosuite" in ROBOTS


@pytest.mark.parametrize("robot", ROBOTS)
def test_every_manifest_loads_with_one_tier_per_entry(robot):
    m = M.load_manifest(robot)
    assert m["primitives"]
    for e in m["primitives"]:
        assert e["tier"] in M.TIERS
        assert e["side"] in M.SIDES
    assert m["digest"] == M.load_manifest(robot)["digest"]


def write(tmp_path, robot: dict, common: dict | None = None):
    (tmp_path / "common").mkdir()
    for name, body in (common or {}).items():
        (tmp_path / "common" / f"{name}.json").write_text(json.dumps(body))
    (tmp_path / f"{robot['robot']}.json").write_text(json.dumps(robot))
    return tmp_path


def entry(**kw):
    return {
        "name": "move",
        "side": "env",
        "method": "env.move",
        "tier": "low",
        "doc": {"code": "Move."},
        **kw,
    }


@pytest.mark.parametrize(
    "bad, match",
    [
        (entry(tier="medium"), "tier must be one of"),
        (entry(side="server"), "side must be one of"),
        (entry(method="env.reset"), "never a primitive"),
        (entry(method="stop"), "never a primitive"),
        (entry(doc={}), "doc must have"),
        (entry(extra=1), "unknown field"),
        (entry(params={"x": {"type": "tensor"}}), "type must be one of"),
        (entry(params={"x": {"type": "enum"}}), "enum needs values"),
        (
            {
                "name": "t",
                "side": "ts",
                "tier": "low",
                "method": "env.t",
                "doc": {"tool": "t"},
            },
            "no RPC method",
        ),
    ],
)
def test_the_loader_refuses_what_the_schema_forbids(tmp_path, bad, match):
    root = write(tmp_path, {"robot": "r", "primitives": [bad]})
    with pytest.raises(M.ManifestError, match=match):
        M.load_manifest("r", root)


def test_shared_entries_resolve_with_overrides_and_bring_their_internal_methods(
    tmp_path,
):
    root = write(
        tmp_path,
        {"robot": "r", "primitives": [{"use": "kit/move", "tier": "high"}]},
        {"kit": {"internal": ["env.kit_state"], "primitives": [entry()]}},
    )
    m = M.load_manifest("r", root)
    assert (
        m["primitives"][0]["tier"] == "high"
        and m["primitives"][0]["method"] == "env.move"
    )
    assert "env.kit_state" in m["internal"]
    # The digest names the bytes of every file used.
    before = m["digest"]
    (root / "common" / "kit.json").write_text(
        json.dumps(
            {
                "internal": ["env.kit_state"],
                "primitives": [entry(doc={"code": "Moves."})],
            }
        )
    )
    assert M.load_manifest("r", root)["digest"] != before


def test_a_name_may_repeat_only_as_its_privileged_variant(tmp_path):
    root = write(
        tmp_path, {"robot": "r", "primitives": [entry(), entry(method="env.move2")]}
    )
    with pytest.raises(M.ManifestError, match="twice"):
        M.load_manifest("r", root)
    root2 = tmp_path / "b"
    root2.mkdir()
    write(
        root2,
        {
            "robot": "r",
            "primitives": [entry(), entry(method="env.truth", tier="privileged")],
        },
    )
    assert len(M.load_manifest("r", root2)["primitives"]) == 2


@pytest.mark.parametrize("robot", ROBOTS)
def test_pi_side_tools_are_tool_mode_only_and_never_reach_code_api(robot):
    """The VLA / skill loops (pi0_pick, rldx_skill, lingbot_act, the Franka VLA grasps, ...) are
    ``side: ts``: pi runs them as tools; no source spec (CaP-X's API, RPent's skills) puts them in a
    program's namespace. Guard the boundary: no code doc, never in the whitelist."""
    m = M.load_manifest(robot)
    pi_side = {e["name"] for e in m["primitives"] if e["side"] == "ts"}
    for e in m["primitives"]:
        if e["side"] == "ts":
            assert "code" not in e["doc"], (
                f"{robot}.{e['name']}: a ts tool has no code doc"
            )
            assert "method" not in e, (
                f"{robot}.{e['name']}: a ts tool has no RPC method"
            )
    served = {p.name for p in M.code_primitives(m, lambda _c: True)}
    other = {e["name"] for e in m["primitives"] if e["side"] != "ts"}
    assert not (pi_side - other) & served, f"{robot}: pi-side tools in code.api"
