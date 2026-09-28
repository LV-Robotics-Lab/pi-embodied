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

"""utils/grasp_chain.py: Metaworld's and Genesis's env.execute_grasp / env.execute_place (the
chain pi's src/primitives/grasp-chain.ts ran before the manifests moved it to the server)."""

from __future__ import annotations

import math

import numpy as np
import pytest

from pi_embodied_services.utils import grasp_chain as chain

DOWN = [0, 0, -1]


def test_split_and_tilt():
    legs = chain.split([0, 0, 0.3], [0, 0.1, 0.05], 0.1)
    assert len(legs) == 3
    assert all(np.linalg.norm(d) <= 0.1 + 1e-12 for d in legs)
    total = np.sum(legs, axis=0)
    assert total[1] == pytest.approx(0.1) and total[2] == pytest.approx(-0.25)
    assert len(chain.split([0, 0, 0], [0, 0, 0], 0.1)) == 1
    assert chain.tilt(DOWN) == 0
    assert chain.tilt([1, 0, 0]) == pytest.approx(math.pi / 2)


def rig(approach, *, stall=False, solved_after=None):
    """A robot whose EEF follows the commanded deltas exactly, unless ``stall`` blocks the
    closed legs."""
    pos = np.array([0, 0.6, 0.3])
    calls: list = []

    def resolve_grasp(grasp_id, standoff=0.0):
        calls.append(("env.resolve_grasp", {"grasp_id": grasp_id}))
        return {"approach": approach}

    def claim_waypoints(grasp_id, **kw):
        calls.append(("env.claim_waypoints", {"grasp_id": grasp_id, **kw}))
        return {
            "kind": "grasp",
            "waypoints": {
                "pre_grasp": [0, 0.6, 0.2],
                "grasp": [0, 0.6, 0.1],
                "lift": [0, 0.6, 0.2],
            },
            "steps": [
                {"to": "pre_grasp", "gripper": -1},
                {"to": "grasp", "gripper": -1},
                {"gripper": 1},
                {"to": "lift", "gripper": 1},
            ],
            "eef_yaw": 0,
            "expired_ids": ["g1"],
        }

    def move(delta, g):
        nonlocal pos
        calls.append(("move", (list(delta), g)))
        if stall and g == "close":
            return {"error": "blocked"}
        pos = pos + delta
        return {"control_steps": 3, "frames": ["f"]}

    def gripper(g):
        calls.append(("gripper", g))
        return {"control_steps": 5}

    moves = lambda: sum(1 for c in calls if c[0] == "move")  # noqa: E731

    def run(kind="grasp", grasp_id="g1", **kw):
        return chain.run_chain(
            kind,
            grasp_id,
            rpc={
                "env.resolve_grasp": resolve_grasp,
                "env.claim_waypoints": claim_waypoints,
            },
            current=lambda: pos,
            max_step=0.04,
            move=move,
            gripper=gripper,
            stop=lambda: False,
            solved=lambda: solved_after is not None and moves() >= solved_after,
            **kw,
        )

    return run, calls


def test_a_grasp_resolves_claims_once_and_runs_bounded_legs():
    run, calls = rig(DOWN)
    out = run(standoff=0.1)
    assert [c for c in calls if c[0].startswith("env.")] == [
        ("env.resolve_grasp", {"grasp_id": "g1"}),
        ("env.claim_waypoints", {"grasp_id": "g1", "standoff": 0.1}),
    ]
    moves = [a for m, a in calls if m == "move"]
    assert all(np.linalg.norm(d) <= 0.04 + 1e-12 for d, _ in moves)
    assert list(dict.fromkeys(g for _, g in moves)) == ["open", "close"]
    assert ("gripper", "close") in calls
    assert [leg.get("to", leg["gripper"]) for leg in out["legs"]] == [
        "pre_grasp",
        "grasp",
        "close",
        "lift",
    ]
    assert "stalled" not in out and out["expired_ids"] == ["g1"]
    assert out["control_steps"] == 3 * len(moves) + 5
    assert out["frames"] == ["f"] * len(moves)


def test_a_tilted_candidate_is_refused_and_a_blocked_leg_stalls():
    run, calls = rig([1, 0, -0.2])
    out = run(grasp_id="g2")
    assert out["refused"] and "next_after" in out["error"]
    assert not any(c[0] in ("env.claim_waypoints", "move") for c in calls)
    run, _ = rig(DOWN, stall=True)
    out = run()
    assert out["stalled"] and out["legs"][-1]["error"] == "blocked"
    assert len(out["legs"]) == 4


def test_a_solved_task_ends_the_chain():
    run, calls = rig(DOWN, solved_after=1)
    out = run()
    assert len(out["legs"]) == 1 and "stalled" not in out
    assert sum(1 for c in calls if c[0] == "move") == 1


def test_a_place_id_is_refused_by_execute_grasp():
    run, _ = rig(DOWN)
    with pytest.raises(ValueError, match="use execute_grasp"):
        run(kind="place")


def test_a_candidate_turned_off_the_fixed_hand_is_refused_and_a_close_notes_the_grasp():
    """main's c306d9765 and ddd7bd92a, on the server's chain."""
    run, calls = rig(DOWN)
    import math

    out = run(yaw=lambda: math.pi / 2)  # the resolved candidate closes at yaw 0
    assert out["refused"] and "cannot turn" in out["error"]
    assert not any(c[0] in ("env.claim_waypoints", "move") for c in calls)
    assert chain.yaw_gap(0.0, math.pi) == 0.0, (
        "a parallel gripper turned half is the same"
    )
    run2, _ = rig(DOWN)
    assert "refused" not in run2(yaw=lambda: 0.1)
    # env.note_grasp_closed is called once the fingers closed (when the server serves it).
    noted = []

    def with_note(**kw):
        return chain.run_chain(
            "grasp",
            "g1",
            rpc={
                "env.resolve_grasp": lambda grasp_id, standoff=0.0: {
                    "approach": DOWN,
                    "eef_yaw": 0.0,
                },
                "env.claim_waypoints": lambda grasp_id, **k: {
                    "kind": "grasp",
                    "waypoints": {"grasp": [0, 0.6, 0.3]},
                    "steps": [{"to": "grasp", "gripper": -1}, {"gripper": 1}],
                },
                "env.note_grasp_closed": lambda: noted.append(1),
            },
            current=lambda: np.array([0, 0.6, 0.3]),
            max_step=0.04,
            move=lambda d, g: {},
            gripper=lambda g: {},
            stop=lambda: False,
            solved=lambda: False,
            **kw,
        )

    assert "stalled" not in with_note()
    assert noted == [1]
