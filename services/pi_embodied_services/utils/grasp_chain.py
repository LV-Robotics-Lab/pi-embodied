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

"""``env.execute_grasp`` / ``env.execute_place`` for a robot whose only motion is a bounded
Cartesian ``move_delta`` with its gripper held pointing down (Metaworld's Sawyer, Genesis's
Panda): the planned grasp chain of utils/grasp.py (``plan_grasp`` -> an id) runs from one
resolution of its id, on the server, so the tool and a program share the limits, the stop
between env steps and the success latch of the robot's own ``move_delta``.

The id is resolved first (``env.resolve_grasp``, read-only) and refused unless its approach
points within ``max_tilt_rad`` of straight down: the gripper cannot turn, so a tilted grasp
would close somewhere else. Then ``env.claim_waypoints`` fixes the whole path from that one
candidate (pre-grasp standoff, grasp, lift; or pre-place, place, retreat) and each leg runs as
deltas of at most ``max_step`` metres holding the leg's gripper command; an in-place step closes
or opens. A leg that ends further than ``tol_m`` from its waypoint, a move that errors, a stop or
a solved task ends the chain (``stalled``). The motions expire the id either way.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from typing import Any

import numpy as np

#: The steepest approach a fixed, downward gripper can take, rad (20 deg).
MAX_TILT_RAD = 0.35
#: The largest yaw difference (mod pi) between a grasp and the fixed hand, rad (20 deg).
MAX_YAW_RAD = 0.35
#: A leg counts as reached within this distance, m.
TOL_M = 0.01


def tilt(approach) -> float:
    """The approach's angle from straight down (world -z), rad."""
    a = np.asarray(approach, dtype=np.float64).reshape(3)
    n = float(np.linalg.norm(a))
    if not n > 0:
        return math.pi
    return math.acos(max(-1.0, min(1.0, -float(a[2]) / n)))


def yaw_gap(a: float, b: float) -> float:
    """The difference of two gripper yaws, rad in [0, pi/2]: a parallel gripper turned half is the same."""
    d = ((a - b) % math.pi + math.pi) % math.pi
    return min(d, math.pi - d)


def split(start, target, step: float) -> list[np.ndarray]:
    """A straight leg as equal deltas of at most ``step`` metres."""
    d = np.asarray(target, dtype=np.float64) - np.asarray(start, dtype=np.float64)
    n = max(1, math.ceil(float(np.linalg.norm(d)) / step - 1e-9))
    return [d / n for _ in range(n)]


def run_chain(
    kind: str,
    grasp_id: str,
    *,
    rpc: dict[str, Callable[..., Any]],
    current: Callable[[], Any],
    max_step: float,
    move: Callable[[np.ndarray, str], dict],
    gripper: Callable[[str], dict],
    stop: Callable[[], bool],
    solved: Callable[[], bool],
    standoff: float | None = None,
    tol_m: float = TOL_M,
    max_tilt_rad: float = MAX_TILT_RAD,
    yaw: Callable[[], float] | None = None,
    max_yaw_rad: float = MAX_YAW_RAD,
) -> dict:
    """Run one planned grasp (``kind`` "grasp") or place ("place") id; see the module doc.

    ``move(delta, "open"|"close")`` is the robot's bounded move_delta (its result's ``error`` or
    ``cancelled`` ends the chain), ``gripper(command)`` closes or opens in place. Returns the
    chain's report: ``legs`` (each ``to``, ``gripper``, ``final_dist_m``[, ``error``]),
    ``stalled`` / ``error`` when a leg stopped short, ``refused`` for a tilted candidate,
    ``eef_yaw``, ``approach_tilt_deg``, ``expired_ids``, ``control_steps`` and ``frames``
    (the motions' frames, in order, where the robot's calls return them)."""
    name = f"execute_{kind}"
    resolved = rpc["env.resolve_grasp"](grasp_id=grasp_id)
    t = tilt(resolved["approach"])
    report: dict[str, Any] = {"name": name, "id": grasp_id}
    if t > max_tilt_rad:
        return {
            **report,
            "refused": True,
            "error": (
                f"{grasp_id} approaches {round(math.degrees(t), 1)} deg from straight down; "
                f"this gripper only points down (at most {round(math.degrees(max_tilt_rad), 1)}"
                " deg). Ask plan_grasp for the next candidate (next_after)."
            ),
        }
    gap = (
        yaw_gap(float(resolved.get("eef_yaw", 0.0)), float(yaw()))
        if kind == "grasp" and yaw is not None
        else 0.0
    )
    if gap > max_yaw_rad:
        return {
            **report,
            "refused": True,
            "error": (
                f"{grasp_id} closes the fingers {round(math.degrees(gap), 1)} deg off the hand's "
                f"fixed direction; this gripper cannot turn (at most "
                f"{round(math.degrees(max_yaw_rad), 1)} deg). Ask plan_grasp for the next "
                "candidate (next_after)."
            ),
        }
    claim = rpc["env.claim_waypoints"](
        grasp_id=grasp_id, **({} if standoff is None else {"standoff": float(standoff)})
    )
    if claim["kind"] != kind:
        raise ValueError(
            f"{grasp_id} is a {claim['kind']} id; use execute_{claim['kind']}"
        )
    waypoints = claim["waypoints"]
    legs: list[dict] = []
    frames: list = []
    control_steps = 0
    stalled = False

    def absorb(r: dict) -> str | None:
        nonlocal control_steps
        frames.extend(r.get("frames") or [])
        control_steps += int(r.get("control_steps") or r.get("steps_used") or 0)
        if r.get("error"):
            return str(r["error"])
        if r.get("cancelled"):
            return "stopped"
        return None

    for step in claim["steps"]:
        g = "close" if (step.get("gripper") or -1) > 0 else "open"
        if stop():
            stalled = True
            legs.append({"gripper": g, "error": "stopped"})
            break
        if not step.get("to"):
            try:
                error = absorb(gripper(g))
            except Exception as exc:  # a refused call ends the chain, like the tool's
                error = str(exc)
            legs.append({"gripper": g, **({"error": error} if error else {})})
            # Closed: the planner measures the grasp's height above its support now.
            if g == "close" and not error and "env.note_grasp_closed" in rpc:
                try:
                    rpc["env.note_grasp_closed"]()
                except Exception:  # noqa: BLE001 - a measurement, not the chain
                    pass
            if error:
                stalled = True
                break
            continue
        target = np.asarray(waypoints[step["to"]], dtype=np.float64)
        error = None
        for delta in split(current(), target, max_step):
            try:
                error = absorb(move(delta, g))
            except Exception as exc:
                error = str(exc)
            if error or solved():
                break
        dist = float(np.linalg.norm(target - np.asarray(current(), dtype=np.float64)))
        legs.append(
            {
                "to": step["to"],
                "gripper": g,
                "final_dist_m": round(dist, 4),
                **({"error": error} if error else {}),
            }
        )
        # A solved task ends the chain where it is: the rest of the path is moot.
        if not error and solved():
            break
        if error or dist > tol_m:
            stalled = True
            break
    out = {
        **report,
        "legs": legs,
        "eef_yaw": claim.get("eef_yaw"),
        "approach_tilt_deg": round(math.degrees(t), 1),
        "expired_ids": claim.get("expired_ids") or [],
        "control_steps": control_steps,
        "frames": frames,
    }
    if stalled:
        out["stalled"] = True
        out["error"] = "a leg stopped short; plan again from the new observation"
    return out


__all__ = [
    "MAX_TILT_RAD",
    "MAX_YAW_RAD",
    "TOL_M",
    "run_chain",
    "split",
    "tilt",
    "yaw_gap",
]
