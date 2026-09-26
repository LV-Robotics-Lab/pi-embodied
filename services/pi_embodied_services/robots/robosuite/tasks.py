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

"""The seven CaP-X robosuite tasks (CaP-X 53e9966 capx/envs/simulators/robosuite_*.py) on
robosuite 1.5: what to build, the task text, the success rule and the motion limits.

Everything that needs no simulator lives here so the success judgement is unit-testable;
``env_server.py`` builds the envs and binds the code-mode primitives. Success is robosuite's own ``_check_success``; Restack adds
CaP-X's rule that both cubes may not be off the table at once (robosuite_cubes_restack.py
task_completed: the stack does not count while the arm carries the whole pile).
"""

from __future__ import annotations

from dataclasses import dataclass, field

#: CaP-X's restack cube: 2 cm half-size (robosuite's Stack uses 2 / 2.5 cm).
RESTACK_CUBE_HALF = 0.02
#: Restack: a cube more than this above the table counts as off the table.
RESTACK_OFF_TABLE_M = 0.04


@dataclass(frozen=True)
class Task:
    #: robosuite env id (``suite.make``).
    env: str
    #: Arms, named as robosuite names them (``robot0``, ``robot1``).
    arms: tuple[str, ...]
    language: str
    #: Extra ``suite.make`` kwargs.
    kwargs: dict = field(default_factory=dict)
    #: The wiping gripper has no fingers: no gripper command, no gripper tools.
    gripper: bool = True
    #: ``robot0_robotview`` for one arm; the two-arm tasks use CaP-X's overhead ``agentview``
    #: (robosuite_two_arm_lift.py: cam_pos [1.5, 0, 2.5], cam_quat [0.653, 0.271, 0.271, 0.653])
    #: so both arms and the object are in view.
    camera: str = "robot0_robotview"
    #: The lowest TCP z the server accepts, m above the table top.
    z_floor_m: float = 0.005
    #: CaP-X's Restack: robosuite's Stack with two 4 cm cubes, the green one placed on the red
    #: one at reset (env_server.RestackStack), and the off-table rule in the success check.
    restack: bool = False


TWO_ARM = {"env_configuration": "opposed"}
TASKS: dict[str, Task] = {
    "Lift": Task("Lift", ("robot0",), "pick up the red cube and lift it"),
    "Stack": Task(
        "Stack",
        ("robot0",),
        "pick up the red cube and gently stack it on top of the green cube, then release it",
    ),
    "Restack": Task(
        "Stack",
        ("robot0",),
        "the green cube sits on the red cube: move the green cube off, then gently place the "
        "red cube on top of the green cube and open the gripper",
        restack=True,
    ),
    # The wiping gripper (a sponge) presses on the table; the floor is the table itself.
    "Wipe": Task(
        "Wipe",
        ("robot0",),
        "wipe up the dirt on the table",
        gripper=False,
        z_floor_m=-0.02,
    ),
    "NutAssemblySquare": Task(
        "NutAssemblySquare",
        ("robot0",),
        "grasp the square nut by its handle and insert it onto the square peg",
    ),
    "TwoArmLift": Task(
        "TwoArmLift",
        ("robot0", "robot1"),
        "the two arms lift the pot together: robot0 grasps one handle, robot1 the other, "
        "then both lift it",
        kwargs=TWO_ARM,
        camera="agentview",
    ),
    "TwoArmHandover": Task(
        "TwoArmHandover",
        ("robot0", "robot1"),
        "robot0 picks up the hammer and hands it over; robot1 grasps the hammer handle and "
        "robot0 lets go",
        kwargs={**TWO_ARM, "prehensile": True},
        camera="agentview",
    ),
}
TASK_NAMES = tuple(TASKS)
#: CaP-X's overhead agentview for the two-arm tasks (world frame, MuJoCo wxyz quaternion).
OVERHEAD_CAMERA = {"pos": [1.5, 0.0, 2.5], "quat_wxyz": [0.653, 0.271, 0.271, 0.653]}
#: OSC_POSE (robosuite BASIC composite controller): action 1.0 = 5 cm / 0.5 rad per step.
OSC_POS_MAX_M = 0.05
OSC_ROT_MAX_RAD = 0.5


def restack_success(stacked: bool, heights_above_table: tuple[float, float]) -> bool:
    """CaP-X's Restack judgement: robosuite's stack check, refused while both cubes are
    more than ``RESTACK_OFF_TABLE_M`` above the table (the arm lifted the pile)."""
    if not stacked:
        return False
    a, b = heights_above_table
    return not (a > RESTACK_OFF_TABLE_M and b > RESTACK_OFF_TABLE_M)


def latch(previous: int | None, success: bool, step: int) -> int | None:
    """The env step of the first success, kept once set (a later release or knock-over does
    not undo it): LIBERO's ``success_once`` convention."""
    if previous is not None:
        return previous
    return step if success else None


def check_move(
    current: tuple[float, ...],
    target: tuple[float, ...],
    *,
    max_move_m: float,
    box: tuple[float, float, float, float] | None,
    z_floor: float,
    z_ceiling: float,
) -> None:
    """The server-side limits of one ``move_to``: the per-call travel cap, the workspace x/y
    box and the z floor / ceiling (world frame). Raises with the offending value; nothing moves."""
    dist = sum((t - c) ** 2 for t, c in zip(target, current)) ** 0.5
    if not dist <= max_move_m:
        raise ValueError(
            f"the move travels {dist:.3f} m; the limit is {max_move_m} m per call. "
            "Split the motion into smaller calls."
        )
    x, y, z = target
    if box is not None and not (box[0] <= x <= box[1] and box[2] <= y <= box[3]):
        raise ValueError(
            f"target x={x:.3f} y={y:.3f} is outside the workspace "
            f"x {box[0]}..{box[1]}, y {box[2]}..{box[3]}"
        )
    if not z >= z_floor:
        raise ValueError(f"target z={z:.3f} is below the floor {z_floor:.3f} m")
    if not z <= z_ceiling:
        raise ValueError(f"target z={z:.3f} is above the ceiling {z_ceiling:.3f} m")
