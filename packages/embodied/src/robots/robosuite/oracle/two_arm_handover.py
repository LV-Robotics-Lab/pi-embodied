# capx: env_configs/human_oracle_code/robosuite/two_arm_handover_oracle.yaml @53e9966
# program: capx/envs/tasks/franka/two_arm_handover.py UNPRIVILEGED_ORACLE_CODE (FrankaHandoverApi)
# task: TwoArmHandover
# tier: low
# CaP-X's task API (not in the high tier) is the prelude over the low tier's primitives.
# prelude: capx_handover.py
# CaP-X's config names no oracle_code, so TwoArmHandoverCodeEnv's class default,
# PRIVILEGED_ORACLE_CODE, would run against FrankaHandoverApi and stop at get_hammer_pose (not
# among that API's functions). This port runs the same file's UNPRIVILEGED_ORACLE_CODE, the
# program written for FrankaHandoverApi.

# ---- CaP-X's program, verbatim ----
import numpy as np

# get poses
handle_pos, handle_quat, _ = get_object_pose('hammer') # unprivileged
handle_pos[2] -= 0.025  # handle_pos is slightly high sometimes

# pickup quat
gripper_down_quat = np.array([0, 1, 0, 0]) # handover quat
gripper_pick_quat = np.array([0, 0.707, 0.707, 0]) # down grip quat

# handover pos
arm0_pos = np.array([0.44, 0.0, 0.0]) # approx init positions
arm1_pos = np.array([1.18, 0.0, 0.0])
handover_pos = (arm0_pos + arm1_pos) / 2
handover_pos[2] = 0.10

# --- Sequence ---
# Arm0: pick up hammer at actual handle pose
open_gripper_arm0()
goto_pose_arm0(handle_pos, gripper_pick_quat, z_approach=0.15)

close_gripper_arm0()
goto_pose_arm0(handle_pos + np.array([0, 0, 0.1]), gripper_pick_quat)
goto_pose_arm0(handle_pos + np.array([0, 0, 0.2]), gripper_pick_quat)

# Arm0: move to handover (shifted toward arm1)
goto_pose_arm0(handover_pos, gripper_down_quat)

# Arm1 approach
arm1_quat = np.array([0, 0, 1, 0])
open_gripper_arm1()
goto_pose_arm1(handover_pos + np.array([0.1, 0, -0.01]), arm1_quat, z_approach=0.12) # account for hammer length
close_gripper_arm1()

# Arm0: release and retract
open_gripper_arm0()
goto_pose_arm0(handover_pos + np.array([-0.1, 0, 0.06]), gripper_down_quat)
