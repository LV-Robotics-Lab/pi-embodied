# capx: env_configs/human_oracle_code/robosuite/franka_robosuite_cube_lifting_privileged_oracle.yaml @53e9966
# program: capx/envs/tasks/franka/franka_lift.py ORACLE_CODE (FrankaControlPrivilegedApi)
# task: Lift
# tier: privileged
# prelude: capx_privileged.py
OBJECTS = {"red cube": {"name": "cube", "extent": [0.05, 0.05, 0.05]}}

# ---- CaP-X's program, verbatim ----
import numpy as np

# Get a grasp pose for the red cube
grasp_pos, grasp_quat = sample_grasp_pose("red cube")

# Open the gripper before approaching
open_gripper()

# Approach the grasp pose from above (0.1 m offset in Z)
goto_pose(grasp_pos, grasp_quat, z_approach=0.1)

# Move to the exact grasp pose
goto_pose(grasp_pos, grasp_quat)

# Close the gripper to grasp the cube
close_gripper()

# Lift the cube slightly to ensure a safe grasp
lift_offset = np.array([0.0, 0.0, 0.1])  # 10 cm lift
lift_pos = grasp_pos + lift_offset
goto_pose(lift_pos, grasp_quat)
