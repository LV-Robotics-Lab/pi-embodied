# capx: env_configs/human_oracle_code/robosuite/franka_robosuite_cube_restack_privileged_oracle.yaml @53e9966
# program: capx/envs/tasks/franka/franka_cube_restack.py ORACLE_CODE (FrankaControlPrivilegedApi)
# task: Restack
# tier: privileged
# The CaP-X API calls are the env server's high-tier primitives (manifests/robosuite.json): no prelude.
# ---- CaP-X's program, verbatim ----
import numpy as np

_, _, green_ext = get_object_pose("green cube", return_bbox_extent=True)
_, _, red_ext = get_object_pose("red cube", return_bbox_extent=True)


# Sample a grasp pose for the green cube and pick it up
pick_pos, pick_quat = sample_grasp_pose("green cube")
goto_pose(pick_pos, pick_quat, z_approach=0.1)
close_gripper()
# Lift the green cube after grasping
post_pick_pos = pick_pos.copy()
post_pick_pos[0] -= 0.15
post_pick_pos[2] += 0.1
goto_pose(post_pick_pos, pick_quat)
open_gripper()

# Sample a grasp pose for the red cube and pick it up
pick_pos, pick_quat = sample_grasp_pose("red cube")
goto_pose(pick_pos, pick_quat, z_approach=0.1)
close_gripper()
# Lift the red cube after grasping
post_pick_pos = pick_pos.copy()
post_pick_pos[0] -= 0.15
post_pick_pos[2] += 0.3
goto_pose(post_pick_pos, pick_quat)

# Compute placement pose on top of the green cube
green_pos, _, _ = get_object_pose("green cube", return_bbox_extent=False)

place_pos = green_pos.copy()
place_pos[2] = green_pos[2] + green_ext[2] + red_ext[2]
# Use down orientation for placement
place_quat = np.array([0.0, 0.0, 1.0, 0.0])

# Approach and place the red cube on the green cube
goto_pose(place_pos, pick_quat, z_approach=0.1)
open_gripper()

# Retract after placing
post_place_pos = place_pos.copy()
post_place_pos[2] += 0.1
goto_pose(post_place_pos, place_quat)
