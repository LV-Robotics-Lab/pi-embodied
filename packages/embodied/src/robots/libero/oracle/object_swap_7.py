# capx: env_configs/human_oracle_code/libero/franka_libero_object_swap_7_oracle.yaml @53e9966
# program: the config's oracle_code (FrankaLiberoApi)
# suite: libero_object_swap
# task: 7
# tier: high
# prelude: capx_libero.py
# The CaP-X API calls are the env server's high-tier primitives (manifests/libero.json); the
# prelude only stubs the unused viser import.
# ---- CaP-X's program, verbatim ----
import numpy as np
import viser.transforms as vtf
use_multiview = True

object_name = "milk carton"
target_name = "woven basket"

open_gripper()
grasp_pos, grasp_quat = sample_grasp_pose(object_name, use_multiview=use_multiview)

# grasp_quat = np.array([0.0, 1.0, 0.0, 0.0])
goto_pose(grasp_pos, grasp_quat, z_approach=0.1)
close_gripper()
goto_pose(grasp_pos+np.array([0.0, 0.0, 0.2]), grasp_quat)

basket_pos, basket_quat = get_object_pose(target_name, use_multiview=use_multiview)
basket_quat = np.array([0.0, 1.0, 0.0, 0.0])
goto_pose(basket_pos, basket_quat, z_approach=0.1)
open_gripper()
