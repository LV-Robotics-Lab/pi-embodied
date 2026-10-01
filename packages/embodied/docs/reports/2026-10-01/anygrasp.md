# AnyGrasp validation — 2026-10-01

## Result

The machine-bound license passed on the bjb2 server. Detection model inference and one LIBERO grasp-and-lift diagnostic passed. This is not a full task-success benchmark or physical-robot acceptance. Tracking was not exercised.

- SDK: `b8eaafc9eca7babd5208e7a5ade3c561060be4c5`.
- Base pi-embodied commit: `9e4017000ca527e2d70295c428df9edc78c8444e`, plus the AnyGrasp adapter and tests in this change.
- Detection weights: the author's supplied Google Drive file `1jNvqOOf_fR3SWkXuz8TAzcHH9x8gE8Et`.
- SHA-256: `a05c3690b95c8b65e78b1bb8a28f1d5ca96613391946e450afacae840bbcf7b2`.
- GPU: RTX 5090, GPU 1. Existing Qwen on GPU 0 remained running.

## Fix

The adapter still used `AnyGrasp(...).load_net()` and the old `get_grasp(points, colors, ...)` signature. The installed licensed SDK exposes `create_detector(config)` and returns a GraspGroup from `get_grasp(points, options)`.

The adapter now uses the current API, rejects a failed detector factory, aligns the target-region mask with valid depth points, and steers grasps within 30 degrees of downward when the environment supplies camera-frame world-up. Existing object-distance filtering, coordinate conversion and motion limits remain in force.

## Evidence

1. The SDK license checker returned `PASSED`.
2. Two RPC predictions on SDK `example_data`, object label 1, each produced 43 candidates with collision filtering enabled. The API returned the top 10. Measured request times were 1.30 s and 0.41 s; these are two diagnostic calls, not a latency benchmark.
3. LIBERO `libero_spatial`, task 0, seed 0: SAM3 located the black bowl (2,435 pixels). Initial unconstrained predictions failed the robot's approach-angle guard. With SDK approach steering, planning returned 10 eligible candidates.
4. The initial direct approach exceeded the 0.30 m lateral-motion limit. The diagnostic moved at carry height through an intermediate waypoint, then segmented and planned again. It did not disable the guard or reuse stale grasp IDs.
5. `execute_grasp` completed in 78 steps. Final waypoint errors were 8.8 mm, 9.7 mm and 10.3 mm; final gripper width was 14.8 mm.
6. Post-lift SAM3 localization changed bowl height from 0.9321 m to 1.0232 m (about +9.1 cm). The saved agent-view image shows the bowl held above the table. Localization points are visible-surface estimates, not ground-truth center-of-mass measurements.
7. The LIBERO task itself remained `terminated: false`: placing the bowl at the task target was not tested.

## Checks

- 47 Python tests passed, including 4 new SDK-boundary tests; Ruff lint and formatting passed.
- The service test wrapper emitted an existing pytest configuration warning (`Unknown config option: timeout`); no test failures occurred.
- Root `npm run check` passed in full.

## Server setup and rerun

License files are stored only under `/root/autodl-tmp/src/anygrasp_sdk/grasp_detection/license/`, directory mode 0700 and file mode 0600. They are not included in this repository or report.

The runtime uses `/root/autodl-tmp/venvs/graspnet1b/bin/python` with PyTorch 2.7.1+cu128 and MinkowskiEngine 0.5.4. GraspNet API dependencies were installed with NumPy pinned to 1.26.4. The SDK's pointnet2 Python package uses the existing compiled pointnet2 extension; real inference verified that combination.

The server-local launcher and diagnostic scripts are in `/root/autodl-tmp/tools/anygrasp-20261001/`. Run `bash /root/autodl-tmp/tools/anygrasp-20261001/start.sh` to acquire the GPU 1 gate and serve AnyGrasp on loopback port 18984. Configure pi/environment with `--anygrasp http://127.0.0.1:18984`.

Raw results are in `/root/autodl-tmp/runs/anygrasp-20261001/` (`inference.json`, `libero.json`, and lift images). Test services are stopped after validation to release GPU 1.
