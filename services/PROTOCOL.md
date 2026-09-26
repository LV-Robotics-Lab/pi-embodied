# Wire protocol

Every service except the LingBot-VLA launcher speaks the same JSON-over-HTTP RPC
(`pi_embodied_services/utils/rpc/http_rpc.py`, `rpc_facade.py`). pi-embodied's client is
`packages/embodied/src/rpc.ts`.

## Transport

- `POST /call` with a JSON body. Any other path answers `404`.
- Request: `{"method": str, "args": [..], "kwargs": {..}, "session_id": str | null}`.
  `args` and `kwargs` may be omitted. `session_id` is only meaningful for session servers
  (RLDX, below); send `null` otherwise.
- Response: always HTTP `200`, body `{"ok": true, "result": <value>}` or
  `{"ok": false, "error": str, "traceback": str}`. On the RoboCasa env server (which runs
  calls on its main thread) `error` of a failed business call is the full server traceback.
- Tuples come back as JSON arrays. A result that is not JSON-encodable (after the numpy
  rules below) makes the server drop the connection instead of answering.
- The server binds `--host` (default `127.0.0.1`) and `--port` (default `0` = any free
  port) and prints `RPC server listening on http://HOST:PORT` to stdout once bound.
- Only HTTP exists. The pickle-framed `socket` transport was removed (unpickling a
  request is remote code execution for anyone who can reach the port); `--transport`
  accepts only `http`.

### numpy encoding (both directions, applied recursively inside lists and dicts)

- ndarray: `{"__ndarray__": base64(arr.tobytes()), "dtype": str(arr.dtype), "shape": [..]}`
  -- raw C-order bytes in the machine byte order (little-endian on x86/ARM), dtype names as
  numpy prints them (`"float32"`, `"uint8"`, `"float64"`, `"bool"`, ...). The server decodes
  such a dict only if it has no other keys.
- numpy scalar of kind bool/int/uint/float: `{"__npscalar__": value, "dtype": "float32"}`.
  Other numpy scalars are not encodable.
- Torch tensors are converted to numpy on the server before encoding.

## Execution model

- **One call at a time per server process.** Every business call (and `session.*`) takes a
  process-wide lock, so no two calls overlap, including across clients and sessions and when
  a client times out, drops the connection and retries while the first call still runs (the
  first call finishes; the retry waits behind it). Earlier versions let "read-only" methods run in
  parallel; that is gone (a concurrent `env.render_camera` broke LIBERO's worker pipe).
- Lock-free methods, answered at once even while a call runs: `healthz`, `stop`, `cancel`,
  `shutdown` (`shutdown` still waits for the running call before it takes effect).

## Framework methods (every RPC server)

| method | args | result |
|---|---|---|
| `healthz` | - | `{"status": "ok", "version": "<pi-embodied-services version>", "service": "<name>"}` |
| `stop` / `cancel` | - | `{"ok": true, "stop_generation": int, "call_in_progress": bool}` |
| `shutdown` | - | `{"ok": true}`; the process exits after answering |

Service names: `libero-env`, `robocasa-env`, `rldx-vla`, `robotwin-env`, `robolab-env`, `franka-env`,
`dual-franka-env`, `franka-polymetis-env`, `ur5e-env`, `pi05-vla`, `sam3`, `molmo`.

### `stop` semantics

`stop` increments a process-wide stop generation and returns immediately. Then:

1. A call that was received *before* the stop and is still waiting for the lock is rejected
   without running: `{"ok": false, "error": "<method>: cancelled by stop before it started"}`.
2. The call that is executing sees the stop at its next step boundary if it has one (table
   below) and returns early with `"cancelled": true` in its result (an ordinary `ok: true`
   envelope; for motion primitives the inner `ok` reflects whether the target was reached).
3. Calls received *after* the stop run normally; nothing needs to be re-armed.

The stop is process-wide: on a shared server (RLDX with several sessions) it affects every
client's queued calls.

| service | what `stop` interrupts | what it cannot interrupt |
|---|---|---|
| libero-env | queued calls | a running `env.chunk_step` (one RLinf `LiberoEnv.chunk_step`, typically 5 actions), `env.step`, `env.reset`, renders |
| robocasa-env | queued calls | any running call (each is a single robosuite operation; there is no server-side chunk loop) |
| robotwin-env | queued calls; `env.chunk_step` before each native action (`info.cancelled = true`) | the native action being executed (one `take_action`, i.e. one planned qpos/ee motion), `env.step`, `env.reset`, `env.plan_arm_path`, renders |
| robolab-env | queued calls; `env.move_delta` / `env.rotate_delta` before each control step; `env.chunk_step` before each action | the Isaac Lab `env.step` in progress (one control step of 8 physics substeps), `env.reset` |
| franka-env | queued calls; `env.move_delta` / `env.rotate_delta` / `env.set_gripper` before each servo step; `env.chunk_step` after each action | the servo step in progress (one RLinf `env.step`: one Cartesian target plus the pacing sleep, and up to 0.6 s when it toggles the gripper); `env.reset` (RLinf go-to-rest / joint reset) |
| franka-polymetis-env | queued calls; `env.move_delta` / `env.rotate_delta` before each servo tick (setpoint advance <= `servo_step_m` / `servo_step_rad`) and during settle; `env.set_gripper` between width polls; `env.reset` between lift ticks and joint-stream ticks (`reset.method: joint_stream`) | the ZeroRPC call in flight (one setpoint); a gripper command already sent; `env.reset` with `reset.method: move_to_joint_positions` (blocking on the NUC) |
| ur5e-env | queued calls; `env.move_delta` / `env.move_pose` / `env.rotate_delta` between polls of the running moveL (`limits.poll_s`, default 20 ms), which is then brought to rest with ur_rtde `stopL`; `env.set_gripper` between Robotiq register polls; `env.reset` between polls of the moveJ (`stopJ`) | the deceleration itself (stopL at 10 m/s^2, stopJ at 2 rad/s^2); a Robotiq command already sent (the fingers finish it). After a stop the setpoint is cleared: the next command starts from the measured pose |
| dual-franka-env | as franka-env; `env.recover_joint_posture` skips its return-to-start moves and reports `cancelled: true, ok: false` | as franka-env; in `recover_joint_posture` the two-arm joint reset and the gripper re-commands that restore the pre-recovery gripper state |
| pi05-vla, rldx-vla, sam3, molmo | queued calls | a running inference |
| LingBot-VLA (RoboTwin) | nothing (WebSocket server, no `/call`) | - |

Real robots: RLinf (pinned `bde6c918`) exposes no stop/hold call for the Franka arm (its arm
parts offer `clear_errors`, `reset_joint`, `reconfigure_compliance_params`,
`wait_until_still`; the ROS backend's `stop_impedance` shuts the controller down rather than
holding). `stop` therefore stops *sending* targets: under Cartesian impedance control the
arm settles on the last commanded target, which is at most one clipped servo step ahead of
the measured pose (single Franka: `action_scale`; dual Franka: `move_max_step_m` = 0.02 m,
`rotate_max_step_rad` = 0.1 by default). Gripper commands already issued complete. The
worker loops run in a Ray actor, possibly on another node, so the facade publishes the stop
through a small Ray actor that each loop polls between steps (one Ray round trip per
step; if the flag cannot be read within 1 s the loop logs a warning and continues).
`stop` is not an emergency stop; use the robot's hardware E-stop for that.

## Sessions (RLDX VLA only)

`rldx-vla` requires a `session_id` on every call. `session.register` (no args) registers
it, `session.close` drops it and resets its policy state; idle sessions expire after
`--session-timeout-s` (default 3600 s, swept every `--session-sweep-s`, default 60 s).
Calls with an unknown or missing session fail.

## Env servers

`obs`, `info`: dicts of numpy arrays / scalars / strings as produced by the simulator
wrapper; single-env servers strip the leading env dimension.

`env.ground_truth_poses` (libero-env, robocasa-env, maniskill-env; simulation only, behind pi's
`--privileged`, CaP-X's S1 tier): kw `names=null` (list of str; null or empty = every object) ->
`{"frame": "world", "poses": {name: {"pos": [x, y, z], "quat_xyzw": [x, y, z, w]}}}`, simulator
world frame, metres, rounded to 1e-5. The names are the simulator's own object list (per server
below); an unknown name is an error that lists them.

`code.api` (read-only; every env server: each robot's `primitives.py`): the server's
primitive registry (`components/code_api.py`), what a code-as-policy caller may use. kw
`tier=null` (`"high"`, `"low"`, `"privileged"` = high plus ground truth; null = every non-privileged
primitive) -> `{"tier": str | null, "primitives": [{"name", "method", "doc", "params": {name:
{"type", "description", "required"}}, "mutating", "tiers"}], "digest": sha256 hex}`. Each
primitive names the `env.*` method that runs it, so a primitive call is the tool's call, with the
same limits; the digest names the API version an episode ran with (pi records it as
`code_api_digest`). A server without a registry answers `unknown RPC method: 'code.api'`.

### libero-env (`robots/libero/env_server.py`)

| method | args | result |
|---|---|---|
| `env.get_env_meta` | - | `{"suite", "task", "seed", "max_episode_steps"}` |
| `env.reset` | - | `[obs, info]` (RLinf `LiberoEnv` obs, env dim stripped) |
| `env.step` | `action` float[7] | `[obs, reward, terminated, truncated, info]` |
| `env.chunk_step` | `actions` float[N,7], kw `return_all_frames=false` | `[obs or list[obs], reward[N], terminated[N], truncated[N], info]` |
| `env.raw_obs` | - | robosuite raw observation dict (e.g. `robot0_eef_quat`) |
| `env.render_camera` | `camera_name="agentview"`, `height=1024`, `width=1024`, `depth=false` | RLinf `render_camera` output: rgb uint8[H,W,3], or `[rgb, depth]` |
| `env.get_camera_meta` | `camera_name="agentview"`, `height=256`, `width=256` | intrinsics/extrinsics dict or `null` |
| `env.get_task_language` | - | str |
| `env.ground_truth_poses` | kw `names=null` | poses of LIBERO's `obj_body_id` bodies (movable objects and fixtures), read in the env worker |

`env.reset` reseeds the env worker's global numpy and Python RNGs with the episode seed, so every
reset restores the same state and the same actions give bitwise-identical transitions in any process.

The env runs with `ignore_terminations: True`: it keeps stepping after success, and `terminated` is
RLinf's latched `success_once` ("LIBERO success at or before this step"). Only `truncated` (max
episode steps) ends an episode.

### robocasa-env (`robots/robocasa/env_server.py`)

| method | args | result |
|---|---|---|
| `env.get_env_meta` | - | `{"task_name", "split", "seed", "camera_h", "camera_w"}` |
| `env.reset` | - | robosuite obs dict (honours `RLDX_RESET_SEED`) |
| `env.step` | `flat_action` float[12] (PandaOmron composite: eef_pos 3, eef_rot 3, gripper 1, base 4, mode 1) | `[obs, reward, done, info]` |
| `env.chunk_step` | - | not implemented (error) |
| `env.render_camera` | `camera_name`, `height`, `width`, `depth` | rgb uint8[H,W,3], or `[rgb, depth_m float[H,W]]` (robosuite orientation) |
| `env.get_camera_meta` | `camera_name`, `height=null`, `width=null` | `{"camera_name", "height", "width", "intrinsic" 3x3, "extrinsic_cam2world" 4x4, "depth_near", "depth_far"}` |
| `env.get_camera_transform` | `camera_name`, `height`, `width` | float64[4,4] pixel-to-world (`inv(K_ext)`) |
| `env.check_success` | - | bool |
| `env.grasp_contact` | - | `[bool, object_name or null]` |
| `env.reassemble_env_action` | `unmap_result` dict of per-part arrays | float[action_dim] |
| `env.get_success_criteria_text` | - | str (<= 9000 chars) |
| `env.get_task_progress` | - | dict of scalar success-check variables |
| `env.get_task_language` | - | str or null |
| `env.ground_truth_poses` | kw `names=null` | poses of the kitchen's objects (`obj_body_id`) and fixtures (root bodies) |

maniskill-env (`robots/maniskill/env_server.py`) serves `env.ground_truth_poses` over the scene's
actors (goal markers included) and its articulations other than the robot.

### robotwin-env (`robots/robotwin/env_server.py`)

`action_type` is `"qpos"` (14-D) or `"ee"` (16-D, eef16).

| method | args | result |
|---|---|---|
| `env.get_env_meta` | - | `env_runtime_contract(...)` (`robots/robotwin/contract.py`) |
| `env.reset` | - | `[obs, info]`; `info` has `instruction`, `instruction_source`, `requested_seed`, `robot_state`, `episode_status` |
| `env.step` | `action` float[14 or 16], kw `action_type="qpos"` | `[obs, reward, terminated, truncated, info]` |
| `env.chunk_step` | `actions` float[N,14 or 16], kw `action_type`, `return_all_frames=false` | `[obs or {"frames", "final"}, rewards[executed], terminated[executed], truncated[executed], info]`; `info.executed_actions`, `info.cancelled` when stopped |
| `env.render_camera` | `camera_name` in head/left_wrist/right_wrist, `depth=false` | rgb, or `[rgb, depth float (NaN = no hit)]` |
| `env.get_camera_meta` | `camera_name` | `{"intrinsic_K", "extrinsic_cv", "cam2world_gl", "width", "height"}` |
| `env.get_task_language` | - | str |
| `env.plan_arm_path` | `arm` left/right, `target_pose` float[7] | `{"status", "position", "velocity"}` |

The env worker runs torch with deterministic algorithms (`CUBLAS_WORKSPACE_CONFIG=:4096:8`) and cuRobo's
L-BFGS step on torch ops instead of its fused CUDA kernel, so cuRobo returns the same plan for the same
start and target (about 150 ms per plan instead of 50), and `env.reset` reseeds the worker's global
Python, numpy and torch RNGs with the episode seed. The same actions then give bitwise-identical
transitions and frames in any process and after any number of resets.

### robolab-env (`robots/robolab/env_server.py`)

One RoboLab (Isaac Lab) task on a Franka + Panda hand under relative IK. The motion primitives
hold the orientation at the reset pose by a per-control-step correction in the IK's rotation
slots; `env.rotate_delta` turns that hold's reference about the base vertical (+yaw = right-handed
about +z, counter-clockwise seen from above) and runs the hold with zero translation until the
heading is reached. Each observation is `{"agentview" uint8[H,W,3], "wrist" uint8[256,256,3]
(fingertips at the top), "eef_pos" float32[3], "eef_quat_wxyz" float32[4], "tilt_deg", "yaw_deg"
(from the reset heading), "gripper_width", "gripper_command", "success", "terminated",
"truncated", "env_steps"[, "subtask"]}`. `code.api` serves the registry of
`robots/robolab/primitives.py`.

| method | args | result |
|---|---|---|
| `env.get_env_meta` | - | `{"task", "seed", "instruction", "instruction_type", "subtask", "robot", "step_m", "command_gain", "steps_per_decision", "gripper_hold_steps", "settle_steps", "episode_length_s", "control_hz", "ik_scale", "output_dir", ...}` |
| `env.reset` | - | `[obs, {"instruction"}]` (opens the gripper, settles, latches the orientation hold) |
| `env.step` | `action` float[7] `[dx, dy, dz, drx, dry, drz, gripper]` (raw relative IK, no hold) | `[obs, 0.0, terminated, truncated, {"success"}]` |
| `env.chunk_step` | `actions` float[N,7], kw `return_all_frames=false` | `[obs or list[obs], terminated, truncated, info[, "cancelled"]]` |
| `env.move_delta` | `delta_xyz` float[3] (m, base frame; refused beyond 0.3 m), kw `gripper` "open"/"close"/null, `return_frames=false` | obs + `{"commanded_m", "moved_m", "decisions", "control_steps"[, "frames" (agentview per decision), "cancelled", "error"]}` |
| `env.rotate_delta` | `yaw` float (rad about base +z; clipped to 0.3), kw `return_frames=false` | obs + `{"requested_yaw", "commanded_yaw", "yaw" (executed, measured), "moved_m" (drift), "decisions", "control_steps"[, "clipped", "frames", "cancelled", "error"]}` |
| `env.state` | - | the obs without images (no stepping) |
| `env.render_camera` | `camera_name="agentview"` or `"wrist"` | the latest frame |
| `env.get_camera_meta` | `camera_name="agentview"` | `{"intrinsic_K" 3x3, "extrinsic_cam2world" 4x4, "width", "height"}` |
| `env.get_task_language` | - | str |
| `env.ground_truth_poses` | kw `names=null` | the scene's object poses (behind pi's `--privileged`) |

### franka-env (`robots/franka/env_server.py`)

| method | args | result |
|---|---|---|
| `env.get_env_meta` | - | `{"ok", "action_dim", "action_scale", "use_relative_frame", "backend", "capabilities"}` |
| `env.reset` | - | `{"ok", "info", "robot_state", "states"}` (moves the arm) |
| `env.get_robot_state` | - | `{"raw_base_state", "action_dim", "action_scale", "use_relative_frame"}` |
| `env.get_observation` | - | live camera frames: `main_images`, `extra_view_images`, `main_depths`, `extra_view_depths` (empty dict on camera failure) |
| `env.get_camera_meta` | - | RLinf camera metadata + `observation_camera_map`, or `{"error", "error_type"}` |
| `env.move_delta` | `delta_xyz` float[3] (m, base frame) | `{"ok", "requested_delta_xyz_base", "start_tcp_pose", "final_tcp_pose", "final_error_m", "steps_used", "states"[, "cancelled"]}` |
| `env.rotate_delta` | `delta_rpy` float[3] (rad) | `{"ok", "requested_delta_rpy_base", "start_tcp_pose", "final_tcp_pose", "final_error_rad", "steps_used", "states"[, "cancelled"]}` |
| `env.set_gripper` | kw `open` bool | `{"ok", "target_gripper_open", "steps_used", "robot_state", "states"[, "cancelled"]}` |
| `env.chunk_step` | `actions` float[N,action_dim], kw `return_all_frames=false` | `{"observation", "terminated", "truncated", "info"[, "cancelled"]}` |

### franka-polymetis-env (`robots/franka_polymetis/env_server.py`)

The franka-env methods, argument and result shapes, on a Polymetis NUC (Show-Harness's
stack). Differences: `env.get_env_meta` / `env.get_robot_state` report `action_dim`,
`action_scale` = null and `backend: "polymetis"`; `env.chunk_step` always errors (no VLA
action space); `states` is null. `env.move_delta` / `env.rotate_delta` refuse (error, nothing
commanded) a call beyond `limits.max_move_m` / `max_rotate_rad` or ending outside the
workspace box / below `z_floor_m`; results add `target_tcp_pose` and, for a descent that
travelled < 70% of the command, `descent_blocked`, `descent_travelled_m`, `note`.
`env.set_gripper` adds `gripper_width_m` and, when a close ends at or below
`gripper.empty_width_m`, reopens and reports `grasp_empty: true, ok: false`. `env.reset`
opens, lifts, moves to `reset.begin_joints` and returns `{"ok", "info", "robot_state",
"states"}`. `env.get_camera_meta` has the franka-env layout (`cameras.<name>.intrinsic_K` of
the letterboxed image, `raw_color_intrinsics`, `letterbox`, `observation_camera_map`).

Both franka servers' `env.get_env_meta` carry `backend` and `capabilities`: `{"backend",
"has_vla", "cameras" (observation key -> camera), "has_depth", "workspace" {min, max} | null,
"z_floor_m", "table_z_m", "max_move_m", "max_rotate_rad" (per-call limits the server
refuses beyond; null = none), "servo_step_m", "servo_step_rad", "empty_grasp_reopen_m"}`.

### ur5e-env (`robots/ur5e/env_server.py`)

One UR5e over ur_rtde (moveL / moveJ at `limits.speed_mps` 0.25 and `accel_mps2` 0.5 by default),
a Robotiq 2F gripper over the URCap socket (port 63352), and cameras through the shared
`components/cameras` layer (RealSense D400 / L515, webcam, RTSP; `--cameras name=type:source,...`
overrides the config's devices). Poses are the UR base frame; `tcp_pose` is `[x, y, z, qx, qy, qz,
qw]`, `tcp_pose_rotvec` the UR `[x, y, z, rx, ry, rz]`. The config is bound to one arm:
`calibration.arm_id` must equal the controller's serial number (`--print-identity`), and each
camera's hand-eye YAML (written by `robots/ur5e/calibrate.py`, applied by a human) must name the
same arm and, when recorded, the same camera serial. `code.api` serves the primitives of
`robots/ur5e/primitives.py` (the motion and state methods below; no privileged tier).

Refusals (error, nothing commanded): a translation beyond `limits.max_move_m`, a turn beyond
`max_rotate_rad`, a target outside the workspace box or below `z_floor_m` (a move from outside
back toward the box is allowed), a tool tilt past `max_tilt_rad`, `env.reset` without
`calibration.begin_joints`, `env.set_gripper` without a gripper. A motion that is stopped, times
out (`move_timeout_s`, stopL), raises in the driver (stopL, `RuntimeError`) or ends farther than
`move_tolerance_m` / `rotate_tolerance_rad` from its target reports `ok: false` and clears the
setpoint (`raw_base_state.setpoint_pose` is null) so the next command starts from the measured pose.

| method | args | result |
|---|---|---|
| `env.get_env_meta` | - | `{"ok", "robot": "ur5e", "backend": "ur_rtde", "arm_id", "config_path", "cameras", "main_camera", "gripper", "has_begin_pose", "limits", "capabilities" (the franka layout plus `camera_depth` {name: bool} and `arm_id`), "tasks"}` |
| `env.reset` | - | `{"ok", "gripper", "move" {ok, target_joints, final_joints, final_error_rad, final_tcp_pose}, "info", "robot_state", "states": null[, "cancelled"]}` (opens the gripper, moveJ to `begin_joints`) |
| `env.get_robot_state` | - | `{"raw_base_state" {tcp_pose, tcp_pose_rotvec, tool_tilt_rad, joints, joint_speeds, setpoint_pose, gripper_position [width_m], gripper_open, gripper_grasped, gripper_commanded_open, gripper {...}, z_floor_m, robot_status}, "backend", "arm_id"}` |
| `env.get_observation` | - | `{"images" {name: uint8[H,W,3]}, "depths" {name: float32[H,W] m, cameras with depth only}, "timestamps" {name: s}}` |
| `env.get_camera_meta` | - | `{"cameras" {name: {role, mount, has_depth, intrinsic_K 3x3 or null, raw_color_intrinsics, output_resolution, extrinsic {frame "tcp" or "base", matrix 4x4 (camera -> that frame), path, arm_id} or null, camera_type, ...}}, "observation_camera_map", "arm_id", ...}` |
| `env.move_delta` | `delta_xyz` float[3] (m, base frame) | `{"ok", "requested_delta_xyz_base", "start_tcp_pose", "target_tcp_pose", "final_tcp_pose", "final_error_m", "final_error_rad", "steps_used", "elapsed_s", "states": null[, "cancelled", "timed_out", "note"]}` |
| `env.move_pose` | `xyz` float[3], kw `rotvec` float[3] or `rpy` float[3] (extrinsic xyz, rad; converted to a rotation vector) or neither (orientation held) | as `env.move_delta` plus `requested_pose_rotvec`; refused beyond `max_move_m` / `max_rotate_rad` from the current setpoint |
| `env.rotate_delta` | `delta_rpy` float[3] (rad, about the base axes) | as `env.move_delta` with `requested_delta_rpy_base` |
| `env.set_gripper` | kw `open` bool | `{"ok", "target_gripper_open", "object_detected", "steps_used", "gripper_width_m", "robot_state", "states": null[, "gripper_jammed", "grasp_empty", "note", "cancelled"]}`; a close ending at or below `gripper.empty_width_m` with nothing detected reopens (`grasp_empty`); fingers that did not move toward the command and hold nothing are `gripper_jammed` |

### dual-franka-env (`robots/dual_franka/env_server.py`)

Same method names; poses are reported in the `right_base` frame. `arm` is `"left"` or
`"right"`.

| method | args | result |
|---|---|---|
| `env.get_env_meta` | - | `{"ok", "action_dim" (20), "per_arm_dim" (10), "action_scale", "arms", "explicit_reset_only", "perception_cameras", "agent_observation", "projection_views"}` |
| `env.reset` | - | `{"ok", "info", "robot_state"}` |
| `env.get_robot_state` | - | `{"coordinate_frame", "left_arm", "right_arm", "joint_health", "action_dim", "per_arm_dim", "action_scale"}` |
| `env.get_observation` | - | wrapped observation + `raw_camera_frames`, `raw_camera_depths`, `<alias>_images`, `<alias>_depths` |
| `env.get_camera_meta` | - | `{"cameras", "observation_camera_map", "agent_observation", "projection_views", ...}` |
| `env.move_delta` | `arm`, `delta_xyz` float[3] | pose/error report `[, "cancelled"]` |
| `env.rotate_delta` | `arm`, `delta_rpy` float[3] | pose/error report `[, "cancelled"]` |
| `env.set_gripper` | `arm`, kw `open` | `{"ok", "arm", "target_gripper_open", "steps_used", "robot_state"[, "cancelled"]}` |
| `env.recover_joint_posture` | `reason=""`, `return_to_start=true` | recovery report `[, "cancelled"]` |
| `env.chunk_step` | `actions` float[N,20], kw `return_all_frames=false` | `{"observation", "terminated", "truncated", "info"[, "cancelled"]}` |

## Model servers

| service | method | args | result |
|---|---|---|---|
| pi05-vla | `vla.predict` | `obs` (openpi wire dict, encoded by the client), `options={"mode": "eval"[, "seed": int]}` | float32 action chunk from openpi's `predict_action_batch`, batch first (LIBERO: 5 steps x 7; dual Franka: 20 x 20) |
| rldx-vla | `vla.get_modality_config` | - | `{"video_delta_indices": [int], "hist_maxlen": int}` |
| rldx-vla | `vla.predict` | `obs_dict`, `options` (must not contain `session_ids`; optional `"seed": int`) | dict of action arrays |
| rldx-vla | `vla.reset_session` | - | `{"ok": true}` |
| sam3 | `sam3.segment` | `image_base64` (PNG/JPEG bytes), exactly one of kw `text_prompt` str or `point` `[row, col]`, `min_score=0.2` | `{"found", "score"?, "box"?, "mask_png_base64"?, "mask_shape"? [H,W], "reason"?}` |
| molmo | `molmo.ground` | `image_base64`, `query` str | `{"point_xy"? [x, y] pixels, "answer", "image_size" [W, H]}` |

LingBot-VLA (`robots/robotwin/vla_server.py`) is the upstream `deploy` WebSocket policy
server: msgpack frames with openpi's numpy extension, server metadata
(`vla_runtime_contract()`) first, then one reply per inference request; `GET /healthz` over
HTTP on the same port. Inferences are serialized through the same one-call lock. An optional
int `seed` key in the observation is removed before the policy sees it and seeds that inference.

A VLA `seed` (int in [0, 2^32)) makes that one inference a function of its inputs: the server
seeds torch (CPU and every CUDA device), numpy and Python `random` for the call and restores
their previous states afterwards. Without it, sampling is unseeded as before. The robots send one
per call (`--vla-seed`, see `packages/embodied/src/vla-seed.ts`) and record it as `vla_seeds` in
the tool result.
