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
- Only HTTP exists. RPent's pickle-framed `socket` transport was removed (unpickling a
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
  first call finishes; the retry waits behind it). RPent let "read-only" methods run in
  parallel; that is gone (a concurrent `env.render_camera` broke LIBERO's worker pipe).
- Lock-free methods, answered at once even while a call runs: `healthz`, `stop`, `cancel`,
  `shutdown` (`shutdown` still waits for the running call before it takes effect).

## Framework methods (every RPC server)

| method | args | result |
|---|---|---|
| `healthz` | - | `{"status": "ok", "version": "<pi-embodied-services version>", "service": "<name>"}` |
| `stop` / `cancel` | - | `{"ok": true, "stop_generation": int, "call_in_progress": bool}` |
| `shutdown` | - | `{"ok": true}`; the process exits after answering |

Service names: `libero-env`, `robocasa-env`, `rldx-vla`, `robotwin-env`, `franka-env`,
`dual-franka-env`, `pi05-vla`, `sam3`, `molmo`.

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
| franka-env | queued calls; `env.move_delta` / `env.rotate_delta` / `env.set_gripper` before each servo step; `env.chunk_step` after each action | the servo step in progress (one RLinf `env.step`: one Cartesian target plus the pacing sleep, and up to 0.6 s when it toggles the gripper); `env.reset` (RLinf go-to-rest / joint reset) |
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

### franka-env (`robots/franka/env_server.py`)

| method | args | result |
|---|---|---|
| `env.get_env_meta` | - | `{"ok", "action_dim", "action_scale", "use_relative_frame"}` |
| `env.reset` | - | `{"ok", "info", "robot_state", "states"}` (moves the arm) |
| `env.get_robot_state` | - | `{"raw_base_state", "action_dim", "action_scale", "use_relative_frame"}` |
| `env.get_observation` | - | live camera frames: `main_images`, `extra_view_images`, `main_depths`, `extra_view_depths` (empty dict on camera failure) |
| `env.get_camera_meta` | - | RLinf camera metadata + `observation_camera_map`, or `{"error", "error_type"}` |
| `env.move_delta` | `delta_xyz` float[3] (m, base frame) | `{"ok", "requested_delta_xyz_base", "start_tcp_pose", "final_tcp_pose", "final_error_m", "steps_used", "states"[, "cancelled"]}` |
| `env.rotate_delta` | `delta_rpy` float[3] (rad) | `{"ok", "requested_delta_rpy_base", "start_tcp_pose", "final_tcp_pose", "final_error_rad", "steps_used", "states"[, "cancelled"]}` |
| `env.set_gripper` | kw `open` bool | `{"ok", "target_gripper_open", "steps_used", "robot_state", "states"[, "cancelled"]}` |
| `env.chunk_step` | `actions` float[N,action_dim], kw `return_all_frames=false` | `{"observation", "terminated", "truncated", "info"[, "cancelled"]}` |

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
| pi05-vla | `vla.predict` | `obs` (openpi wire dict, encoded by the client), `options={"mode": "eval"}` | float32 action chunk from openpi's `predict_action_batch`, batch first (LIBERO: 5 steps x 7; dual Franka: 20 x 20) |
| rldx-vla | `vla.get_modality_config` | - | `{"video_delta_indices": [int], "hist_maxlen": int}` |
| rldx-vla | `vla.predict` | `obs_dict`, `options` (must not contain `session_ids`) | dict of action arrays |
| rldx-vla | `vla.reset_session` | - | `{"ok": true}` |
| sam3 | `sam3.segment` | `image_base64` (PNG/JPEG bytes), exactly one of kw `text_prompt` str or `point` `[row, col]`, `min_score=0.2` | `{"found", "score"?, "box"?, "mask_png_base64"?, "mask_shape"? [H,W], "reason"?}` |
| molmo | `molmo.ground` | `image_base64`, `query` str | `{"point_xy"? [x, y] pixels, "answer", "image_size" [W, H]}` |

LingBot-VLA (`robots/robotwin/vla_server.py`) is the upstream `deploy` WebSocket policy
server: msgpack frames with openpi's numpy extension, server metadata
(`vla_runtime_contract()`) first, then one reply per inference request; `GET /healthz` over
HTTP on the same port. Inferences are serialized through the same one-call lock.
