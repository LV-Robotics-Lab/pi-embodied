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

Service names: `libero-env`, `robocasa-env`, `maniskill-env`, `metaworld-env`, `genesis-env`, `rldx-vla`,
`robotwin-env`, `robolab-env`, `behavior-env`, `franka-env`, `dual-franka-env`, `franka-polymetis-env`, `ur5e-env`, `pi05-vla`, `sam3`, `molmo`,
`unidepth`, `openvla`, `openvla-oft`, `gr00t`, `ik`.

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
| robosuite-env | queued calls; `env.move_to` / `env.move_delta` / `env.set_gripper` before each control step and `env.chunk_step` before each action (`cancelled: true`) | the control step in progress (one robosuite `step`), `env.reset`, renders |
| metaworld-env | queued calls; `env.chunk_step` before each action and `env.move_delta` / `env.set_gripper` before each control step (`cancelled: true`) | the control step in progress (one MuJoCo `step`), `env.reset`, renders |
| robotwin-env | queued calls; `env.chunk_step` before each native action (`info.cancelled = true`) | the native action being executed (one `take_action`, i.e. one planned qpos/ee motion), `env.step`, `env.reset`, `env.plan_arm_path`, renders |
| robolab-env | queued calls; `env.move_delta` / `env.rotate_delta` before each control step; `env.chunk_step` before each action | the Isaac Lab `env.step` in progress (one control step of 8 physics substeps), `env.reset` |
| behavior-env | queued calls; every primitive (`env.navigate_to_pose`, `env.move_hand`, `env.grasp_object`, `env.open_gripper`, `env.close_gripper`) between control steps (`cancelled: true`, `ok: false`); `env.chunk_step` before each action | the OmniGibson `env.step` in progress (one action, 4 physics substeps), a cuRobo plan being computed, `env.reset` |
| franka-env | queued calls; `env.move_delta` / `env.rotate_delta` / `env.set_gripper` before each servo step; `env.chunk_step` after each action | the servo step in progress (one RLinf `env.step`: one Cartesian target plus the pacing sleep, and up to 0.6 s when it toggles the gripper); `env.reset` (RLinf go-to-rest / joint reset) |
| franka-polymetis-env | queued calls; `env.move_delta` / `env.rotate_delta` before each servo tick (setpoint advance <= `servo_step_m` / `servo_step_rad`) and during settle; `env.set_gripper` between width polls; `env.reset` between lift ticks and joint-stream ticks (`reset.method: joint_stream`) | the ZeroRPC call in flight (one setpoint); a gripper command already sent; `env.reset` with `reset.method: move_to_joint_positions` (blocking on the NUC) |
| ur5e-env | queued calls; `env.move_delta` / `env.move_pose` / `env.rotate_delta` between polls of the running moveL (`limits.poll_s`, default 20 ms), which is then brought to rest with ur_rtde `stopL`; `env.set_gripper` between Robotiq register polls; `env.reset` between polls of the moveJ (`stopJ`) | the deceleration itself (stopL at 10 m/s^2, stopJ at 2 rad/s^2); a Robotiq command already sent (the fingers finish it). After a stop the setpoint is cleared: the next command starts from the measured pose |
| dual-franka-env | as franka-env; `env.recover_joint_posture` skips its return-to-start moves and reports `cancelled: true, ok: false` | as franka-env; in `recover_joint_posture` the two-arm joint reset and the gripper re-commands that restore the pre-recovery gripper state |
| pi05-vla, openvla, openvla-oft, gr00t, rldx-vla, sam3, molmo, unidepth | queued calls | a running inference |
| ik | queued calls | a running solve or plan (milliseconds with PyRoKi; up to the plan `timeout` with cuRobo) |
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

`env.ground_truth_poses` (libero-env, robocasa-env, robosuite-env, maniskill-env, metaworld-env, genesis-env; simulation only, behind pi's
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
| `env.preview_reach` | `pos` float[3] (m, world frame), kw `quat_xyzw=null` (null = the current gripper orientation) | reach preview (below); `status: "unknown"` without `--ik` |

`env.preview_reach` (libero-env, franka-env, franka-polymetis-env; `--ik <url>` names the `ik`
service, robot model `panda_libero` for LIBERO and `panda` for the Frankas) solves IK for the
TCP target from the current joints without touching the sim or the arm:
`{"status": "reachable" | "unreachable" | "unknown", "reachable": bool | null, "q": [7 joints] |
null, "position_err" m, "orientation_err" rad, "message", "target": {"frame": "world" | "base",
"pos", "quat_xyzw"[, "base_pose"]}, "robot", "backend", "path_checked": false}`. `unknown` means
the check could not run (no `--ik`, the service is down or errored) and is not approval. With
`--ik`, the Frankas' `env.move_delta` / `env.rotate_delta` check the end pose after the delta
first and refuse an `unreachable` one as an error without moving; pi's LIBERO `move_to` tool asks
`env.preview_reach` itself, and a server-side LIBERO or Robosuite `move_to` should call
`utils/reach.py`'s `require_reachable(self.preview_reach(xyz), "move_to")` before stepping. The
LIBERO server converts the world target into the `robot0_base` frame (read from the worker once
per reset). `path_checked` is always false: only the end pose is solved, not the path. Each
robot's `primitives.py` declares `preview_reach` in `code.api` (high and low tiers).

`env.reset` reseeds the env worker's global numpy and Python RNGs with the episode seed, so every
reset restores the same state and the same actions give bitwise-identical transitions in any process.

#### Code mode (`code.run`, `code.helpers`; pi's `--code`, CaP-X's run_code)

libero-env serves `code.run` over its registry (`code.api`, above): the runner is
`utils/code_exec.py` (`CodeRunner` + `registry_primitives`), and every call a program makes goes
through `CodeApi.resolve` (the declared name, parameters and tier) to the registered `env.*`
method, so a program's `move_to` steps the env under the same limits and stop generation as the
robot's tools. The primitives it reaches are `robots/libero/primitives.py`: the raw surface
(`LIBERO_PRIMITIVES`) plus `CODE_PRIMITIVES`, high tier `get_state`, `get_observation` (512x512
upright rgb + metric depth + `intrinsic_K` + `extrinsic_cam2world` per camera: `agentview`,
`wrist`), `back_project(row, col, camera)`, `move_to(xyz, gripper=None, tol, max_steps)`,
`rotate_wrist(target_yaw | delta_yaw, gripper)`, `set_gripper(close, steps)` and, with
`--sam3 <url>`, `segment(prompt, camera, min_score)`; low tier `get_state`, `get_observation`,
`move_delta(dxyz <= 0.10 m, gripper)`, `rotate_delta(delta_yaw <= pi/2)`, `set_gripper`; the
privileged tier is the high tier plus `ground_truth_poses`. `gripper=None` keeps the last command
(unlike the `move_to` tool, whose default opens). Each motion primitive stops at the episode's end
(success latched or truncated) and between env steps on `stop`.

| method | args | result |
|---|---|---|
| `code.run` | kw `code` str, `timeout_s=60`, `tier="high"` (`high`, `low`, `privileged`), `max_calls=50`, `max_move_m=null`, `helpers=false` | `{"status": "ran" | "error" | "timeout", "stdout", "stderr" (8 KB each), "traceback", "error", "result" (the program's `RESULT`, JSON-able), "calls": [{name, args, kwargs, ms, error?, refused?, move_m?, cancelled?}], "n_calls", "move_m", "limit"?, "cancelled"?, "stop_issued"?, "timeout_s", "ms"}` plus the server's run fields (libero-env: `steps`, `success_step` (within the run, or null), `terminated`, `truncated`, `obs` (pi's `Obs`), `frames` (one agentview image per mutating primitive, at most 32)) |
| `code.helpers` | - | `[{"name", "signature", "doc", "kind": "helper"}]`: CaP-X's nine numpy helpers a run with `helpers=true` injects (pure computation) |

`code.run` spawns a fresh Python process (`multiprocessing` spawn) that holds no env object:
its globals are `np`, `math`, `RESULT` and one stub per primitive of the tier (positional
arguments fill the declared parameters in order); a stub sends the call over a pipe and the
parent resolves and executes it. The child gets `RLIMIT_AS` (4 GiB beyond what the interpreter
had mapped when the program starts), `RLIMIT_CPU` (the timeout + 2 s beyond the CPU already
spent) and a temporary working directory. Past `timeout_s` the child is killed, `status` is
`timeout` and the server issues itself a `stop` (`stop_issued`). Past `max_calls` calls or
`max_move_m` metres of commanded translation (estimated per call before it runs: `move_to`'s
distance to the target, `move_delta`'s norm, a raw `step`'s clipped translation) the call is
refused: the program gets a `CodeLimitError` and the result carries `limit`. A `stop` while a run
executes kills the child and the running primitive returns at its next env step
(`cancelled: true`). A program that raises ends with `status: "error"` and its `traceback`;
`error` is the traceback's last line.

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

### metaworld-env (`robots/metaworld/env_server.py`)

One Metaworld MT50 task (`metaworld==3.1.1`, MuJoCo 3.3.0; the 50 `*-v3` names and their
instructions are the server's `INSTRUCTIONS` table). Action `[dx, dy, dz, gripper]` in [-1, 1]:
a world-frame hand translation, 1.0 = 1 cm per control step, and the gripper effort (+1 close,
-1 open). `obs` is `{"agentview" uint8[256,256,3] (corner4, turned 180 deg so +z is up),
"wrist" uint8[256,256,3] (gripperPOV), "tcp_pos" float32[3], "gripper_width" float (m, the
finger pad distance: 0.095 open, 0.023 closed on nothing), "obs" float32[39]}`; `info` is the
task's metrics as scalars (`success`, `grasp_success`, `near_object`, `obj_to_target`, ...) plus
`success_once`.

| method | args | result |
|---|---|---|
| `env.get_env_meta` | - | `{"task", "seed", "metaworld", "agentview", "wrist", "view_size", "action_scale_m", "max_move_m", "gripper_steps", "action_space", "workspace" {min, max} (after a reset)}` |
| `env.reset` | kw `seed=null` (default the launch seed) | `[obs, info]`; the hand at the task's start pose, gripper open |
| `env.step` | `action` float[4] | `[obs, reward, terminated (= success), truncated, info]` |
| `env.chunk_step` | `actions` float[N,4], kw `return_all_frames=false` | `[obs or list[obs], reward[N], terminated[N], truncated[N], info]`; stops at success or `stop` |
| `env.move_delta` | `delta_xyz` float[3] (m, world), kw `gripper=null` ("open" / "close" first, held 20 steps), `tol_m=0.006` | `{"ok", "requested_delta_xyz", "start_tcp_pos", "final_tcp_pos", "final_error_m", "moved_m", "gripper", "gripper_width", "steps_used", "frames" list[obs] (one per control step), "info"[, "cancelled"]}`; refuses (error, nothing commanded) more than 0.2 m or a target outside the workspace box (the mocap bounds, in TCP coordinates). The mocap target moves 1 cm per step and the hand settles behind it |
| `env.set_gripper` | kw `open` bool | as `env.move_delta` with `target_gripper_open` |
| `env.state` | - | `{"tcp_pos", "gripper_width", "gripper_command", "success", "success_once", "info", "workspace"}` (no stepping) |
| `env.raw_obs` | - | `{"obs" float64[39], "info"}` |
| `env.render_camera` | `camera_name="agentview"`, `height=256`, `width=256`, `depth=false` | rgb uint8[H,W,3] rows top-first, or `[rgb, depth_m float32[H,W]]` |
| `env.get_camera_meta` | `camera_name`, `height=256`, `width=256` | `{"camera_name", "height", "width", "intrinsic_K" 3x3 (OpenCV), "extrinsic_cam2world" 4x4}` (the agentview's includes its 180 deg turn) |
| `env.get_task_language` | - | str |
| `env.ground_truth_poses` | kw `names=null` | poses of the scene's named MuJoCo bodies outside the robot and its stand, plus `goal` (the task's target position) |
| `code.api` | kw `tier=null` | the registry of `robots/metaworld/primitives.py`: `state`, `move_delta`, `set_gripper` (high and low), `render_camera`, `get_camera_meta`, `step`, `chunk_step` (low), `ground_truth_poses` (privileged) |

`env.reset` reseeds the env's RNG and the process's global numpy and Python RNGs with the episode
seed, so a seed draws the same layout in any process and every reset restores the same state.
Metaworld's own 500-step horizon is lifted; the planner's budget ends an episode.

### genesis-env (`robots/genesis/env_server.py`)

One Genesis 1.4 scene: a Franka Panda, a table plane and the task's objects (`cube_pick`: a 4 cm
cube sampled over the reachable table by the seed). The server owns the motion: `env.move_delta`
runs a base-frame delta as ~2 cm IK decisions with the reset orientation held; `env.set_gripper`
opens or closes and holds. Limits are checked before anything moves (a refusal is an error and
nothing moves): at most `max_move_m` (0.2) per call, the TCP above `z_floor_m` (0.012) and inside
`workspace`. Each observation is `{"agentview" uint8[256,256,3], "wrist" uint8[256,256,3],
"tcp_pos" float32[3], "tcp_quat_wxyz" float32[4], "gripper_width", "gripper_command",
"qpos" float32[9], "success", "is_grasped", "lift_m", "env_steps"}`; `success` (cube_pick: the
cube's bottom face >= `lift_m` (0.08) above the table for 5 control steps) latches. `stop`
interrupts `env.move_delta` / `env.set_gripper` before each control step and `env.chunk_step`
before each action (`cancelled: true`), not the Genesis step in progress.

| method | args | result |
|---|---|---|
| `env.get_env_meta` | - | `{"task", "seed", "instruction", "backend", "dt", "substeps", "view_size", "agentview", "wrist_offset", "wrist_fov_deg", "step_m", "workspace" {min, max}, "z_floor_m", "max_move_m", "lift_m", "empty_width_m", "objects"[, "layout", "visible_px"]}` |
| `env.reset` | kw `seed=null` (default: the launch seed) | `[obs, {"instruction", "seed"}]`; refuses (error) when the front camera does not show the task object |
| `env.step` | `action` float[4] `[dx, dy, dz, gripper]` (m, base frame; +1 open / -1 close), one control step | `[obs, 0.0, success, false, {"success"}]` |
| `env.chunk_step` | `actions` float[N,4], kw `return_all_frames=false` | `[obs or list[obs], reward[n], terminated[n], truncated[n], {"success"[, "cancelled"]}]`, n = executed actions (stops at success or `stop`) |
| `env.move_delta` | `delta_xyz` float[3] (m), kw `gripper` "open"/"close"/null (first, holding still), `return_frames=false` | obs + `{"commanded_m", "moved_m", "decisions", "control_steps"[, "frames" (both views side by side, one per decision), "cancelled"]}` |
| `env.set_gripper` | kw `open` bool, `return_frames=false` | obs + `{"control_steps"[, "frames", "grasp_empty" (a close ending at or below `empty_width_m`), "cancelled"]}` |
| `env.state` | - | the obs without images (no stepping) |
| `env.render_camera` | `camera_name="agentview"` or `"wrist"`, `depth=false` | rgb uint8[256,256,3] as the model sees it, or `[rgb, depth_m float32[256,256]]` |
| `env.get_camera_meta` | `camera_name="agentview"` | `{"intrinsic_K" 3x3, "extrinsic_cam2world" 4x4 (OpenCV), "width", "height"}` |
| `env.back_project` | `camera_name`, kw `pixels` [[row, col], ...] | list of `[x, y, z]` (world, m) or null where the depth is missing |
| `env.get_task_language` | - | str |
| `env.ground_truth_poses` | kw `names=null` | poses of the task objects (`cube`) |
| `code.api` | kw `tier` | the registry of `robots/genesis/primitives.py` (high: state, move_delta, set_gripper, back_project; low adds render_camera, get_camera_meta, step, chunk_step) |

### robosuite-env (`robots/robosuite/env_server.py`)

CaP-X's seven robosuite tasks (`robots/robosuite/tasks.py`: `Lift`, `Stack`, `Restack`, `Wipe`,
`NutAssemblySquare`, `TwoArmLift`, `TwoArmHandover`) on upstream robosuite 1.5 with Pandas under the
BASIC composite controller's OSC_POSE (the OpenETA convention; not CaP-X's joint-position
controller), its own venv (the `robosuite` extra conflicts with LIBERO's robosuite 1.4).
`Restack` is robosuite's Stack with two 4 cm cubes and the green one placed on the red one at reset.
`arm` is `robot0` or `robot1` on the two-arm tasks and omitted on one arm; `Wipe`'s sponge gripper
has no fingers (no gripper command, action dim 6). Every observation is `{"agentview" uint8[512,512,3]
(the task camera: `robot0_robotview`, or CaP-X's overhead `agentview` on the two-arm tasks), "wrist"
uint8[512,512,3] (`robot0_eye_in_hand`), per arm "<arm>_eef_pos", "<arm>_eef_quat" (xyzw),
"<arm>_joint_pos", "<arm>_gripper_qpos", "<arm>_gripper_width", "<arm>_gripper_command", "success",
"success_step", "truncated", "env_steps"}`; images are upright. `success` is robosuite's
`_check_success` (Restack: refused while both cubes are more than 0.04 m above the table, CaP-X's
rule), latched at its first control step; the episode keeps stepping after it. Object poses leave the
server only through `env.ground_truth_poses`: `env.raw_obs` holds the robots' state alone, and the
handover task renders no instance segmentation.

| method | args | result |
|---|---|---|
| `env.get_env_meta` | - | `{"task", "seed", "env", "arms", "gripper", "action_dim", "controller", "camera", "wrist_camera", "camera_size", "max_episode_steps", "max_move_m", "language", "capabilities" {segment, reach, grasp...}, "box" [xmin, xmax, ymin, ymax], "z_floor", "z_ceiling", "table_z"}` |
| `env.reset` | - | `[obs, {"language"}]`; reseeds the global numpy / Python RNGs (robosuite 1.5's samplers draw from them) with the episode seed, then settles 20 steps with the grippers open |
| `env.step` | `action` float[action_dim] (per arm: 6 OSC_POSE deltas in [-1, 1], then the gripper, +1 close / -1 open) | `[obs, reward, success, truncated, state]` |
| `env.chunk_step` | `actions` float[N, action_dim], kw `return_all_frames=false` | `[obs or list[obs], reward[n], success[n], truncated[n], info]`; stops at `stop` (`info.cancelled`) |
| `env.move_to` | `target_xyz` float[3] (m, world), kw `arm`, `quat_xyzw` or `rotvec` (a world-frame turn applied to the current orientation), `gripper` "open"/"close"/null (set first), `tol_m=0.005`, `tol_rad=0.03`, `step_m=0.02`, `step_rad=0.2`, `max_steps=100` | `{"obs", "info": {"ok", "arm", "target_xyz", "final_eef_pos", "final_dist_m", "final_rot_err_rad"?, "steps_used", "success", "cancelled", "frames" list[uint8[256,256,3]] (the task camera every 4 steps)}}`; refuses (error, nothing moves) a target more than `--max-move` (0.3 m) away, outside the table box (footprint + 0.1 m, widened 0.15 m around each arm's start), below `z_floor` (table + 0.005 m; Wipe: table - 0.02) or above `z_ceiling`, and with `--ik` an unreachable one |
| `env.move_delta` | `delta_xyz` float[3] (m, world), kw as `env.move_to` | as `env.move_to` |
| `env.set_gripper` | `close` bool or "open"/"close", kw `arm`, `steps=15` | `{"obs", "info": {"ok", "arm", "gripper", "gripper_width", "steps_used", "cancelled", "frames"}}`; the fingers stop early once they no longer move |
| `env.state` / `env.get_state` | - | the obs without images plus `home_eef_pos`, `table_z` (no stepping; `get_state` as plain lists) |
| `env.raw_obs` | - | robosuite's `robot*_` observations only |
| `env.render_camera` | `camera_name="agentview"` or `"wrist"`, `height=512`, `width=512`, `depth=false` | upright rgb uint8[H,W,3], or `[rgb, depth_m float32[H,W]]` |
| `env.get_camera_meta` | `camera_name`, `height=512`, `width=512` | `{"camera_name", "height", "width", "intrinsic_K" 3x3, "extrinsic_cam2world" 4x4 (robosuite's, for the upright image), "depth_metric": true}` |
| `env.get_observation` | - | `{"agentview", "wrist": {"rgb", "depth", "intrinsic_K", "extrinsic_cam2world"}}` at 512 px plus `get_state`'s fields |
| `env.back_project` | `row`, `col`, kw `camera="agentview"` | `{"camera", "pixel", "world_xyz"}` through the current depth |
| `env.segment` | `prompt`, kw `camera`, `min_score=0.2` | SAM3 (`--sam3`) mask: `{"found", "score", "box", "mask" bool[512,512], "n_pixels", "centroid_rowcol", "world_xyz"}` |
| `env.get_task_language` | - | str |
| `env.ground_truth_poses` | kw `names=null` | poses of robosuite's task objects (`cube`; `cubeA`, `cubeB`; `SquareNut`, `RoundNut`, `peg1`, `peg2`; `pot` + `pot_handle0` / `pot_handle1`; `hammer` + `hammer_handle`; Wipe's dirt markers) |
| `env.preview_reach` | `pos`, `quat_xyzw=null`, kw `arm` | the `--ik` reach preview (robot model `panda_libero`), or the `unknown` answer without it |
| `code.api` | kw `tier=null` | the primitive registry (`robots/robosuite/primitives.py`): high = `get_state`, `get_observation`, `segment`, `back_project`, `preview_reach`, `move_to`, `set_gripper`; low = `get_state`, `get_observation`, `move_delta`, `set_gripper`, `raw_obs`, `render_camera`, `get_camera_meta`, `step`; privileged adds `ground_truth_poses`; a grasp server (`--graspnet` ...) adds its planner's |

`stop` interrupts `env.move_to` / `env.move_delta` / `env.set_gripper` between control steps
(`info.cancelled`) and `env.chunk_step` between actions.

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

### behavior-env (`robots/behavior/env_server.py`)

One BEHAVIOR-1K 2025-challenge activity on the R1Pro in OmniGibson (Isaac Sim), the primitives
of OmniGibson's StarterSemanticActionPrimitives (cuRobo plans) run one control step at a time.
`--task` is the activity (`robots/behavior/tasks.py`: CaP-X's `turning_on_radio`,
`picking_up_trash`, then the other 48), `--seed` its pre-sampled instance id, `--gpu-id` the
simulator's GPU (`OMNIGIBSON_GPU_ID`). Each observation is `{"head", "left_wrist", "right_wrist"
uint8[S,S,3], "<camera>_depth" float32[S,S] (m), "base_pos" float32[3], "base_quat_xyzw"
float32[4], "base_yaw", "eef": {left/right: {"pos", "quat_xyzw", "gripper_width"}}, "success"
(the BDDL task's own termination), "q_score" (BEHAVIOR's partial credit: goal predicates newly
satisfied over all, 1 on success), "goals": {"satisfied", "total"}, "terminated", "truncated",
"env_steps", "privileged": {"in_hand": {left, right: object name or null}, "picked"}}`.
`privileged` is what only the simulator knows (OmniGibson's grasp state and CaP-X's
pick_up_*_reward judgement, a reference field, never success); the pi robot shows `in_hand` to the
planner only under `--privileged` and records `picked` in `robot_result.reference`. Every
primitive's result is the observation plus `{"primitive", "ok", "phase", "steps"[, "error"
(OmniGibson's ActionPrimitiveError or a stuck primitive; nothing more moved), "cancelled"]}`.

| method | args | result |
|---|---|---|
| `env.get_env_meta` | - | `{"task", "task_index", "seed", "instruction", "scene_model", "instances", "robot", "cameras", "image_size", "grasping_mode", "max_steps", "gpu_id"}` |
| `env.reset` | - | `[obs, {"instruction", ...}]` (reloads the instance, settles, latches the goal predicates and object heights) |
| `env.step` | `action` float[action_dim] (the R1Pro's full action vector) | `[obs, 0.0, terminated, truncated, {"success"}]` |
| `env.chunk_step` | `actions` float[N,action_dim], kw `return_all_frames=false` | `[obs or list[obs], terminated, truncated, info[, "cancelled"]]` |
| `env.navigate_to_pose` | kw `x`, `y`, `yaw` (world, m / rad; refused beyond 5 m) | obs + primitive report + `{"goal", "reached_pos", "reached_yaw", "distance_left_m", "yaw_left_rad"}` |
| `env.move_hand` | kw `arm` left/right, `position` float[3], `quat_xyzw=null` (default: the current orientation; refused beyond 1.5 m of the base in xy) | obs + report + `{"arm", "eef_pos", "eef_quat_xyzw", "distance_left_m", "gripper_width"}` |
| `env.grasp_object` | kw `arm`, `position`, `quat_xyzw=null`, `pregrasp_offset_m=0.1` | obs + report (phases open, pregrasp, close/approach, settle, lift) + `{"grasping_mode", ...as move_hand}` |
| `env.open_gripper` / `env.close_gripper` | kw `arm` | obs + report + `{"arm", "gripper_width"}` |
| `env.get_robot_position` | - | `{"pos", "quat_xyzw", "yaw", "eef": {left/right: {"pos", "quat_xyzw"}}}` (no stepping) |
| `env.state` | - | the obs without images |
| `env.raw_obs` | - | `{"proprio", "joint_positions", "joint_names"}` (no task state: the BDDL low-dim observation is privileged) |
| `env.render_camera` | `camera_name="head"`, `left_wrist`, `right_wrist`, kw `depth=false` | rgb uint8[S,S,3], or `[rgb, depth_m float32[S,S]]` |
| `env.get_camera_meta` | `camera_name` | `{"camera", "intrinsic_K" 3x3, "extrinsic_cam2world" 4x4, "convention": "opengl" (looks along -Z, +Y up), "width", "height"}` at the current pose |
| `env.get_task_language` | - | str (the challenge's task description) |
| `env.ground_truth_poses` | kw `names=null` | poses of the task's BDDL object instances (`task.object_scope` names) |
| `code.api` | kw `tier` | the registry of `robots/behavior/primitives.py` (high: CaP-X's motions and get_robot_position; low adds state, raw_obs, render_camera, get_camera_meta, step, chunk_step) |

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
| `env.preview_reach` | `pos` float[3] (m, base frame), kw `quat_xyzw=null` (null = the current TCP orientation) | reach preview (see libero-env); with `--ik`, `env.move_delta` / `env.rotate_delta` refuse an unreachable end pose |

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

#### Perception primitives (`--sam3 <url>`, `--unidepth <url>`; `utils/perception.py`)

Both franka servers compose the model servers over their *current observation*: the frames
the last `env.get_observation` returned (the client's latest state step). Without the flags
none of these methods exist; `capabilities.perception` = `{"segment": bool, "enhance_depth":
bool}` says which are on, and `code.api` lists them (`robots/franka/primitives.py`:
`SEGMENT_PRIMITIVES`, `ENHANCE_DEPTH_PRIMITIVES`) only then. `camera` is `"wrist"` (main) or
`"third_person"` (extra_0).

| method | args | result |
|---|---|---|
| `env.segment` | `camera="wrist"`, kw exactly one of `text_prompt` / `point` `[row, col]`, `min_score=0.2`, `all=false` | `{"found", "observation", "camera", "count", "detections": [{"id", "rank", "score", "box", "area_px", "centroid_rc", "depth_m", "point_camera" (OpenCV camera frame, m), "mask_png_base64", "prompt", "point", "observation"}], "ids", "invalidated", "overlay"? uint8[H,W,3], "reason"?}` |
| `env.select_detection` | `id` | `{"ok", "detection"?, "error"?, "observation", "ids", "selected", "rejected", "invalidated"}` |
| `env.reject_detection` | `id` | same shape; the id stays resolvable and is listed in `rejected` |
| `env.enhance_depth` | `camera="wrist"` | `{"ok", "observation", "camera", "depth" float32[H,W] (fused, m, 0 = none), "report" {mode `filled` / `mono_only` / `sensor_only`, scale, overlap_pixels, filled_pixels, ...}, "estimate", "invalidated"}` |

Short ids (`d7`) are OpenETA's evidence chain: the planner passes ids, the server resolves
them to the mask and frame they came from (stage 8's grasp and place primitives take these
ids). Every `env.get_observation` invalidates all ids; a stale id is refused with the
observation it belonged to, and `invalidated` on the next perception result lists the ids
dropped since the previous one. Ids are never reused within a server process.
`env.enhance_depth` replaces that camera's depth in the current observation (sensor holes
filled with the UniDepth estimate scaled to their overlap; a camera without depth takes the
estimate as-is), so later `env.segment` calls project through it.

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
| openvla, openvla-oft, gr00t | `vla.predict` | `obs` (the Pi0.5 LIBERO wire dict: `main_images` uint8[1,H,W,3], `wrist_images` uint8[1,H,W,3] or null, `states` float32[1,8], `task_descriptions` [str]), `options={"mode": "eval"[, "seed": int]}` | float32[1, horizon, 7] in LIBERO's OSC action space (gripper -1 open / +1 close), ready for `env.chunk_step`: OpenVLA 1 step, OpenVLA-OFT 8, GR00T its action horizon (16). The adapter does the model's own preprocessing (LIBERO's upside-down frame flipped, OpenVLA's 224 resize / OFT's 0.9 center crop, prompt template) and denormalisation |
| openvla, openvla-oft, gr00t | `vla.reset` | - | `{"ok": true}` (per-episode state; a no-op for these stateless policies) |
| openvla, openvla-oft, gr00t | `vla.info` | - | `{"service", "model" (repo id or path), "revision" (pinned commit hash or null), "horizon", "action_dim": 7, "wrist": bool}`; the robot records `service model@revision` per grasp tool as `vla` in `robot_result` |
| rldx-vla | `vla.get_modality_config` | - | `{"video_delta_indices": [int], "hist_maxlen": int}` |
| rldx-vla | `vla.predict` | `obs_dict`, `options` (must not contain `session_ids`; optional `"seed": int`) | dict of action arrays |
| rldx-vla | `vla.reset_session` | - | `{"ok": true}` |
| sam3 | `sam3.segment` | `image_base64` (PNG/JPEG bytes), exactly one of kw `text_prompt` str or `point` `[row, col]`, `min_score=0.2`, `all=false` | `{"found", "score"?, "box"?, "mask_png_base64"?, "mask_shape"? [H,W], "reason"?}`; with `all=true`: `{"found", "count", "detections": [{"index", "score", "box"?, "area_px", "mask_png_base64", "mask_shape"}], "reason"?}`, best first, every non-empty mask at or above `min_score` (a point prompt gives SAM3's three multimask candidates) |
| molmo | `molmo.ground` | `image_base64`, `query` str | `{"point_xy"? [x, y] pixels, "answer", "image_size" [W, H]}` (byte-identical on both backends) |
| molmo (`--model molmopoint`) | `molmo.ground_set` | `images_base64` list of 1-4 base64 PNG/JPEG (Image 1, 2, ... in the prompt), `query` str (the pointing prompt as authored) | `{"points": [{"id", "image_index" (0-based), "pixel_x", "pixel_y"}], "point_count", "answer", "image_sizes" [[W, H], ...], "coordinate_convention"}` |
| unidepth | `depth.estimate` | `rgb` uint8[H,W,3] (or base64 PNG/JPEG), kw `K` 3x3 pinhole intrinsics of that image or null | `{"depth" float32[H,W] metres (0 = none), "confidence"? float32[H,W], "model", "resolution_level", "used_intrinsics", "valid_ratio", "depth_range_m", "inference_s"}` |
| ik | `ik.robots` | - | `{"backend": "pyroki" \| "curobo", "robots": {name: {"description", "ee_link", "tool_offset_xyz", "arm_joints", "fixed_joints", "home_q", "note", "loaded"[, "lower", "upper"][, "curobo_config", "supported"]}}}` |
| ik | `ik.solve` | `robot` str, `target_pose` (`{"pos": [x, y, z], "quat_xyzw": [...]}` or flat `[x, y, z, qx, qy, qz, qw]`, TCP in the robot base frame, m), kw `seed_q=null` (arm joints, rad) | `{"robot", "q": [arm joints] \| null, "ok", "error" str \| null, "position_err" m, "orientation_err" rad, "solve_ms"}` plus `seeds_tried` (pyroki) or `self_collision_checked` (curobo); `ok` needs both errors within `--pos-tol` (5 mm) / `--ori-tol` (0.05 rad) and the joints within limits; `q` on failure is the best attempt |
| ik | `ik.plan` | `robot`, `start_q`, exactly one of kw `goal_pose` / `goal_q`, kw `obstacles=null` (list of `{"type": "box", "position", "extent"[, "quat_xyzw"]}`, `{"type": "sphere", "center", "radius"}`, `{"type": "capsule", "position", "radius", "height"[, "quat_xyzw"]}`, `{"type": "halfspace", "point", "normal"}`, m, base frame), kw `waypoints=20` | `{"robot", "path": [[arm joints], ...] \| null, "ok", "error", "plan_ms", "collision_free": bool \| null (null = no obstacles given)}` plus `max_joint_step_rad`, `min_clearance_m` (pyroki) or `status`, `dt`, `obstacles` (curobo) |

`ik` (`components/ik_server.py`) is an internal dependency of the env servers (`env.preview_reach`,
their motion primitives' reach check, a Robosuite `move_to`), not an agent tool. Robot models:
`panda` (TCP = libfranka's O_T_EE, flange + 0.1034 m; the Frankas' `tcp_pose`), `panda_libero`
(robosuite's `robot0_eef_pos` grip site 0.097 m below `panda_hand`, `robot0_eef_quat`; poses in
the `robot0_base` frame), `ur5e` (TCP = `tool0`), `piper` (TCP = `gripper_base`). Backends:
`--backend pyroki` (default; JAX on the CPU, MIT) solves IK as a least-squares problem from the
seed, the home pose and 6 random restarts, and plans by interpolating the pose and solving each
waypoint from the previous one; with obstacles it adds PyRoKi's collision costs and verifies the
path against them, so it only avoids obstacles locally (no graph search; `ok` is false when the
path penetrates one). `--backend curobo` (GPU, `ik-curobo` extra, Panda and UR5e) uses cuRobo's
IKSolver (self-collision aware) and MotionGen (collision-free trajectory optimisation with graph
seeds; `path` is the interpolated trajectory `dt` apart, or `waypoints` samples of it). The first
solve per robot compiles (PyRoKi about 1-2 s; cuRobo's planner warm-up about a minute); `--robots`
(default `panda,panda_libero`) does that before serving.

`molmo_server --model` picks the backend: `molmo2` (default; `allenai/Molmo2-8B`, `molmo.ground`
only) or `molmopoint` (`allenai/MolmoPoint-8B` rev `188130f`, OpenETA's pointing model; the HF ids
are accepted as aliases). `molmo.ground` is its one-image case; `molmo.ground_set` is OpenETA's
Pointing Image Set (`tools/molmopoint_core.py`): one prompt over an ordered set of up to four
images in a single inference, the model's `extract_image_points` tagging each point with the
0-based index of the image it lies in. `MOLMO_CHECKPOINT_PATH` names that backend's weights.
`unidepth_server` serves OpenETA's checkpoint `lpiccinelli/unidepth-v2-vitl14` (`UNIDEPTH_MODEL`
overrides; a local directory works) at `--resolution-level 4`.

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
