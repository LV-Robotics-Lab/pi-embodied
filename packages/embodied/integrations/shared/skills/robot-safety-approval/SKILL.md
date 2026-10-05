---
name: robot-safety-approval
description: The safety and approval rules for pi-embodied robot tools - which motions need the operator, what stop/resume and the hardware lock do, and what to do when a call is refused. Use before the first motion tool call and whenever a motion is refused or asks for confirmation.
---

# Safety and approval

## Who decides

The env server enforces the robot's limits for every caller (per-call move and rotation caps,
workspace box, z floor, reach and collision checks where IK is on); a refusal is final for that
call. On a real robot (Franka, dual Franka, Piper, UR5e) the MCP server itself refuses every motion
tool and `reset` unless the operator authorised it outside this session: either the server was
started with `PI_EMBODIED_MOTION_CONFIRMED=1` (the whole session), or the operator wrote the tool's
name into the server's confirm file (`--confirm-file` / `PI_EMBODIED_CONFIRM_FILE`) right before
the call, which that one call consumes. The refusal names both channels; you cannot satisfy either:
tell the operator which tool you want to call and why, then wait. Never write to the confirm file
yourself, and never ask for the variable to be set to get past one refusal.

On top of that the plugin's PreToolUse hook asks the operator before high-risk motions, with the
same risk classes as pi's `--approval standard`:

- grasp or place execution (`execute_grasp`, `execute_place`, `release`, VLA skills) and resets,
  including the built-in `reset` (a simulator's new episode; on a real arm the gripper opens and the
  arm moves to its start pose, which is why a real arm is never reset when the server connects);
- a move to an absolute target (`move_to`, `move_pose`, `move_grip`, `navigate_to`, `move_to_joints`);
- a relative move commanding more than 0.1 m (`--large-move`);
- on a real robot (Franka, dual Franka, Piper, UR5e): every motion.

Claude Code prompts the operator (`permissionDecision: ask`). Codex 0.160 runs no plugin hooks at
all; there the plugin's `.mcp.json` marks every motion tool `approval_mode: "prompt"` so Codex itself
asks before each one, and on a real robot the server's gate above is what protects the arm (with
`--approve-for-me` the model answers Codex's prompt, not the operator). Do not look for a way around
a denial: report it and wait.

## Stop, resume, E-stop

- `stop` sends the env server's `stop` (the running call ends at its next step boundary; the arm
  settles on its last target) and latches the motion tools off. It is not an emergency stop: on a
  real arm the hardware E-stop is the emergency stop.
- `resume` clears the latch. Observe first.
- Motion tools also refuse when the env server does not answer `healthz` right now, or when another
  process answers on the port (`robot_status.same_process: false`): a server restarted behind your
  back is never moved blind.

## Hardware lock

One env server per physical arm per machine (`hardware_lock.py`): a second server for a held arm
exits at startup naming the holder (`robot busy: arm ... is driven by another env server`). The MCP
server then fails to start and serves nothing. Ask the operator to stop the other server; do not
try another port or a different robot name.

## When the episode ends

`finish` ends the episode: motion tools answer "The episode is finished." A server that stopped
answering ends it too ("The robot failed: ... The episode is over."). Either way, report the last
observation and the claim; a new episode is a new MCP server session started by the operator.
