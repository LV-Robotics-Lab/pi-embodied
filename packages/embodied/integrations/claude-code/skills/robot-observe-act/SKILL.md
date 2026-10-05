---
name: robot-observe-act
description: Drive a pi-embodied robot through its MCP tools (observe, the manifest's motion and perception primitives, finish). Use when the pi-embodied MCP server is connected and the task is to look at or manipulate a scene with the robot.
---

# Observe, act, observe

The `pi-embodied` MCP server exposes one tool per primitive the robot's manifest declares for the
env server (`packages/embodied/src/primitives/manifests/<robot>.json`), plus five built-ins:
`robot_status`, `observe`, `finish`, `stop`, `resume`. Which primitives appear depends on the tier
(`--tier high|low|raw`, default all non-privileged), on `--privileged` (simulators only) and on what
the env server was started with (a `requires` the server does not meet hides the tool). Tool names
are the manifest names: `move_to`, `set_gripper`, `release`, `segment`, `plan_grasp`, ...
(in Claude Code they appear as `mcp__pi-embodied__<name>`).

## The loop

1. `robot_status` once: the env server answers, `same_process` is true, `halted` is false,
   `tools` lists what you can call. If `reachable` is false, stop: nothing moves.
2. `observe` before the first motion. Read the images and the state; name the objects you see.
3. One motion tool per step, then `observe` again. A motion result reports what the controller did
   (`ok`, `steps_used`, `final_dist_m`, `stopped`, `refused`, `cancelled`); only an observation
   shows what the scene looks like now.
4. Perception tools (`segment`, `detect`, `plan_grasp`, `back_project`, `view_points`, ...) are
   look-only: use them to turn pixels into metric targets before moving.
5. When the task is done, or cannot be done, call `finish` with `status` (`success`, `failure`,
   `blocked`, `aborted`) and a `summary` that cites the last observation. After `finish` only
   `observe`, `robot_status`, `stop` and `finish` run.

## Rules that hold on every robot

- Never chain motions without observing in between; never guess a pose from an old image.
- Keep relative moves small (the manifest's `max_move`-style limits are enforced by the env server
  and a refusal names the limit); split a long motion into steps.
- A `refused` result is the server's safety check (workspace box, z floor, reach, collision);
  change the target, do not retry the same call.
- A `stopped: contact` or `stopped: stalled` result means the arm met something: observe, then
  back off before anything else.
- If a tool answers "The episode is finished." or "The robot failed: ...", the session is over;
  report it and do not try other tools.
- If `robot_status.left_out` lists tools, they need a `--var` (robot variables such as `cameras`
  or `arms`) on the server command line; tell the operator rather than working around it.

## What is not here

Action units, Python code mode (`run_code`), visual differencing, memory, exploration, replay,
evaluation and `result.json` are pi features: run `pi` with the robot extension for them. The MCP
tools are the robot's tools mode only.
