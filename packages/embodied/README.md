# pi-embodied

Robots as pi extensions. Each robot calls `defineRobot(pi, spec)` (`src/robot.ts`) and then
registers only its flags, tools and observations. The base owns the rest: the task (flags, or a
`robot_task` entry from `/robot-task` or the dashboard), fail-closed startup, the finish rules and
the --max-turns / --time-limit budget, one `robot_result` entry per episode, the env server's
lifecycle, pruning of old camera frames, the file-tool guard, the status published on `pi.events`
for the dashboard, and the shared modules below. Everything else (the agent loop, models,
sessions, interactive/print/json/rpc modes) is pi.

Load one robot per process with `-e`. Robots share flag and tool names (`--seed`, `--task`,
`finish`, `move_to`, and the shared modules' `--operator`, `--memory-dir`, ...), and pi rejects
two loaded extensions that register the same flag or tool, so they cannot all be listed in
`pi.extensions`; `package.json` lists only the dashboard, which works with any robot.

| Robot | Extension | Success signal | Shared modules |
| --- | --- | --- | --- |
| LIBERO / LIBERO-PRO | `src/libero` | LIBERO `terminated` | memory, explore, video, flywheel, operator, Flash |
| RoboCasa | `src/robocasa` | `env._check_success()` | memory, explore, video |
| RoboTwin | `src/robotwin` | `eval_success` | memory, explore, video |
| Franka (real) | `src/franka` | operator verdict (`--operator`) | memory guard, operator |
| Dual Franka (real) | `src/dual_franka` | operator verdict (required) | memory guard, operator |

Shared modules:

- `src/memory/`: memory (`MEMORY.md`, global/suite/task layers,
  validation, merge, index), synced from `RLinf/RPent-memory` into `$PI_EMBODIED_MEMORY`
  (default `~/.pi/embodied/memory`); `/memory sync|validate|index|merge`.
  Its guard denies file tools by default: only the memory, the cell's files in the output dir,
  and (real robots) the robot's own step artifacts are reachable.
- `src/explore.ts`: exploration mode (`--explore`, `/explore`): reset budget, finish
  guard, attempt archive, cross-session handoff; each robot supplies its exploration prompt
  (LIBERO also its DISTIL pass).
- `src/operator.ts`: human-in-the-loop (`--operator`): verdict and scene-reset requests as
  `ctx.ui.select` dialogs (TUI or RPC client), plus `/success /failure /abort /done /continue /operator`.
- `src/video.ts`: episode video (ffmpeg). `src/flywheel.ts`: Flywheel data in the
  LIBERO-only schema (default root `~/.pi/embodied/datacollection`), with LeRobot export
  (LIBERO only; `/flywheel-export` runs `pi_embodied_services.flywheel` with `--python`).
- `src/dashboard/`: live web dashboard (`--dashboard`) for any robot.
- `src/libero/flash.ts`: Flash replay without an LLM (`--model flash/replay`).

Every robot result carries `robot`, `claimed`, `summary`, `turns`, `planner_budget_exhausted`,
`planner_error` and `env_error`; the eval scripts count an episode only when the environment
produced a result and the planner did not fail, and rerun the others.

## LIBERO

Needs the repository's Python services (`services/`, package `pi_embodied_services`) installed
with the `[libero]` extra (see services/README.md); the extension speaks their HTTP RPC directly.
Robots spawn their env servers from `--services` (env `PI_EMBODIED_SERVICES`, default the repo's
`services/`) with `PYTHONPATH` set to it; `serve.sh` and `eval.sh` default to the same directory.

```bash
export PI_EMBODIED_PYTHON=services/.venv-libero/bin/python
PI05_CHECKPOINT_PATH=... SAM3_CHECKPOINT_PATH=... packages/embodied/src/libero/serve.sh

pi -e packages/embodied/src/libero --suite libero_10_task --task 2 --seed 0             # interactive
packages/embodied/src/libero/eval.sh runs/l10 libero_10_task 0-9 0-2 --model <provider/model>
pi -p -e packages/embodied/src/libero --explore --suite libero_10_task --task 2 --seed 0 "/explore"
```

Each session ends with a `robot_result` entry (`terminated` is LIBERO's success flag,
`claimed` is the agent's own status). `/robot-task <suite> <task> <seed>` starts a new episode
in a new session. Boolean flags take the next word as their value; write them as
`--flag=true` before a prompt.

## Other robots

- RoboCasa: the services' `[robocasa]` extra (Python 3.10), kitchen assets, the RLDX-1-FT-RC365
  checkpoint; `src/robocasa/serve.sh`, `src/robocasa/eval.sh` runs Target50.
- RoboTwin: the services' `[robotwin]` extra (Python 3.11), RoboTwin assets, the LingBot-VLA
  RoboTwin checkpoint; `src/robotwin/serve.sh`, `src/robotwin/eval.sh`.
- Franka / dual Franka: the services' `[franka]` extra, a Ray cluster on the controller nodes,
  hand-eye calibration, and an operator at the emergency stop. Flags use a `--robot-`
  prefix (`--robot-env`, `--robot-vla`, `--robot-sam3`, `--robot-config`).

An abort (Esc, `/abort`, a session switch) stops a robot between RPC calls and asks the server to
`stop` its running call; what each server can interrupt mid-call is listed in
services/PROTOCOL.md (the rest finish the call, bounded by their own step clips and timeouts). At session end the base asks the
server it started to `shutdown`, which closes the environment (and the real arm's RLinf worker)
before the process exits.

