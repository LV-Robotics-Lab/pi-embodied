# pi-embodied

Robots as pi extensions. A robot extension registers the robot's tools and flags,
replaces the system prompt, prunes old camera frames from context, and records the
environment's own success signal in the session. Everything else (the agent loop,
models, sessions, interactive/print/json/rpc modes) is pi.

Load one robot per process with `-e`; robots share tool and flag names.

| Robot | Extension | Success signal |
| --- | --- | --- |
| LIBERO / LIBERO-PRO | `src/libero` | LIBERO `terminated` |
| RoboCasa | `src/robocasa` | `env._check_success()` |
| RoboTwin | `src/robotwin` | `eval_success` |
| Franka (real) | `src/franka` | operator verdict (`--operator`) |
| Dual Franka (real) | `src/dual_franka` | operator verdict (required) |

Shared features, used by the robots:

- `src/memory/`: RPent-compatible memory (`MEMORY.md`, global/suite/task layers,
  validation, merge, index), synced from `RLinf/RPent-memory`; `/memory sync|validate|index|merge`.
- `src/explore.ts`: exploration mode (`--explore`, `/explore`): reset budget, finish
  guard, attempt archive, cross-session handoff, DISTIL.
- `src/operator.ts`: human-in-the-loop (`--operator`): `/success /failure /abort /done /continue /operator`.
- `src/video.ts`, `src/flywheel.ts`: episode video (ffmpeg) and flywheel data with LeRobot export.
- `src/dashboard/`: live web dashboard (`--dashboard`).
- `src/libero/flash.ts`: Flash replay without an LLM (`--model flash/replay`).

## LIBERO

Needs an [RPent](https://github.com/RLinf/RPent) checkout with its LIBERO, Pi0.5 and SAM3
services installed; the extension speaks their HTTP RPC directly.

```bash
export RPENT_ROOT=/path/to/RPent RPENT_PYTHON=/path/to/venv/bin/python
PI05_CHECKPOINT_PATH=... SAM3_CHECKPOINT_PATH=... packages/embodied/src/libero/serve.sh

pi -e packages/embodied/src/libero --suite libero_10_task --task 2 --seed 0             # interactive
packages/embodied/src/libero/eval.sh runs/l10 libero_10_task 0-9 0-2 --model <provider/model>
pi -p -e packages/embodied/src/libero --explore --suite libero_10_task --task 2 --seed 0 "/explore"
```

Each session ends with a `libero_result` entry (`terminated` is LIBERO's success flag,
`claimed` is the agent's own status). Boolean flags take the next word as their value;
write them as `--flag=true` before a prompt.

## Other robots

- RoboCasa: RPent's `[robocasa]` extra (Python 3.10), kitchen assets, the RLDX-1-FT-RC365
  checkpoint; `src/robocasa/serve.sh`, `src/robocasa/eval.sh` runs Target50.
- RoboTwin: RPent's `[robotwin]` extra (Python 3.11), RoboTwin assets, the LingBot-VLA
  RoboTwin checkpoint; `src/robotwin/serve.sh`, `src/robotwin/eval.sh`.
- Franka / dual Franka: RPent's `[franka]` extra, a Ray cluster on the controller nodes,
  hand-eye calibration, and an operator at the emergency stop. Flags use a `--robot-`
  prefix (`--robot-env`, `--robot-vla`, `--robot-sam3`, `--robot-config`).

The RPC format and tool semantics follow RPent (Apache-2.0).
