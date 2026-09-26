# pi-embodied

Robots as pi extensions. Each robot calls `defineRobot(pi, spec)` (`src/robot.ts`) and then
registers only its flags, tools and observations. The base owns the rest: the task (flags, or a
`robot_task` entry from `/robot-task` or the dashboard), fail-closed startup, the finish rules and
the --max-turns / --time-limit budget, one `robot_result` entry per episode, the env server's
lifecycle, pruning of old camera frames, the file-tool guard, the status published on `pi.events`
for the dashboard, and the shared modules below. Everything else (the agent loop, models,
sessions, interactive/print/json/rpc modes) is pi.

Load one robot per process (`-e`, or an experiment directory's settings). Robots share flag and
tool names (`--seed`, `--task`, `finish`, `move_to`, and the shared modules' `--operator`,
`--memory-dir`, ...), and pi rejects two loaded extensions that register the same flag or tool, so
they cannot all be listed in `pi.extensions`; a robot also replaces the coding tools. The package
manifest therefore loads only the onboarding extension (`src/setup`) and the
`embodied-quickstart` skill.

## Install and quickstart

```bash
pi install ./packages/embodied            # from a checkout; or npm:@lv-robotics/pi-embodied once published
pi                                        # first start in a project: a notice points to /embodied-setup
```

`/embodied-setup` asks for the robot or benchmark (and what it needs: GPU, Isaac Sim, real hardware
and an operator), the mode (tools, units, fine-tuned, flash), where the services run (here, or a
remote GPU box over ssh) and the planner model. After you confirm, it writes the experiment
directory's `.pi/settings.json` and asks the agent to follow the `embodied-quickstart` skill: run
`services/setup.sh <robot>` (venv, pyproject extra, assets, `--weights` for checkpoints), then the
preflight (`node src/check.ts <robot>`; `/robot-check` inside a robot session), and report the
launch command. The agent runs each step with its bash tool, under your normal approvals.

The experiment directory's settings load the robot through `extensions` and drop the onboarding
extension with a delta entry for this package (pi's package filters only narrow what the manifest
declares, so they cannot add a robot):

```json
{
  "packages": [{ "source": "<package>", "autoload": false, "extensions": ["-src/setup/index.ts"] }],
  "extensions": ["<package>/src/libero/index.ts", "<package>/src/dashboard/index.ts"],
  "defaultProvider": "<provider>",
  "defaultModel": "<model>"
}
```

Settings cannot carry extension flags, so the task and mode stay on the command line, e.g.
`cd <experiment dir> && source services/.venv-libero-pro/pi-embodied.env && pi --suite libero_10 --task 0 --seed 0 --units=true --dashboard=true`.
Run the eval scripts outside that directory: they load the robot with `-e`, and loading it twice
fails on duplicate tools.

The package is publishable to npm on its own (`keywords: ["pi-package"]`, host packages as `*` peers,
`files` = src, skills, README, `publishConfig.access: public`): `cd packages/embodied && npm publish`.
It is not part of pi's lockstep release (`scripts/release-packages.mjs` takes only the
`@earendil-works/*` packages), so its version moves independently. An npm install has no `services/`:
setup clones this repository for them (`--services` / `PI_EMBODIED_SERVICES` point the robots at it).

Docker images (planned, not built): one image per services venv, since the extras pin conflicting
Torch/Transformers stacks. LIBERO / LIBERO-PRO / ManiSkill / RoboTwin on
`nvidia/cuda:12.8.1-cudnn-devel-ubuntu22.04` (EGL and Vulkan runtime libraries; cuRobo builds need
the devel image), RoboCasa on the same base with Python 3.10, RoboLab on NVIDIA's Isaac Sim 6.1
container, Piper on `ros:noetic` with the `[piper]` extra. Each image runs `services/setup.sh
<robot> --weights` at build time with weights in a mounted volume; pi runs outside and attaches
with `--env` / `--vla` / `--sam3`. Real-arm robots (Franka, dual Franka) stay on the controller host.

| Robot | Extension | Success signal | Shared modules |
| --- | --- | --- | --- |
| LIBERO / LIBERO-PRO | `src/libero` | LIBERO `terminated` | all below, plus flywheel, operator, Flash |
| RoboCasa | `src/robocasa` | `env._check_success()` | all below |
| RoboTwin | `src/robotwin` | `eval_success` | all below |
| ManiSkill | `src/maniskill` | ManiSkill `success` | all below |
| RoboLab | `src/robolab` | RoboLab's task predicate | all below |
| Franka (real) | `src/franka` | operator verdict (`--operator`) | all below but `--privileged`; explore resets through the operator |
| Dual Franka (real) | `src/dual_franka` | operator verdict (required) | all below but `--privileged`; explore resets through the operator |
| Piper / dual Piper (real) | `src/piper` | operator verdict (required) | all below but `--privileged`; explore resets through the operator |

Every robot mounts memory, explore, video, units (so GUMI and the fine-tuned provider), VDM
(`--vdm`), the primitive registry (`code.api`) and, in simulation, `--privileged`. ManiSkill,
RoboLab and the Piper publish no memory corpus: they default to the local one exploration writes.

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
  (LIBERO only; `/flywheel-export` runs `pi_embodied_services.flywheel` with `--flywheel-python`,
  default `--python`; LeRobot needs its own venv, see services/README.md).
- `src/units/`: Show-Harness action units. `--units=true` hides the robot's tools: the model drives
  the arm with `act` (one unit: MV_FWD/BACK/LEFT/RIGHT/UP/DOWN, ROTATE_CW/CCW where the robot has
  yaw, GRASP, RELEASE, STOP, DONE; optional repeat `n`), `finish`, and the plugins' `point` / `plan`,
  under the ported zero-shot prompt. `--units=both` adds `act` (and `point` / `plan`) to the robot's
  tools and appends the units section to its prompt. `--stateless` keeps only the task and the latest
  observation turn (the paper's no-history setting). `--units-plugins` (default: Show-Harness's
  zero-shot Franka set `recovery,auto_release,proprioception,variable_step,action_chunk,rotation,plan`;
  `point` is opt-in) picks the plugins; `--units-coarse-step` is variable_step's coarse step. A robot
  opts in with `units` in its spec (base-frame unit vectors, step, optional yaw step, `apply`,
  `state`); the moves go through its own safety checks (Franka and dual Franka: `--max-move`,
  `--workspace-xy`, `--z-floor`).
- `src/dashboard/`: live web dashboard (`--dashboard`) for any robot.
- `src/libero/flash.ts`: Flash replay without an LLM (`--model flash/replay`).

Every robot result carries `robot`, `claimed`, `summary`, `turns`, `planner_budget_exhausted`,
`planner_error` and `env_error` (also set when the env server exits or a service stops answering
mid-episode); the eval scripts count an episode only when the environment produced a result and
the planner did not fail, and rerun the others. LIBERO and RoboTwin episodes get
`--time-limit ${TIME_LIMIT:-1800}` s (the episode ends as a failure) and a `timeout` backstop
900 s later (the episode is invalid and rerun).

`src/eval-parallel.sh` runs a robot's eval.sh matrix on N workers (`-j N --gpus 1`: several workers
may share a GPU; each gets CUDA_VISIBLE_DEVICES and the EGL device on the same PCI bus), as an A/B
over `--variant NAME=ARGS` (same cells, one subdirectory each), and reports success rate, Pass@k and
invalid cells per variant; `--min-success N` fails a regression run, `--max-api-concurrency M` caps
model calls across workers. Every cell is its own eval.sh call, so validity and reruns are eval.sh's.

## LIBERO

Needs the repository's Python services (`services/`, package `pi_embodied_services`) installed
with the `[libero]` extra (`services/setup.sh libero`, or see services/README.md); the extension
speaks their HTTP RPC directly.
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

