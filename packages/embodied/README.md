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
| LIBERO / LIBERO-PRO | `src/libero` | LIBERO `terminated` | all below, plus flywheel, operator, Flash (Molmo re-anchoring) |
| RoboCasa | `src/robocasa` | `env._check_success()` | all below, plus flywheel, recipe Flash (Molmo re-anchoring) |
| RoboTwin | `src/robotwin` | `eval_success` | all below, plus flywheel, recipe Flash (Molmo re-anchoring) |
| ManiSkill (`--robot`, below) | `src/maniskill` | ManiSkill `success` | all below, plus recipe Flash (Molmo + ray-plane re-anchoring of delta waypoints) |
| RoboLab | `src/robolab` | RoboLab's task predicate | all below, plus recipe Flash (Molmo + ray-plane re-anchoring of delta waypoints) |
| Robosuite | `src/robosuite` | robosuite `_check_success` (Restack adds CaP-X's off-table rule), latched | video, units, VDM, code.api, `--privileged` |
| Metaworld | `src/metaworld` | Metaworld `info["success"]`, latched | video, units, VDM, code.api, `--privileged` |
| Genesis | `src/genesis` | the task predicate (cube_pick: an 8 cm lift) | video, units, code.api, `--privileged` |
| BEHAVIOR-1K / R1Pro | `src/behavior` | the BDDL activity's `success`, latched (`q_score` = partial credit) | video, VDM, code.api, `--privileged` |
| Franka (real) | `src/franka` | operator verdict (`--operator`) | all below but `--privileged`; explore resets through the operator |
| Dual Franka (real) | `src/dual_franka` | operator verdict (required) | all below but `--privileged`; explore resets through the operator |
| Piper / dual Piper (real) | `src/piper` | operator verdict (required) | all below but `--privileged`; explore resets through the operator |
| UR5e (real) | `src/ur5e` | operator verdict (required) | all below but `--privileged` and memory/explore; bound to one arm (`--arm-id`) |

ManiSkill's `--robot` picks the arm (ManiSkill 3.0.1 agents with a parallel gripper that the stock
table scene places), all in translation-only `pd_ee_delta_pos` with the same MV_* vectors, 2 cm step
and servo (each measured at 19.7 mm per unit along its axis). The RLinf rigs (BlockPAP-v1 /
BlockStack-v1) run their own Panda. Results record a non-Panda arm as `maniskill_robot`, and
`maniskill/eval.sh` keeps each arm in its own out dir.

| `--robot` | ManiSkill uid | Env ids | Gripper | Wrist view |
| --- | --- | --- | --- | --- |
| `panda` (default) | `panda_wristcam` | all 12 stock ids and the rigs | mimic, +1 open / -1 close | Show-Harness's centred D415, turned 270 deg |
| `xarm6_robotiq` | `xarm6_robotiq` | PickCube, StackCube, PullCube, LiftPegUpright, PlaceSphere, StackPyramid, PullCubeTool, PlugCharger | Robotiq 2F-85 in delta mode, +1 close / -1 open | the wristcam variant's camera on `camera_link`, turned 90 deg |
| `widowxai` | `widowxai` (+ `pd_ee_delta_pos` on its six arm joints) | PickCube | carriages, +1 open / -1 close; 10-step hold | none: the agentview alone |

The other ids are refused per arm: the xArm6 cannot reach PushCube's and PokeCube's goals,
PegInsertionSide resets a Panda joint vector, PickSingleYCB has no xArm6 layout, and with its
gripper held pointing down the WidowX AI reaches only ~0.37 m from its base (PickCube's own layout).
Not offered: `so100` and `koch-v1.1` (joint control only and no TCP link), `fetch` (mobile base),
`ur_10e` / `widowx250s` (joint control only; the table scene has no placement for them), and the
`*_wristcam` uids of the xArm6 and WidowX AI (the table scene does not place the former, its arm
spawns inside the table; PickCube gives the latter the Panda's layout, out of its reach).

"All below" is memory, explore, video, units (so GUMI and the fine-tuned provider), VDM (`--vdm`),
the primitive registry (`code.api`) and, in simulation, `--privileged`. ManiSkill, RoboLab and the
Piper publish no memory corpus: they default to the local one exploration writes. Robosuite,
Metaworld, Genesis and BEHAVIOR mount only what their rows list: none of them has memory or
explore, Genesis has no VDM, and BEHAVIOR has no units (its motions are cuRobo-planned primitives
of a mobile two-arm robot, not fixed-frame 2 cm steps).

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
- `src/video.ts`: episode video (ffmpeg). `src/flywheel.ts`: Flywheel data (`--collect-flywheel-data`,
  default root `~/.pi/embodied/datacollection`): every env step with what the robot's VLA reads and
  emits, on LIBERO, RoboCasa and RoboTwin (the robots whose VLA runs agent-side). `/flywheel-export
  [selection]` writes a LeRobot v3.0 dataset with the shared feature names (`--flywheel-python`,
  default `--python`; LeRobot needs its own venv, see services/README.md, which also covers GUMI runs).
- `src/units/`: Show-Harness action units. `--units=true` hides the robot's tools: the model drives
  the arm with `act` (one unit: MV_FWD/BACK/LEFT/RIGHT/UP/DOWN, ROTATE_CW/CCW where the robot has
  yaw, GRASP, RELEASE, STOP, DONE; optional repeat `n`), `finish`, and the plugins' `point` / `plan`,
  under the ported zero-shot prompt. `--units=both` adds `act` (and `point` / `plan`) to the robot's
  tools and appends the units section to its prompt. `--stateless` keeps only the task and the latest
  observation turn (the paper's no-history setting). `--units-plugins` (default `auto`: the robot's
  set, else Show-Harness's zero-shot Franka set
  `recovery,auto_release,proprioception,variable_step,action_chunk,rotation,plan,mem_text`; `point` is
  opt-in) picks the plugins. A configuration without a wrist camera (ManiSkill `--robot widowxai`,
  dual Franka without an inline wrist camera, a UR5e or Piper streaming none) runs `auto` without
  variable_step and action_chunk and refuses to start when they are named; `--units-coarse-step` is variable_step's coarse step. A robot
  opts in with `units` in its spec (base-frame unit vectors, step, optional yaw step, `apply`,
  `state`); the moves go through its own safety checks (Franka and dual Franka: `--max-move`,
  `--workspace-xy`, `--z-floor`).
- `src/code/`: code mode (CaP-X's run_code). `--code=true` hides the robot's tools: the model
  writes Python programs that `run_code` executes on the env server against its primitive registry
  (`code.api`; `--code-api=high|low`, CaP-X's S2/S3; `--privileged` runs the privileged tier, S1),
  in a spawned subprocess with no env object whose calls the server resolves through the registry
  (JSON over the pipe, never pickle; the child starts without the server's secret-looking
  environment variables, in its own process group, with no new processes or threads allowed;
  it can still open sockets, so run the server in a container to keep programs off the network),
  killed at `--code-timeout` (a stop is issued, also inside a running primitive), refused past
  `--code-max-calls` calls or `--code-max-move` metres; `--code-helpers` injects CaP-X's numpy
  helpers. `--code=both` adds
  `run_code` to the robot's tools. Mutually exclusive with `--units`; `--stateless` applies. Real
  robots need `--code-real` and `--operator`, and every program is confirmed by the operator.
  LIBERO today (services/PROTOCOL.md, code mode).
- `src/dashboard/`: live web dashboard (`--dashboard`) for any robot.
- `src/libero/flash.ts`: Flash replay without an LLM (`--model flash/replay`).

Every robot result carries `robot`, `claimed`, `summary`, `turns`, `planner_budget_exhausted`,
`planner_error` and `env_error` (also set when the env server exits or a service stops answering
mid-episode); the eval scripts count an episode only when the environment produced a result and
the planner did not fail, and rerun the others. LIBERO and RoboTwin episodes get
`--time-limit ${TIME_LIMIT:-1800}` s (the episode ends as a failure) and a `timeout` backstop
900 s later (the episode is invalid and rerun).

`src/eval-parallel.sh` runs a robot's eval.sh matrix (LIBERO, ManiSkill, Metaworld, Robosuite, Genesis,
BEHAVIOR, RoboLab, RoboTwin, RoboCasa) on N workers (`-j N --gpus 1`: several workers
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
- Metaworld: the services' `[metaworld]` extra (Python 3.11, metaworld 3.1.1, no assets); the 50 MT50
  Sawyer tasks (`--task reach-v3 --seed 0`), a world-frame `move_delta` plus `gripper`, depth tools
  (`back_project`, `segment` with a SAM3 server), units mode and `--privileged`; `src/metaworld/eval.sh`.
- Robosuite: the services' `[robosuite]` extra (Python 3.11, robosuite 1.5 in its own venv: LIBERO and
  RoboCasa pin 1.4 forks); CaP-X's seven tasks (`--task Lift --seed 0`, two-arm tasks take `arm`),
  closed-loop `move_to` / `move_delta` under `--max-move`, `gripper`, depth tools, units mode and
  `--privileged`; `src/robosuite/eval.sh`.
- Genesis: the services' `[genesis]` extra (Python 3.11, Genesis 1.4, a GPU for rendering, no assets);
  OpenETA's Franka `cube_pick` (`--task cube_pick --seed 0`), a base-frame `move_delta` plus `gripper`,
  depth tools, units mode and `--privileged`; `src/genesis/eval.sh`.
- BEHAVIOR-1K: the venv `services/setup.sh behavior` builds (Isaac Sim, OmniGibson and BDDL from a
  BEHAVIOR-1K checkout, the challenge dataset; see services/pi_embodied_services/robots/behavior/README.md);
  the 50 2025-challenge activities on the R1Pro (`--task turning_on_radio --seed <instance> --gpu-id N`),
  OmniGibson's semantic primitives as tools (`navigate_to_pose`, `move_hand`, `grasp_object`, the
  grippers), `segment` / `point` / `back_project` on three cameras, `--grasping-mode` and
  `--privileged`; `src/behavior/eval.sh`.
- Franka / dual Franka: the services' `[franka]` extra, a Ray cluster on the controller nodes,
  hand-eye calibration, and an operator at the emergency stop. Flags use a `--robot-`
  prefix (`--robot-env`, `--robot-vla`, `--robot-sam3`, `--robot-config`).
- UR5e: the services' `[ur5e]` extra (ur_rtde, the shared `components/cameras` layer: RealSense
  D400 / L515 with `[realsense-l515]`, webcams, RTSP; `--robot-cameras name=type:source,...`), a
  Robotiq gripper over the URCap socket, `--operator` and `--arm-id <controller serial>` (the config,
  its limits and its camera calibrations are bound to that arm; `env_server --print-identity`), and
  the hand-eye tool `robots/ur5e/calibrate.py` (`capture`, `solve` -> `<calibration>.new.yaml`,
  `apply --yes` after a human reviewed the residuals).

An abort (Esc, `/abort`, a session switch) stops a robot between RPC calls and asks the server to
`stop` its running call; what each server can interrupt mid-call is listed in
services/PROTOCOL.md (the rest finish the call, bounded by their own step clips and timeouts). At session end the base asks the
server it started to `shutdown`, which closes the environment (and the real arm's RLinf worker)
before the process exits.

## OpenETA extras

Features ported from OpenETA beyond the core harness; each is off by default and registers nothing then.

| Feature | Enable | pi mechanism | Robots |
|---|---|---|---|
| Human as the model (OpenETA's manual VLM console) | `--model human/operator` (or any VLM flag set to it, e.g. `--attach-vlm-model human/operator`) | a provider (`src/human.ts`) asking through `ctx.ui` select/input (TUI dialogs, RPC `extension_ui_request`) and the dashboard's composer | any |
| Web search / page fetch | `pi install npm:pi-web-search npm:@zeldrisho/pi-web-fetch`, then `--web-tools` | pi packages; `src/web.ts` keeps their tools active next to the robot's | any |
| Object memory | `--object-memory` [`--object-memory-dir <dir>`] | tools + `object_record` session entries (`src/objects.ts`) | any |
| Multi-waypoint route | `--waypoints` | `follow_waypoints` (`src/primitives/waypoints.ts`) | LIBERO, Franka |
| Wrist-view alignment | `--align-wrist` | `align_wrist` (`src/primitives/wrist.ts`) | LIBERO, Franka |
| Grasp advisor | `--grasp-advisor` [`--grasp-advisor-model`], with a plan_grasp backend | `suggest_grasp` (`src/primitives/advisor.ts`), a side VLM call | LIBERO |
| Task skills | `/skill:embodied-pick` ... | pi skills (`skills/`) | any |
| One server per arm | always, real robots | `utils/hardware_lock.py` flock per arm (`--lock-id`, `--lock-dir`, `$PI_EMBODIED_LOCK_DIR`) | Franka (RLinf, Polymetis), dual Franka, Piper, UR5e |

The flags that change what the agent can do are recorded in the result row as `extras`.
