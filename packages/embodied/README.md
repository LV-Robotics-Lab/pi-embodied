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
manifest therefore loads only the onboarding extension (`src/infra/setup`) and the
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
preflight (`node src/infra/check.ts <robot>`; `/robot-check` inside a robot session), and report the
launch command. The agent runs each step with its bash tool, under your normal approvals.

The experiment directory's settings load the robot through `extensions` and drop the onboarding
extension with a delta entry for this package (pi's package filters only narrow what the manifest
declares, so they cannot add a robot):

```json
{
  "packages": [{ "source": "<package>", "autoload": false, "extensions": ["-src/infra/setup/index.ts"] }],
  "extensions": ["<package>/src/robots/libero/index.ts", "<package>/src/capabilities/dashboard/index.ts"],
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

Source layout (`src/`, one directory per layer; `src/robot.ts`, the `defineRobot` base, stays at the top):

| Directory | Holds |
| --- | --- |
| `robots/<robot>/` | one pi extension per robot (sims and real arms), its prompts, eval.sh, serve.sh, oracles |
| `primitives/` | shared hand-written tools (grasp, motion, perception), the `code.api` registry types, IK, the VLA and XPolicyLab adapters |
| `modes/` | alternative action interfaces: `units/` (Show-Harness), `code/` (CaP-X run_code), `finetuned/` |
| `capabilities/` | opt-in features a robot mounts by flag: memory, explore, operator, flywheel, Flash, GUMI, dashboard, replay, object memory, web tools |
| `planner/` | what shapes or guards the planner's calls: fallback, ensemble, API gate, VLA seeding, context version, closed-loop contract, human-as-model |
| `observation/` | what the model sees and what is recorded: episode video, VDM, Viser |
| `infra/` | RPC client, encoders, model-service auto-start, `/robot-check`, onboarding (`setup/`) |
| `scripts/` | `eval-parallel.sh` |

| Robot | Extension | Success signal | Shared modules; notes |
| --- | --- | --- | --- |
| LIBERO / LIBERO-PRO | `src/robots/libero` | LIBERO `terminated` | memory, explore, video, units, VDM, code, `--privileged`, flywheel, operator, Flash; Flash re-anchors with Molmo |
| RoboCasa | `src/robots/robocasa` | `env._check_success()` | memory, explore, video, units, VDM, code, `--privileged`, flywheel, Flash; recipe Flash (Molmo re-anchoring) |
| RoboTwin | `src/robots/robotwin` | `eval_success` | memory, explore, video, units, VDM, code, `--privileged`, flywheel, Flash, XPolicyLab; recipe Flash (Molmo re-anchoring), XPolicyLab (`aloha_agilex`, joint and ee) |
| ManiSkill (`--robot`, below) | `src/robots/maniskill` | ManiSkill `success` | memory, explore, video, units, VDM, code, `--privileged`, flywheel, Flash; recipe Flash (Molmo + ray-plane re-anchoring of delta waypoints) |
| RoboLab | `src/robots/robolab` | RoboLab's task predicate | memory, explore, video, units, VDM, code, `--privileged`, Flash; recipe Flash (Molmo + ray-plane re-anchoring of delta waypoints) |
| RoboDojo (two ARX X5) | `src/robots/robodojo` | RoboDojo's `is_episode_end` (`score` = partial credit) | memory, explore, video, units, VDM, code, `--privileged`, flywheel, Flash, XPolicyLab; flywheel in joint space, recipe Flash (Molmo + depth back-projection), XPolicyLab (`arx_x5`, joint and ee) |
| Robosuite | `src/robots/robosuite` | robosuite `_check_success` (Restack adds CaP-X's off-table rule), latched | memory, explore, video, units, VDM, code, `--privileged`, flywheel |
| Metaworld | `src/robots/metaworld` | Metaworld `info["success"]`, latched | memory, explore, video, units, VDM, code, `--privileged`, flywheel |
| Genesis | `src/robots/genesis` | the task predicate (cube_pick: an 8 cm lift) | memory, explore, video, units, VDM, code, `--privileged`, flywheel |
| HumanCLAW (SMPL-X humanoid, HSSD homes) | `src/robots/humanclaw` | HumanClawBench's paper metrics (FindSR, NavSR@20cm/@1m, InteractSR; `success` = NavSR@20cm) | video, units, VDM, `--privileged`; units in its own vocabulary (WALK, TURN_*, SIDE_*, STEP_BACK, CLIMB_UP, WALK_DOWN, SIT, STOP; `--units=both` with `look`), GUMI keys, video ego (exo on the server), `code.api` recorded; `--humanclaw-mode paper` (default) plans with `humanclaw-psv/<base>` (HumanCLAW's prompt v4 + verifier v3 verbatim, no SYSTEM.md/VDM/memory), `pi` with our SYSTEM.md, `act`'s `target_visible`, and optional `--units-verify` / VDM. Later: `--api low` (segment, point, navmesh `navigate_to`). |
| BEHAVIOR-1K / R1Pro | `src/robots/behavior` | the BDDL activity's `success`, latched (`q_score` = partial credit) | memory, explore, video, units, VDM, code, `--privileged`; units on `env.move_hand_delta` |
| Franka (real) | `src/robots/franka` | operator verdict (`--operator`) | memory, explore, video, units, VDM, code, operator; explore resets through the operator |
| Dual Franka (real) | `src/robots/dual_franka` | operator verdict (required) | memory, explore, video, units, VDM, code, operator, XPolicyLab; explore resets through the operator; XPolicyLab (`franka`, ee; fine-tuning required) |
| Piper / dual Piper (real) | `src/robots/piper` | operator verdict (required) | memory, explore, video, units, VDM, code, operator; explore resets through the operator; XPolicyLab on the dual rig only (`piper/dual.ts`: `piper`, ee; fine-tuning required) |
| UR5e (real) | `src/robots/ur5e` | operator verdict (required) | memory, explore, video, units, VDM, code, operator; explore resets through the operator; bound to one arm (`--arm-id`) |

Every robot but the Frankas (whose `segment` does this with `--robot-sam3` / `--robot-unidepth`) takes
`--detections` (SAM3 masks with ids on the env server: `detect`, `select_detection`,
`reject_detection`, through its `--sam3`) and `--unidepth <url>` (`enhance_depth`: UniDepth depth
fused with the sensor's, or the only depth the Piper / UR5e webcams have; UR5e's `back_project` then
reads it); `src/primitives/detections.ts`, the env servers' `env.detect` & co. The simulators render
metric sensor depth (ManiSkill and RoboLab included); ManiSkill hands UniDepth the intrinsics of its
oriented, letterboxed views (`view_intrinsics`).
Metaworld and Genesis take the grasp flags too (`--contact-graspnet` & co): `plan_grasp`, `plan_place`,
`check_attached` over their env server's planner, and `execute_grasp` / `execute_place`, which run a
planned id's claimed path as bounded `move_delta` legs on the env server (`services/.../utils/grasp_chain.py`; a candidate
more than 20 deg from straight down, or turned more than 20 deg off the hand, is refused: the
grippers cannot turn). `plan_place` places upright by default, keeping only AnyPlace's turn about
the vertical (unlike upstream AnyPlace); `keep_tilt: true` keeps its full rotation for tilted
insertions, which only LIBERO's full-orientation `execute_place` can run (services/PROTOCOL.md).
ManiSkill (the Panda and the xArm6) and Genesis take `--ik <url>` too: `preview_reach` over the env
server's `env.preview_reach` (the ik service gained an `xarm6` model); Genesis's `move_delta` then
refuses an unreachable target before it moves.
`--point` adds Molmo's `point` on the same robots (BEHAVIOR has it by default) over `--molmo`: one
camera through `molmo.ground`, several at once through MolmoPoint's `molmo.ground_set`, each point
with its camera and, where the robot has depth, its world point (`src/primitives/pointing.ts`).

ManiSkill's `--robot` picks the arm (ManiSkill 3.0.1 agents the stock table scene places, and the
robots the other scenes of OpenETA's ManiSkill table are built for), all translation-only with the same
MV_* vectors, 2 cm step and servo (each measured at 19.5-20.3 mm per unit along its axis). The RLinf
rigs (BlockPAP-v1 / BlockStack-v1) run their own Panda; a task built for another robot runs only on it.
Results record a non-Panda arm as `maniskill_robot`, and `maniskill/eval.sh` keeps each arm in its own
out dir.

| `--robot` | ManiSkill uid | Env ids | Gripper | Wrist view |
| --- | --- | --- | --- | --- |
| `panda` (default) | `panda_wristcam` | the rigs, the 12 stock tabletop ids and FMBAssembly1Easy | mimic, +1 open / -1 close | Show-Harness's centred D415, turned 270 deg |
| `xarm6_robotiq` | `xarm6_robotiq` | PickCube, StackCube, PullCube, LiftPegUpright, PlaceSphere, StackPyramid, PlugCharger | Robotiq 2F-85 in delta mode, +1 close / -1 open | the wristcam variant's camera on `camera_link`, turned 90 deg |
| `widowxai` | `widowxai` (+ `pd_ee_delta_pos` on its six arm joints) | PickCube, PickCubeWidowXAI | carriages, +1 open / -1 close; 10-step hold | none: the agentview alone |
| `panda_stick` | `panda_stick` | PushT, DrawTriangle, DrawSVG | none: a stick (no GRASP / RELEASE units) | none |
| `panda_pair` | `("panda", "panda")` | TwoRobotPickCube, TwoRobotStackCube | mimic per arm; `move_delta` / `act` take `arm` (left / right), world frame | none |
| `widowx250s` | the bridge scenes' own WidowX 250 S (pose controller, zero rotation) | PutCarrotOnPlateInScene, PutEggplantInBasketScene, StackGreenCubeOnYellowCubeBakedTexInScene, PutSpoonOnTableClothInScene | mimic, +1 open / -1 close; 4-step hold | none; the agentview is the scene's `3rd_view_camera`, world frame |

The other ids are refused per arm: the xArm6 cannot reach PushCube's and PokeCube's goals,
PegInsertionSide resets a Panda joint vector, PickSingleYCB has no xArm6 layout, PullCubeTool's
"cube within 0.6 m of the base" already holds at reset on ~8 % of seeds with the xArm6's nearer base, and with its
gripper held pointing down the WidowX AI reaches only ~0.37 m from its base (PickCube's own layout).
Not offered: `so100` and `koch-v1.1` (joint control only and no TCP link), `fetch` (mobile base),
`ur_10e` (joint control only; the table scene has no placement for it), and the
`*_wristcam` uids of the xArm6 and WidowX AI (the table scene does not place the former, its arm
spawns inside the table; PickCube gives the latter the Panda's layout, out of its reach).
Of OpenETA's ManiSkill table (sim/envs/maniskill at 7d4a0a1), not offered: AssemblingKits,
PickClutterYCB, TurnFaucet and OpenCabinetDrawer / OpenCabinetDoor (their assets are on
storage1.ucsd.edu, unreachable, and PartNet-Mobility's mirror is gated; PickClutterYCB also never
reports success), TableTopFreeDraw (no success condition), RollBall (its robot faces -y with the goal
out of the shared camera's view), the SO100 scenes (joint control only, no TCP link), and the scene,
locomotion, humanoid and dexterous-hand envs.

The module list (before the first `;`) is what each robot mounts: memory, explore, video, units
(so GUMI and the fine-tuned provider), VDM (`--vdm`), code (`--code` over the primitive registry,
`code.api`), `--privileged` (simulation only), flywheel (`--collect-flywheel-data`), operator,
Flash (`--model flash/replay`) and XPolicyLab (`--xpolicy`). `test/readme-modules.test.ts` checks it
against the flags each robot registers. ManiSkill, RoboLab, RoboDojo and the Piper publish no memory
corpus: they default to the local one exploration writes. BEHAVIOR's units run as small end-effector
steps (`env.move_hand_delta`) of its mobile two-arm robot.

Shared modules:

- `src/capabilities/memory/`: memory (`MEMORY.md`, global/suite/task layers,
  validation, merge, index), synced from `RLinf/RPent-memory` into `$PI_EMBODIED_MEMORY`
  (default `~/.pi/embodied/memory`); `/memory sync|validate|index|merge`.
  Its guard denies file tools by default: only the memory, the cell's files in the output dir,
  and (real robots) the robot's own step artifacts are reachable.
- `src/capabilities/explore.ts`: exploration mode (`--explore`, `/explore`): reset budget, finish
  guard, attempt archive, cross-session handoff; each robot supplies its exploration prompt
  (LIBERO also its DISTIL pass).
- `src/capabilities/operator.ts`: human-in-the-loop (`--operator`): verdict and scene-reset requests as
  `ctx.ui.select` dialogs (TUI or RPC client), plus `/success /failure /abort /done /continue /operator`.
- `src/observation/video.ts`: episode video (ffmpeg). `src/capabilities/flywheel.ts`: Flywheel data (`--collect-flywheel-data`,
  default root `~/.pi/embodied/datacollection`): every env step with what the robot's VLA reads and
  emits, on LIBERO, RoboCasa and RoboTwin (the robots whose VLA runs agent-side), and every control
  step of a motion with the env action it applied on Metaworld, Genesis, Robosuite and ManiSkill
  (their env servers return the steps; one dataset per Robosuite arm layout and ManiSkill `--robot`,
  its `--space`). The UR5e's session data has export rules but no converter. `/flywheel-export
  [selection]` writes a LeRobot v3.0 dataset with the shared feature names (`--flywheel-python`,
  default `--python`; LeRobot needs its own venv, see services/README.md, which also covers GUMI runs).
- `src/modes/units/`: Show-Harness action units. `--units=true` hides the robot's tools: the model drives
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
  `--workspace-xy`, `--z-floor`). A robot that is not an arm declares its own `vocabulary` instead
  (units with an optional enum or clamped number parameter, a terminal unit; src/modes/units/custom.ts): the
  arm plugins are off, plan / mem_text / the verifier / --stateless / GUMI stay. Experimental plugins (Show-Harness `coords`, `mcq`,
  `action_ablation`; never in `auto`): `coords` states the directions in base-frame axes, `mcq` makes
  `act` answer with an option letter, `action_ablation` with `--units-ablation bare|letters|letters_blind`
  runs the paper's action-representation ablation (src/modes/units/experimental.ts).
- `src/modes/code/`: code mode (CaP-X's run_code). `--code=true` hides the robot's tools: the model
  writes Python programs that `run_code` executes on the env server against the robot's primitive
  manifest (`src/primitives/manifests/<robot>.json`; `--code-api=high|low|low-noexamples|raw`:
  CaP-X's semantic functions (S2), the perception / IK / motion parts (S3), the same without
  examples (S4), the raw step; default: the robot's highest tier, e.g. low on Metaworld;
  `--privileged` adds ground truth: the privileged tier (S1), or `low+privileged`),
  in a spawned subprocess with no env object whose calls the server resolves through the registry
  (JSON over the pipe, never pickle; the child starts without the server's secret-looking
  environment variables, in its own process group, with no new processes or threads allowed;
  it can still open sockets, so run the server in a container to keep programs off the network),
  killed at `--code-timeout` (a stop is issued, also inside a running primitive), refused past
  `--code-max-calls` calls or `--code-max-move` metres; `--code-helpers` injects CaP-X's numpy
  helpers. `--code=both` adds
  `run_code` to the robot's tools. Mutually exclusive with `--units`; `--stateless` applies. Real
  robots need `--code-real` and `--operator`, and every program is confirmed by the operator.
  `--code-oracle <file>` runs a human reference program once instead of the model (CaP-X's 17 oracles,
  ported in `src/robots/robosuite/oracle/` and `src/robots/libero/oracle/`; the result records `code_oracle`).
  Every simulator serves it (LIBERO, Robosuite, MetaWorld, ManiSkill, Genesis, BEHAVIOR, RoboCasa,
  RoboLab, RoboTwin, RoboDojo), and the real robots (Franka, dual Franka, Piper, UR5e) when their
  server runs with `--code`; eval.sh records `--code`, `--code-api` and `--code-oracle`
  (services/PROTOCOL.md, code mode).
- `src/primitives/xpolicy.ts`: XPolicyLab policies (github.com/XPolicyLab/XPolicyLab, pinned `d6332bf`).
  pi-embodied is only the environment client: start the policy server the XPolicyLab way
  (`policy/<name>/setup_eval_policy_server.sh`) and pass `--xpolicy ws://host:port`
  (`--xpolicy-action joint|ee`, default joint). The session start connects a new trial (and fails
  closed) and adds `xpolicy_act {chunks}`: prepare_case + reset before the first chunk and after a
  scene reset, then per chunk update_obs + get_action and every action, with update_obs between two
  actions (XPolicyLab's deploy loop), until the episode ends. The protocol (handshake, request ids
  reused across a reconnect, `ServerRestartedError`, msgpack-numpy) stays in a Python bridge
  (services `components/xpolicy_bridge.py`, holding XPolicyLab's own `WsModelClient`), started with
  `--xpolicy-python` and `--xpolicylab <checkout>` or attached with `--xpolicy-bridge`;
  `--xpolicy-encode-images` sends JPEGs, `--xpolicy-timeout` / `--xpolicy-connect-timeout` are
  XPolicyLab's request and cold-start budgets. Dimensions come from the services' env_cfg
  (`components/xpolicy_env_cfg`: XPolicyLab's own robot table; `aloha_agilex`, `piper`, `franka`
  and `arx_x5` are all two-armed, keys prefixed `left_` / `right_`). A robot opts in with `xpolicy`
  in its spec (env_cfg type, action types, how it builds the observation and executes one action).
  RoboTwin runs joint (qpos14) and ee (ee16) actions natively; the dual Piper and dual Franka rigs
  run ee targets as their bounded relative moves (Piper: yaw only) and have no joint command.
  XPolicyLab publishes no weights for them (nor real-robot evaluation): a policy has to be
  fine-tuned on the rig's own data first. The single-arm Piper and Franka have no XPolicyLab
  env_cfg and no `xpolicy_act`. RLDX (RoboCasa) and LingBot (RoboTwin) keep their own clients for
  now; new VLAs go through XPolicyLab.
- `src/infra/model-services.ts`: model-service auto-start (RPent's `robots/runtime.py`), opt-in with
  `--serve-models vla,sam3,molmo` (or `all`; RoboCasa's VLA is `rldx`) on the simulators: each named
  server starts on its endpoint flag's loopback port (`--vla`, `--sam3`, `--molmo`, `--rldx`), all
  at once, and the robot starts only once every one answers `healthz`; one that exits or misses
  `--serve-timeout` stops them all and the start fails closed. A port that already serves is refused
  (attach to it without `--serve-models`). `--serve-python molmo=<venv>/bin/python` (Molmo's own
  venv), `--serve-cuda-device`, and `--serve-lock <file>` (default `$PI_EMBODIED_GPU_LOCK`; on a
  shared GPU box its gpu1.lock) holds flock(1) on the file while the models load and run, after
  `--serve-min-free <MiB>` found that much free on the GPU without the lock (checked again under it;
  not with eval-parallel.sh's `LOCK` on the same file, which that run already holds). Logs go to
  `--serve-log-dir`. `serve.sh` stays the way to share one server across an eval batch.
- `src/capabilities/dashboard/`: live web dashboard (`--dashboard`) for any robot. Operator tools: withdraw a
  message still in pi's queue (`POST /message/withdraw`), call one robot tool by hand while the agent
  is idle or taken over (`GET /primitives`, `POST /primitive`; the robot's gates apply, one robot call
  at a time, recorded in the session), a real 1-token model check (`POST /llm-check`), and downloads
  (`GET /download/session?format=jsonl|html`, pi's `/export`; `GET /downloads`,
  `GET /download/video/<session>/<file>.mp4`). Every request's Host header must name this machine
  (DNS-rebinding defense; `--dashboard-allowed-hosts` adds names); off loopback (`--dashboard-host`)
  every request also needs the token in the printed URL (`--dashboard-token`, else a random one).
  `/gumi-replay <run>` replays a GUMI recording as operator steps (asks first).
  Its GUMI teleop panel (src/capabilities/gumi/) drives action units by key, records demonstrations
  (`--gumi-record`), and with `--gumi-operator <provider/model>` a VLM operator drives the same teleop
  path (Run / Step once / Pause), recorded as `gpt-operator` (src/capabilities/gumi/operator.ts).
- Operator CLIs in `services/` (hardware, unverified on a rig): `robots.dual_franka.manual_call`
  (one facade call, dry-run by default), `robots.franka.capture` (`z-floor`, `pose`),
  `robots.piper.capture_z_floor`, `robots/piper/ros_launch.sh` (`can`, `arms`, `cameras`), and
  `flywheel.gumi_tools` (`rebuild-video`, `step-timing` of GUMI runs).
- `src/observation/viser.ts`: live 3D view (`--viser`, CaP-X's Viser scene): point clouds of the RGB-D cameras,
  camera frustums, the EEF and each `plan_grasp` / `plan_place` result's candidates, served by
  `services/.../components/viser_view.py` (the `viser` extra; `--viser-python`) at
  `http://<host>:8080/` (`--viser-port`), linked from the dashboard's header. LIBERO and Franka.
- `src/observation/vdm.ts`: `--vdm-video` describes each change from the step's episode frames
  (`--vdm-video-frames` sampled) instead of the before/after pair (CaP-X's video differencing).
- `src/robots/libero/flash.ts`: Flash replay without an LLM (`--model flash/replay`).

Every robot result carries `robot`, `claimed`, `summary`, `turns`, `planner_budget_exhausted`,
`planner_error` and `env_error` (also set when the env server exits or a service stops answering
mid-episode); the eval scripts count an episode only when the environment produced a result and
the planner did not fail, and rerun the others. LIBERO and RoboTwin episodes get
`--time-limit ${TIME_LIMIT:-1800}` s (the episode ends as a failure) and a `timeout` backstop
900 s later (the episode is invalid and rerun).

`src/scripts/eval-parallel.sh` runs a robot's eval.sh matrix (LIBERO, ManiSkill, Metaworld, Robosuite, Genesis,
BEHAVIOR, RoboLab, RoboTwin, RoboCasa) on N workers (`-j N --gpus 1`: several workers
may share a GPU; each gets CUDA_VISIBLE_DEVICES and the EGL device on the same PCI bus), as an A/B
over `--variant NAME=ARGS` (same cells, one subdirectory each), and reports success rate, Pass@k and
invalid cells per variant; `--min-success N` fails a regression run, `--max-api-concurrency M` caps
model calls across workers. Every cell is its own eval.sh call, so validity and reruns are eval.sh's.

Developer guides: [adding a robot](docs/adding-a-robot.md) and [adding a primitive](docs/adding-a-primitive.md).
`test/gpu-e2e.test.ts` is the GPU end-to-end suite (real simulators and model servers, no model API;
skipped unless `PI_EMBODIED_E2E` names a robot and a GPU answers); `test/gpu-e2e.sh` runs it robot by
robot on a GPU box (docs/adding-a-robot.md, "Testing on a GPU").

## LIBERO

Needs the repository's Python services (`services/`, package `pi_embodied_services`) installed
with the `[libero]` extra (`services/setup.sh libero`, or see services/README.md); the extension
speaks their HTTP RPC directly.
Robots spawn their env servers from `--services` (env `PI_EMBODIED_SERVICES`, default the repo's
`services/`) with `PYTHONPATH` set to it; `serve.sh` and `eval.sh` default to the same directory.

```bash
export PI_EMBODIED_PYTHON=services/.venv-libero/bin/python
PI05_CHECKPOINT_PATH=... SAM3_CHECKPOINT_PATH=... packages/embodied/src/robots/libero/serve.sh

pi -e packages/embodied/src/robots/libero --suite libero_10_task --task 2 --seed 0             # interactive
packages/embodied/src/robots/libero/eval.sh runs/l10 libero_10_task 0-9 0-2 --model <provider/model>
pi -p -e packages/embodied/src/robots/libero --explore --suite libero_10_task --task 2 --seed 0 "/explore"
```

Each session ends with a `robot_result` entry (`terminated` is LIBERO's success flag,
`claimed` is the agent's own status). `/robot-task <suite> <task> <seed>` starts a new episode
in a new session. Boolean flags take the next word as their value; write them as
`--flag=true` before a prompt. The system prompt is RPent's LIBERO prompt and guides
(`--libero-prompt rpent`, default; `compact` is the short one; see `src/robots/libero/PROMPT_PORT.md`).

## Other robots

- RoboCasa: the services' `[robocasa]` extra (Python 3.10), kitchen assets, the RLDX-1-FT-RC365
  checkpoint; `src/robots/robocasa/serve.sh`, `src/robots/robocasa/eval.sh` runs Target50.
- RoboTwin: the services' `[robotwin]` extra (Python 3.11), RoboTwin assets, the LingBot-VLA
  RoboTwin checkpoint; `src/robots/robotwin/serve.sh`, `src/robots/robotwin/eval.sh`.
- RoboDojo (robodojo-benchmark/RoboDojo @726e9aa, eval only): `services/setup.sh robodojo` (Isaac Sim
  6.1 / Isaac Lab 3.0 with RoboDojo patched by `robodojo-isaac61.patch`, its cuRobo v2 fork, 41 GB of
  assets); `--task <name> --seed <layout id>` on RoboDojo's two ARX X5 arms. 54 runnable tasks: 42 base
  tasks in five capability dimensions (Generalization 12, Memory 6, Precision 8, Long-Horizon 8, Open 8)
  plus 12 `_random` variants of the Generalization tasks; the 55th yml, `config/_task.yml`, is the shared
  per-task settings file (data source, scene/robot config, render interval, self-collision, eval count).
  The seed is an eval layout (pre-generated, 25 or 50 per task). Per-arm `move_to` / `move_delta` /
  `rotate_delta` / `set_gripper`, `go_home` (most tasks need both arms back home), `locate` (depth
  back-projection of head pixels), units per arm; `src/robots/robodojo/eval.sh` records
  `--eval-seed` and averages RoboDojo's score. One env per process: RoboDojo's heterogeneous parallel
  simulation (several envs and tasks in one Kit process, up to 10 per GPU in its config) is not used,
  so every episode pays its own Kit start (about 10 s warm, minutes cold) and cuRobo warmup (20-50 s),
  and holds its own Kit, renderer and cuRobo memory (8.6 GB of GPU memory measured on stack_bowls
  with the three 640x480 cameras and depth). Parallelism is episodes as processes (`eval-parallel.sh -j`, which takes
  `LOCK` for RoboDojo like RoboLab): budget that memory per worker. On Isaac Sim 6.1 48 of the 54 tasks run;
  the cloth, liquid and charger tasks do not (the per-task table and the reasons: services/pi_embodied_services/robots/robodojo/README.md).
- Metaworld: the services' `[metaworld]` extra (Python 3.11, metaworld 3.1.1, no assets); the 50 MT50
  Sawyer tasks (`--task reach-v3 --seed 0`), a world-frame `move_delta` plus `gripper`, depth tools
  (`back_project`, `segment` with a SAM3 server), units mode and `--privileged`; `src/robots/metaworld/eval.sh`.
- Robosuite: the services' `[robosuite]` extra (Python 3.11, robosuite 1.5 in its own venv: LIBERO and
  RoboCasa pin 1.4 forks); CaP-X's seven tasks (`--task Lift --seed 0`, two-arm tasks take `arm`),
  closed-loop `move_to` / `move_delta` under `--max-move`, `gripper`, depth tools, planned grasps
  (`plan_grasp` / `plan_place` / `check_attached` with `--contact-graspnet` & co), `preview_reach` with `--ik`,
  units mode and `--privileged`; `src/robots/robosuite/eval.sh`.
- Genesis: the services' `[genesis]` extra (Python 3.11, Genesis 1.4, a GPU for rendering, no assets);
  OpenETA's Franka `cube_pick` (`--task cube_pick --seed 0`), a base-frame `move_delta` plus `gripper`,
  depth tools, units mode and `--privileged`; `src/robots/genesis/eval.sh`.
- BEHAVIOR-1K: the venv `services/setup.sh behavior` builds (Isaac Sim, OmniGibson and BDDL from a
  BEHAVIOR-1K checkout, the challenge dataset; see services/pi_embodied_services/robots/behavior/README.md);
  the 50 2025-challenge activities on the R1Pro (`--task turning_on_radio --seed <instance> --gpu-id N`),
  OmniGibson's semantic primitives as tools (`navigate_to_pose`, `move_hand`, `grasp_object`, the
  grippers), `segment` / `point` / `back_project` on three cameras, `--grasping-mode` and
  `--privileged`; `src/robots/behavior/eval.sh`.
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
| Human as the model (OpenETA's manual VLM console) | `--model human/operator` (or any VLM flag set to it, e.g. `--attach-vlm-model human/operator`) | a provider (`src/planner/human.ts`) asking through `ctx.ui` select/input (TUI dialogs, RPC `extension_ui_request`) and the dashboard's composer | any |
| Web search / page fetch | `pi install npm:pi-web-search npm:@zeldrisho/pi-web-fetch`, then `--web-tools` | pi packages; `src/capabilities/web.ts` keeps their tools active next to the robot's | any |
| Object memory | `--object-memory` [`--object-memory-dir <dir>`] | tools + `object_record` session entries (`src/capabilities/objects.ts`) | any |
| Multi-waypoint route | `--waypoints` | `follow_waypoints` (`src/primitives/waypoints.ts`) | LIBERO, Franka |
| Wrist-view alignment | `--align-wrist` | `align_wrist` (`src/primitives/wrist.ts`) | LIBERO, Franka |
| Grasp advisor | `--grasp-advisor` [`--grasp-advisor-model`], with a plan_grasp backend | `suggest_grasp` (`src/primitives/advisor.ts`), a side VLM call | LIBERO |
| Task skills | `/skill:embodied-pick` ... | pi skills (`skills/`) | any |
| One server per arm | always, real robots | `utils/hardware_lock.py` flock per arm (`--lock-id`, `--lock-dir`, `$PI_EMBODIED_LOCK_DIR`) | Franka (RLinf, Polymetis), dual Franka, Piper, UR5e |

The flags that change what the agent can do are recorded in the result row as `extras`.


## Deliberately dropped

What the four source repositories have that pi-embodied does not port, and why.

RPent (eecf206):

- Its planner loop, CLI / TUI and session layer (`rpent/planner`, `rpent/cli`, `rpent/session`): the
  agent loop, models, sessions and modes are pi's; a robot is a `defineRobot` extension.
- The dashboard's own session and state store: the dashboard follows pi's events and rebuilds its
  state from the session branch, so resume and fork show the right history.
- The pickle-framed socket RPC: only JSON `POST /call` is served (unpickling a request frame is
  remote code execution for anyone who can reach the port).

CaP-X (53e9966):

- GRPO / VeRL training (`third_party/verl`, `verl_agent_reward`): training stays outside the
  repository; pi-embodied exports sessions in a format VeRL and RLinf read, with the environment's
  success as the reward, never "the program ran without error = 0.1".
- Its LLM client and model proxies (`capx/llm/client.py`, `serving/{openrouter,vllm}_server.py`):
  replaced by pi-ai providers and pi's model registry (`--model <provider>/<id>`, models.json), which
  serve the planner and the side VLMs.
- FastAPI / msgpack model serving (`serving/launch_*_server.py`): replaced by the services' JSON
  `POST /call` with `healthz` / `stop` (services/PROTOCOL.md); a CaP-X server's endpoint maps to one
  RPC method of the matching `components/*_server.py`.
- In-process `exec` with the env in the program's globals: `run_code` runs in a sandboxed
  subprocess that holds no env object and reaches the robot only through the primitive registry.
- Ground truth in ordinary observations (`cube_poses` / `nut_poses`, TwoArmHandover's instance
  segmentation): object poses leave the server only through `ground_truth_poses` under
  `--privileged`, which the result marks, so a score without it is a clean non-privileged score.
- `pick_up_radio_reward` as BEHAVIOR success (BDDL's `success` decides; CaP-X's judgement is a
  reference field) and `move_hand(ignore_all_obstacles=True)` (the R1Pro planner keeps its obstacles).

Show-Harness (137d571):

- Its runners and VLM client (`core/runners`, `core/vlm`): the planner is the model; MvTokenRunner
  is the `finetuned/<adapter>` provider, and the verifier and video_ref calls go through pi's model
  registry.

OpenETA (7d4a0a1):

- Its Python agent runtime (sessions, resume, compaction, skills, TUI), XML decision parsing, the
  MCP layer and the Codex plugin packaging: pi core, pi's native tool calls and pi extensions
  replace them.
- The environment shells copied from RLinf (CALVIN, RoboVerse, Isaac Lab, Polaris, Habitat,
  EmbodiChain, FrankaSim): empty shells; connect them from RLinf directly when needed. RoboTwin and
  RoboLab are native here.
- D4RL (locomotion, not manipulation) and the tactile branch (Isaac Sim 5.1 + TacEx, a separate
  research direction).
- Its tool-contract maturity reviews and grasp-policy / calibration promotion workflows: OpenETA's
  own project governance.
- Its UR5e safety layer (a 0.6 m / 180 deg per-move check, collision checking ignored, rpy passed as
  a rotation vector): replaced by the Franka / Piper standard (operator gate, workspace box, Z floor,
  refuse-not-clamp limits, `stopL` on stop). Its hand-eye calibration writing back unreviewed:
  the solver writes an arm-bound `.new.yaml` that a human applies in a separate step.
