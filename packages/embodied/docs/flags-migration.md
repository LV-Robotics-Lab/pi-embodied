# Flags renamed or moved to the deployment config

Where a service listens, which Python runs an env server and where outputs go are no longer flags
(no aliases: an old flag fails with "unknown flag"). They live in the deployment config,
`~/.pi/agent/embodied.json` or `<cwd>/.pi/embodied.json`, under `deployments.<name>`, picked with
`--deployment <name>` (README, "Where things run"; `src/infra/config.ts`). A service that changes
results keeps a flag, which now says what to use rather than where it listens.

| Removed flag | Now |
|---|---|
| `--sam3 URL`, `--robot-sam3 URL` | `services.sam3` (sims: always used; real arms: `--segment`) |
| `--molmo URL` | `services.molmo` |
| `--molmo off` (Flash) | `--flash-reanchor off` (a value, not `=false`: pi reads a boolean extension flag as true whatever follows it) |
| `--vla URL` (LIBERO) | `services.vla` |
| `--robot-vla URL` (Franka, dual Franka) | `--vla` + `services.vla` |
| `--openvla URL`, `--openvla-oft URL`, `--gr00t URL` | `--vla-adapter openvla[,openvla-oft,gr00t]` + `services.openvla` / `.openvla_oft` / `.gr00t` |
| `--ik URL` | `--ik` + `services.ik` |
| `--contact-graspnet URL`, `--graspgenx URL`, `--anygrasp URL`, `--graspnet1b URL` | `--grasp contact_graspnet[,graspgenx,anygrasp,graspnet1b]` + `services.<backend>` |
| `--anyplace URL` | `--place anyplace` + `services.anyplace` |
| `--unidepth URL`, `--robot-unidepth URL` | `--depth unidepth` + `services.unidepth` |
| `--rldx URL` | `services.rldx` |
| `--lingbot URL` | `services.lingbot` |
| `--ft-endpoint URL` | `services.finetuned` |
| `--services DIR` | `services_dir` (or `PI_EMBODIED_SERVICES`) |
| `--python P` | `python.<robot>`, else `python.default` (or `PI_EMBODIED_PYTHON`) |
| `--robocasa-python P` | `python.robocasa` (or `ROBOCASA_PYTHON`) |
| `--flywheel-python P` | `python.flywheel` (or `PI_EMBODIED_FLYWHEEL_PYTHON`) |
| `--viser-python P` | `python.viser` |
| `--xpolicy-python P` | `python.xpolicy` (or `PI_EMBODIED_XPOLICY_PYTHON`) |
| `--serve-python name=P,...` | `python.<service name>` |
| `--serve-log-dir DIR`, `--log-dir DIR` | `dirs.logs` (or `PI_EMBODIED_DIRS_LOGS`) |
| `--out DIR` (real arms) | `dirs.artifacts` |
| `--memory-dir DIR` | `dirs.memory` (ignored with `--memory-profile hf`) |
| `--output-dir DIR` | `dirs.memory_out` |
| `--video-dir DIR` | `dirs.video` |
| `--flywheel-root DIR` | `dirs.flywheel` |
| `--flash-plans DIR` | `dirs.flash_plans` |
| `--api-slots DIR` | `dirs.api_slots` (eval-parallel.sh sets `PI_EMBODIED_DIRS_API_SLOTS`) |
| `--ffmpeg BIN` | `ffmpeg` |
| `--cuda-device N`, `--gpu-id N` | `cuda_device` (or `PI_EMBODIED_CUDA_DEVICE`, which eval-parallel.sh sets per worker) |
| `--robot-ros-setup FILES` (Piper) | `ros_setup` |

## Renamed flags (one concept, one name)

| Old | New |
|---|---|
| `--env URL` (simulators), `--robot-env URL` (real arms) | `--env-url URL` |
| `--task-name` (RoboCasa, RoboTwin), `--env-id` (ManiSkill) | `--task` |
| `--robot` (ManiSkill's arm), `--arm-id` (UR5e) | `--arm` |
| `--robot-backend` (Franka) | `--backend` |
| `--eval-seed` (RoboDojo's layout set) | `--layout-set` |
| `--robot-cameras` (UR5e) | `--cameras` |
| `--units-vlm-model`, `--vdm-model`, `--attach-vlm-model` | `--aux-model` (one model for VDM, the units verifier and video_ref, check_attached); a role that needs another names it in the deployment's `aux.vdm` / `aux.verify` / `aux.attach`. A result records each role's model as `aux_models`. |

A result's `robot_task` carries the new names (`task`, not `task-name` / `env-id`). The eval scripts
refuse every old flag above before any cell runs (`src/scripts/old-flags.sh`).

The OpenETA extras keep their switches (`--waypoints`, `--align-wrist`, `--grasp-advisor`, `--object-memory`, `--web-tools`;
the result records them as `extras`) and `--grasp-advisor-model` stays suggest_grasp's own override of `--aux-model`.
`--object-memory-dir` changes what the robot remembers and is a parameter; `--asset-references-dir` is a location and,
like the removed directory flags, is left out of `params`.

Precedence: built-in ports and paths < environment (`PI_EMBODIED_SERVICES`, `PI_EMBODIED_PYTHON`, ...)
< deployment file, except `PI_EMBODIED_CUDA_DEVICE` and `PI_EMBODIED_DIRS_<KIND>`, which win over the
file (per-worker and per-episode values). A venv's own variable (`ROBOCASA_PYTHON`, `PI_EMBODIED_FLYWHEEL_PYTHON`, ...) sits
above `python.default`: a setting for that venv beats the generic one. `/embodied-config` prints what is in effect; a result
records `deployment` and `deployment_sha`.

A launcher that must run both older and newer checkouts can test for
`packages/embodied/src/infra/config.ts` in the checkout and pass the old flags only when it is missing.

## `--tier` and `--preset`: one flag for a whole setting

`--tier S1..M4` (CaP-X's tiers) and `--preset showharness|humanclaw|rpent|openeta|xpolicylab|capx-<tier>`
(the source repositories' native settings) are not a new mechanism and add no aliases: each stands for
values of the flags above (`src/infra/tiers.ts`, the README's two tables), which read as those values
unless given themselves. A flag given with another value is a start-time error naming both
(`--tier S3 sets --code-api=low, but --code-api=high was given`); a choice the robot cannot serve is
refused at start too. The result records `tier` or `preset` and `axes`, and `params` the expanded
flags, so `params-match.mjs` keeps a `--tier S3` run apart from the same flags spelled out. The eval
scripts expand a choice through `src/scripts/tier-flags.mjs` (`eval-options.sh`) and refuse a
contradicting argument before any cell runs.

| Setting | Flag |
|---|---|
| CaP-X S1 ... M4 | `--tier S1` ... `--tier M4` (or `--preset capx-S1`) |
| Show-Harness zero-shot | `--preset showharness` (= `--units=true`) |
| HumanCLAW paper mode | `--preset humanclaw` (= `--units=both --humanclaw-mode paper`) |
| RPent | `--preset rpent` (the robot's tools with memory; no mode flag) |
| OpenETA | `--preset openeta` (= `--vdm --anchor-image`) |
| XPolicyLab / RoboDojo | `--preset xpolicylab` (with `--xpolicy ws://host:port`) |
