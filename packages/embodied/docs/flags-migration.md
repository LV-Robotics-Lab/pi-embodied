# Flags moved to the deployment config

Where a service listens, which Python runs an env server and where outputs go are no longer flags
(no aliases: an old flag fails with "unknown flag"). They live in the deployment config,
`~/.pi/agent/embodied.json` or `<cwd>/.pi/embodied.json`, under `deployments.<name>`, picked with
`--deployment <name>` (README, "Where things run"; `src/infra/config.ts`). A service that changes
results keeps a flag, which now says what to use rather than where it listens.

| Removed flag | Now |
|---|---|
| `--sam3 URL`, `--robot-sam3 URL` | `services.sam3` (sims: always used; real arms: `--segment`) |
| `--molmo URL` | `services.molmo` |
| `--molmo off` (Flash) | `--flash-reanchor=false` |
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

Precedence: built-in ports and paths < environment (`PI_EMBODIED_SERVICES`, `PI_EMBODIED_PYTHON`, ...)
< deployment file, except `PI_EMBODIED_CUDA_DEVICE` and `PI_EMBODIED_DIRS_<KIND>`, which win over the
file (per-worker and per-episode values). A venv's own variable (`ROBOCASA_PYTHON`, `PI_EMBODIED_FLYWHEEL_PYTHON`, ...) sits
above `python.default`: a setting for that venv beats the generic one. `/embodied-config` prints what is in effect; a result
records `deployment` and `deployment_sha`.

A launcher that must run both older and newer checkouts can test for
`packages/embodied/src/infra/config.ts` in the checkout and pass the old flags only when it is missing.
