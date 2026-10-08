---
name: embodied-quickstart
description: Install, preflight and run pi-embodied robots (simulators LIBERO, ManiSkill, Metaworld, Robosuite, Genesis, RoboCasa, RoboTwin, RoboLab, RoboDojo, BEHAVIOR-1K, HumanCLAW; real Franka, dual Franka, Piper, UR5e). Use when setting up a robot or benchmark, after /embodied-setup, or when a robot run fails to start.
---

# pi-embodied quickstart

Paths below are relative to the services checkout (`services/`) and the package (`packages/embodied/`
in the repository; the package root in an npm install). An npm install has no `services/`: clone
`https://github.com/LV-Robotics-Lab/pi-embodied.git` and use its `services/`. On a remote box run
every command over `ssh <host>`; pi and the experiment directory live on the box too.

## 1. Pick

`/embodied-setup` asks for the robot, the mode, where services run and the planner, and writes the
experiment directory's `.pi/settings.json`: the robot extension, the dashboard, and this package
without its onboarding extension. A robot extension replaces the coding tools, so start pi for the
robot only in that directory. Settings cannot hold flags; the task and mode go on the command line.

Modes: tools (default), units (`--units=true`; `--units=both` keeps the tools), code (`--code=true`,
CaP-X run_code; `--tier S1..S4|M1..M4` sets a whole CaP-X tier, `--preset showharness|rpent|openeta|
xpolicylab|humanclaw|capx-<tier>` another repository's setting), fine-tuned (`src/modes/finetuned`
extension, `--model finetuned/<adapter>`), flash (`--model flash/replay`: LIBERO, RoboCasa, RoboTwin,
ManiSkill, RoboLab, RoboDojo).

Fine-tuned mode needs two more steps after the install: `FT_ADAPTER=<adapter> services/setup.sh
finetuned` (default `qwen3_5_2b_sim`) fetches the released adapter and its base model with
`finetuned/download.py` (pinned, verified), and `services/pi_embodied_services/finetuned/serve.sh`
(`MODEL`, `LORA`, `VLLM_VENV` as setup.sh prints them) serves them on the vLLM endpoint the
deployment's `services.finetuned` names (default http://127.0.0.1:8010/v1). Start it before the
episodes and leave it running. LIBERO's adapters are the user's own (`--model finetuned/local
--ft-model <served adapter>`).

The user confirms the plan once in `/embodied-setup`; pi does not gate each tool call, so ask before
sudo, anything touching real hardware, or a download the plan does not list.

## 2. Install

```bash
services/setup.sh <target> [--weights] [--venv DIR] [--weights-dir DIR] [--no-assets] [--dry-run]
services/setup.sh --help        # targets and what each fetches
```

Run `--dry-run` first and show the user what it will do. Robot targets: libero, libero-pro,
libero-plus, maniskill, metaworld, robosuite, genesis, robocasa, robotwin, robolab, robodojo,
behavior, humanclaw, franka, franka-polymetis, dual-franka, piper, ur5e; others: finetuned,
llamafactory, flywheel (LeRobot export), graspnet1b. Most create `services/.venv-<target>` with the
matching extra; robolab, robodojo, behavior (Isaac Sim 6.1) and humanclaw (Habitat-Sim) run their
install scripts on a checkout under `$HOME` (or `<NAME>_ROOT`). Assets (`--no-assets` skips them) and
`--weights` go to `$PI_EMBODIED_WEIGHTS` (default `~/.cache/pi-embodied`). It writes
`<venv>/pi-embodied.env`; `source` it before pi, `serve.sh` and `eval.sh`. Mirrors:
`HF_ENDPOINT=https://hf-mirror.com`, `UV_INDEX_URL` / `PIP_INDEX_URL`. Gated data (SAM3, HSSD) needs
`HF_TOKEN` with the license accepted.

Where things run is the deployment config, not flags: `deployments.<name>` in `~/.pi/agent/embodied.json`
or `<cwd>/.pi/embodied.json` (`services.*` URLs, `python.<robot>`, `cuda_device`, `dirs.*`),
`--deployment NAME`, `/embodied-config` prints it; its `python.<robot>` beats the env file's
`PI_EMBODIED_PYTHON`. Removed flags (`--env`, `--sam3 URL`, `--python`, ...): docs/flags-migration.md.

Tools mode also needs the model servers: `packages/embodied/src/robots/<robot>/serve.sh` (LIBERO:
Pi0.5 + SAM3, and OpenVLA / OpenVLA-OFT / GR00T when their `*_PYTHON` is set; RoboCasa: RLDX-1, also
in units mode; RoboTwin: LingBot; dual Franka: Pi0.5 + SAM3), or `--serve-models vla,sam3,molmo` per
episode; other robots attach to the deployment's `services.*` (SAM3 for `segment`, Molmo for `--point`).

## 3. Preflight

```bash
source services/.venv-<target>/pi-embodied.env
node packages/embodied/src/infra/check.ts <robot> [--deployment NAME] [--units] [--model provider/id]
```

`<robot>` is the extension's directory name (`dual_franka`; libero-pro / libero-plus are `libero`).
Fix every FAIL before running. Inside a robot session the same check is `/robot-check`.

## 4. One episode

```bash
cd <experiment dir> && source <venv>/pi-embodied.env
pi <task flags> [--units=true] --dashboard=true
```

The first pi in the experiment directory asks whether to trust the project: its `.pi/settings.json`
loads the robot extension, so the user answers yes (`/trust` saves it); declined, pi starts without
the robot. `pi -p` cannot ask and runs untrusted (no robot, the coding tools, the global default
model) unless trust was saved or `-a` / `--approve` is given. The dashboard URL is shown at startup.
Boolean flags take the next word: write `--flag=true`. Without the experiment settings:
`pi -e packages/embodied/src/robots/<robot> ...`. Task flags: `--task` (and `--seed`) everywhere;
LIBERO `--suite`, `--libero-type standard|pro|plus`; HumanCLAW `--units=both --episode`; real arms
`--robot-config`, `--env-url`, UR5e `--arm <serial>`.

Optional, per launch (each registers nothing when off): `--web-tools` keeps web search and page
fetch for the robot, from pi packages installed once in the experiment directory
(`pi install -l npm:pi-web-search@1.6.0 npm:@zeldrisho/pi-web-fetch@0.9.2`; web_search uses the planner provider's
native search, so it needs a provider that has one); `--object-memory` (`--object-memory-dir` to keep
records per scene), and on LIBERO / Franka `--waypoints`, `--align-wrist`, `--grasp-advisor`.
`--replay-thinking=true` sends every earlier turn's thinking back (default false; it is most of a
Qwen / vLLM planner's input). `--model human/operator` lets a person answer as the planner (pi's
dialogs, or the dashboard). The task skills `/skill:embodied-pick`, `embodied-place`,
`embodied-push-pull`, `embodied-stack` and `embodied-object-memory` go in front of the prompt
(`pi ... "/skill:embodied-pick Solve the task."`): a robot's system prompt replaces pi's, so the model
does not see the skill list itself.

## 5. Evaluate

```bash
packages/embodied/src/robots/<robot>/eval.sh <out-dir> <cells...> --model <provider/model> --thinking low
```

Each robot's `eval.sh` header gives its cells (LIBERO: suite, tasks, seeds; most sims: tasks, seeds).
Run it from a directory without the experiment settings (it loads the robot with `-e`; loading it
twice fails on duplicate flags). Rerunning retries only invalid episodes; a run with other flags in
the same out dir is refused. `src/scripts/eval-parallel.sh [-j N --gpus LIST] [--durable] <robot>
<out-dir> <cells...>` runs the matrix on N workers (`--durable` resumes an interrupted one).

## Common failures

- `[robot] unavailable: ...` at start: the env server did not come up; run the preflight.
- No robot (no `/robot-task`, the coding tools are back) in the experiment directory: project trust
  was declined, or `pi -p` ran without `-a`; start pi there again and accept, or `/trust`.
- Import errors in the preflight: wrong venv (or the deployment's `python.<robot>` names another).
  Each robot family has its own venv; the extras conflict (Torch/Transformers pins).
- RTX 5090 / sm_120: install `torch==2.7.1 torchvision==0.22.1` from the cu128 index before the extra
  (Genesis: a cu128 torch>=2.8, graspnet1b: a cu128 torch, first; HumanCLAW: `HUMANCLAW_TORCH`); RoboTwin also needs
  cuRobo built against it (services/README.md).
- LIBERO renders black or crashes: `MUJOCO_GL=egl` needs an NVIDIA EGL driver; pick the GPU with
  `MUJOCO_EGL_DEVICE_ID`. MuJoCo must stay `3.3.0`.
- LIBERO-PRO assets "ready" but tasks missing files: rerun `setup.sh libero-pro`; its patched
  downloader verifies every file. `FileNotFoundError` under an old venv path: `~/.liberopro/config.yaml`
  (and `~/.libero/`) keep absolute paths; rewrite them after moving the venv, or set `LIBERO_CONFIG_PATH`.
- SAPIEN (ManiSkill, RoboTwin): needs `libvulkan1`; on sm_120 its OIDN denoiser logs "unsupported
  device type": replace SAPIEN's `oidn_library` with OIDN >= 2.3.
- `robotwin-download-assets` fails behind hf-mirror (tree API): download the asset zips directly.
- RoboLab / RoboDojo / BEHAVIOR: Isaac Sim takes about a minute to start (minutes cold); the
  `*_ROOT` checkout must be the patched one.
- A model server port already answers: `serve.sh` leaves it running; `serve.sh stop` stops only its own.
- Molmo and SAM3 on one GPU: start Molmo with `--offload-blocks 20`.
- Real robots: an operator at the e-stop (`--operator=true`; motions need approval, `--approval`);
  Franka needs Ray on the controller node. From Codex / Claude Code (README, "Using the robots from
  Codex or Claude Code") a real arm moves only with `PI_EMBODIED_CONFIRM_FILE` tickets or
  `PI_EMBODIED_MOTION_CONFIRMED=1`.
