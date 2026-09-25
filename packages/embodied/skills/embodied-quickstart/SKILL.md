---
name: embodied-quickstart
description: Install, preflight and run pi-embodied robots (LIBERO, ManiSkill, RoboCasa, RoboTwin, RoboLab, Franka, Piper). Use when setting up a robot or benchmark, after /embodied-setup, or when a robot run fails to start.
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

Modes: tools (default), units (`--units=true`), fine-tuned (`src/finetuned` extension,
`--model finetuned/<adapter>`, served by `services/pi_embodied_services/finetuned/serve.sh`),
flash (`--model flash/replay`, LIBERO only).

## 2. Install

```bash
services/setup.sh <target> [--weights] [--venv DIR] [--weights-dir DIR] [--dry-run]
services/setup.sh --help        # targets and what each fetches
```

Run `--dry-run` first and show the user what it will do. Targets: libero, libero-pro, libero-plus,
maniskill, robocasa, robotwin, robolab, franka, franka-polymetis, dual-franka, piper, finetuned,
llamafactory. It creates `services/.venv-<target>` (uv when present), installs the matching extra,
fetches assets, and with `--weights` the checkpoints into `$PI_EMBODIED_WEIGHTS`
(default `~/.cache/pi-embodied`). It writes `<venv>/pi-embodied.env`; `source` it before pi,
`serve.sh` and `eval.sh`. Mirrors: `HF_ENDPOINT=https://hf-mirror.com`, `UV_INDEX_URL` /
`PIP_INDEX_URL`. Gated weights (SAM3) need `HF_TOKEN` with the license accepted.

Tools mode also needs the shared model servers: `packages/embodied/src/<robot>/serve.sh`
(LIBERO: Pi0.5 + SAM3; RoboCasa: RLDX-1; RoboTwin: LingBot). Units mode does not.

## 3. Preflight

```bash
source services/.venv-<target>/pi-embodied.env
node packages/embodied/src/check.ts <robot> --python "$PI_EMBODIED_PYTHON" --services services [--units]
```

Fix every FAIL before running. Inside a robot session the same check is `/robot-check`.

## 4. One episode

```bash
cd <experiment dir> && source <venv>/pi-embodied.env
pi <task flags> [--units=true] --dashboard=true
```

The dashboard URL is shown at startup. Boolean flags take the next word: write `--flag=true`.
Without the experiment settings: `pi -e packages/embodied/src/<robot> ...`.

## 5. Evaluate

```bash
packages/embodied/src/<robot>/eval.sh <out-dir> <cells...> --model <provider/model> --thinking low
```

Each robot's `eval.sh` header gives its cells (LIBERO: suite, tasks, seeds). Run it from a directory
without the experiment settings (it loads the robot with `-e`; loading it twice fails on duplicate
flags). Rerunning retries only invalid episodes. Use `eval-parallel.sh` next to it where present.

## Common failures

- `[robot] unavailable: ...` at start: the env server did not come up; run the preflight.
- Import errors in the preflight: wrong venv. Each robot family has its own venv; the extras conflict
  (Torch/Transformers pins).
- RTX 5090 / sm_120: install `torch==2.7.1 torchvision==0.22.1` from the cu128 index before the extra;
  RoboTwin also needs cuRobo built against it (services/README.md).
- LIBERO renders black or crashes: `MUJOCO_GL=egl` needs an NVIDIA EGL driver; pick the GPU with
  `MUJOCO_EGL_DEVICE_ID`. MuJoCo must stay `3.3.0`.
- LIBERO-PRO assets "ready" but tasks missing files: rerun `setup.sh libero-pro`; its patched
  downloader verifies every file.
- SAPIEN (ManiSkill, RoboTwin): needs `libvulkan1`; on sm_120 its OIDN denoiser logs "unsupported
  device type": replace SAPIEN's `oidn_library` with OIDN >= 2.3.
- `robotwin-download-assets` fails behind hf-mirror (tree API): download the asset zips directly.
- RoboLab: Isaac Sim takes about a minute to start; `ROBOLAB_ROOT` must point at the patched checkout.
- A model server port already answers: `serve.sh` leaves it running; `serve.sh stop` stops only its own.
- Molmo and SAM3 on one GPU: start Molmo with `--offload-blocks 20`.
- Real robots: an operator at the e-stop (`--operator=true`); Franka needs Ray on the controller node.
