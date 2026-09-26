# pi-embodied services

The Python side of pi-embodied: the simulator/robot **env servers** and the **model servers**
(VLA policies, SAM3, Molmo) that the TypeScript robots in `packages/embodied` talk to over
HTTP.

License: Apache-2.0, like the rest of the repository (`LICENSE` at the root).

Wire protocol, method list and `stop` semantics: [PROTOCOL.md](PROTOCOL.md).

## Layout

```
services/
  pyproject.toml            pi-embodied-services, extras per robot
  pi_embodied_services/
    utils/                  config, logging, EGL device mapping, RPC (HTTP only)
    components/             facade bases, Pi0.5 VLA, SAM3 and Molmo servers
    robots/
      libero/               env_server.py
      robocasa/             env_server.py, vla_server.py (RLDX-1), eval/target50.json
      robotwin/             env_server.py, vla_server.py (LingBot), rlinf_env.py, contract.py,
                            eval/demo_randomized.json
      franka/               env_server.py, runtime_config.py, tasks.py, perception.py
                            (calibration only), franka_env.py, config/example.yaml
      dual_franka/          env_server.py, runtime_config.py, tasks.py, perception.py
                            (calibration only), config/example.yaml
  tests/                    dispatch lock / stop / healthz (no simulator)
```

`franka/perception.py` and `dual_franka/perception.py` hold the calibration and base-frame
helpers.

## Behavior

- One call at a time per server (process-wide lock).
- New lock-free `stop` / `cancel` method; Franka/dual-Franka servo and chunk loops and the
  RoboTwin chunk loop poll it between steps. See PROTOCOL.md for exactly what it can stop.
- `healthz` returns `{status, version, service}`.
- `--transport` accepts only `http`; results of interrupted calls carry `"cancelled": true`.

## Install

Each robot extra pins its own VLA runtime (Torch/Transformers versions differ), so use one
venv per robot family. `setup.sh <robot> [--weights] [--dry-run]` does the steps below for one
robot (venv, extra, assets, checkpoints) and writes `<venv>/pi-embodied.env`; by hand, from the
repository root:

```bash
# LIBERO (+ SAM3 + Pi0.5 via RLinf's openpi fork; mujoco==3.3.0)
uv venv services/.venv-libero --python 3.11 && source services/.venv-libero/bin/activate
uv pip install -e "services[libero]"          # or [libero-pro] / [libero-plus]

# RoboCasa (RLDX-1 needs Python 3.10; install a CUDA torch>=2.7/torchvision>=0.22 pair first)
uv venv services/.venv-robocasa --python 3.10 && source services/.venv-robocasa/bin/activate
uv pip install -e "services[robocasa]" \
    --constraint services/pi_embodied_services/robots/robocasa/eval/target50-constraints.txt
# Kitchen assets come from Box.com and RLDX weights from Hugging Face; where those are
# unreachable, ModelScope `exuan2/robocasa` (kitchen_assets/) and `Twilighted/RLDX-Robocasa365`
# hold byte-identical copies (same sizes / sha256 as Box and HF rev 587e9ec).

# ManiSkill: the stock scenes need no assets; the Show-Harness real2sim rigs BlockPAP-v1
# (default) / BlockStack-v1 come from github.com/AaronCaoZJ/RLinf (Apache-2.0):
#   bash services/pi_embodied_services/robots/maniskill/fetch_real2sim.sh
# (sparse-clones the rig code, fetches the table textures with sha256 checks, and writes
# BlockStack's extended-finger Panda URDF, which is not published anywhere).

# BEHAVIOR-1K / R1Pro (OmniGibson + BDDL; Isaac Sim from NVIDIA's index, the challenge dataset:
# tens of GB): robots/behavior/install.sh <venv> <BEHAVIOR-1K checkout> [--dataset]; the env
# server takes --gpu-id (OMNIGIBSON_GPU_ID). See robots/behavior/README.md for what runs where.
bash services/pi_embodied_services/robots/behavior/install.sh services/.venv-behavior ~/BEHAVIOR-1K --dataset

# Metaworld (Sawyer MT50; metaworld==3.1.1 pins mujoco==3.3.0, no assets; MUJOCO_GL=egl)
uv venv services/.venv-metaworld --python 3.11 && source services/.venv-metaworld/bin/activate
uv pip install -e "services[metaworld]"

# Genesis (OpenETA's Franka cube_pick; Genesis 1.4 wants torch>=2.8, on sm_120 a cu128 build)
uv venv services/.venv-genesis --python 3.11 && source services/.venv-genesis/bin/activate
uv pip install -e "services[genesis]"

# RoboTwin (SAPIEN 3.0.0b1, LingBot runtime, cuRobo built from GitHub against torch==2.7.1)
uv venv services/.venv-robotwin --python 3.11 && source services/.venv-robotwin/bin/activate
uv pip install -e "services[robotwin]"
robotwin-download-assets --output ~/.robotwin/assets

# Franka / dual Franka (RLinf franka branch at bde6c918, openpi, franky); Ray must be running
uv pip install -e "services[franka,sam3]"

# Molmo: its own venv (transformers>=4.57 conflicts with openpi's 4.53.2)
uv pip install -e "services[molmo]"

# Flywheel LeRobot export: its own venv, Python >= 3.10 (lerobot 0.4 pins numpy 2 /
# huggingface-hub); pass its python to pi as --flywheel-python
uv pip install -e "services[flywheel]"

# IK / reach preview (components/ik_server.py, PyRoKi on the CPU): its own venv; the env
# servers reach it over RPC (pi: --ik http://127.0.0.1:18400). [ik-curobo] adds the GPU backend.
uv venv services/.venv-ik --python 3.12 && source services/.venv-ik/bin/activate
uv pip install -e "services[ik]"
python -m pi_embodied_services.components.ik_server --port 18400   # --backend curobo (GPU)

# Third-party VLAs (components/openvla_server.py, openvla_oft_server.py, gr00t_server.py): one venv
# each; their repos pin torch 2.2 / their own transformers, which no robot extra shares. On sm_120
# install the cu128 torch first, then the extra without its torch pin:
uv venv services/.venv-openvla --python 3.10 && source services/.venv-openvla/bin/activate
uv pip install torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu128
uv pip install -e "services[openvla]"       # or [openvla-oft] (Python 3.10/3.11) / [gr00t] (Python 3.12)
# Checkpoints are pinned by commit hash in each server (OPENVLA_CHECKPOINTS, OPENVLA_OFT_CHECKPOINTS,
# GR00T_CHECKPOINTS) and fetched by huggingface_hub at that revision when --model-path is not a
# directory: set HF_ENDPOINT=https://hf-mirror.com (never a gateway proxy) where huggingface.co is
# unreachable, or snapshot_download them yourself and pass the directory.
```

Every dataset pi-embodied exports is LeRobot v3.0 with the same feature names:
`observation.images.<camera>`, `observation.state`, `action` (their dimensions named in
`meta/info.json`), plus `action_source` (VLA or scripted) for Flywheel data and `actor`,
`dagger`, `action_repeat` for GUMI runs. One dataset holds one robot's action space.
RoboTwin episodes also export in joint space (`--space joint`) in XPolicyLab's LeRobot layout
(its `scripts/transform_lerobot_v30_format.py`): `observation.state` the measured joints and
`action` the commanded joint targets, both `left_joint_0..6, right_joint_0..6` (6 joints and the
gripper per arm), channel-first `cam_high` / `cam_left_wrist` / `cam_right_wrist` video,
`robot_type` `unified_robot`, and `action_source` kept as an extra column (XPolicyLab reads its
features by name), so scripted steps can be filtered out. The joint-space export always encodes
video: lerobot encodes with PyAV's bundled FFmpeg, but reads it back through `torchcodec` when
that is installed, which needs FFmpeg's shared libraries (4 to 8, e.g. `libavutil.so.60`) on
the system; with them missing, uninstall `torchcodec` to read through PyAV, or install FFmpeg.

```bash
# Flywheel (LIBERO, RoboCasa, RoboTwin; pi's /flywheel-export runs the same)
python -m pi_embodied_services.flywheel.cli export-lerobot --data-root ~/.pi/embodied/datacollection \
  --robot robocasa --select target/PnPCounterToCab
# RoboTwin for XPolicyLab's training scripts (datasets/lerobot-joint/robotwin/<select>/<id>)
python -m pi_embodied_services.flywheel.cli export-lerobot --data-root ~/.pi/embodied/datacollection \
  --robot robotwin --select demo_randomized/beat_block_hammer --space joint
# GUMI teleop / DAgger runs of one task
python -m pi_embodied_services.flywheel.cli export-gumi <gumi-record>/<MMDD>/task_<id> --output-root runs/lerobot
# A LeRobot v2.1 dataset exported before (lerobot 0.3) converts in place
python -m lerobot.datasets.v30.convert_dataset_v21_to_v30 --repo-id <id> --root <dir> --push-to-hub=false
```

`uv` reads the `[tool.uv]` conflict table, so `uv sync --extra <robot>` inside `services/`
works as well. Plain `pip install -e "services[...]"` also works.

**RTX 5090 / Blackwell (sm_120):** the default `torch==2.7.1` wheels (pulled in by the openpi
fork and used to build cuRobo) do not include sm_120 kernels. Install the CUDA 12.8 build of
the same version first, then the extra:

```bash
uv pip install torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu128
```

RoboTwin on sm_120 (verified on RTX 5090): `[robotwin]` alone builds cuRobo against the default
torch, so install the runtime first and build cuRobo against the cu128 torch with a CUDA 12.8
toolkit (the pip nvcc 12.8 wheel has no `nvcc`; CUDA 13 does not match a cu128 torch):

```bash
uv pip install -e "services[rlinf]" rlinf-robotwin-runtime==0.1.1 rlinf-lingbotvla==0.1.1
CUDA_HOME=<cuda 12.8 toolkit> TORCH_CUDA_ARCH_LIST=12.0 SETUPTOOLS_SCM_PRETEND_VERSION=0.7.8 \
  uv pip install --no-build-isolation "nvidia-curobo @ git+https://github.com/NVlabs/curobo.git@v0.7.8"
```

SAPIEN needs `libvulkan1`, and its bundled OIDN 2.0.1 denoiser has no sm_120 code (every frame
logs "OIDN Error: unsupported device type: CUDA" and stays noisy): replace the libraries under
SAPIEN's `oidn_library` with OIDN >= 2.3 (keep SAPIEN's file names); redo after reinstalling SAPIEN.
`robotwin-download-assets` lists files through the Hugging Face tree API, which hf-mirror breaks;
download the asset zips directly and unpack them with the runtime's extraction functions.

## Run

Every RPC server takes `--transport http --host 127.0.0.1 --port <port>` (port `0` picks a
free one and prints `RPC server listening on http://...`) and `--parent-watch` (exit when
stdin closes, i.e. when the spawning pi process dies). Run them as modules; if the package is
not installed, put `services/` on `PYTHONPATH`.

```bash
# LIBERO env (one per episode; pi spawns it)
python -m pi_embodied_services.robots.libero.env_server --suite libero_spatial --task 9 --seed 0 \
    [--max-episode-steps 10000] [--cuda-device N]

# Pi0.5 VLA (shared)
PI05_CHECKPOINT_PATH=/ckpt/rlinf-pi05-libero-130-fullshot-sft \
python -m pi_embodied_services.components.pi05_vla_server --embodiment libero --port 18200
# dual Franka: --embodiment dual_franka --model-path CKPT --repo-id ORG/DATASET [--norm-stats-path DIR]

# OpenVLA / OpenVLA-OFT / GR00T (own venvs; pi mounts openvla_act / openvla_oft_act / gr00t_act with
# --openvla / --openvla-oft / --gr00t <url>; libero/serve.sh starts them when OPENVLA_PYTHON etc. are set)
OPENVLA_CHECKPOINT_PATH=/ckpt/openvla-7b-finetuned-libero-spatial \
python -m pi_embodied_services.components.openvla_server --port 18600        # [--suite libero_spatial|...] [--unnorm-key K]
OPENVLA_OFT_CHECKPOINT_PATH=/ckpt/openvla-7b-oft-finetuned-libero-spatial \
python -m pi_embodied_services.components.openvla_oft_server --port 18700    # [--no-center-crop] [--repo <openvla-oft clone>]
GR00T_CHECKPOINT_PATH=/ckpt/RLinf-Gr00t-N1.6-SFT-Spatial \
python -m pi_embodied_services.components.gr00t_server --port 18800          # [--embodiment libero_panda]

# SAM3 (shared)
SAM3_CHECKPOINT_PATH=/ckpt/sam3/sam3.pt python -m pi_embodied_services.components.sam3_server --port 18300

# Molmo (own venv; only needs this package on PYTHONPATH)
MOLMO_CHECKPOINT_PATH=/ckpt/Molmo2-8B PYTHONPATH=services \
python -m pi_embodied_services.components.molmo_server --port 18400
# On a GPU shared with the Pi0.5 VLA and SAM3 add `--offload-blocks 20` (the last 20 text
# layers stay in host memory, ~7.7 GB less GPU); without it SAM3 runs out of memory and
# every Flash anchor fails. Flash's `--molmo off` replays a recorded plan exactly instead.
# RoboTwin's LingBot serve.sh also defaults to 18400: give one of them another port.

# RoboCasa env (one per episode) and RLDX-1 VLA (shared, per-session state)
python -m pi_embodied_services.robots.robocasa.env_server --task-name OpenDrawer --split target --seed 0
python -m pi_embodied_services.robots.robocasa.vla_server --model-path /ckpt/rldx-1-ft-rc365 --port 18500 \
    [--session-timeout-s 3600] [--session-sweep-s 60]

# RoboTwin env (one per episode) and LingBot-VLA (WebSocket, shared)
python -m pi_embodied_services.robots.robotwin.env_server --task-name beat_block_hammer \
    --task-config demo_randomized --seed 0 --assets-path ~/.robotwin/assets [--max-episode-steps 10000]
QWEN25_PATH=$M/qwen_base python -m pi_embodied_services.robots.robotwin.vla_server --model-path $M \
    --norm-path $M/norm_stats/robotwin_eef.json \
    --lingbot-robot-config $M/configs/robot_configs/robotwin_eef.yaml --use-length 50 --port 18400

# Franka / dual Franka (Ray cluster up, RLINF_NODE_RANK set on each node)
python -m pi_embodied_services.robots.franka.env_server --task-description "..." [--robot-config YAML] [--print-config]
python -m pi_embodied_services.robots.dual_franka.env_server --task-description "..." [--robot-config YAML]
```

Environment variables read by the servers: `PI05_CHECKPOINT_PATH`,
`PI05_NORM_STATS_PATH`, `OPENVLA_CHECKPOINT_PATH`, `OPENVLA_OFT_CHECKPOINT_PATH`, `GR00T_CHECKPOINT_PATH`
(and `HF_ENDPOINT` for their pinned downloads), `SAM3_CHECKPOINT_PATH`, `MOLMO_CHECKPOINT_PATH`, `LIBERO_ROBOT_BASE`,
`ROBOT_PLATFORM`, `MUJOCO_EGL_DEVICE_ID`, `RLDX_RESET_SEED`, `RLDX_ATTN_IMPL`, `HF_HOME` /
`HF_HUB_CACHE` (RLDX backbone metadata), `ROBOTWIN_ASSETS_PATH`, `QWEN25_PATH` (LingBot),
`PI_EMBODIED_RLINF` / `RLINF_REPO_PATH` (an RLinf checkout the dual-Franka server puts on
`sys.path`; default `<services>/../rlinf`), `PI_EMBODIED_SERVICES` (overrides the project root,
default `services/`).

From pi, the robots in `packages/embodied` start the env servers themselves (`--services` /
`PI_EMBODIED_SERVICES`, default this directory; `--python` / `PI_EMBODIED_PYTHON` for the venv),
and `packages/embodied/src/<robot>/serve.sh` starts the shared model servers the same way.

## Fine-tuned mode

Show-Harness's fine-tuned mode (`packages/embodied/src/finetuned`: a small VLM + LoRA picks one
action unit per step) uses the scripts in `pi_embodied_services/finetuned/`. Training needs
LLaMA-Factory in its own venv; install it once on the box with
[`setup_llamafactory.sh`](pi_embodied_services/finetuned/setup_llamafactory.sh) (pinned
LLaMA-Factory, torch 2.8.0+cu129, China mirrors; `LF_VENV` / `LF_ROOT` default to
`/root/autodl-tmp/venvs/llamafactory` and `$LF_VENV/LlamaFactory`). `GIT_PROXY` routes its
GitHub fetch of the pinned commit through a proxy:

```bash
GIT_PROXY=http://127.0.0.1:1056 bash services/pi_embodied_services/finetuned/setup_llamafactory.sh
```

Then `train.sh` turns GUMI recordings (or `--from-lerobot` datasets, or `--from-rollouts`
generated ones) into a LoRA adapter, `serve.sh` serves base + adapter on an OpenAI-compatible vLLM
endpoint, and `episode.sh` runs pi episodes against it (usage in each script's header). The
Show-Harness files training reads (github.com/showlab/Show-Harness @137d571, Apache-2.0: the
`rollouts_to_alpaca.py` converter and its `prompts/v3`, `v4`, `register_dataset.py`, the four
`train/configs`, the LLaMA-Factory extension and their `train/scripts/train.sh`) are vendored under
`finetuned/showharness/` and are the default; `SH=<checkout>` still points at another copy.
`download_dataset.py` fetches their training data (Show-Harness-Data at a pinned revision, via
hf-mirror, every file checked against a pinned listing digest) and can register it.

Training data from simulation (Show-Harness's real2sim, action units executed closed-loop and
recorded frame-before-unit; the core is `finetuned/atomic.py`):
`robots/maniskill/real2sim.py` (Scheme A, a privileged oracle) and `real2sim_follow.py` (Scheme D:
`record` continuous demos, `follow` them in 2 cm units) on the BlockPAP/BlockStack rigs;
`robots/robolab/real2sim.py` (`oracle` / `record` / `follow`, any single-object pick-and-place
task, in the RoboLab venv). Shards merge with `finetuned/merge_shards.py`; `finetuned/check_dataset.py`
gates a dataset (ending, stalled units, empty grasps, success) and `finetuned/make_dataset.py`
converts it and reports the step-size statistics.

Not ported from Show-Harness, on purpose: the GPT web operator and the web/controller prompts
(pi is the operator), `generate_subgoals.py` / `generate_affordance.py` and the converter's
`--use-subgoal` / `--use-affordance` modes (they need Show-Harness's planner plugins), the gemma4
venv of their `setup_llamafactory.sh`, the real-robot collectors (`scripts/trajectory/collect_*`,
replay, rebuild_video; GUMI records here), the preview-video renderer of the generators, RoboLab's
fingertip calibration sweep and short-finger asset builder (the USD ships in `robots/robolab/assets`),
and the Scheme D stock-ManiSkill demos (PickCube/StackCube).

## External dependencies that remain

Not vendored; installed by the extras or provided by the host:

- MuJoCo (`mujoco==3.3.0` for LIBERO), robosuite (RLinf fork, `rpent` branch), LIBERO /
  LIBERO-PRO / LIBERO-plus (RLinf forks), RoboCasa365 (`rpent-robocasa365`).
- RoboTwin runtime (`rlinf-robotwin-runtime==0.1.1`, SAPIEN 3.0.0b1) and cuRobo v0.7.8.
- RLinf (`rpent-rlinf`; for Franka the `rlinf[franka]` GitHub pin `bde6c918`), Ray.
- openpi fork (`rpent-openpi`, `rpent` branch; brings `torch==2.7.1`, `transformers==4.53.2`).
- SAM3 fork (`RLinf/sam3@main`), Hugging Face `transformers>=4.57` (Molmo), RLDX-1
  (`rlinf-rldx`, `rpent` branch), `flash_attn` (optional; RLDX falls back to SDPA),
  LingBot-VLA deploy package (`rlinf-lingbotvla==0.1.1`, provides `deploy.*`).
- Real robots: `franky-control` (libfranka 0.19), `pyrealsense2`, `gymnasium`, the RLinf
  Franka controller stack (real-time kernel, ROS/easy_handeye calibration files).
- Checkpoints and assets: `RLinf/RLinf-Pi05-LIBERO-130-fullshot-SFT`, `facebook/sam3`
  (`sam3.pt`), `allenai/Molmo2-8B`, `RLWRLD/RLDX-1-FT-RC365` (rev `587e9ec`) plus
  `RLWRLD/RLDX-1-VLM` metadata (rev `4b9f870`), `RLinf/LingBot-VLA-RoboTwin-EEF-ckpt1500`
  (rev `e727b46`), the RoboTwin asset snapshot, the dual-Franka Pi0.5 checkpoint and its SFT
  dataset repo id.
- Third-party VLAs: `openvla/openvla-7b-finetuned-libero-{spatial,object,goal,10}` (revs in
  `openvla_server.OPENVLA_CHECKPOINTS`, spatial `962318c`), `moojink/openvla-7b-oft-finetuned-libero-*`
  (`openvla_oft_server.OPENVLA_OFT_CHECKPOINTS`, spatial `6d0231a`), `RLinf/RLinf-Gr00t-N1.6-SFT-Spatial`
  (rev `e39614a`, the `libero_panda` embodiment) and the `nvidia/GR00T-N1.6-3B` / `GR00T-N1.7-3B` bases
  (`gr00t_server.GR00T_CHECKPOINTS`).

Known caveat: `dual_franka/env_server.py` uses RLinf attributes
(`_left_ctrl`/`_right_ctrl`, `get_raw_camera_snapshot`, `get_raw_camera_metadata`) that the
pinned RLinf `bde6c918` does not define; they come from the RLinf checkout given by
`PI_EMBODIED_RLINF`.

## Tests

```bash
cd services && pip install -e ".[test]" && pytest
```
