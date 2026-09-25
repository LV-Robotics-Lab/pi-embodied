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
venv per robot family. From the repository root:

```bash
# LIBERO (+ SAM3 + Pi0.5 via RLinf's openpi fork; mujoco==3.3.0)
uv venv services/.venv-libero --python 3.11 && source services/.venv-libero/bin/activate
uv pip install -e "services[libero]"          # or [libero-pro] / [libero-plus]

# RoboCasa (RLDX-1 needs Python 3.10; install a CUDA torch>=2.7/torchvision>=0.22 pair first)
uv venv services/.venv-robocasa --python 3.10 && source services/.venv-robocasa/bin/activate
uv pip install -e "services[robocasa]" \
    --constraint services/pi_embodied_services/robots/robocasa/eval/target50-constraints.txt

# RoboTwin (SAPIEN 3.0.0b1, LingBot runtime, cuRobo built from GitHub against torch==2.7.1)
uv venv services/.venv-robotwin --python 3.11 && source services/.venv-robotwin/bin/activate
uv pip install -e "services[robotwin]"
robotwin-download-assets --output ~/.robotwin/assets

# Franka / dual Franka (RLinf franka branch at bde6c918, openpi, franky); Ray must be running
uv pip install -e "services[franka,sam3]"

# Molmo: its own venv (transformers>=4.57 conflicts with openpi's 4.53.2)
uv pip install -e "services[molmo]"
```

`uv` reads the `[tool.uv]` conflict table, so `uv sync --extra <robot>` inside `services/`
works as well. Plain `pip install -e "services[...]"` also works.

**RTX 5090 / Blackwell (sm_120):** the default `torch==2.7.1` wheels (pulled in by the openpi
fork and used to build cuRobo) do not include sm_120 kernels. Install the CUDA 12.8 build of
the same version first, then the extra:

```bash
uv pip install torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu128
```

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

# SAM3 (shared)
SAM3_CHECKPOINT_PATH=/ckpt/sam3/sam3.pt python -m pi_embodied_services.components.sam3_server --port 18300

# Molmo (own venv; only needs this package on PYTHONPATH)
MOLMO_CHECKPOINT_PATH=/ckpt/Molmo2-8B PYTHONPATH=services \
python -m pi_embodied_services.components.molmo_server --port 18400

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
`PI05_NORM_STATS_PATH`, `SAM3_CHECKPOINT_PATH`, `MOLMO_CHECKPOINT_PATH`, `LIBERO_ROBOT_BASE`,
`ROBOT_PLATFORM`, `MUJOCO_EGL_DEVICE_ID`, `RLDX_RESET_SEED`, `RLDX_ATTN_IMPL`, `HF_HOME` /
`HF_HUB_CACHE` (RLDX backbone metadata), `ROBOTWIN_ASSETS_PATH`, `QWEN25_PATH` (LingBot),
`PI_EMBODIED_RLINF` / `RLINF_REPO_PATH` (an RLinf checkout the dual-Franka server puts on
`sys.path`; default `<services>/../rlinf`), `PI_EMBODIED_SERVICES` (overrides the project root,
default `services/`).

From pi, the robots in `packages/embodied` start the env servers themselves (`--services` /
`PI_EMBODIED_SERVICES`, default this directory; `--python` / `PI_EMBODIED_PYTHON` for the venv),
and `packages/embodied/src/<robot>/serve.sh` starts the shared model servers the same way.

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

Known caveat: `dual_franka/env_server.py` uses RLinf attributes
(`_left_ctrl`/`_right_ctrl`, `get_raw_camera_snapshot`, `get_raw_camera_metadata`) that the
pinned RLinf `bde6c918` does not define; they come from the RLinf checkout given by
`PI_EMBODIED_RLINF`.

## Tests

```bash
cd services && pip install -e ".[test]" && pytest
```
