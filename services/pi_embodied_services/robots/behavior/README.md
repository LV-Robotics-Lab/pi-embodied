# BEHAVIOR-1K / R1Pro env server

One 2025-challenge activity on the R1Pro in OmniGibson (Isaac Sim), served over the HTTP RPC of
the other env servers (`env_server.py`); `sim.py` is the OmniGibson glue, `tasks.py` the 50
activities and their text, `primitives.py` the `code.api` registry. The pi robot is
`packages/embodied/src/behavior`.

## Install

```bash
bash install.sh services/.venv-behavior ~/BEHAVIOR-1K --dataset    # or: services/setup.sh behavior
```

`install.sh <venv> <BEHAVIOR-1K checkout> [--dataset]` builds the venv (torch, Isaac Sim from
NVIDIA's index, `bddl3` and `OmniGibson[primitives]` from the checkout, cloned at
`v$B1K_VERSION` when missing) and, with `--dataset`, fetches OmniGibson's datasets and the
`2025-challenge-task-instances` (tens of GB) into `OMNIGIBSON_DATA_PATH` (default
`<checkout>/datasets`). Two stacks: `B1K_VERSION=3.9.0` (default: Isaac Sim 5.1, Python 3.11)
and `3.7.2` (Isaac Sim 4.5, Python 3.10, CaP-X's fork). Isaac Sim 6.1 is not supported by
OmniGibson. Mirrors: `PIP_INDEX`, `NVIDIA_INDEX`, `TORCH_INDEX`.

## What runs where

- The env server (Isaac Sim, the scene, OmniGibson's cuRobo primitives) takes one GPU:
  `--gpu-id N` (`OMNIGIBSON_GPU_ID`). Loading a house takes minutes; every episode of
  `eval.sh` starts its own server, so on a shared GPU run the whole eval under that GPU's lock.
- The perception servers the robot attaches to (SAM3 for `segment`, Molmo for `point`) are
  started once, before pi, and should sit on another GPU.
- pi itself runs anywhere that reaches them: `--env` attaches to a running env server.

```bash
source services/.venv-behavior/pi-embodied.env
pi -e packages/embodied/src/behavior --task turning_on_radio --seed 0 --gpu-id 1
packages/embodied/src/behavior/eval.sh runs/b1k turning_on_radio,picking_up_trash 0-4 --model <provider/model> --gpu-id 1
```

`--seed` is the task's pre-sampled instance id (the server lists the instances it finds and
refuses one that is not there). Success is the BDDL activity's own `success`, latched at the
first control step it holds; `q_score` is BEHAVIOR's partial credit. What only the simulator
knows (the object OmniGibson's grasping holds, CaP-X's "picked" judgement) reaches the planner
only under `--privileged`.

## Known problems

- Isaac Sim 5.x and 4.5 segfault at RTX startup on NVIDIA driver 595.x
  (isaac-sim/IsaacSim#677); use an older driver.
- Kit is not thread-safe: the server runs every call on its main thread, one at a time.
