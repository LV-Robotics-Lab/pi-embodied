# BEHAVIOR-1K / R1Pro env server

One 2025-challenge activity on the R1Pro in OmniGibson (Isaac Sim), served over the HTTP RPC of
the other env servers (`env_server.py`); `sim.py` is the OmniGibson glue, `tasks.py` the 50
activities and their text. The primitives (pi's tools and `code.api`) are declared once in
`packages/embodied/src/primitives/manifests/behavior.json`; the server checks itself against it
at start. The pi robot is `packages/embodied/src/robots/behavior`.

## Install

Two stacks. The env server is written for either; only the Isaac Sim 6.1 one has been run.

```bash
# Isaac Sim 6.1, OmniGibson 3.9.0 patched by behavior-isaac61.patch, its cuRobo by curobo-isaac61.patch
# (run on driver 595.71.05, an RTX 5090)
bash install_isaac61.sh services/.venv-behavior ~/BEHAVIOR-1K --dataset    # or: services/setup.sh behavior
# OmniGibson's own stack, unpatched: Isaac Sim 5.1 (3.9.0) or 4.5 (3.7.2, CaP-X's fork)
bash install.sh services/.venv-behavior51 ~/BEHAVIOR-1K-51 --dataset    # or: B1K_STACK=isaac51 services/setup.sh behavior
```

`install_isaac61.sh <venv> <BEHAVIOR-1K checkout> [--dataset]` builds the venv (torch 2.11 cu128,
Isaac Sim 6.1 from NVIDIA's index, OmniGibson's runtime deps against that stack), clones
BEHAVIOR-1K at `v3.9.0` when missing and applies `behavior-isaac61.patch`, installs `bddl3` and
`OmniGibson` from it, and builds OmniGibson's pinned cuRobo (StanfordVL/curobo @78612f45 with
`curobo-isaac61.patch`) for the GPU's architecture; it needs a CUDA 12.x `nvcc` (`CUDA_HOME`).
Isaac Sim 5.x and 4.5 segfault at RTX startup on NVIDIA driver 595.x (isaac-sim/IsaacSim#677),
which is why the 6.1 port exists; `install.sh` (the 5.1 / 4.5 stacks) has not been run on this
repository's only BEHAVIOR host, which has that driver. Mirrors: `PIP_INDEX`, `NVIDIA_INDEX`,
`TORCH_FIND_LINKS` (`TORCH_INDEX` for install.sh).

### The data

OmniGibson 3.9 reads everything from one directory, `OMNIGIBSON_DATA_PATH` (the server's
`--data-path`; default `<checkout>/datasets`):

- `omnigibson-robot-assets/` (the R1Pro) and `behavior-1k-assets/` (scenes and objects, 31 GB,
  encrypted: the BEHAVIOR Data Bundle license puts the key at `<data>/omnigibson.key`, which is
  never copied or committed);
- `2026-challenge-task-instances/`: the task instances OmniGibson 3.9 loads
  (`get_task_instance_path` reads only this set). Per split (`scenes/` = train,
  `scene_test/public/` = public test) a scene holds one full template per task,
  `<scene>_task_<activity>_0_0_template.json`, and an overlay per instance,
  `<scene>_task_<activity>_instances/<scene>_task_<activity>_0_<instance>_template-tro_state.json`
  (the task-relevant objects' states and the robot pose). `--seed` is that instance id; the server
  lists the ids it finds across both splits and refuses one that is not there, builds the env on
  instance 0's template and puts the overlay in place as OmniGibson's challenge evaluator does
  (`sim.load_task_instance`), then makes it the scene's initial state (`instance_split` in the
  meta says which split held it). The 2025 set (`2025-challenge-task-instances/`, the demos'
  metadata) is not needed to run.
- CaP-X's OmniGibson 3.7 layout, `og_dataset/scenes/<scene>/json/..._<instance>_template.json`
  (a full template per instance), is still read when present.

`--dataset` fetches all of it with OmniGibson's downloader (`HF_ENDPOINT` honoured). A box that
already holds the zips unpacks them into the same layout.

## Isaac Sim 6.1: what differs from OmniGibson 3.9.0 as released

**Runs on Isaac Sim 6.1; scores are not comparable with the official BEHAVIOR challenge**, which
is evaluated on Isaac Sim 5.1. What differs from upstream (StanfordVL/BEHAVIOR-1K v3.9.0):

- Simulator: Isaac Sim 6.1 / Kit 110, PhysX 110, torch 2.11, warp 1.16 (upstream: Isaac Sim 5.1,
  torch 2.7). Contacts, solver and rendering are other versions.
- The port (`behavior-isaac61.patch`, header in the file): an `omnigibson_6_1_0.kit` (Kit 110's
  extension registry; the deprecated `isaacsim.core.*` extensions still ship); the PhysX
  interfaces from `omni.physx` (no longer on `PhysicsContext`); `SdrShaderNode.GetShaderInput`
  (OpenUSD without Ndr); articulation handles invalidated through `carb.eventdispatcher` stop
  events. One OmniGibson 3.9.0 bug on compute capability 12.0 (RTX 50-series), not an Isaac Sim
  one: `CuRoboMotionGenerator` drops the R1Pro's DEFAULT embodiment there (its cuRobo warmup hits
  an illegal memory access), and `update_obstacles` then raised `KeyError` on every plan; it now
  updates the shared collision world through the MotionGen that exists. Planning uses the arm
  embodiments only.
- cuRobo (`curobo-isaac61.patch`): `wp.device_from_torch` (warp 1.16 dropped `warp.torch` from
  the top-level namespace); built for the host GPU only.
- GPU selection: on Isaac Sim 6 the server leaves `OMNIGIBSON_GPU_ID` unset (`sim.set_render_gpu`):
  RTX numbers all Vulkan devices, so an explicit `active_gpu` 0 after `CUDA_VISIBLE_DEVICES`
  names the hidden GPU and no render device is created; Kit then renders on the visible GPU.
- The challenge evaluator's light synchronisation (a teleoperation aid) is not reproduced when an
  instance overlay is loaded.

Measured on the box (RTX 5090, driver 595.71.05): the R1Pro in an empty scene renders its three
cameras with metric depth, a joint target moves the asked 0.150 rad, cuRobo moves either hand
5 cm to within 0.1 mm; `turning_on_radio` instance 0 (train split) loads with its overlay (the
house takes about 5 minutes), BDDL's `success` reads false before anything is done, and a cuRobo
`_move_hand` runs in it.

## What runs where

- The env server (Isaac Sim, the scene, OmniGibson's cuRobo primitives) takes one GPU:
  `--gpu-id N`. Loading a house takes minutes; every episode of `eval.sh` starts its own server,
  so on a shared GPU run the whole eval under that GPU's lock.
- The perception servers the env server calls (SAM3 for `segment`, `--sam3`; Molmo for
  `point`, `--molmo`; `--molmo ""` when none runs) are started once, before pi, and should sit
  on another GPU.
- pi itself runs anywhere that reaches them: `--env` attaches to a running env server.

```bash
source services/.venv-behavior/pi-embodied.env
pi -e packages/embodied/src/robots/behavior --task turning_on_radio --seed 0   # GPU: cuda_device / PI_EMBODIED_CUDA_DEVICE
packages/embodied/src/robots/behavior/eval.sh runs/b1k turning_on_radio,picking_up_trash 0-4 --model <provider/model>
```

`--seed` is the task's pre-sampled instance id (the server lists the instances it finds and
refuses one that is not there). Success is the BDDL activity's own `success`, latched at the
first control step it holds; `q_score` is BEHAVIOR's partial credit. What only the simulator
knows (the object OmniGibson's grasping holds, CaP-X's "picked" judgement) reaches the planner
only under `--privileged`.

## Known problems

- Isaac Sim 5.x and 4.5 segfault at RTX startup on NVIDIA driver 595.x
  (isaac-sim/IsaacSim#677); use `install_isaac61.sh` there.
- `OMNIGIBSON_DATA_PATH` must exist before `omnigibson` is imported (its macros assert it): the
  server sets it from `--data-path`, so a missing or wrong directory fails at start with
  OmniGibson's own message.
- Kit is not thread-safe: the server runs every call on its main thread, one at a time.
