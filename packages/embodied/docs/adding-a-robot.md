# Adding a robot

A robot is two halves that talk over the services' RPC (`POST /call`, services/PROTOCOL.md):

- a Python env server in `services/pi_embodied_services/robots/<robot>/`, which owns the simulator
  or the arm, every motion limit, and the execution of every primitive;
- a primitive manifest, `src/primitives/manifests/<robot>.json`, the one declaration of the robot's
  tools and code primitives, read by both halves ([Adding a primitive](adding-a-primitive.md));
- a pi extension in `packages/embodied/src/robots/<robot>/`, which registers the robot's flags and tools with
  `defineRobot` (`src/robot.ts`) and nothing else.

The agent loop, models, sessions, modes, `/export`, compaction and the message queue are pi's. Before
building anything that is not robot-specific, check pi core (`packages/coding-agent/docs/`) and the
shared modules below; a feature a robot needs is a flag, a tool, a hook (`pi.on`), a `pi.events`
message or a session entry, and a feature that is off registers no tool.

Genesis is the smallest complete robot to copy from: `src/robots/genesis/index.ts` (one file, 420 lines),
`services/.../robots/genesis/env_server.py`, `src/primitives/manifests/genesis.json`, `test/genesis.test.ts`.

## 1. The env server

`robots/<robot>/env_server.py`:

- A facade class on `BaseEnvFacade` (`components/env_facade_base.py`). Mix in `MainThreadServeMixin`
  first when the renderer needs one thread (MuJoCo EGL, Genesis). The base registers `env.reset`,
  `env.step`, `env.chunk_step`, `env.get_env_meta`, `env.get_task_language`, `env.get_camera_meta`
  and `env.render_camera`; add the robot's own methods in `_register_rpc` and mark read-only ones in
  `_readonly_methods` (they may run next to a motion).
- Motion methods check every limit before anything moves and refuse, not clamp: a per-call
  translation cap, a workspace box, a Z floor (see Genesis `check_target`). Long motions poll
  `self.stop_requested()` between control steps so an abort stops them (`stop` semantics,
  PROTOCOL.md).
- `env.get_env_meta` returns the launch arguments; the TS side compares them with its task flags
  and refuses a server running another task or seed.
- A simulator serves `env.ground_truth_poses` (`utils/ground_truth.py`) for `--privileged` only;
  ground truth never leaks into ordinary observations.
- `main()` takes `--transport http --host --port --parent-watch` plus the task arguments and calls
  `facade.serve(...)`. Port 0 is the default: the server binds a free port and prints
  `RPC server listening on http://127.0.0.1:<port>`, which is how `robot.serve` finds it.
- Serve the manifest: mix `CodeRunMixin` (`utils/code_exec.py`) in first and call
  `self._manifest_code_run("<robot>", have=self._has, move_m=..., check=..., reply=..., begin=...,
  finish=...)` at the end of `_register_rpc`; `_has(capability)` says which `requires` this server
  meets. At `serve` the server checks its RPC methods against the manifest and refuses to start on
  a mismatch.

Install: a `[<robot>]` extra in `services/pyproject.toml` (its own venv when its Torch, MuJoCo or
robosuite pin conflicts with another robot's) and a target in `services/setup.sh`. Document the
methods in `services/PROTOCOL.md` under "Env servers".

## 2. The extension

`src/<robot>/index.ts` exports `default function <robot>(pi: ExtensionAPI)`:

1. Register the task flags (`--task`, `--seed`, ...) and `--env-url` (attach to a running server).
   Where things live is deployment config, not flags (`src/infra/config.ts`): the services dir is
   `servicesDir(pi)`, the env server's Python `python(pi, "<robot>")`, a model server's endpoint
   `service(pi, "sam3")`, an output directory `dir(pi, "logs")`. A service that changes results
   gets a switch that says what, never where (`--detections`, `--grasp contact_graspnet`). Experiment
   parameters are flags; enums use `StringEnum` from `@earendil-works/pi-ai`.
2. Call `defineRobot(pi, spec)` before registering anything else. The spec names the robot, its task
   flags, `start` (bring the robot up and return the tools to activate; throwing fails closed),
   `result` (the outcome fields of `robot_result`), `status` (the dashboard's step and success),
   `finish`, and `prompt` (the robot's `SYSTEM.md`, `[tool:name]` blocks follow the active tools).
3. In `start`, launch the env server with `robot.serve({python, args, cwd, env, log})` (or
   `attach(--env-url)`), check `env.get_env_meta` against the task flags, reset, and return the tools.
4. Name the manifest in the spec (`manifest: "<robot>"`) with its `vars` (the cameras, arms and
   limits its descriptions and enums refer to) and `capabilities` (which `requires` this run
   meets; they must agree with the server's `_has`). Register tools with
   `robot.tool(name, "", Type.Object({}), run)`: the schema and description are the manifest's. Tools run
   sequentially; pass `robot.signal` (or the `signal` argument) to every RPC so an abort stops the
   robot between calls and asks the server to `stop` the running one.
5. Opt in to shared modules through the spec; each one mounts only what the spec asks for:

| Spec field | Module | What it adds |
| --- | --- | --- |
| `units` | `src/modes/units` | Show-Harness `act` (`--units`), GUMI, the fine-tuned provider |
| `code` / `codeApi` | `src/modes/code`, `src/primitives/registry.ts` | `run_code` (`--code`), the recorded `code.api` |
| `groundTruth` | `src/robot.ts` | `--privileged` and `ground_truth_poses` (simulators only) |
| `memory`, `explore` | `src/capabilities/memory`, `src/capabilities/explore.ts` | memory corpus, `--explore` |
| `operator` | `src/capabilities/operator.ts` | `--operator` verdicts and scene resets (real robots) |
| `video`, `vdm` | `src/observation/video.ts`, `src/observation/vdm.ts` | episode video, `--vdm` |
| `flywheel` | `src/capabilities/flywheel.ts` | `--collect-flywheel-data` (needs `robots/<robot>/flywheel.py`) |
| `flash` | `src/capabilities/flash` | `--model flash/replay` |
| `services` | `src/infra/model-services.ts` | `--serve-models`: start its model servers itself |

`services` lists the model servers the robot attaches to (`SAM3`, `MOLMO`, `pi05("<embodiment>")`
from `src/infra/model-services.ts`, or the robot's own `ModelService`); each one's `flag` is the endpoint
flag the robot already reads, so nothing else changes.

Shared tools live in `src/primitives/` (motion, perception, grasp); mount them instead of writing a
second `move_delta`.

## 3. Wiring and docs

- `src/infra/check.ts` `SPECS`: the imports, paths and endpoints `/robot-check` and `node src/infra/check.ts
  <robot>` verify.
- `src/<robot>/eval.sh` for the eval matrix, and its case in `src/scripts/eval-parallel.sh` (light or heavy
  job: see the GPU lock note there).
- The robot table in `README.md`, the setup target in `src/infra/setup` (`/embodied-setup`).
- `test/tool-schemas.test.ts`: add the robot to `ROBOTS` and regenerate the snapshot with
  `UPDATE_TOOL_SCHEMAS=1 node --test --experimental-strip-types test/tool-schemas.test.ts`; the diff
  must show only the new robot.

## 4. Tests

- `test/<robot>.test.ts`: load the robot into a stub pi (see `test/genesis.test.ts`), drive its tools
  against a fake HTTP server that answers the env methods, and check refusals, limits and the
  result entry. No simulator, no model API.
- `services/tests/`: the env server's pure functions (limits, geometry, the registry) with mocks.
- The GPU end-to-end suite (below).

## Testing on a GPU

`test/gpu-e2e.test.ts` runs real env servers and model servers with no model API: each robot starts
from its venv, runs one unit (`act MV_UP`), checks that the env stepped, calls `finish` and reads the
result entry; LIBERO also starts SAM3 through `--serve-models` and segments with it. It is skipped
unless `PI_EMBODIED_E2E` names the robot and `nvidia-smi` answers. Add a `test(...)` for a new robot
with its task flags.

`test/gpu-e2e.sh` runs it robot by robot, each with its environment file sourced (the venv's
`pi-embodied.env` from `services/setup.sh`, or the box's own scripts), against this checkout's
services. On the shared bjb2 box (GPU0 is vLLM; GPU1 is shared):

```bash
cd /root/autodl-tmp/pi-<name>/packages/embodied
export PATH=/root/autodl-tmp/tools/node/bin:$PATH
test/gpu-e2e.sh metaworld=/root/autodl-tmp/tools/metaworld-env.sh
E2E_GPU=1 MIN_FREE=8000 PI_EMBODIED_GPU_LOCK=/root/autodl-tmp/locks/gpu1.lock \
  test/gpu-e2e.sh libero=/root/autodl-tmp/tools/embodied-env.sh robosuite=<env> maniskill=/root/autodl-tmp/tools/maniskill-env.sh
```

`E2E_GPU` makes a light simulator wait (without the lock) until that GPU has `MIN_FREE` MiB free.
The model load takes the GPU lock itself: `--serve-lock` (default `$PI_EMBODIED_GPU_LOCK`) after
`--serve-min-free` found the memory. Never start it under eval-parallel.sh's `LOCK` on the same file.
