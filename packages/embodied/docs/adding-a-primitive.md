# Adding a primitive

A primitive is one robot action or query (`move_delta`, `set_gripper`, `back_project`, `segment`,
...). The model can reach it three ways, and all three must end in the same server method under
the same limits:

- a pi tool the robot registers (`robot.tool`, or a shared one from `src/primitives/`);
- a Show-Harness unit (`act`), through the robot's `units.apply`;
- a code-mode program (`run_code`, `--code`), through the server's primitive registry (`code.api`).

So a primitive starts on the server, is declared once in its registry, and only then gets a tool.

## 1. The server method

In `services/pi_embodied_services/robots/<robot>/env_server.py`:

- Implement it on the facade and register it in `_register_rpc`:
  `self._rpc["env.<name>"] = self.<name>`. A method that does not move the robot goes into
  `self._readonly_methods` so it may run next to a motion.
- Check every limit before anything moves and refuse instead of clamping: per-call translation and
  rotation caps, the workspace box, the Z floor, finite numbers. Say in the error what the caller
  should do ("Split the motion.").
- A motion loop polls `self.stop_requested()` between control steps and returns `cancelled: true`
  when it stops early, so an abort or a code-mode timeout ends it within one step.
- Return plain data (numbers, lists, `np.ndarray`); images as arrays, which the RPC encodes.

## 2. The registry entry

Declare it in `robots/<robot>/primitives.py` (see `robots/genesis/primitives.py`):

```python
Primitive(
    "push",                       # the name a program calls; a Python identifier
    "env.push",                   # the RPC method it resolves to
    "Push the TCP along a base-frame direction (m); refused beyond 0.1 m or outside the box.",
    {"direction": Param("vec3", "unit vector"), "distance": Param("number", "m")},
    mutating=True,                # it moves the robot
    tiers=("high",),              # high, low; ground truth alone is privileged
)
```

`register_code_api(self, PRIMITIVES)` (in `_register_rpc`) validates the declaration when the
server starts: the name is an identifier and unique, the method is registered, the tiers are known
(`privileged` only on its own), the parameter types are `number|integer|boolean|string|vec3|array|object`.
The declaration's digest names the API version: each episode records it (`code_api` entry,
`code_api_digest` in the result), so a changed primitive shows up in the results.

Tiers follow CaP-X's levels: `high` is perception plus pose-level motion, `low` is raw observations
and small motions, `privileged` is the high tier plus ground truth (simulators, `--privileged`).

## 3. Code mode

Code mode runs a program in a sandboxed subprocess whose only way to the robot is the registry
(`utils/code_exec.py`, PROTOCOL.md "Code mode"). On a server that serves `code.run` (LIBERO today),
a new mutating primitive also needs:

- its translation estimate in the server's `_code_move_m(method, kwargs)`, so `--code-max-move`
  bounds it before it runs (a primitive it does not know counts 0 m);
- a refusal in `_code_check(method, kwargs)` for anything `--code-timeout` could not bound (a long
  chunk, a large render).

## 4. The tool

- One robot: `robot.tool("push", description, TypeBox schema, run)` in `src/<robot>/index.ts`; `run`
  checks what the TS side knows (the operator gate, `checkMove`) and calls
  `env.call("env.push", kwargs, timeout, [], signal)`. Enums use `StringEnum`.
- Two or more robots: put it in `src/primitives/` as a `toolDef(...)` that takes a rig (the robot's
  env call, limits, checks: see `MotionRig` in `src/primitives/motion.ts`) and let each robot mount
  it through its own `tool()`. Do not copy a tool between robots.
- Describe it in the robot's `SYSTEM.md` inside `[tool:push]...[/tool:push]`, so the prompt mentions
  it only when the tool is active (`toolSections`, `src/robot.ts`).
- Add it to the list `start` returns (or leave it behind a flag: a feature that is off registers no
  tool).
- A unit that should use it goes through `units.apply` in the robot's spec, not around it.

## 5. Tests and docs

- `services/tests/`: the limits and refusals of the method (mock the simulator), and that the
  registry validates (`test_code_api.py` shows the pattern).
- `test/<robot>.test.ts`: the tool against a fake env server: the RPC it sends, a refusal, an abort.
- The tool-schema snapshot: `UPDATE_TOOL_SCHEMAS=1 node --test --experimental-strip-types
  test/tool-schemas.test.ts`, and check the diff is only the new tool.
- `services/PROTOCOL.md`: the method's row in the robot's table.
- If the primitive needs a model server, add it to the robot's `services` spec
  (`src/model-services.ts`) so `--serve-models` can start it, and a case to `test/gpu-e2e.test.ts`
  when it only runs on a GPU.
