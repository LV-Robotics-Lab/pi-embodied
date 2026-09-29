# Adding a primitive

A primitive is one robot action or query (`move_delta`, `set_gripper`, `back_project`,
`get_object_pose`, ...). The model reaches it three ways, and all three end in the same server
method under the same limits:

- a pi tool the robot registers;
- a Show-Harness unit (`act`), through the robot's `units.apply`;
- a code-mode program (`run_code`, `--code`), through the server's whitelist (`CodeApi.resolve`).

A primitive is declared **once**, in the robot's manifest
`packages/embodied/src/primitives/manifests/<robot>.json`. pi reads it at load (tool schemas and
descriptions, the code-mode prompt); the env server reads the same file
(`services/.../components/manifest.py`) for `code.api`, the whitelist every program call goes
through, and its startup self-check. There is no second declaration: no `primitives.py`, no
TypeBox schema in the robot's `index.ts`.

## 1. The server method

In `services/pi_embodied_services/robots/<robot>/env_server.py`:

- Implement it on the facade and register it in `_register_rpc`:
  `self._rpc["env.<name>"] = self.<name>`. A method that does not move the robot goes into
  `self._readonly_methods`.
- Check every limit before anything moves and refuse instead of clamping: per-call translation and
  rotation caps, the workspace box, the Z floor, finite numbers. Say in the error what the caller
  should do ("Split the motion.").
- A motion loop polls `self.stop_requested()` between control steps and returns `cancelled: true`
  when it stops early.
- A tool whose behaviour needs several server calls or a servo loop runs it here, in one method:
  the tool and the program then share the limits, the stop handling and the success latching.
- A composite that calls another primitive calls it through `self._rpc["env.<other>"]` (the
  registered, possibly wrapped method: the grasp planners' id expiry, for one).

## 2. The manifest entry

```jsonc
{
	"name": "push",                  // the tool's and the program's name; a Python identifier
	"side": "env",                   // env: this RPC method runs it for both; ts: a pi-side tool (never in code mode); code: no tool
	"method": "env.push",
	"tier": "low",                   // exactly one of high | low | raw | privileged (CaP-X's levels)
	"mutating": true,
	"requires": ["ik"],              // capabilities the run needs (flags, services, server state)
	"params": {
		"direction": { "type": "vec3", "required": true, "description": "unit vector, base frame" },
		"distance": { "type": "number", "minimum": 0, "maximum": 0.1, "description": "m" },
		"arm": { "type": "enum", "values": "{{arms}}" },          // a robot variable; [] drops the parameter
		"steps": { "type": "integer", "maximum": 400, "modes": ["code"] }  // program-only
	},
	"doc": {
		"tool": "Push the TCP along a direction; refused beyond {{max_move}} m.",
		"code": "Push the TCP along a direction.\n\nExample:\n    push([1, 0, 0], 0.05)"
	},
	"result": "motion"               // the tool's display: motion (new images and state) or read
}
```

- **Tiers** (PARAMS.md 4.1): `high` is CaP-X's semantic functions (`get_object_pose`,
  `sample_grasp_pose`, `goto_pose`, `home_pose`, `open_gripper`, `close_gripper`); `low` the parts
  (perception, IK, joint and Cartesian motion, the gripper); `raw` the step and the raw
  observation; `privileged` ground truth (a same-name privileged entry replaces the high one under
  `--privileged`; `low+privileged` adds ground truth to the low tier).
- **Examples** go into `doc.code` as a Google `Example:` section: the S4 tier
  (`--code-api=low-noexamples`) drops them.
- **Shared entries** (`common/*.json`: grasp planner, perception, geometry, the simulators' ground
  truth, the modules' tools) are used by name with overrides: `{"use": "grasp/plan_grasp", "doc": {...}}`.
- Types: `number | integer | boolean | string | enum | vec3 | quat | array | object`; `minimum`,
  `maximum` and enum `values` are enforced on both sides.
- Methods that are never primitives (`env.reset`, `stop`, `code.*`) are refused by the loader;
  the robot's other non-primitive RPC methods are listed in the manifest's `"internal"`.

The server checks itself at start (`serve`): a declared, available `env` / `code` entry without its
RPC method, or a registered business method that is neither declared nor internal, stops it with
the list. `code.api` answers the manifest's digest (pi refuses a server of another version) and the
primitives of a tier available this run; pi checks that its own view (the robot's `capabilities`)
agrees, and records `code_tier_digest`.

## 3. Code mode

A new mutating primitive also needs, on the server:

- its translation estimate in `_code_move_m(method, kwargs)`, so `--code-max-move` bounds it;
- a refusal in `_code_check(method, kwargs)` for anything `--code-timeout` could not bound;
- in `_code_reply(method, out)`, whatever a program must not see (object state) or need not
  carry (images, video frames: they go to the run's video).

## 4. The tool

In `src/robots/<robot>/index.ts`, register the execution only:
`robot.tool("push", "", Type.Object({}), async (params, signal) => call("env.push", params, [], signal))`
(or `robot.primitive("push", run)`). The schema and description are the manifest's; registering a
name the manifest does not declare throws. Describe it in the robot's `SYSTEM.md` inside
`[tool:push]...[/tool:push]`, and add it to the list `start` returns; a tool whose `requires` the
run cannot meet is not activated.

A `ts` entry is a tool only: pi executes it (its schema comes from the manifest) and it never
reaches a program: no `code` doc, no RPC method, never in `code.api`
(`test_pi_side_tools_are_tool_mode_only_and_never_reach_code_api`). That is where the VLA and skill
loops belong (`pi0_pick`, `rldx_skill`, `lingbot_act`, ...): they run in pi against a policy
server and are tool-mode only, by design. A behaviour a program should reach is an `env` entry
with a server method.

## 5. Tests and docs

- `services/tests/`: the method's limits and refusals (mock the simulator), and the robot's server
  self-check against its manifest.
- `test/<robot>.test.ts`: the tool against a fake env server (`test/helpers/code-api.ts` answers
  `code.api` from the manifest).
- The tool-schema snapshot: `UPDATE_TOOL_SCHEMAS=1 node --test --experimental-strip-types
  test/tool-schemas.test.ts`; the diff is the new tool.
- `services/PROTOCOL.md`: the method's row in the robot's table.
- A model server it needs goes into the robot's `services` spec (`src/infra/model-services.ts`).
