# pi-embodied

Embodied-agent harness built on pi. An LLM planner drives a robot through tools;
robot, VLA and perception services are reached over the robot service RPC
(`POST /call` with `{method, args, kwargs}`, numpy arrays as `__ndarray__`).

## Layout

- `src/rpc.ts`: robot service RPC client.
- `src/toolkit.ts`: robot tools as pi tools. Mutating tools return a fresh state
  capture with camera images; `finish` ends the run; success comes from the
  environment, never from the agent's claim.
- `src/image-budget.ts`: keeps camera frames in context under a byte budget, with
  hysteresis so provider prefix caches keep hitting.
- `src/extension.ts`: pi extension for one episode (tools + context hygiene).
- `src/eval.ts`: batch evaluation, one pi session per episode; provider and
  gateway failures are reported as `infra_error`, retried, and excluded from the
  success rate.
- `src/robots/`: robot registry.
- `test/`: unit tests.

## Commands

```bash
cd packages/embodied
npm run check                                   # typecheck
npm test                                        # unit tests
npm run eval -- --robot <name> --endpoint http://127.0.0.1:18100 \
  --tasks 0-3 --seeds 0-4 --model selfhost/muse-glimmer-30b --thinking medium
```

## Compatibility and credits

The RPC wire format and the tool execution contract (post-action state capture,
environment-judged success) follow [RPent](https://github.com/RLinf/RPent)
(Apache-2.0), so RPent's LIBERO env, pi0.5 VLA and SAM3 servers can be attached
without modification.
