# pi-embodied

Robots as pi extensions. A robot extension registers the robot's tools, replaces the
system prompt, and records the environment's own success signal in the session.
Everything else (the agent loop, models, sessions, interactive/print/json/rpc modes)
is pi.

## LIBERO

Needs an [RPent](https://github.com/RLinf/RPent) checkout with its LIBERO, Pi0.5 and
SAM3 services installed; the extension speaks their HTTP RPC directly.

```bash
export RPENT_ROOT=/path/to/RPent RPENT_PYTHON=/path/to/venv/bin/python
PI05_CHECKPOINT_PATH=... SAM3_CHECKPOINT_PATH=... packages/embodied/src/libero/serve.sh

pi -e packages/embodied/src/libero --suite libero_10 --task 2 --seed 0          # interactive
packages/embodied/src/libero/eval.sh runs/l10 libero_10 0-9 0-2 --model <provider/model>
```

Each session ends with a `libero_result` entry (`terminated` is LIBERO's success flag,
`claimed` is the agent's own status).

The RPC format and tool semantics follow RPent (Apache-2.0).
