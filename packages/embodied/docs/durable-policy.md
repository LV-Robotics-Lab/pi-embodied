# Persistent embodied policy prototype

The optional policy extension keeps one Pi Durable conversation, goal ledger and
working memory across MetaWorld episodes. A host event starts or resumes work;
`policy_finish` settles the current goal or puts it into a waiting state. Waiting
does not poll the model. A later event continues the same conversation.

The host is the existing pi session. This prototype is loaded explicitly and is
not registered as a default extension. It currently supports MetaWorld tool mode.

## Start

From the repository root on the Linux workspace, using its Node >=22.19 runtime,
installed workspace dependencies, and an already configured MetaWorld environment:

```bash
NODE_OPTIONS="--conditions=source" pi \
  -e packages/embodied/src/robots/metaworld \
  -e packages/embodied/src/capabilities/policy \
  --task reach-v3 --seed 0 --units=false --code=false \
  --policy-store "$PWD/runs/policy/reach.sqlite" \
  --model provider/model
```

Use the provider/model configured in your environment. Ask the host agent to call
`policy_goal` with these arguments:

```json
{
  "goal_id": "reach-001",
  "goal": "Complete the current MetaWorld reach task",
  "event_id": "start"
}
```

On interruption, reopen with the same SQLite file and model, and repeat exactly
the same goal and event. Event IDs deduplicate admission. Changed content requires
a new event ID. If the goal is waiting, send a new event such as `scene-ready`
with an `input` field describing the new evidence or instruction. After success
or failure, a new episode uses a new goal ID and the same SQLite file. An unfinished
goal must be concluded before another goal can start.

Normal MetaWorld session startup may reset the simulator. Resuming the policy
retains reasoning and memory, not the old simulator state. A new process or reset
invalidates observations and requires another observation before any action.
The policy store retains goals and transcript; it does not hold a running process
alive. External wakeup scheduling is outside this prototype.

## Execution and recovery

- `policy_observe` calls `view_env_state(fresh=true)` through the host tool pipeline.
  This refreshes server state and both cameras instead of reusing cached images.
- `policy_act` delegates only `move_delta` and `set_gripper` through
  `ExtensionToolContext.executeTool`. Schema validation, tool hooks, permission
  checks and robot limits remain in that path. An observation admits one action.
- Action intent is committed before dispatch, and the action is not replay-safe.
  If execution is interrupted after a physical effect but before acknowledgement,
  the saved outcome remains `unknown`. The policy must observe again.
- `policy_finish(success)` requires fresh environment-confirmed task success.
  Finishing a policy goal preserves the conversation. The outer robot episode's
  ordinary `finish` and result recording remain the host's responsibility.
- `policy_remember` replaces bounded working memory shared across goals.
- Model requests are capped at 32 per external event; actions at 64 per goal.
  Exhausted request budget puts the goal into waiting. A new event grants another
  request budget, but does not reset the goal's action budget. Nested model cost
  is reported to the host's existing side-model cost accounting.
- SQLite ownership uses Linux `flock`; concurrent owners of the same canonical
  path are refused. Use a distinct store for each independently controlled robot.
  Keep the same path; symlink/hardlink aliases to one file are unsupported.

The episode/revision check is local to this host. It does not fence another
process controlling the same environment. Use one controller for the environment.
Fresh state and camera reads are sequential RPCs, not an atomic simulator snapshot.
There is no server-side exactly-once action ledger, physical emergency stop,
simulator checkpoint restoration or recovery of lost step-count telemetry here.

Do not treat this prototype as a benchmark mode: the existing evaluator does not
fingerprint policy history or initial memory. Cross-episode memory changes the
experimental conditions and needs a separate protocol before comparing scores.

## Validation

```bash
node --conditions=source --test \
  packages/embodied/test/policy.test.ts \
  packages/embodied/test/policy-tool-world.test.ts \
  packages/embodied/test/metaworld.test.ts
npm run check
```

The tests exercise real Durable scheduling and SQLite persistence with a scripted
model, plus the actual MetaWorld extension against a fake HTTP RPC server. They
cover goal/event persistence, waiting, interrupted actions, stale observation
rejection, environment-confirmed success, budgets, exclusive ownership, host
pipeline delegation, and fresh state/camera reads. They do not establish learned
policy quality or real simulator/robot task success.
