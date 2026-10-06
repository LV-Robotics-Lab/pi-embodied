# Durable evaluation

`eval-parallel.sh --durable` uses Pi Durable for persistent batch, worker-lane and
episode tasks. It runs the existing robot `eval.sh` commands, including their
configuration checks, valid-result skipping and invalid-episode reruns. The
ordinary scheduler remains the default.

From the repository root, with the robot environment already configured:

```bash
bash packages/embodied/src/scripts/eval-parallel.sh --durable \
  -j 2 --gpus 1 --variant base= --variant vdm=--vdm=true \
  maniskill runs/ms-durable PickCube-v1 0-4 --model provider/model
```

Requirements: Linux, Bash, `flock`, Node >=22.19, and the workspace's Pi Durable
and Chord packages. The shell selects the workspace source exports. An installed
package needs its optional `@earendil-works/pi-durable` and `@earendil-works/chord`
peers installed separately. Node must provide `node:sqlite`.

## Recovery

Run the same command again after an interruption. Keep its working directory,
selection, variants, worker count, GPU assignments, evaluator code, and robot /
model configuration unchanged. The scheduler checks the worker command, job list
and evaluation environment fingerprint before resuming unfinished tasks. External
model settings files and service state are not snapshotted.

Each slot runs one episode task at a time. A restart continues the saved batch
and lane checkpoints. If a process died after writing a result but before the
Durable completion commit, the evaluator validates that result and skips the
episode. Without a valid result, it starts that episode from its beginning.
RoboTwin retains the existing evaluator's task-level unit covering all its seeds.

The shell takes an exclusive output-directory lock and checks earlier worker
process groups before opening SQLite. It refuses recovery while those processes
are alive. INT, TERM and HUP stop the current run's process group; a killed
Durable scheduler also triggers group cleanup. No existing external service is
stopped by recovery.

After a batch is terminal, another invocation creates a new batch that revalidates
all selected cells. This repairs missing or invalid results and rejects valid
results from another configuration. Infrastructure failures and failed success
thresholds retain the original evaluator's nonzero exit behavior. Durable task
completion describes scheduling; environment success is determined by result.json.

## State and outputs

- `.parallel/durable.sqlite`: task checkpoints and the current batch identity.
- `.parallel/durable-status.json`: current batch ID, job count, worker count,
  live task graph and terminal scheduling outcome. Written by atomic rename.
- `.parallel/w<N>.log`: existing per-worker evaluator logs.
- `summary.json`: existing success rate, Pass@k and invalid-result summary.

SQLite has one owner, and recovery requires the original storage. This is
episode-level recovery: simulator snapshots, robot action replay, operator
confirmation and Dashboard integration are separate work. Completed artifacts
should remain unchanged while a batch is interrupted; resume skips tasks whose
completion was already committed. A later invocation after the batch finishes
revalidates all artifacts.

## Validation

```bash
node --conditions=source --test packages/embodied/test/eval-parallel.test.ts
npm run check
```

The integration tests drive the actual evaluation shell scripts with a stand-in
`pi` that writes robot results. They cover A/B parity, repeat execution, missing
artifacts, infrastructure failure, resume identity, changed-input rejection,
concurrent invocation, orphan protection and scheduler death. They require no
GPU, robot connection or model API calls.
