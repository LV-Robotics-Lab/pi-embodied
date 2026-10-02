# Pi boundary and default regression

The development baseline is pi 1.0.0 (release tag `v1.0.0`, commit
`a13d35a742c6ef8462812a28fbe1d8c8b7431c32`), including upstream fixes through
`319fecb89b17b7bf8a4b62734de9a6cc6ceabe49`. Peer dependencies target pi 1.x;
the range is installation metadata, not a claim that every pi 1.x release has
passed the same regression checks.

The agent loop, model registry, messages, sessions, tools and extension events
belong to pi. Robot task state, motion gates, camera context and results belong
to the embodied extension. Python RPC services own simulation, perception and
bounded robot execution. New robotics features should use these boundaries
before proposing another core patch. Pi 1.0 removes the experimental harness and
session-storage exports from `pi-agent-core`; durable sessions belong to
`pi-durable`. Embodied adapters use the retained `Agent` and coding-agent
extension APIs.

`infra/service-process.ts` owns Python process startup, readiness, shutdown,
attachment and environment construction. `defineRobot` owns episode lifecycle
and decides whether a service exit invalidates its episode. Existing robot
helper exports remain available to robot extensions.

## Official pi and fork capabilities

The fork adds `ExtensionContext.exportSession` and `withdrawQueuedMessage`.
Dashboard checks these capabilities per attached session. Its SSE snapshot
reports `capabilities.sessionExport` and `capabilities.queueWithdrawal`.
An unavailable operation returns HTTP 501 with an explanation; sending a
steering message and calling an available robot tool continue to work.
An unsuccessful withdrawal leaves the message in pi's queue.

Tests cover contexts with and without these optional APIs. This is contract
coverage, not an end-to-end certification of every official pi installation.
Before claiming standalone upstream support, run extension loading and session
lifecycle tests against an unmodified official distribution at the stated
version. The deployed, integrated target remains this fork.

## Default workflow

The ordinary extension workflow uses the robot's native tools. Units, Python
code mode, fallback, ensemble, exploration, replay and privileged simulator
state are explicit modes. They should not silently change ordinary sessions.
Python code mode executes the robot service's bounded primitive API; pi's
QuickJS codemode serves a different execution environment.

The fixed Qwen comparison profile deliberately uses units mode to preserve the
existing six-cell diagnostic: MetaWorld reach-v3, ManiSkill PickCube-v1 and
LIBERO libero_spatial task 0, each at seeds 0 and 1. It fixes low thinking,
40 turns and 300 seconds, with a fresh local memory per cell. All cells run
sequentially. This diagnostic is too small to estimate general task ability.

Run from a clean, built checkout under the host's GPU reservation:

```sh
bash packages/embodied/src/scripts/fixed-regression.sh /path/to/new-results /path/to/env-files
```

The environment directory provides `metaworld-env.sh`, `maniskill-env.sh` and
`libero-env.sh`, with backend Python, rendering, GPU and asset locations. Model
credentials remain in pi's existing registry. The runner selects this checkout's
CLI and services, rejects reused output directories, and records Git version
before and after. Build from that same clean version before running. Report
successes out of all six planned cells as well as valid cells; model/transport
errors and missing results are invalid, never silently dropped or retried.

Separately, `test/gpu-e2e.sh` covers 11 simulator adapters with fixed startup,
observation, one-action and result checks. These checks use a stub planner and
measure integration, not Qwen success. HumanCLAW WALK coverage does not certify
sitting on the intended seat. BEHAVIOR dataset incompatibility must remain a
failed check until resolved. Physical robot acceptance is separate and deferred.

Fallback-to-VirtualModels migration and further separation of episode budgets,
context pruning and robot tool registration require their own behavior tests;
they are not implied by this first process-lifecycle extraction.
