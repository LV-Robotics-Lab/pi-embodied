---
description: Check the pi-embodied robot - env server health, the tools served, the stop latch - and show the current camera view
allowed-tools: mcp__pi-embodied__robot_status, mcp__pi-embodied__observe
---

Check the robot the `pi-embodied` MCP server is connected to, without moving it:

1. Call `robot_status`. Report: `robot`, `endpoint`, `reachable`, `healthz.service` / `healthz.pid`,
   `same_process`, `tier`, `privileged`, `halted`, `claimed`, `broken`, how many `tools` are served
   and any `left_out` entries (they need a `--var`).
2. If `reachable` is true, call `observe` and describe what each camera shows in one sentence.
3. If `reachable` is false, `same_process` is false, `halted` is true or `broken` is set, say so
   first and what the operator has to do (restart or attach the right env server, `resume`, or a
   new session). Do not call any motion tool from this command.

Keep the report short: one line per item, the images as they come.
