---
name: robot-results-evidence
description: How to read a pi-embodied tool result (text JSON, structuredContent, attached images, refused/stopped/cancelled fields) and what counts as evidence for a finish claim. Use when interpreting a robot tool result or writing the finish summary.
---

# Reading results and evidence

Every tool answers one text block of JSON, then zero or more PNG images. The JSON is the env
server's reply rendered as pi renders it: small arrays inline, arrays over 256 elements as
`{"ndarray": "<dtype>", "shape": [...]}`, each `[H, W, 3]` uint8 image replaced by
`{"image": n, "shape": [H, W, 3]}` where `n` is the 1-based position of the attached image. The
same object is also returned as `structuredContent`. Text over 60 kB ends in `[truncated]`.

## Motion results

| field | meaning |
| --- | --- |
| `ok` | the controller reached its target within tolerance |
| `refused` | the server's safety check refused the call before moving; the message names the limit |
| `stopped` | `contact` (predicted or measured contact), `stalled` (no progress), or `cancelled` (a `stop`) |
| `steps_used`, `final_dist_m` | how long the servo ran and how far from the target it ended |
| `cancelled: true` | a `stop` interrupted the call (then `ok` is false) |
| `error` (`isError`) | the server raised: the one-line message is the exception's last line |

`ok: true` says the arm went where it was told, not that the object moved with it. Only an
`observe` after the motion shows the scene.

## Observation results

`observe` returns the server's `get_observation` (cameras with `rgb`, often `depth` and calibration,
plus proprioception such as `eef_pos`, `gripper_width`), or, on servers without it, `image` from
`render_camera` and `state` from `get_state`. Depth and calibration arrays come back as shape stubs;
ask the perception tools (`back_project`, `view_points`, `mark_point`) for metric values.

## Evidence for finish

A `finish` summary should state, from the last observation: where the target object is, the gripper
state, and the motion results that got there (counts of refused or stopped calls). `status` is
`success` only when the observation shows the goal; `failure` when it shows the goal was not reached;
`blocked` when every attempt was refused or the hook denied the motions; `aborted` after `stop`.
The MCP session records nothing durable: pi's `robot_result` entries, `result.json` and the
evaluation scripts are pi features, so a measured success rate comes from running `pi`, not from
these tools.
