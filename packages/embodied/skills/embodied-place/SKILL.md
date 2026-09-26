---
name: embodied-place
description: Carry a held object and release it on or into a target with a pi-embodied robot. Use for the second half of pick-and-place, put-in, put-on tasks.
license: Apache-2.0 (adapted from OpenETA agent/skills/place.md)
---

# Place

Guidance, not a macro; the tool descriptions define the parameters. Use the tools your robot has.

## Before the grasp

1. In a pick-and-place task, plan the placement from the pre-grasp scene when the placement tool needs
   the object and the target from the same view (`plan_place` takes the grasp id and the region).
2. Segment and confirm the destination separately from the object; a high score does not make it
   the requested destination.
3. Memory of an earlier success is a prior only; check the current scene, free space and holding.

## Choose where to release

4. Prefer a placement with clear margins from walls and edges that fits the object's footprint.
   Rank is a heuristic.
5. A placement pose is a low release reference, not a path: never carry straight to it.

## Carry

6. Start only while the object is still held (fresh evidence). Keep the gripper closed and reason
   about collisions with the whole object, not only the fingers.
7. Lift clear, move sideways above clutter, descend as a separate step (`follow_waypoints` with
   --waypoints does this in one checked call). Keep the orientation unless a change is needed.
8. Look at the result of each motion. When blocked, go higher, more central or around the named
   obstacle rather than replaying the same diagonal.
9. Re-check holding after the carry; a check from before a long carry is history.

## Release

10. Descend when the opening or surface is clear. A short drop from just above is acceptable only
    into a visibly open container for a sturdy object that clearly fits.
11. Stop moving sideways before opening. Open only over the destination with the object still held.
12. Retreat, look, and confirm the object rests on or in the target. In benchmark episodes the
    environment's success flag decides.

## Recovery

- Destination unclear: another view, or ask.
- Carry blocked: route around with newly checked geometry.
- Object dropped in transit: stop, find it again, return to embodied-pick.
- Still held after opening: look at the contact before retrying.
