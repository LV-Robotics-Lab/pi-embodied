---
name: embodied-stack
description: Stack one object on another with a pi-embodied robot. Use for stack, put-on-top-of and tower tasks.
license: Apache-2.0 (adapted from OpenETA agent/skills/stack.md)
---

# Stack

Guidance, not a macro. Combine embodied-pick and embodied-place, and reason about stability before release.

1. Identify and confirm both the object to move and the support; look at the support's usable top
   and at nearby obstacles.
2. Grasp with embodied-pick, preferring a contact that leaves the object's base and the release
   view unobstructed.
3. Choose a release point that centres the object's base over a level, large enough part of the
   support, with room for the fingers.
4. Carry above clutter, align without sweeping either object, descend as its own step. Stop if the
   support shifts or the held object slips.
5. Release only when fresh images show stable contact. Retreat so the stack is visible, and check
   both objects stay still and the task condition holds.

If the support is too small, tilted, moving or unclear, get more evidence or ask rather than forcing it.
