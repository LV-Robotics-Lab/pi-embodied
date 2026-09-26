---
name: embodied-object-memory
description: Keep track of where objects are during a long pi-embodied episode with remember_object / recall_objects (--object-memory). Use in long-horizon or multi-object tasks, after occlusions, or when a lesson from a failure should inform later steps.
license: Apache-2.0 (adapted from OpenETA agent/skills/memory_extract.md)
---

# Object memory

`remember_object`, `recall_objects` and `forget_object` (pi started with --object-memory) keep one
record per object name: its last world position (and orientation), the env step and time it was
seen, and your note. They are session entries, so a resumed or forked session keeps them; with
--object-memory-dir they carry over to the next episode of the same scene, marked `from_earlier_episode`.

Record:
- an object's position after you localized it (segment + back_project, plan_grasp) and after you
  moved it (placed, pushed), with a note of its state (in the bowl, on the plate, lid open);
- where an object went when it leaves the view (held, occluded, dropped behind something);
- a short lesson on the object that failed (slips when grasped at the rim, blocked from the left).

Do not record guesses, whole tool outputs, or images. Recalled poses are as old as their step: look
again before acting on one, and `forget_object` a record that turned out wrong. Records from an
earlier episode are hints about the layout only; the scene was reset since.

The robot's memory corpus (`/memory`, --explore) is different: task knowledge across episodes (recipes,
notes) in files. Object memory is what is where in this scene now.
