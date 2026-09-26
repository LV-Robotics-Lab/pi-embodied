---
name: embodied-push-pull
description: Push an object across a surface, or pull a drawer, door or handle, with a pi-embodied robot. Use for push, slide, open-drawer, close-drawer and pull tasks.
license: Apache-2.0 (adapted from OpenETA agent/skills/push.md and pull.md)
---

# Push and pull

Guidance, not a macro; the tool descriptions define the parameters. Use the tools your robot has.

## Push

1. Look: the object, the surface, where it must go, what is in the way.
2. Segment the object when its edge or the contact face is unclear.
3. Choose one short segment: an approach point behind the object, the contact point, the direction
   and a distance of a few centimetres.
4. Move to the approach point, then push one short segment (`move_to`, or `follow_waypoints` with
   --waypoints for approach plus push), then look again before the next one.
5. If the object turns, slips or catches, stop and plan from its new pose; if the path is blocked,
   push from another side. A finished motion is not progress: check the object moved.

## Pull (drawer, door, handle)

1. Look at the handle, the direction it travels, and obstacles. A drawer is a pull with the gripper
   closed on the handle, not a pick-and-place.
2. Segment the handle when unclear; choose a grasp or hook that fits it and the gripper opening.
3. Approach and close as in embodied-pick. Then test the hold with a short move along the travel
   direction (straight for a drawer, a small arc for a hinge) and look before the real pull.
4. Keep the gripper closed and the orientation compatible with the mechanism. Pull one short segment,
   then look: displacement, still attached, new obstacles.
5. Continue only while attached and moving as expected; stop when the images or the environment's
   success flag show the goal.
6. Lost contact or unexpected rotation: stop and replan. Motion without the mechanism moving means
   change the contact or the direction, not repeat it.

With a VLA skill for contact tasks (LIBERO `pi0_doubled`), prefer it for knobs, stoves, drawers and buttons.
