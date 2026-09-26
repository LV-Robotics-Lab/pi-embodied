---
name: embodied-pick
description: Grasp and lift one object with a pi-embodied robot (LIBERO, ManiSkill, RoboCasa, Franka, UR5e, Piper, ...). Use at the start of a pick, pick-and-place or stack task, or after a failed grasp.
license: Apache-2.0 (adapted from OpenETA agent/skills/pick.md)
---

# Pick

Guidance, not a macro: the robot's tool descriptions define the exact parameters. Use the tools your
robot has; skip a step whose tool is not active.

## Know the target

1. Look before choosing (`view_env_state`). Name the target in a short English phrase.
2. Segment it (`segment`) and compare the overlay with the image: a high score does not prove it is
   the right object. Keep the same physical object across views; do not switch to a neighbour when
   the target moves or is occluded. If segmentation misses, point at it and `back_project` a pixel.
3. Memory (the robot's memory corpus, `recall_objects`) is a prior about this scene, not evidence:
   check the current image before acting on it.

## Choose the grasp

4. With a grasp backend, `plan_grasp` the target. Choose for holding through lift and transport, not
   for the highest score alone: aperture, contact depth, clearance, where the contacts land on the object.
5. `suggest_grasp` (--grasp-advisor) gives a second opinion from a separate vision model. Treat it as
   evidence. Contacts on a rim, cap, neck, thin edge or barely on the object slip; if every candidate
   looks like that, get another view or plan again instead of taking the least bad one.
6. A candidate that fails for a structural reason (unreachable, collision, empty close):
   `plan_grasp` with `next_after` for the next rank, not the same one again.

## Approach

7. Come in from behind the grasp along its approach direction. Avoid long sideways sweeps near the
   object: they push it away. Set the orientation before the last inward motion.
8. One short clear move is fine; around an obstacle or for a precise approach use a short route
   (`follow_waypoints`, --waypoints: lift, carry, descend as separate segments) and look again from
   where the robot actually stopped before extending it.
9. Near the object, when the wrist view shows a small sideways offset and the depth and orientation
   are right, `align_wrist` (--align-wrist) on the target's wrist pixel and move to `aligned_xyz`.
   When the target is clipped or the orientation is doubtful, move for a better wrist view or plan again.

## Close and confirm

10. Close only when fresh images show the fingers at the intended contact. A reported collision with
    unrelated scenery does not by itself make the close unsafe.
11. A close or a gripper width is not proof of holding. Lift a little and check: `check_attached`,
    the object moving with the gripper and its old place empty.
12. Carry on only with positive evidence. Ambiguous: look again. Empty close or slip: reopen and choose
    a materially different contact.
13. Once held, clear the whole object from nearby clutter: lift first, then move sideways. Do not
    descend towards a container while still carrying sideways.

## Recovery

- Wrong or unclear target: identify it again before planning another grasp.
- Weak candidates: change the view or the evidence rather than cycling ranks of the same plan.
- Motion stopped short or missed: reason from where the robot is now.
- A tool error, timeout or missing calibration is not evidence against the grasp; follow the error.
