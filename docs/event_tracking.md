# Event Tracking

The `EventTracker` class monitors and records significant events that occur during robot manipulation tasks. It provides a unified mechanism to detect anomalies, failures, and state transitions during task execution.

## Overview

The tracker maintains internal state to detect transitions and ensures each event type is only recorded once per occurrence. When a condition clears (e.g., gripper lifts off table), the tracker resets for that event, allowing it to be recorded again if it reoccurs.

Events are returned as `(info_string, StatusCode)` tuples, where the info string provides human-readable context and the status code categorizes the event type.

## Tracked Event Modes

RoboLab Verified reworked this vocabulary so that each line names one physical transition
(see [Changes → Event vocabulary](verified/changes.md#event-vocabulary)). A pick-and-place
reads `OBJECT_GRIPPED → OBJECT_CARRIED → OBJECT_GRABBED_SUCCESS → OBJECT_RELEASED →
OBJECT_IN_CONTAINER_SUCCESS → SUBTASK_COMPLETED`. Every event is stamped at its **onset**;
the step the detector fired is kept as `detected_step`.

| Event | Code | Severity | Description |
|-------|------|----------|-------------|
| **OBJECT_GRIPPED** | 284 | neutral | The jaws closed on this object (emitted once it became a carry, stamped when the jaws closed) |
| **OBJECT_CARRIED** | 283 | neutral | A carry was established: contact held 0.2 s with the object coupled to the hand while the hand moves |
| **OBJECT_GRABBED_SUCCESS** | 139 | success | The subtask ladder's grasp rung was credited |
| **GRASP_ATTEMPT_FAILED** | 266 | failure | The hand touched / closed on the object but never established a carry; attempts within 2 s are one line with a count. Never on containers or fixtures, never within 2 s of releasing the same object |
| **OBJECT_RELEASED** | 267 | neutral | A carried object left the hand while the hand was commanded open |
| **OBJECT_DROPPED** | 268 | failure | A carried object left the hand while it stayed closed (slip) |
| **TOWED_WITHOUT_GRASP** | 275 | failure | The object moved with an *open* hand, off-centre on one finger, lifted clear — a physics artifact; the episode is marked `physics_artifact` |
| **WRONG_OBJECT_GRABBED** | 250 | failure | A carried object is not in the task's target set (targets resolved from the condition arguments across all stages; containers excluded) |
| **WRONG_OBJECT_DETACHED** | 257 | failure | A wrong object left the hand without a release (folded into the release when one follows within 0.5 s) |
| **WRONG_OBJECT_PLACED** | 270 | failure | A non-target the hand had held was released inside a goal container; objects already inside at reset never count |
| **TARGET_OBJECT_BUMPED** | 282 | neutral | The policy nudged an object the task is about |
| **OBJECT_BUMPED** | 258 | failure | A non-target moved ≥ 2 cm and stopped; not while the policy holds it or within 1 s of releasing it |
| **OBJECT_MOVED** | 259 | failure | A non-target moved ≥ 50 cm and stopped |
| **OBJECT_OUT_OF_SCENE** | 260 | failure | An object left the workspace bounding box |
| **OBJECT_FELL_OFF_TABLE** | 272 | failure | An object dropped 15 cm below its starting height |
| **TARGET_LOST** | 273 | failure, terminal | The success condition can no longer be met (a required object or the destination left the table); the episode ends |
| **SCENE_SETTLING** | 269 | neutral | Objects moved during the 1 s reset warm-up without hand contact; one line per env |
| **GRIPPER_HIT_TABLE** | 255 | failure | Gripper made contact with the table surface |
| **GRIPPER_HIT_OBJECT** | 264 | failure | Gripper collided with a non-target object |
| **GRIPPER_FULLY_CLOSED** | 256 | neutral | "Gripper closed on nothing": ≥ 98 % of the closure span with no object in contact |
| **MULTIPLE_OBJECTS_GRABBED** | 265 | failure | A single gripper is in contact with multiple objects simultaneously |
| **SUBTASK_COMPLETED** | 190 | success | A ladder stage finished |

Not emitted by this fork: `TARGET_OBJECT_DROPPED` (replaced by released / dropped),
`OBJECT_GRABBED_FAILURE` as a tracker line (replaced by `GRASP_ATTEMPT_FAILED`),
`WRONG_OBJECT_PUSHED_IN` (merged into `WRONG_OBJECT_PLACED`), `PLACED_WITHOUT_LIFT` (off by
default; no true positive observed). `OBJECT_STARTED_MOVING` and `OBJECT_TIPPED_OVER` exist
in the enum and are effectively dead, as upstream.

The upstream table, for reference:

| Event | Description |
|-------|-------------|
| **WRONG_OBJECT_GRABBED** | Gripper grasped an object not in the intended target list |
| **GRIPPER_HIT_TABLE** | Gripper made contact with the table surface |
| **GRIPPER_FULLY_CLOSED** | Gripper closed completely without grasping an object (potential failed grasp) |
| **OBJECT_STARTED_MOVING** | A non-target object transitioned from stationary to moving |
| **OBJECT_BUMPED** | Object stopped after minor displacement (< move threshold) |
| **OBJECT_MOVED** | Object stopped after significant displacement (>= move threshold) |
| **OBJECT_OUT_OF_SCENE** | Object moved outside the defined workspace bounding box |
| **OBJECT_TIPPED_OVER** | An object that should remain upright has fallen over |
| **TARGET_OBJECT_DROPPED** | Target object was successfully grasped but released mid-transport |
| **GRIPPER_HIT_OBJECT** | Gripper collided with a non-target object |
| **MULTIPLE_OBJECTS_GRABBED** | A single gripper is in contact with multiple objects simultaneously |

### Multi-gripper robots

Contact events resolve gripper names through the robot's `contact_gripper` declaration
(see [Robots](robots.md#contact-gripper)), and the two contact events treat the
`"gripper"` alias group differently on purpose:

- **GRIPPER_HIT_OBJECT** uses the group as-is: touching a non-target object with *any*
  gripper is a hit, whichever hand did it.
- **MULTIPLE_OBJECTS_GRABBED** is evaluated **per concrete gripper**: the violation is one
  gripper clutching several objects at once. Contacts are counted separately for each
  member of the group, so a bimanual robot holding one object in each hand does not
  trigger it — but either hand individually touching two objects does.
