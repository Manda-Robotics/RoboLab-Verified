# RoboLab Verified

A fork of [NVIDIA RoboLab](https://github.com/NVlabs/RoboLab) v0.3.1 that changes what the
benchmark **reports**, not what it simulates. Same 120 tasks, same scenes, same success
predicates, same physics defaults. What is different is that the numbers and the per-episode
event log now mean what they say, and every change carries the measurement that motivated
it and the test that guards it.

| | |
|---|---|
| [Changes](changes.md) | every change, one row each: what upstream did, what changed, whether a score can move, how it was verified |
| [Findings](findings.md) | what was wrong, with the evidence — including what is *not* fixed |
| [Migration](migration.md) | what a downstream tool has to know: new fields, new event codes, new files, the knobs that restore upstream behaviour |
| [Verification](verification.md) | how the evidence was produced and how to reproduce it |
| [Physics](../physics.md) | friction as a run parameter, and why the arm controller is left alone |

## Why

Running the benchmark and watching episodes frame by frame, rather than reading the summary
numbers, turned up defects in the evaluation harness itself. A few, each with its measurement
in [findings.md](findings.md):

- **Success fired on the first frame the predicate held.** 88 of 88 successes in a 328-episode
  corpus ended while a target was still moving, 61 % faster than 10 cm/s. Whether the object
  would have stayed in the bowl was never observed.
- **"Object grabbed" meant "object touched".** No closure, no lift, no coupling — 64 % of logged
  grabs happened with the hand open, and a fumble at a banana produced three identical lines
  per step. Reviewers stopped reading the event log.
- **Five of 120 tasks could never report a grasp**, because the grasp line was a side effect of
  a particular subtask ladder shape.
- **`BananasOutOfBin` completed its subtask ladder at 0.07 s**, before the arm moved, then
  flagged its own targets as wrong objects for the rest of the episode.
- **Four of four `BowlStackingRightOnLeft` "successes" had stacked the wrong bowl** — nested
  identical bowls satisfy `object_in_container` in either order.
- **62 (task, object) pairs sink into their support at reset**, before the robot moves; five of
  them by more than half a metre.

None of these change what a policy *does*. All of them change what the benchmark *says* it did.

## What changed

Organised the way [changes.md](changes.md) is. Numbers are from the runs recorded in
[verification.md](verification.md).

**Scoring and termination.** A success is confirmed only once every target has been at rest
for 0.2 s (an already-still object ends the episode on the spot; a moving one makes it wait).
An episode ends as a failure once its success condition can no longer be met — a required
object or the destination has left the table. Open-top containment is capped at the rim plus
5 cm, and nested containment requires the object above the container's base.

**Subtask crediting.** A ladder rung already true at reset earns no credit, and nothing is
credited before the scene has settled. A placement is credited only after the object comes to
rest. The final `score` is judged on the last frame, with the live monotone number kept as
`score_peak`. List-form ladders are sequences, as their authors evidently intended.

**Event vocabulary.** A grasp is a *carry*: contact held for 0.2 s with the object coupled to
the hand while the hand moves. A pick reads `OBJECT_GRIPPED → OBJECT_CARRIED →
OBJECT_GRABBED_SUCCESS`; a failed attempt is one line with a count; a release is distinguished
from a drop by the commanded gripper channel; a bump of the task's own target is a neutral
note; a non-target delivered into the goal container is `WRONG_OBJECT_PLACED`; an object moving
with an *open* hand is `TOWED_WITHOUT_GRASP` and marks the episode a physics artifact. Per
episode, π0.5's event count fell from 45.5 to 31.0 on the same tasks, and every remaining line
names one physical transition. Every line is stamped at its onset, with detection time kept
alongside.

**Contact sensing and recording.** Both Robotiq pads carry a contact sensor (upstream watched
the left pad only). The HDF5 records per-pad contact *force* and object-to-destination
contact, so most flag rules can be re-checked against an existing recording.

**Physics.** Friction is a run parameter (`--friction`), applied at start-up and read back from
PhysX into every run directory. The default is upstream's authored materials; a 32-episode
sweep per condition found the success rate insensitive to a 4× change in μ while the
behaviour metrics were not ([physics.md](../physics.md)). Five objects that were authored
inside their support were placed where they settle.

**Harness.** A `run_complete.json` marker, line-buffered stdout, `pytest` exit codes that
survive Isaac's shutdown, LFS pointer detection, an install guide that works on a fresh
machine.

**Dashboard.** One transport under all camera tiles with the event timeline as the scrubber,
permalinks with `?t=`, a subgoal checklist that fills with the playhead, a per-task page across
experiments, runs grouped by policy family, HTTP range requests so seeking works.

**Embodiments and policy backends.** A dual-Franka rig and a bimanual ViperX (ALOHA) rig, each
one articulation with per-arm end-effector recording, tracking wrist cameras, and unchanged
success predicates; a connector for running a pointing-capable VLM as a policy; a scripted
runner that proves the two-arm stack turns end to end. No released checkpoint drives two arms
well, and the docs say so.

**Tooling.** Offline audits that need no simulator — task definitions against scene spawn
state, scene interpenetrations, objects that sink at reset, open-hand carries — and a
verifier that turns every flag change into a PASS / FAIL / N/A predicate over recorded runs.

## What did not change, on purpose

- **Task definitions and success predicates**, except where a ladder contradicted its own
  success term (`GrabABagel`, `GrabAFruit`, `FoodPackingByColor`, five spatial Rubik's-cube
  ladders) — each such case is a row in [changes.md](changes.md) with its measurement.
- **Physics defaults**: friction, gravity handling, PD gains, solver settings. The arm
  controller is Isaac Lab's reference high-PD configuration ([physics.md](../physics.md)).
- **Scene geometry**, beyond moving five objects out of the table they were authored inside.
  Objects authored interpenetrating their containers are *reported* (`SCENE_SETTLING`), not
  re-authored — one attempt to nudge a lemon ejected 27 unrelated objects and was reverted.
- **The benchmark's task set.** No task was added or removed. The bimanual rigs ship without
  benchmark tasks; users write their own.

## Status, read before citing a number

The offline suite (222 tests, no simulator) runs in CI and is green. Offline tests are not
evidence that a change behaves correctly at runtime: three changes in this fork passed every
unit test and were wrong on hardware. [changes.md](changes.md) therefore marks every row
**RUNTIME** (a recorded GPU run demonstrates it, judged by `scripts/verify_patches.py` with a
matching FAIL on the pre-patch baseline, or confirmed by a human on a linked episode),
**OFFLINE** (unit-tested or replayed over recorded logs, never executed on a GPU) or **NONE**.
About 25 of the 120 tasks have been run against the patched code, most of them with π0.5
only. Treat anything not marked RUNTIME as a proposal with evidence, not a verified fix.

Withdrawn and reverted changes are kept in the register with the measurement that disproved
them.

## Method

1. A human reviews a sample of episodes on the dashboard and writes down, in their own words,
   what the log got wrong.
2. Each complaint becomes an anomaly class with a detector, run over the whole recorded corpus
   to size it.
3. A change is written against a clean upstream checkout, with an offline test, and with the
   corpus measurement in its commit message.
4. The change runs on a GPU. `scripts/verify_patches.py` states, per recorded run, whether the
   claimed behaviour is present (PASS), the old behaviour survived (FAIL), or the run could
   not exercise it (N/A — never a pass).
5. The reviewer looks at linked episodes and says whether the flags are right. Proposals
   without an episode link were rejected.

Human verdicts on specific episodes live in `analysis/flag_labels.jsonl` and are replayed
as regressions by `scripts/flag_regression.py`.

## Relationship to upstream

RoboLab is developed by NVIDIA's Seattle Robotics Lab; the paper, leaderboard and original
repository are linked from the top-level [README](../../README.md). This fork tracks v0.3.1.
Changes that do not alter benchmark semantics are candidates for upstream pull requests;
changes that do are what makes this a separate, versioned benchmark and are listed as such.
Results produced with this fork should say so and name the tag.
