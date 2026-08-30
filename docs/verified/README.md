# RoboLab Verified

A fork of [NVIDIA RoboLab](https://github.com/NVlabs/RoboLab) v0.3.1. The fork keeps
upstream's 120 tasks, scenes, success predicates and physics defaults. It changes how episodes
are scored, how subtasks are credited, what the per-episode event log records, and what the
sensors and recordings capture. Each change is documented with the measurement behind it and
the test that covers it.

| | |
|---|---|
| [Changes](changes.md) | every change, one row each: what upstream did, what changed, whether a score can move, how it was verified |
| [Findings](findings.md) | what was wrong, with the evidence, including what is not fixed |
| [Migration](migration.md) | what a downstream tool has to know: new fields, new event codes, new files, the options that restore upstream behaviour |
| [Verification](verification.md) | how the evidence was produced and how to reproduce it |
| [Physics](../physics.md) | friction as a run parameter, and why the arm controller is unchanged |

## Why

Inspection of recorded episodes from the upstream harness found defects in the evaluation
itself. The measurements are in [findings.md](findings.md).

- Success fired on the first frame the predicate held. 88 of 88 successes in a 328-episode
  corpus ended while a target was still moving, 61 % of them faster than 10 cm/s. Whether the
  object stayed in the bowl was not observed.
- "Object grabbed" meant "object touched". No closure, lift or coupling was required. 64 % of
  logged grabs occurred with the hand open, and a fumble at a banana produced three identical
  lines per step. Reviewers stopped reading the event log.
- Five of 120 tasks could never report a grasp, because the grasp line was a side effect of a
  particular subtask ladder shape.
- `BananasOutOfBin` completed its subtask ladder at 0.07 s, before the arm moved, then flagged
  its own targets as wrong objects for the rest of the episode.
- Four of four `BowlStackingRightOnLeft` "successes" had stacked the wrong bowl. Nested
  identical bowls satisfy `object_in_container` in either order.
- 62 (task, object) pairs sink into their support at reset, before the robot moves; five of
  them by more than half a metre.

## What changed

Organised as in [changes.md](changes.md). Numbers are from the runs recorded in
[verification.md](verification.md).

**Scoring and termination.** A success is confirmed only once every target has been at rest
for 0.2 s. An object that is already still ends the episode immediately; a moving one delays
it. An episode ends as a failure once its success condition can no longer be met, i.e. a
required object or the destination has left the table. Open-top containment is capped at the
rim plus 5 cm. Nested containment requires the object to be above the container's base.

**Subtask crediting.** A ladder rung already true at reset earns no credit. Nothing is
credited before the scene has settled. A placement is credited only after the object comes to
rest. The final `score` is judged on the last frame; the live monotone value is kept as
`score_peak`. List-form ladders are evaluated as sequences.

**Event vocabulary.** A grasp is a carry: contact held for 0.2 s with the object coupled to
the hand while the hand moves. A pick is logged as `OBJECT_GRIPPED → OBJECT_CARRIED →
OBJECT_GRABBED_SUCCESS`. A failed attempt is one line with a count. A release is distinguished
from a drop by the commanded gripper channel. A bump of the task's own target is a neutral
note. A non-target delivered into the goal container is `WRONG_OBJECT_PLACED`. An object
moving with an open hand is `TOWED_WITHOUT_GRASP` and marks the episode as a physics artifact.
On the same tasks, π0.5's event count per episode fell from 45.5 to 31.0, and each remaining
line corresponds to one physical transition. Every line is stamped at its onset, with the
detection time recorded alongside.

**Contact sensing and recording.** Both Robotiq pads carry a contact sensor (upstream read
the left pad only). The HDF5 records per-pad contact force and object-to-destination contact,
so most flag rules can be re-evaluated on an existing recording.

**Physics.** Friction is a run parameter (`--friction`), applied at start-up and read back from
PhysX into every run directory. The default is upstream's authored materials. A 32-episode
sweep per condition found the success rate insensitive to a 4× change in μ; the behaviour
metrics were not ([physics.md](../physics.md)). Five objects that were authored inside their
support were moved to where they settle.

**Harness.** A `run_complete.json` marker, line-buffered stdout, `pytest` exit codes that
survive Isaac's shutdown, LFS pointer detection, and an install guide that works on a fresh
machine.

**Dashboard.** One transport under all camera tiles with the event timeline as the scrubber,
`?t=` permalinks, a subgoal checklist that fills with the playhead, a per-task page across
experiments, runs grouped by policy family, and HTTP range requests so that seeking works.

**Embodiments and policy backends.** A dual-Franka rig and a bimanual ViperX (ALOHA) rig. Each
is one articulation with per-arm end-effector recording, tracking wrist cameras, and unchanged
success predicates. A connector for running a pointing-capable VLM as a policy. A scripted
runner that exercises the two-arm stack end to end. No released checkpoint drives two arms
well.

**Tooling.** Offline audits that need no simulator: task definitions against scene spawn state,
scene interpenetrations, objects that sink at reset, and open-hand carries. A verifier that
evaluates each flag change as a PASS / FAIL / N/A predicate over recorded runs.

## What did not change, on purpose

- Task definitions and success predicates, except where a ladder contradicted its own success
  term (`GrabABagel`, `GrabAFruit`, `FoodPackingByColor`, five spatial Rubik's-cube ladders).
  Each such case is a row in [changes.md](changes.md) with its measurement.
- Physics defaults: friction, gravity handling, PD gains, solver settings. The arm controller
  is Isaac Lab's reference high-PD configuration ([physics.md](../physics.md)).
- Scene geometry, apart from the five objects moved out of the table they were authored
  inside. Objects authored interpenetrating their containers are flagged (`SCENE_SETTLING`)
  and left as authored. One attempt to nudge a lemon ejected 27 unrelated objects and was
  reverted.
- The benchmark's task set. No benchmark task was added or removed. The bimanual rigs ship
  with three smoke-test tasks under `robolab/tasks/bimanual/`, outside the benchmark set,
  loaded with `--task-dirs bimanual`.

## Status

The offline suite (222 tests, no simulator) runs in CI and is green. Offline tests are not
evidence that a change behaves correctly at runtime: three changes in this fork passed every
unit test and were wrong on hardware. [changes.md](changes.md) marks every row RUNTIME (a
recorded GPU run demonstrates it, judged by `scripts/verify_patches.py` with a matching FAIL
on the pre-patch baseline, or confirmed by a human on a linked episode), OFFLINE (unit-tested
or replayed over recorded logs, never executed on a GPU) or NONE. About 25 of the 120 tasks
have been run against the patched code, most of them with π0.5 only. Changes not marked
RUNTIME have not been demonstrated at runtime.

Withdrawn and reverted changes are kept in the register with the measurement that disproved
them.

## Method

1. A human reviews a sample of episodes on the dashboard and writes down what the log got
   wrong.
2. Each complaint becomes an anomaly class with a detector, which is run over the whole
   recorded corpus to size it.
3. A change is written against a clean upstream checkout, with an offline test, and with the
   corpus measurement in its commit message.
4. The change runs on a GPU. `scripts/verify_patches.py` states, per recorded run, whether the
   claimed behaviour is present (PASS), the old behaviour survived (FAIL), or the run could
   not exercise it (N/A). N/A is not a pass.
5. The reviewer checks the linked episodes and records whether the flags are right. Proposals
   without an episode link were rejected.

Human verdicts on specific episodes are in `analysis/flag_labels.jsonl` and are replayed as
regressions by `scripts/flag_regression.py`.

## Relationship to upstream

RoboLab is developed by NVIDIA's Seattle Robotics Lab. The paper, leaderboard and original
repository are linked from the top-level [README](../../README.md). This fork tracks v0.3.1.
Changes that do not alter benchmark semantics are candidates for upstream pull requests.
Changes that do alter semantics make this a separate, versioned benchmark and are listed as
such. Results produced with this fork should say so and give the tag.
