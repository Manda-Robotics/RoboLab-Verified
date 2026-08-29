# RoboLab Verified

A fork of [NVIDIA RoboLab](https://github.com/NVlabs/RoboLab) v0.3.1 that changes what
the benchmark *reports*, not what it simulates.

## Why

Running the benchmark and watching episodes frame by frame — rather than reading the
summary numbers — turned up defects in the evaluation harness itself. Some examples,
each with its measurement in `docs/VERIFIED_PATCHES.md`:

- **Five of 120 tasks could never report a grasp.** The "object grabbed" line was a
  side effect of a task's subtask ladder, so tasks whose ladder is a single placement
  condition — the whole stacking family — logged releases and drops but never a pick.
- **`BananasOutOfBin` completed its subtask ladder at 0.07 s**, before the arm moved,
  and then logged its own targets as *wrong* objects.
- **`GrabAFruit` logged "Completed subtask" on episodes the task itself scored as
  failures** — success needs a 50 mm lift, the ladder only needed contact.
- **In one `BlackItemsInBin` episode, 69 of 74 events were `GRIPPER_FULLY_CLOSED`** —
  the threshold read "at least 75 % closed", so a hand stalled on the object it was
  holding counted as fully closed.
- **62 (task, object) pairs sink into their support at reset**, before the robot moves;
  five of them fall 675–817 mm.

## What this fork does and does not do

**Does:** correct the event vocabulary and the subtask crediting rules; end episodes
that have become unwinnable; record the signals needed to re-check a flag decision
without re-running the simulator; add two bimanual embodiments and a VLM policy
connector; ship offline tooling that audits task definitions and scenes.

**Does not:** change the physics *defaults* -- the friction table, gravity, gains -- the
scene geometry, or any task's success condition. Friction becomes a run parameter
(`--friction`, off by default) so its effect can be measured rather than argued. Where a defect is in the scene rather than the harness, this fork
*reports* it — see "Known and deliberately unchanged" in `CHANGELOG.md`.

## Status — read before citing any number

This fork is **not release-verified**. The offline suite (221 tests) runs in CI and is
green, but offline tests are not evidence that a change behaves correctly at runtime:
two changes in this fork passed every offline test and were wrong on hardware (one
printed every grasp twice; one made a scene measurably worse and was reverted), and a
third died two minutes into its first GPU boot on a config-class rule no unit test saw.

`docs/VERIFIED_PATCHES.md` carries a port-readiness table marking each change
**RUNTIME** / **OFFLINE** / **NONE**, regenerated from `scripts/verify_patches.py`, which
turns each patch's claim into a PASS / FAIL / N/A predicate over recorded runs (an N/A is
never a pass). As of 2026-08-28 the event-vocabulary and crediting patches (P61, P71–P77)
and the friction run parameter (P79) are RUNTIME on the tasks they were run on; about 25
of the 120 benchmark tasks have been run against patched code, most of them on π0.5
only. Treat anything not marked RUNTIME as a proposal with evidence, not a verified fix.

Withdrawn claims are kept in the register rather than deleted, with the measurement
that disproved them.

## Layout

    docs/VERIFIED_PLAN.md      findings register — what is wrong, with evidence
    docs/VERIFIED_PATCHES.md   patch register — one row per change, with its measurement
    analysis/flag_labels.jsonl human verdicts on specific episodes, used as regressions
    scripts/                   offline audits, lints, and the re-flagger
    offline_tests/             the suite CI runs; no Isaac, no GPU

## Licence

Apache-2.0, as upstream. See `LICENSE` and `NOTICE`.
