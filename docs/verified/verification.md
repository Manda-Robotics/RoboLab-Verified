# Verification

Every row in [changes.md](changes.md) carries a verification level. This page defines the
levels, names the tools that produce them, and states what has and has not been run.

## Levels

| level | meaning | produced by |
|---|---|---|
| RUNTIME | a recorded GPU run demonstrates the behaviour | `scripts/verify_patches.py` PASS on a patched run and FAIL on the same predicate over a pre-patch run of the same task; or a human confirming the flag on a linked episode |
| OFFLINE | unit-tested, or replayed over recorded logs | `offline_tests/`, `scripts/reflag.py` + `scripts/flag_regression.py` |
| NONE | written, never executed | — |

Two rules apply throughout:

1. N/A is not a pass. A run in which no episode lost a destination provides no evidence
   about `TARGET_LOST`. The verifier reports it as N/A, and the summary counts a patch that
   is N/A everywhere as untested.
2. A PASS requires a baseline FAIL. The same predicate is run over the pre-patch
   recording; a predicate that passes on both has not measured the change.

## Offline tests

`python -m pytest offline_tests` runs 222 tests with no simulator and no GPU in ~5 s. They
run in CI on every push (`.github/workflows/offline-tests.yml`). In addition to unit tests of
each change, they check three classes of bug that each cost a GPU launch to discover:

- every Python source under `robolab/`, `policies/`, `scripts/`, `dashboard/` must parse;
- every option a runner reads (`args_cli.<name>`) must be declared;
- every robot-config label must be shadowed to `None` on the generated scene config, or
  `InteractiveScene` rejects it at boot.

`tests/` is upstream's install-verification suite and boots Isaac Sim.

## Verifying a recorded run

```bash
scripts/verify_patches.py output/<run>                      # per-patch detail for one run
scripts/verify_patches.py --baseline output/<pre> output/<post>   # before / after, side by side
scripts/verify_patches.py --summary output/rc4_* output/rc5_*     # one table across runs; exit 1 on any FAIL
```

Predicates today: P61 (onset stamping), P71 (tracker grab is neutral), P72 (no attempts on
containers), P73 (no attempt right after a release), P74 (nothing credited before settle),
P75 (`PLACED_WITHOUT_LIFT` retired), P76 (a lost destination ends the episode), P77 (contact
force recorded; needs `h5py`), P79 (friction override applied; judged from the PhysX
readback only).

Each predicate reads only the run directory: `*/log_*.json`, `run_*.hdf5`,
`friction_applied.json`. The script does not import the runtime constants of the build it
judges, so it can be run on a recording made by another version.

## Replaying flag rules over recordings

```bash
scripts/reflag.py output/<run>                               # re-derive post-processable lines
scripts/flag_regression.py --output-dir output               # score against human verdicts
```

`analysis/flag_labels.jsonl` holds 23 labelled instances from the review sessions: run,
env, time, flag, verdict (`present` / `absent` / `missing` / `ambiguous`), and the reviewer's
own words. The labelled recordings (the rc-series runs) are not bundled; on another machine every label reports `no-episode`. `flag_regression.py` reports `pass` / `fail` / `cannot-check` per label and
explains every `cannot-check` (a rule that needs a re-run, a label that needs a recording with
pad forces). Current: pass 6, fail 1, cannot-check 13, ambiguous 2. The failure is
open: a quick grasp at ~88 s in `FoodPacking2Cans` env 2 that the detector still misses.

## Auditing definitions and scenes without a simulator

```bash
scripts/audit_task_definitions.py            # exit 1 on a conflict; notes are printed only; needs usd-core
scripts/check_scene_intersections.py         # authored overlaps; differential by design
scripts/check_rest_heights.py --tasks <T,...>  # drops / rises / rolls at reset, from recordings
scripts/find_sinking_objects.py output/<run> ...
scripts/find_open_hand_carries.py output/<run> ...
```

The task audit reads each task's step-0 poses before asserting a success-vs-ladder conflict.
Over all 120 definitions it finds no real conflict; two earlier candidates were withdrawn when
the spawn state was read. Candidates it cannot verify are listed as not asserted.

## What has been run

| run | code | policy | tasks | what it established |
|---|---|---|---|---|
| corpus (upstream v0.3.1) | — | π0.5, Cosmos3 | 41 | the findings; every corpus figure in [findings.md](findings.md) |
| rc2 | P01–P56 | π0.5 | 3 | event rate 45.5 → 31.0 per episode on the same tasks; release vs drop consistent with the commanded channel in 12 of 12 |
| rc3 | P01–P66 | π0.5 | 13 | the pre-Batch-14 baseline; every P71–P75 FAIL below is measured on it |
| rc4, rc5, rc6 | P01–P78 | π0.5 | 10 runs | P61, P71–P77 PASS on every run, with FAILs on rc3 for the same predicates; P72/P73 met their worst cases (`BananasOutOfBin`, `FoodPacking2Cans`) |
| rc7 | P01–P79 | π0.5 | 8 tasks × 4 friction conditions, 32 runs | P79 PASS on all 24 override runs; the friction sensitivity table in [physics.md](../physics.md); `TOWED_WITHOUT_GRASP` fired for the first time at runtime |
| bimanual YAM | rig + MolmoAct 2 client | MolmoAct 2 (Ai2, released) | 1 | parity task 7/8 in RoboLab (final acceptance run; 5–6/8 in earlier rig variants) vs 4/8 in Ai2's ManiSkill harness, same checkpoint; BananaInBowl, BBQSauceInBin, FoodPacking2Cans 0/4 each (policy hovers, picks the wrong object, or releases) |
| bimanual | rigs + scripted client | scripted | 1 | the dual-Franka stack turns end to end (6/6 clean lifts); ALOHA: rig turns, π0.5 base 0/6 |

About 25 of the 120 benchmark tasks have been run against patched code. Most runs are 4
episodes per task; the friction sweep is 32 episodes per condition, which resolves a
25-point difference in success rate but not a 10-point one. The runs exist to verify the
changes; they are too small to rank policies.

## What has not been run

- The other ~95 tasks under the patched harness.
- Any policy other than π0.5 under the patched harness (Cosmos3 and Gemini pointing ran on
  earlier revisions).
- The full 120-task re-run under the patched harness, and the per-task comparison of old and new scores that comes with it. Planned; not yet scheduled.

## Reproducing

The runs above were made on a single 48 GB GPU (L40) with Isaac Sim 5.1 / Isaac Lab
2.3.2.post1, π0.5 served from the `pi05_droid_jointpos` checkpoint
(`scripts/serve_pi05.sh`; the `--env DROID` convenience flag serves delta actions, which
this action space does not accept), one process per task, 4 envs. A whole-task run takes
5–17 minutes depending on the episode length. Sanity check for a new run: the median
hand-to-target distance over the run (1.3 cm on the corpus, 95 % within 20 cm). A value far
from that indicates a harness or checkpoint problem.
