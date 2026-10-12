# Experimental Newton integration

This branch combines RoboLab-Verified at `790c1a5` with the four Newton commits
from `ymetz/RoboLab`, ending at `56d461a`. Their common base is `82725b5`;
Verified has 186 commits after that base. The merge retains both histories.
Only `pyproject.toml` and `robolab/eval/runner.py` had textual conflicts.

**The small-sample GPU/contact integration checks pass; the backend remains experimental.**
See the [October 11–12 GPU report](newton-gpu-2026-10-12.md),
[passive-contact investigation](newton-rocking-2026-10-12.md), and
[validated settling fixes](newton-settling-2026-10-12.md). The latest contact
work passed 76 Newton unit cases locally, 238 simulator/integration cases on
an RTX PRO 5000, replicated-world contact/recording checks, and paired bottle
and cube grasp/release replays. In one reused scene with restored starting states,
immovable-scenery filtering stops the shelf disturbance, and support-local
rolling resistance reduces the sideways jug's late rocking from 14.65 to 0.031 mm.
These are now the experimental Newton defaults. The [wider scene audit](newton-mesh-audit-2026-10-12.md)
adds narrow mesh-import repairs and compact Newton contact storage. That
follow-up passed 98 Newton tests, 313 Verified offline tests, 238 simulator/integration
cases, the two-world contact/recording audit, bottle/cube recorded-action replays,
and five 20-second passive scenes. The [compiled shelf follow-up](newton-shelf-2026-10-13.md)
replaces the rack's 35,903 collision pieces with 44 prebuilt convex pieces. The
three shelf task configurations initialize in 18–19 seconds with less than 8 GiB
peak host RSS, versus about ten minutes and 92 GiB for the detailed rack. All
106 Newton regression tests, three 20-second shelf task runs and eight isolated
GPU drop/slide probes pass; body mass and inertia are preserved exactly.
This is sampled integration coverage, not a complete benchmark or fresh-policy
validation. See the audit for exact revisions, settings, limits and evidence.
The RUNTIME evidence in the Verified change register was collected with PhysX and
does not transfer to Newton. PhysX is still the default. Select Newton with
`--physics newton_mjwarp` in a shared evaluation runner.

## What was retained and adapted

- Verified's success-rest gate, final-frame scores, grasp/carry tracker,
  termination rules, scene corrections, both pad sensors, dense recordings,
  bimanual rigs, dashboard and run-completion marker are retained.
- Newton's pinned source-stack installer, robot inertia/gravity overlays,
  gripper limiter, contact preset, capacity guard, renderer compatibility,
  backend output marker and diagnostic scripts are imported.
- Both CLI argument sets and Verified's `usd-core` test dependency are retained.
  Verified-only policy launchers also use the launch-flag compatibility helper.
- Newton's object-specific contact queries already compose with Verified's
  two-pad alias. The extra-body recorder needed a separate fix: its batch-column
  labels assume requested object order, which Newton does not guarantee. Enabled
  extra-body sensors now become one sensor per named object under Newton. Their
  magnitudes are summed over that object's resolved filter bodies, retaining the
  `contact_body/<body>__<object>` recording keys. PhysX keeps batch sensors.
  Force arrays are converted through the existing tensor compatibility helper.
- Verified's `--friction` overrides use PhysX material events. Newton accepts
  only `--friction upstream`, with explicit rejection of uniform/table overrides.
  Its `friction_applied.json` records `status: unsupported` and the backend;
  verification P79 returns N/A. This is not a Newton coefficient measurement.
- `ROBOLAB_ROBOT_USD` variants, including Verified's solid-pad asset, are rejected
  by the Newton preset. Newton's single-hull fingertip setting is a separate
  experimental treatment; the PhysX towing-fix evidence does not validate it.
- CI runs both the Verified offline suite and the Newton CPU suite. The latter
  uses `--confcutdir=tests/newton` to avoid starting Isaac Sim.

## Existing evidence and remaining gaps

The source branch's [Newton log](../newton.md) reports three fresh pi0.5 seed-0
successes on September 15 (cube/bowl, banana/plate, bottle/pail), each with five
seconds of retained success, plus one ordinary-runner cube success. It also
reports observation/action conversion checks and isolated gripper diagnostics.
Those are reported results from RoboLab, not runs of this merged Verified build;
the external recordings were not reviewed during this integration.

That log also leaves open a tiny red-onion hull compilation failure, a non-target
pail leaving the scene, earlier obscured camera scenes, uncalibrated inertias and
contact forces, and zero recorded end-effector velocities despite motion. The
October GPU run did not reproduce the zero-velocity issue and checked recorded
velocities against pose differences. It established sampled two-environment
contact routing, force clearing, recording and repeat resets. Long runs, broader
scene coverage, calibration and policy success-rate parity remain unestablished.

The added CPU tests exercise reversed object ordering, separate environments,
multiple filter bodies, release to zero, tensor conversion, unchanged PhysX
recording keys, and unsupported-friction provenance. They cannot prove that the
simulator produces correct forces or that Verified's thresholds are appropriate
for Newton.

## Reproduce the local checks

Integration checks on October 11, 2026: **313 Verified offline tests passed,
61 Newton CPU tests passed**, and the task-definition audit reported zero
conflicts (10 notes and four spawn-state checks not asserted without recordings).
No simulator was run on this macOS host.

Use a CPU Python 3.12 environment with `pytest`, `numpy`, `scipy`, `h5py`,
`pyyaml`, `usd-core` and `torch`, and materialize the Git LFS assets/fixtures:

```bash
python -m pytest offline_tests -q
python -m pytest --confcutdir=tests/newton tests/newton -q
python scripts/audit_task_definitions.py
```

The offline suite's fake policy server needs localhost socket access. Do not
interpret a sandbox bind failure as a simulator failure.

## GPU validation before benchmark use

Follow the pinned installation in [the Newton guide](../newton.md): Isaac Sim
6.1.0.0 and Isaac Lab source revision
`4ed09d414e93f35b8573db8850f8b27572779176`. Do not re-sync the older `isaac60`
wheel extra over that environment. Use the same pinned stack for a solver-control
comparison; keep legacy PhysX wheel stacks in a separate environment. Give every
attempt a fresh output directory.

1. Run the no-policy smoke on `RubiksCubeTask`, first with one environment, then
   two. Check exit status **and** `ROBOLAB_SMOKE_PASS`/`smoke_result.json`, RGB
   visibility, no early success under a stationary arm, supported objects and
   finite state. Inspect the geometric contact samples; a sampled minimum is
   not a bound on transient penetration.

   ```bash
   .venv/bin/python examples/run_newton_smoke.py --headless \
     --physics newton_mjwarp --task RubiksCubeTask --num-envs 1 --steps 120 \
     --output output/verified_newton_smoke_1
   .venv/bin/python examples/run_newton_smoke.py --headless \
     --physics newton_mjwarp --task RubiksCubeTask --num-envs 2 --steps 120 \
     --output output/verified_newton_smoke_2
   ```

2. Reproduce the free-space and loaded-block gripper diagnostics from the guide.
   These explicitly tune the diagnostic config; neither proves that a free object
   can be grasped. Then test the production preset on a free object through
   approach, close, lift, hold and release.
3. Isolate left-pad, right-pad, wrist/knuckle and object/destination contact.
   Deliberately touch different objects in different environments. Compare raw
   pairwise sensor forces, `WorldState` queries, `contact` and `contact_body`
   HDF5 channels and the event log. Check no swapped labels or cross-environment
   contact, sensible force direction/magnitude, and zero after separation.
   Static support should agree with the object's weight within a predeclared
   tolerance; video alone cannot validate force scale.
4. Replay the three reported simple tasks with Verified's success-rest gate,
   both-pad recording and carry/release logic enabled. Audit carry/drop/tow and
   placement events against HDF5 and video; explicitly observe stability after
   release. Include reset/repeated-run checks and compare EE velocities to pose
   differences. A reported success alone is insufficient.
5. Triage the known hull issue and observed scene differences, then compare a
   declared task/seed set against PhysX using the same policy and observation/action
   settings. Report successes, grasp/slip behavior, contact evidence and non-target
   object motion separately.
   Do not infer parity from matching seeds or the earlier three successes.

Example policy launch after the smoke/contact checks, with the policy server
already running:

```bash
.venv/bin/python policies/pi0_family/run.py --policy pi05 --headless \
  --physics newton_mjwarp --friction upstream --task RubiksCubeTask \
  --seed 0 --num-envs 1 --num-runs 1 --remote-host localhost --remote-port 8000 \
  --video-mode sensor --output-folder-name verified_pi05_newton_cube
```

The [GPU report](newton-gpu-2026-10-12.md) records the merged-build results and
remaining unknowns. Preserve the exact repository commit, package versions,
resolved config, MJCF where available, videos and HDF5 for every validation run.
