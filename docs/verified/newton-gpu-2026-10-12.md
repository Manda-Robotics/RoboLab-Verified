# Newton GPU validation — October 11–12, 2026

The merged `newton-port` runs on an RTX PRO 5000 and passes the tested contact,
recording, and evaluation contracts. It remains experimental: a stationary
bottle-scene control reproduces a Newton-specific non-target pail fall.
Target success must not be treated as validation of every object in the scene.

## Build and environment

- RoboLab-Verified base: `f4ca3b1ca3eb3fd0746b4f6257c807504698b73e`.
- Isaac Sim 6.1.0.0; Isaac Lab source
  `4ed09d414e93f35b8573db8850f8b27572779176`.
- Newton 1.5.2, MuJoCo and MuJoCo-Warp 3.11.0, Warp 1.16.0,
  Torch 2.11.0+cu128, Python 3.12.12. Full versions are in `stack.json` and
  `install-record/final-freeze.txt` in the validation bundle.
- Ubuntu 22.04, RTX PRO 5000 Blackwell 48 GB, NVIDIA driver 595.84.
- One Vast instance, `55449995`, 160 GB disk, approximately $0.711/hour while
  running, excluding transfer charges. The instance was stopped after downloading
  and verifying the results; its installed environment is retained and disk charges
  continue.
- Policy: `xuningy/openpi` revision
  `aa6420561529593114160d05e5ad155792b272f3`, checkpoint
  `gs://openpi-assets-simeval/pi05_droid_jointpos`, separate Python environment.
- Standard DROID Newton preset, upstream friction, RTX cameras, environment seed
  0, one environment for policy runs. The policy server RNG was not reset between
  episodes. The PhysX control used the same Sim/Lab stack; this is not a comparison
  with the older PhysX benchmark stack or a success-rate estimate.

## Automated checks

- 313 offline tests passed. The initial fresh host skipped four Node-dependent
  checks; all four passed after installing Node 22.14.0.
- 61 Newton unit tests passed.
- 237 simulator integration tests passed after correcting three stale assertions.
- A new actual MP4 encode/decode test passed; all six video tests passed.
  This makes 612 distinct passing tests across the exercised suites.
- A deliberately failing pytest probe returned exit status 1 and wrote its JUnit
  report after the shutdown-hook fix. The unmodified harness had returned status
  0 despite three reported failures, so the initial process exit alone was invalid
  evidence of success.
- Newton cube smoke: 120 steps each at one and two environments; finite state,
  nonconstant RGB from all three cameras, supported objects, and contact queries
  passed. The two-environment cube-into-bowl drop also passed for 120 steps.
- The final two-environment contact audit passed. Raw contact matrices were
  finite and had shape `(N, B, M, 3)`. It exercised both gripper labels and all
  seven extra gripper-body labels, correct object lookup, positive dense-recorder
  channels, zero after separation, and no response in the untouched environment.
- A 0.2 kg cube's final audit support-force ratios to `m*g` were 0.99930 and
  1.00003 (acceptance interval declared in the fixture: 0.7–1.3). Lifting only
  environment 0 cleared its table contact while environment 1 retained contact.
- Both-pad and extra-body HDF5 channels matched 50 live samples per environment
  during support/separation; two subsequent resets completed with finite joints.

The injected pad/body probes deliberately overlap a cube with collision geometry.
Their large transient forces test identity, routing, and clearing; they are not
physical force calibration or successful grasps. Static support provides the
separate force-scale check. Early audit attempts had two fixture bugs (waiting
long enough for the lifted cube to land, and using USD paths for MuJoCo's sanitized
body names); those were corrected without changing simulator behavior.

## Policy results

| Backend | Task | Runs | Result | Episode steps |
| --- | --- | ---: | --- | --- |
| Newton | RubiksCubeTask | 2 | 2 successes | 155, 284 |
| Newton | BananaOnPlateTask | 1 | success | 173 |
| Newton | OneBottleInSquarePailTask | 1 | target success; collateral scene failure | 297 |
| PhysX | RubiksCubeTask | 1 | success | 156 |

All five completed recordings contained finite numeric data. The cube runs
recorded both-pad contact, a 13–15 cm lift, release, bowl contact, and low final
object speed. The second run recorded a drop and successful regrasp. Cube pad
force peaks were about 46–54 N; those values are measurements, not hardware
calibration. The bottle recording ended with no pad contact, but its final
bottle/pail contact Boolean was false; that task's success predicate does not
require contact. Do not infer supported placement from that success flag alone.

Recorded EE velocities were nonzero. For the cube runs, reported peak speeds
were 0.245/0.199 m/s versus pose-difference peaks of 0.246/0.193 m/s. The historical
all-zero EE-velocity problem did not reproduce in these recordings. This is a
sampled consistency check, not validation of every velocity-derived metric.

`verify_patches.py` found no failures in the exercised event checks: P61, P71,
P74, P75 and P77 passed on every run directory; P72/P73 were exercised by the
banana run and passed. P76 was not exercised. P79 was N/A with upstream friction;
Newton friction overrides remain unsupported. N/A is not a pass.

## Remaining scene failure

In `OneBottleInSquarePailTask`, the policy run recorded `plasticpail_a02` leaving
the scene at step 62 and `utilityjug_a01` at step 192. Paired no-policy controls
held the initial arm target for 120 steps (8 simulated seconds), with identical
authored initial object positions:

| `plasticpail_a02` | Newton | PhysX |
| --- | ---: | ---: |
| Net displacement | 0.934 m | 0.0246 m |
| Height change | -0.762 m | -0.0245 m |
| Final root height | -0.626 m | 0.112 m |

This reproduces the pail instability without policy motion. The generic smoke
program's default 1 m displacement bound passes both controls, so its pass marker
alone does not establish scene stability. The saved trajectories expose the fall.
The utility jug's policy-run failure was not independently attributed.

Investigate the pail/rack collision import, support geometry, and contact/material
behavior before promoting this scene. The tiny red-onion hull issue, broad scene
coverage, long-duration stability, force/inertia calibration, and success-rate
parity remain outside this validation. No five-second post-success policy hold was
run; ordinary Verified rest gates were used.

## Fixes made during validation

1. Declare `imageio[ffmpeg]` so fresh installs can record MP4 videos. All initial
   policy attempts failed on the missing backend; fresh reruns with the dependency
   completed. Add an encode/decode regression test using the real backend.
2. Run pytest's reporting hooks before closing Kit and preserve a nonzero exit
   with `os._exit` after flushing reports. Validate with a deliberate failure.
3. Update hull tests for Verified's existing rim-height cap and test the actual
   launcher compatibility wrapper instead of requiring the removed Lab 2 option.

These fixes do not retune the Newton solver or modify task success rules.

## Reproduction and evidence

Follow the pinned installer in [the Newton guide](../newton.md). On hosts with
slow package downloads, set `UV_HTTP_TIMEOUT=300` and reduce concurrent downloads;
the first installation attempt here timed out while downloading a wheel.

The external validation bundle is named `newton-verified-20261011`. It contains
commands/scripts, JUnit XML, stdout, package resolution records, source patch,
MJCF exports, camera PNGs, policy videos, HDF5, event/phase/tow logs, numerical
recording audits and `bottle-static-comparison.json`. The final diagnostic script
is `contact-audit-final.py`; use a fresh `--output` directory and `--headless`.
The bundle is kept outside Git because it contains generated evidence. The
23 MB archive, `newton-validation-results.tar.gz`, was verified after download:

```text
SHA256 73564b242690a70f6c24ea321992f5f79449c24100ceb2f253893796879db094
```

```bash
python -m pytest offline_tests -q
python -m pytest --confcutdir=tests/newton tests/newton -q
python -m pytest tests --ignore=tests/newton -q
python examples/run_newton_smoke.py --headless --physics newton_mjwarp \
  --task RubiksCubeTask --num-envs 2 --steps 120 --output output/newton_smoke
python policies/pi0_family/run.py --policy pi05 --headless \
  --physics newton_mjwarp --friction upstream --task RubiksCubeTask \
  --seed 0 --num-envs 1 --num-runs 2 --video-mode sensor \
  --output-folder-name newton_cube_validation
```
