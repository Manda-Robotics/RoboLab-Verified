# Newton failure triage and candidate-buffer repair (P128)

Validated on 2026-10-14 on the retained RTX PRO 5000 Blackwell 48 GB node.
The isolated repair checkout starts at `7857803`. The original sweep and its
recordings are preserved; the experiments below are separate follow-up evidence.

## Sweep triage

The canonical sweep covers 120 tasks in 47 scenes: all 120 physics probes passed,
119 policy episodes completed, and 45 succeeded. Of the 74 completed failures,
62 logged failed grasp attempts, 37 drops, 25 wrong-object grasps, and nine target
losses. These categories overlap. They are evidence for triage, not a conclusion
that every failure is caused by the policy. No claim is made that all failed
videos have received manual review.

One policy case, `FruitsOnionTask`, failed at the simulator level: the retained
retry reports **66,695 CCD candidates against 65,536 slots**. Three completed
failures carry explicit contact-artifact flags: `BagelsOnPlateTask`,
`FruitsOnPlateTask`, and `FruitsOrangesOnPlateTask`. A passive breakfast-table
recording also showed persistent apple motion during its 20-second window.

## Accepted repair: candidate storage

MuJoCo-Warp initially assigns the same capacity to broadphase candidate arrays
and contact outputs, although not every candidate becomes a contact. Its
broadphase increments the candidate count even after its arrays fill. The existing
RoboLab guard correctly aborts before CCD could read beyond those arrays.

The new scoped adapter grows only the three temporary candidate arrays and
reruns broadphase before narrowphase or solver side effects. SAP and NXN are
covered; primitive, heightfield and SDF launch dimensions cover the enlarged
candidate list. The original contact-output, constraint and CCD work budgets
remain unchanged. Growth is bounded to eight times the original contact budget;
an unsupported layout or larger requirement fails closed. No pair is silently
truncated. The existing pre-CCD guard remains active.

Default geometry, materials, contact response, gripper control and timing are
unchanged. This remains an eager-execution adapter for the pinned private
MuJoCo-Warp interface, not a general implementation for arbitrary versions or
CUDA graph replay. Recovering an aborted episode can change benchmark completion
and scores; this does not establish a policy-quality improvement.

## Verification

- Complete offline suite: **353 passed**, no skips.
- Complete Newton suite before adding retained GPU parametrization: **127 passed**,
  no skips. The final candidate regression file then passed all **four CPU/GPU
  cases**, including its two additional GPU cases.
- Real CPU and GPU synthetic probes, spheres and boxes, two worlds: 132 candidates
  exceed 32 contact slots. Candidate storage grows to 256 while all contact pairs
  and depths match a larger-buffer reference, including contacts beyond the old
  candidate cutoff.
- Saved onion scene on GPU: the final recorded state has **1,825 candidates and
  1,389 contacts**. With a deliberately reduced 1,607-slot budget, the old guarded
  path raises an overflow. The repair grows candidate capacity to 2,048 and
  reproduces every contact pair and depth (rounded to one micrometre) from the
  131,072-slot reference while preserving the 1,607 contact-output budget.
- A fresh public-runner pi0.5 onion episode succeeds after **499 steps / 33.27 s**.
  Independent audit passes JSON/HDF5 success agreement, finite values, step
  lengths, unit quaternions, both pads, 119 extra contact channels and video
  lengths. No artifact, motion-review or pre-satisfied flag is present.

The fresh onion run peaked at 14,207 candidates and did not need growth. Replaying
30 steps from the saved failed episode also stayed below the old limit in both
modes. **The original 66,695-candidate trajectory has not been reproduced**. The
smaller-capacity negative control proves the repair mechanism on the real scene;
the fresh success establishes integration and recording, not causation.

## Unresolved contacts and rejected experiments

Five-Hz geometric reconstruction of the flagged episodes matches recorded finger
poses within 1.24 micrometres. It finds substantial native pad/object penetration:

| Flagged contact | Worst penetration | Median sampled penetration |
|---|---:|---:|
| Bagel, first flagged segment | 18.55 mm | 9.84 mm |
| Bagel, later flagged segment | 10.52 mm | 6.46 mm |
| Pomegranate | 3.69 mm | 2.31 mm |
| Lemon | 8.89 mm | 5.18 mm |

This reconstruction uses the compiled collision geometry and recorded poses;
it is not a readback of the original run's GPU contacts. A subsequent live GPU
bagel replay confirms the problem. Four 12-second replays use the same original
actions and recorded initial state, with contacts sampled at 5 Hz:

| Experiment | Worst penetration | Median sampled penetration |
|---|---:|---:|
| Current native pads, 4 substeps | 19.40 mm | 10.71 mm |
| Solid bounding-box pads, 4 substeps | 21.42 mm | 13.43 mm |
| Native pads, 16 substeps | 18.53 mm | 11.07 mm |
| Native pads, 16 substeps, 1.1 ms contact time constant | 12.34 mm | 7.14 mm |

The box substitution worsens the result. Finer steps alone do not resolve it;
the stiffer response helps but remains unacceptable. None of these parameter or
geometry experiments is included in P128, and none of the three artifact flags
is cleared. A focused investigation of pad contact forces, solver convergence,
and constrained gripper motion remains necessary before changing defaults.

## Apple settling

The earlier 20-second passive probe showed a 36.10 mm tail XY span and
1.25 rad/s angular RMS for the apple. In a new 60-second run with unchanged
contact defaults, the apple rolls off its initial plate placement, then settles:
0.86 mm XY span from 5–10 seconds, 0.07 mm from 10–20 seconds, and 0.09 mm from
50–60 seconds; final-window angular RMS is about 0.04 rad/s. This different outcome
is a reproducibility concern, not evidence of a repaired placement or universal
settling guarantee. No global damping or friction increase was made.

## Retained evidence

The [verification JSON](evidence/newton-repairs-p128-verification.json) links the baseline count, negative control, source hashes,
regression results, native-contact measurements, and recorded follow-up results.
The archive also contains scripts, failed diagnostic attempts, HDF5/video files,
and the original onion model/state needed to reproduce the candidate comparison.
The archive is `/workspace/newton-repairs-20261014/repair-evidence.tar.gz`
(72,211,124 bytes, SHA-256
`f015efd96ca25570f771746b2fe55ee7d1bd6bb40d1033fef36491d690e874fd`).
It contains 168 hashed evidence files. All 78 preserved baseline files remain
unchanged. The downloaded
archive and every manifest entry are hash-verified locally.

The node is left running. No all-task sweep with altered contact parameters has
been performed, and no changes have been pushed.
