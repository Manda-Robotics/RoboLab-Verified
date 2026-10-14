# Newton contact compliance and release follow-up

Tested on 2026-10-14 on the retained RTX PRO 5000 node, using the production
runtime at `7a2a722`. This follow-up qualifies the contact conclusions in
[P128](newton-repairs-2026-10-14.md). No physics defaults or benchmark scores
were changed.

## Finding

The measured overlap is real overlap of rigid collision geometry, but overlap
alone does not establish a physics defect. MuJoCo permits soft contact; its
contact model can approximate compression without deforming the visible mesh.
The [MuJoCo solver-parameter documentation](https://mujoco.readthedocs.io/en/stable/modeling.html#solver-parameters)
describes that distinction. These tests establish unloading behavior, not a
calibrated model of bagel or fruit material, and do not establish author intent.

Pomegranate and lemon unloaded cleanly. The bagel showed substantial one-finger
retention: one shorter withdrawal remained loaded at the end, while replaying
from the beginning eventually released. Both longer withdrawals along the
finger axis released. Thus the tested bagel contact is not permanently trapped,
but its large overlap and delayed, configuration-dependent release still need
review. It would be premature to label every flagged episode either physically
invalid or physically validated.

## Controlled experiments

Restore the recorded post-step state, hold its original command for 2 seconds,
open the gripper for 1 second, withdraw over 3 seconds, and wait another
3 seconds. No state is teleported after initialization. The first four trials
command 80 mm along the average loaded contact-normal direction. The two bagel
controls command 120 mm opposite the finger-extension axis, verified from the
compiled Robotiq geometry. These controls vary both direction and distance;
they do not isolate which change caused release.

Two bagel trials instead replay all 79 original actions from the recorded
initial scene before applying the diagnostic sequence. This checks sensitivity
to restoring a loaded state without its solver history. Recorded state indices
are 78 for bagel, 464 for pomegranate, and 1153 for lemon (post-step times
5.267, 31.000, and 76.933 seconds respectively).

| Trial | Peak overlap during initial hold | Peak summed normal load during hold | Measured base travel during withdrawal | Last active robot contact after diagnostic start | Contact-free at end |
|---|---:|---:|---:|---:|---|
| Bagel, saved state | 16.41 mm | 20.74 N | 71.45 mm | Still active at 9.00 s | No |
| Pomegranate, saved state | 4.53 mm | 31.04 N | 63.24 mm | 3.87 s | Yes |
| Lemon, saved state | 6.32 mm | 33.73 N | 62.48 mm | 3.67 s | Yes |
| Bagel, original prefix | 16.42 mm | 18.43 N | 65.36 mm | 7.53 s | Yes |
| Bagel, axial withdrawal, saved state | 16.41 mm | 20.74 N | 96.41 mm | 6.27 s | Yes |
| Bagel, axial withdrawal, original prefix | 16.41 mm | 18.42 N | 98.34 mm | 6.07 s | Yes |

Peak depth and peak load are separate maxima, not necessarily simultaneous.
Loads sum positive normal forces over active robot/object contacts; they are
not net object force or motor effort. Actual travel differs from commanded
travel because this is joint-target control under load. Contacts and forces
are sampled at the 15 Hz control rate, not at every solver substep. For the
fruits, the entire final 3-second wait was contact-free. Bagel controls released
during that wait, so their contact-free windows are shorter.

![Measured force and overlap during unloading](evidence/newton-contact-unloading.png)

The original episodes also show a positive force/depth association, strongest
for pomegranate (Pearson r = 0.885; first bagel segment 0.348; lemon 0.220).
Those samples combine reconstructed geometry with recorded pad force and vary
in pose and contact topology. They cannot identify a material stiffness.

## Contact law and interpretation limits

Live GPU contacts confirm the common response: `solref = [0.0045, 1]`,
`solimp = [0.99, 0.999, 0.001, 0.5, 2]`, and `includemargin = 0`.
All runs use 120 Hz simulation, four Newton substeps (480 Hz solver rate),
and 15 Hz control. This is a shared response, not food-specific calibration.
The pinned MuJoCo-Warp collision code treats `geom_gap` as an expanded detection
envelope; it is not an allowed-penetration offset in the contact force law.

Sampled solver iteration counts stay at six or fewer, without a divergence or
overflow reported by these diagnostics. Iteration count alone is not proof of
adequate physical accuracy or a measured convergence residual. The bagel video
shows contact with its outer ring; this evidence does not establish that the
finger simply passed through the central hole. Nor does the rigid model
explicitly represent puncture, tearing, or permanent deformation.

The appropriate next validation is a controlled indentation/unloading curve
with an explicit material acceptance range, followed by a fresh policy run
if contact parameters change. The present evidence does not justify globally
stiffening contact solely to minimize geometric overlap. Original detector
flags remain preserved as review signals; no task was rescored.

## Verification and retained evidence

All six diagnostic runs complete. Their HDF5 files contain 968 total action
steps, finite numeric data, and matching lengths for both camera recordings
(1,936 total video frames). All four restored post-step states match their
recorded state exactly on readback. Withdrawal targets pass CPU kinematic
preflight. The 292 tracked RoboLab Python runtime files match the local commit,
and all 78 preserved baseline files retain their previous hashes.

Early diagnostic attempts failed because of an output-directory error and a
coordinate-frame error in the diagnostic IK controller. They are retained but
excluded from the six completed trials. These were diagnostic harness failures,
not additional benchmark failures. The corrected controller uses a world-frame
orientation error with the world-frame Jacobian.

The [verification JSON](evidence/newton-contact-compliance-verification.json)
contains phase statistics, contact laws, recording audits, source integrity,
and limitations. The downloaded archive contains 138 hashed files including
scripts, failed attempts, results, models, HDF5 trajectories and videos:

`/workspace/newton-compliance-20261014/compliance-evidence.tar.gz`

Size: 156,047,745 bytes. SHA-256:
`cc1618481f467e758ef57ba83b0f927668b00d0312e232591544b6144909e96d`.
The archive and every manifest entry were verified locally. The report plot is
a separately retained presentation revision with corrected labels; its source
and digest are included in the verification metadata.

The node remains running. This is a contact diagnostic, not a new policy sweep
or a claim that all benchmark failures have been resolved.
