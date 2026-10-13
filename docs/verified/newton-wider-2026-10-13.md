# Wider Newton validation — October 13, 2026

The final Newton preset completed **47 distinct scene layouts**, representing
120 task definitions, on one RTX PRO 5000. Fifteen layouts ran for 20 simulated
seconds and the remaining 32 for six seconds. Fresh pi0.5 succeeded on cube,
banana and bottle, but failed on all three tested shelf tasks after robot–rack
collisions. Four parallel cube policy episodes all succeeded and exported
independent recordings. Default contact buffers support the tested batches up
to 16 environments; explicit smaller buffers also pass at 32.

These are sampled integration and capability results. They do not establish
complete benchmark coverage, a policy success rate, PhysX parity or calibrated
hardware dynamics. PhysX remains the default backend.

## Changes and diagnosed failures

- **Orientation:** integrated the intent of `fix-dome-rot` at
  `e86eb12f825933d4f5b494ee9cfaee045d92b71d`. Dome lights, frame offsets and
  affected configurations now explicitly use WXYZ identity quaternions. An AST
  audit replaces a regex scan. The GPU check confirms the live dome identity;
  changing backdrop orientation leaves the frozen physics state bitwise
  unchanged. Bimanual pose export also uses the current tensor compatibility
  helper and WXYZ orientation, including translated/yawed robot roots.
- **Desk startup:** deferred the redundant CPU staging contact solve when
  constructing the GPU/native-contact solver. It previously overflowed MuJoCo's
  CPU arena at 35,723 contacts. Kinematics and equalities still initialize on
  CPU; the CONTACT flag is restored before upload. A 240-step GPU comparison
  checks loaded and falling boxes against the original initialization,
  including trajectories and final constraint forces.
- **Keyboard collision asset:** replaced 5,503 native CoACD fragments with 110
  prebuilt component hulls; the desk model drops from 8,832 to 3,439 geoms.
  Key surfaces and gaps remain, closing only open keycap bottoms. Geometry
  checks bound native-face-centroid inset to 0.10 micrometres and check the
  outward convex surfaces. All 33 body masses, inertias, inertial frames and
  authored poses in the compared desk models are bitwise identical.
  Source fingerprints and geometry guards prevent silent reuse after changes.
  Rendering is unchanged. See the [asset README](../../assets/objects/hot3d/README.md)
  for provenance, license, tolerances and rebuild commands.
- **Static extras:** added lazy, read-only USD pose views for `AssetBaseCfg`
  extras, which previously broke the cleaning-shelf task. Tests cover translated
  and rotated environments, scale, cache invalidation and unchanged USD data.
- **Authored scene defects:** deactivate abandoned bowl/banana parents in
  `objects_around_table` that remained active 123 m below the table because old
  edits named obsolete child meshes. Remove their unused contact channels.
  Raise `orange_02` in `fruits_out_of_basket` out of a roughly 56 mm tabletop
  penetration to 0.5 mm clearance, based on transformed collision vertices.
  Horizontal position and orientation remain. These corrections affect both
  backends.
- **Convex mesh rocking:** enable native multi-point contacts
  (`enable_multiccd=True`). Newton 1.5.2 defaults to single-point contacts for
  convex mesh pairs, so broad flat supports can behave like pivots. In the
  stacked-cube control, this was spontaneous motion: zero initial velocity,
  no robot contact, and only table/cube and cube/cube contacts. The change
  reduces the top cube's late XY span from **20.631 to 0.196 mm**.

The pinned Isaac Lab config omits the multi-point-contact option. A scoped
constructor bridge enables it only for native GPU contacts and restores the
constructor on exit/error. An undeclared dynamic config field is insufficient:
`cfg.copy()` drops it on this pinned version. CPU and non-native solvers keep
their original options. Newton handles its zero-margin requirement. The default
can be reverted with `enable_multiccd=False` for a diagnostic control.

![Cube stack contact comparison](evidence/newton-wider-stack-contact-comparison.png)

The isolated stack comparison verifies bitwise equality of mass, inertia, COM,
inertial/body frames, initial generalized poses, mesh vertices/faces, geom
shapes/transforms/materials/margins and contact-response arrays. The only changed
disable flag is MULTICCD. A repeat using the production adapter gives a 0.215 mm
top-cube span. Middle-cube span falls from 15.960 to 0.107 mm and bottom-cube span
from 2.832 to 0.045 mm in the isolated comparison. No added mass or friction
produced this improvement.

## Scene coverage and residual motion

The first six-second sweep had 44 successful exits and three failed exits:
desk CPU staging overflow, static-extra pose access, and a probe that incorrectly
assumed every task had exactly one success term. Those issues were corrected.
Trajectory review, beyond checking exit codes, found the below-floor props and
embedded orange. All 47 layouts were then rerun with the final multi-point
preset; five additional long probes bring the total to 52 final trajectories.

Independent checks recompute finite poses/velocities, quaternion norms, expected
step counts and duration, workspace bounds, late-motion metrics and nonconstant
RGB. Contact samples contain no immovable–immovable contacts, and no scene
reports an unearned passive task success. All 47 final scene frames were also
reviewed. One representative per layout does **not** exercise all 120 task
configurations, resets, variations or seeds. The representative-task coverage list is in
[evidence/newton-wider-summary.json](evidence/newton-wider-summary.json).

Selected 20-second cases below use the largest object XY span over 17–20 seconds;
the table is a measured residual, not a universal pass threshold.

| Representative task | Largest late XY span |
|---|---:|
| UnstackRubiksCubeTask | 0.974 mm |
| FoodPacking3BoxesTask | 0.176 mm |
| FoodPackingByColorTask | 0.314 mm |
| AppleAndYogurtInBowlTask | 0.435 mm |
| PutTwoMugsOnShelfTask | 0.280 mm |
| FruitsGreenLimesOnPlateTask | 0.662 mm |
| AnimalsInBinTask | 0.299 mm |
| BBQSauceInBinTask | 1.084 mm |
| SauceBottlesCrateTask | 1.055 mm |
| BananaInBowlTask | 0.209 mm |
| BananasInBinOneMoreTask | 0.785 mm |
| BigPumpkinInBinTask | 1.844 mm |
| CookingClearPlateTask | 0.484 mm |
| FruitsMovingOrangeOrLimeTask | 1.206 mm |

FoodPacking3BoxesTask improves from 9.616 to 0.176 mm and
FoodPackingByColorTask from 2.166 to 0.314 mm relative to the earlier preset.
Longer checks distinguish settling from persistent motion: the pumpkin scene's
red onion falls from an authored collision-bottom height about 6 cm above the
table, tumbles and then settles (0.774 mm late span). Its avocado retains 1.844 mm
of late movement. Some crowded fruit/bottle scenes still show millimetre-scale
motion; this is not a claim of perfect sleeping or calibrated rolling resistance.
The largest residual in the six-second samples is 2.845 mm in
ReorientWhiteMugsTask; longer or manipulated episodes can expose further issues.

![Additional scene examples](evidence/newton-wider-additional-scenes.png)

Startup measurements for the compiled-keyboard tasks below were isolated and
preceded the multi-point-contact change. The final sweep rechecks their physical
behavior with multi-point contacts. Time includes Kit and cameras; ordinary
native hull caches were warm. These are not clean-machine installation times.

| Task | First recorded state | Peak host RSS |
|---|---:|---:|
| AppleAndYogurtInBowlTask | 35.18 s | 8.85 GiB |
| MarkerInMugTask | 29.14 s | 8.21 GiB |
| BlackItemsInBinTask | 28.15 s | 8.12 GiB |

The earlier native-keyboard attempt took 172.95 s and 19.19 GiB, but ran with
other GPU work; it is not a valid isolated timing ratio. The compiled shelf
from the [preceding audit](newton-shelf-2026-10-13.md) remains enabled, retaining
its 20 kg mass, exact inertia and original render mesh.

## Fresh policy evaluation

All six single-environment episodes used the ordinary shared runner, the public
`gs://openpi-assets-simeval/pi05_droid_jointpos` checkpoint, seed 0,
`--friction upstream`, sensor videos, both-pad contacts and seven extra-body
recording channels. The policy server remained alive across these six episodes.
The final multi-point-contact results are:

| Task | Success | Steps | Simulated time | Observed outcome |
|---|---|---:|---:|---|
| RubiksCubeTask | Yes | 138 | 9.20 s | Grasp, carry and release into bowl. |
| BananaOnPlateTask | Yes | 167 | 11.13 s | Grasp, carry and release onto plate; one table hit. |
| OneBottleInSquarePailTask | Yes | 686 | 45.73 s | Three grasps and two drops before successful placement. |
| TakeMugsOffOfShelfTask | No | 60 | 4.00 s | Arm contact tips the rack; mugs fall. |
| PutMugsOnShelfTask | No | 327 | 21.80 s | One placement, then another grasp and robot–rack collision; rack falls. |
| PutBowlOnShelfTopTask | No | 50 | 3.33 s | Gripper hits/lifts rack before a bowl grasp; rack falls. |

These are **three successes out of six individual examples**. Videos and HDF5
show robot contact before shelf motion; passive shelf cases remain stable.
The rack was not anchored or given artificial inertia to help the policy.
Robot–shelf interaction is still unresolved evaluation work: these failures
cannot establish whether the limiting issue is policy robustness or a contact
model mismatch. A matched PhysX/control study is needed to distinguish them.
The automated physics-artifact flag stayed false, which alone proves neither.

All six runs emitted completion markers. Independent HDF5 audits checked every
numeric dataset for finite values, per-step lengths, WXYZ quaternion norms,
workspace bounds, nonzero EE velocity and contact channels. Sensor videos have
one fewer frame than HDF5 steps because the runner omits the terminal frozen
frame. The videos and event histories were reviewed for all six cases.
Earlier single-contact controls (two successes out of six) remain in the raw
archive and are not mixed into the final results.

A separate **four-environment cube policy run succeeded in all four worlds**,
at steps 167, 150, 160 and 146 (11.13, 10.00, 10.67 and 9.73 simulated seconds).
Each `demo_0`–`demo_3` contains exactly its own terminal step count, matching its
success flag, and each video has exactly one fewer frame. The four initial
local object poses match, all 83 numeric datasets per demo are finite, and the
21 extra-body/object channels are present. Different completion times exercise
per-environment inference exclusion and recording finalization. This does not
prove that physical bodies remain immobile after an environment is marked done.
The runner reported 71.811 s for inference, stepping and video writing together,
excluding startup. The resolved environment contains three cameras per world;
the saved policy videos show the shoulder and wrist views. The scaling benchmark
below excludes policy inference.

## Parallel-environment scaling

Each entry measures 60 synchronized control steps after 20 warmup steps, with
no other GPU workload. `normal_rtx` uses ordinary `env.step`, cameras, managers
and recorders. `physics_and_state_only` also omits managers and recorders, so it
is not just a camera-off comparison. Three cameras per environment are enabled:
egocentric 480×864, shoulder 720×1280, and wrist 720×1280 (height×width). **Policy inference is
excluded.** Rates below are aggregate environment control steps per wall second.

| Scene | Envs | Full environment | Physics + state | Peak GPU | Peak host RSS |
|---|---:|---:|---:|---:|---:|
| Cube | 1 | 3.39 | 3.75 | 4.19 GiB | 7.68 GiB |
| Cube | 2 | 6.56 | 7.37 | 5.67 GiB | 7.78 GiB |
| Cube | 4 | 12.19 | 14.59 | 8.44 GiB | 7.95 GiB |
| Cube | 8 | 21.84 | 28.65 | 14.75 GiB | 8.42 GiB |
| Cube | 16 | 37.47 | 56.01 | 24.63 GiB | 9.37 GiB |
| Cube | 32, default buffers | GPU allocation failure | — | — | — |
| Cube | 32, smaller buffers | 55.35 | 111.04 | 33.22 GiB | 11.30 GiB |
| Shelf | 1 | 2.62 | 2.92 | 4.41 GiB | 7.89 GiB |
| Shelf | 2 | 5.06 | 5.78 | 5.92 GiB | 8.02 GiB |
| Shelf | 4 | 9.46 | 11.47 | 8.79 GiB | 8.21 GiB |
| Shelf | 8 | 16.76 | 22.46 | 15.13 GiB | 8.87 GiB |
| Shelf | 16 | 28.07 | 44.12 | 25.35 GiB | 10.10 GiB |
| Shelf | 32, default buffers | GPU allocation failure | — | — | — |
| Shelf | 32, smaller buffers | 41.40 | 87.81 | 33.76 GiB | 12.57 GiB |

![Measured scaling](evidence/newton-wider-scaling.png)

The default allocates 65,536 contact slots and 131,072 constraint slots per
world. At 32 worlds that is 2,097,152 aggregate contact/CCD slots; native contact
workspace allocation fails on this 48 GB GPU. This is a buffer-allocation limit,
not an observed scene explosion. The separate successful 32-world probes set:

```python
cfg.newton_contact_settings = {
    "contact_capacity": 8192,
    "constraint_capacity": 32768,
}
```

They retain multi-point contacts, geometry, material settings and overflow
guards. The cube/shelf probes observed at most 4,757/9,105 aggregate active
contacts and 201/405 constraints in any world. These overrides are validated
only for these short cases; the production defaults stay unchanged because
more complex scenes or manipulation can need more space. Do not infer that
32 environments plus a co-resident policy model fit from these inference-free
runs. At 16 with unchanged buffers, aggregate throughput is roughly 11 times
the single-environment rate. This is throughput, not real-time speed per world.

GPU memory is sampled every three seconds inside the benchmark after any
auxiliary-test wait ends; samples taken by the outer queue while waiting are
excluded. Host RSS is the process high-water mark. Checks cover all worlds'
finite state/contact arrays and nonconstant RGB, actual model world count,
contact/constraint limits and repeated resets. Robot/object reset poses relative
to environment origins agree within 0.1 mm. First/last-world images are reviewed.
The cube additionally checks support force against weight, lifts only one
world's cube and verifies the others retain contacts/poses. World 15/31 is also
probed to catch ordering errors beyond single-digit IDs.

Separate four- and eight-world audits compare all 50 exported contact steps
against live values, check support force against weight, reset twice, and inject
contacts independently for both pads and all seven extra-body channels. Every
positive channel remains confined to its intended world and clears after
separation. Measured support force/weight ranges from 0.9889 to 1.0004. The
intentional overlap injections test routing; their large impulses are not force
calibration evidence.

## Regression tests and reproducibility

| Suite | Result |
|---|---:|
| Verified offline (macOS) | 317 passed |
| Newton regression (GPU node) | 123 passed |
| Simulator/integration (GPU node) | 241 passed |

The final Newton suite also ran locally: 121 passed, two GPU cases skipped.
The task-definition audit found zero conflicts, ten notes and four spawn-state
checks not asserted without recordings. The keyboard regression originally
required identical coplanar triangle indices on macOS and Linux. It now checks
identical vertices, a closed outward convex surface and equivalent volume;
the asset did not change. Failed diagnostic attempts are retained in the archive.

Node: Vast `55712744`, RTX PRO 5000 Blackwell 48 GB, about 90 GiB host RAM and
32 logical CPU cores. Rental rate: $0.835185/hour including storage, before
transfer fees. The node remains running for follow-up work.
Stack: Isaac Sim 6.1.0.0; Isaac Lab
`4ed09d414e93f35b8573db8850f8b27572779176`; Newton 1.5.2; MuJoCo/MJWarp 3.11;
Warp 1.16; PyTorch 2.11/CUDA 12.8; Python 3.12.12. OpenPI checkout:
`aa6420561529593114160d05e5ad155792b272f3`.

Local implementation commits run from `05b4f5c` through `c513131`, on top of
`be9332c`. The remote checkout stayed at base `be9332c` with deployed changes;
`tested-source.json`, `tested-source.tar.gz` and `tested-working-tree.patch`
identify the exact tested source. Source hashes were compared with the local
implementation; later documentation-only changes are excluded from that claim.
Scene-sweep jobs sometimes ran concurrently and their startup measurements are
not isolated performance results. Policy and scaling phases ran sequentially.

The raw archive `wider-evidence-final.tar.gz` is retained locally under
`/Users/metz/Documents/manda/state-of-robot-aug-2026/output/newton-wider-20261013/`,
along with the extracted `evidence-final/` directory and derived review videos.
Its size is 1,106,294,180 bytes; SHA-256:
`bfcdbc98220127b26b62e4f92cddc0bcae9596da63e9527ebfa14f03aaf52f5d`.
All 47,550 archived files passed individual size/hash verification.

`local-implementation-source.tar.gz` (7,655,082 bytes) additionally
preserves all changed files at local commit `c513131`, including licensing/docs
and the AST test run locally rather than deployed to the GPU node. SHA-256:
`cae2f03eb564264a5e87e6df48e6c15c28158417178d35df17129169c7f814e2`.
The [source comparison](evidence/newton-wider-source-verification.json) records
matching runtime files and identifies these local-only documentation/test files.
Local review copies include `stack-contact-comparison.mp4` and
`parallel-cube-policy.mp4`; the latter holds each shorter recording's last frame
for the side-by-side presentation.

The archive contains queue/probe scripts, resolved configs, logs, resource
samples, exit/result markers, images/videos, HDF5, state trajectories, test XML,
independent audit summaries and per-file SHA-256 hashes. In the extracted root,
`scenes.json` identifies representatives; `followup_manifold_*` and
`followup_long_manifold_*` contain final scene probes; `policy_manifold_*` the six
single-world policies; `policy_parallel_RubiksCubeTask_4` the batch;
`scale_manifold_*` default-buffer runs; `scale_sized_*` the explicit smaller
buffers; `contact_scale_manifold_*` the routing audits. The queue scripts retain
exact commands. `run_multiccd_validation.py` runs the final main matrix;
`run_long_manifold.py`, `run_parallel_policy.py` and `run_capacity_probe.py`
record the additional stages. Give any rerun fresh output directories.

Next evaluation priorities are robot–shelf collision behavior under a matched
control, broader manipulated/cluttered scenes and seeds, and longer batched
policy runs with scene-specific capacity measurements. Full 120-task policy
coverage, contact-force/friction/inertia calibration and matched PhysX
success-rate estimates remain open.
