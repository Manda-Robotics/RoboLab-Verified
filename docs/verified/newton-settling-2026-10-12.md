# Newton scenery contacts and settling — October 12, 2026

Follow-up to [the passive-contact investigation](newton-rocking-2026-10-12.md).
The background pail and sideways jug have separate causes. In a controlled
full-scene comparison, contacts between immovable scenery perturb the rack's
internal solver state. Removing only those response pairs keeps the rack fixed
and the pail supported. Contact-local rolling resistance reduces the jug's late rocking by over 99%; it
does not require global velocity damping or changing scene assets.

## Implementation

`newton_collision_filter.py` uses Newton's actual MuJoCo body mapping and
kinematic flags. Jointless children inherit their parent's mobility. The adapter
rejects differing mobility across replicated worlds, where one shared pair table
could not represent the requested filtering. It updates both the full SAP and
compact NXN collision caches. It retains collision-sensor entries and every
original dynamic contact-pair permission. The GPU filter is not represented in
CPU geometry recomputed from `mj_model`.

The rolling-contact option changes immovable support geoms and explicit contact
pairs involving supports. Sliding and torsional coefficients are retained;
object/finger and other dynamic/dynamic parameters remain unchanged. It requests
contact dimension six and a rolling-friction length of 0.001 m. This is an
experimental contact parameter, not a measured material coefficient and not a
replacement for sliding friction. Existing MuJoCo priority/mixing rules still
apply. Actual jug contact dimensions are checked in the full-scene test.

MJWarp 3.11 caches maximum contact dimension and allocates the contact constraint
address array accordingly. Increasing dimensions after construction requires
reserving ten address columns and updating both dimension maxima. The initial
adapter omitted this and failed with a non-finite state on its first rolling
step. The repaired adapter reserves the storage before changing parameters;
a two-world native-engine regression compares it to allocation at dimension six
from the start. Other constraint arrays use the separate configured `njmax`
budget (131072 per world). CUDA graphs remain disabled, as required by this
pinned runtime's eager guards and adapters.

The changes are scoped to the experimental Newton preset. They do not alter USD
geometry, scene poses, densities, masses, COM/inertia, task success predicates,
robot actions, or PhysX settings. Mobility must remain fixed during a runtime.
This adapter relies on private Newton 1.5.2 / MJWarp 3.11 APIs and needs revalidation
when upgrading the pinned simulator stack.

## Scene-authoring audit

The authored residual velocities are tiny (jug linear speed about 2.8e-5 m/s,
pail about 5.4e-5 m/s); USD angular velocities are in degrees/second. Sustained
rocking reproduces from zero velocity in the isolated model, so removing the
authored residuals would not address its mechanism. The body materials already
have sliding friction: 2.0 on the jug/pail bodies, 0.2 on caps/handles. Their
densities are 950 and 800 kg/m³ respectively, with a denser jug cap. A zero or
unspecified authored mass requests geometric mass computation; these are not
massless dynamic objects. The rack and table are authored kinematic.

The previous investigation found different automatically computed mass/COM/inertia
across engines. This work retains the imported Newton values and does not claim
mass-matched engine parity or physical ground truth.

## Controlled CPU settling probes

`scripts/newton_contact_decay.py` reproduces the isolated-model tests from an
exported MJCF. Geometry, density-derived mass, COM and inertia stay fixed.
All 22 controlled runs completed with finite states and no MuJoCo warnings:
two 60-second settling preparations and twenty 20-second probes. Rest and nudge
comparisons use the exact same saved settled pose for each object and both modes.

Jug horizontal position span during seconds 15–20:

| Probe | Original dimension 3 | Dimension 6, rolling length 0.001 m |
| --- | ---: | ---: |
| Authored pose, zero velocity | 15.952 mm | 0.032 mm |
| Rest from shared settled pose | 0.093 mm | 0.068 mm |
| Linear nudge, 0.05 m/s | 1.406 mm | 0.070 mm |
| Angular nudge, 0.5 rad/s | 0.771 mm | 0.041 mm |
| Drop from 2 cm above authored pose | 15.591 mm | 0.037 mm |

The isolated pail is essentially stationary with either setting. Its full-scene
excursion is instead explained by the support interaction below. A coefficient
sweep finds native rolling length 0.0001 m insufficient; 0.0005 m reduces but
does not eliminate visible late rocking, while 0.001–0.002 m settles this fixture.
That sweep supports a candidate for this scene, not universal material calibration.

These CPU probes set dimension six on the isolated fixture's geoms. The full
scene changes only support geoms, preserving manipulation contact parameters.
An additional 20-second CPU comparison for each isolated fixture verifies that
support-only changes produce exactly the same trajectories as changing all geoms.
Small differences in imported coefficient precision change detailed stick/slip
trajectories. Early exploratory runs rounded the native torsional coefficient;
those artifacts are retained but excluded from the table above, which uses the
repository script and preserves native precision.

## GPU environment and controls

Replacement Vast instance 55558470: RTX PRO 5000 Blackwell 48 GB, driver 610.43.02,
160 GB disk, approximately $0.68593/hour including disk, plus transfer charges.
The previous node could not restart. No old instance was destroyed by this work.

Pinned stack: Isaac Sim 6.1.0.0; Isaac Lab
`4ed09d414e93f35b8573db8850f8b27572779176`; Newton 1.5.2;
MuJoCo/MuJoCo-Warp 3.11.0; Warp 1.16.0; Torch 2.11.0+cu128;
Python 3.12.12. Implementation through `eb45cb7` is used for the GPU comparisons.

Each comparison constructs a scene once and restores a frozen initial state
between phases. It asserts solver identity, matching initial poses, and a hash
of geometry/mass/COM/inertia. Thus fresh-import CoACD variability cannot confound
a paired result. Four substeps, 100 solver iterations, 50 line-search iterations,
4.5 ms response time and impedance (0.99, 0.999) remain fixed. The robot holds its
initial joint targets; passive experiments have no policy. Videos record three
cameras, while live GPU contacts, poses and velocities are logged separately.

The first completed 20-second geometry-controlled baseline/filter comparison
(`paired_full_v2`) establishes the scenery mechanism:

| Quantity | Original | Immovable-contact filter |
| --- | ---: | ---: |
| Rack maximum internal linear speed | 0.830803 m/s | 2.88e-6 m/s |
| Rack maximum internal position error | 1.731 mm | 0 mm at sampled precision |
| Maximum active immovable/immovable contacts | 32 | 0 |
| Pail horizontal span, seconds 15–20 | 15.015 mm | 0.043 mm |
| Pail angular-speed RMS, seconds 15–20 | 0.26225 rad/s | 0.00886 rad/s |

The pail remains on the rack with the filter. The filter alone leaves the jug
rocking. Geometry hash for this comparison:
`e08ff8288b296c3e44bb7ebed7fe62b522da1a37d26f588f5e9997666e06a5ac`.
The subsequent rolling phase of that attempt failed because of the allocation
bug; it is not counted as successful validation.


## Final full-scene settling comparison

The repaired `paired_full_v3` run reserves dimension-six storage for every phase,
including the dimension-three baseline, so capacity does not vary between phases.
Geometry hash: `5daf3f32ca00fd3934d42f154ae7bea863ef5c4471f022b4b9d7eae4a1462a2f`.

| 20-second phase | Jug span, seconds 15–20 | Jug angular RMS | Pail span, seconds 15–20 |
| --- | ---: | ---: | ---: |
| Original | 14.646 mm | 0.31943 rad/s | 295.721 mm |
| Scenery filter | 14.062 mm | 0.28126 rad/s | 0.042 mm |
| Filter + support rolling | 0.031 mm | 0.01658 rad/s | 0.040 mm |

The baseline pail leaves the rack and moves across the table; both filtered
phases keep it supported. The rack has zero sampled position drift after
filtering, and all sampled active immovable/immovable contacts are gone.
Actual jug contacts have dimension three in the baseline/filter-only phases
and six with rolling resistance. These results demonstrate improved numerical
settling in this scene; they do not establish real-world material calibration.

The full-scene 0.05 m/s world-X nudge at ten seconds also settles: the jug spans
0.190 mm during seconds 15–20, with angular RMS 0.02205 rad/s. Restoring original
contact settings restores the problem (jug span 14.122 mm, rack drift 1.731 mm,
32 active immovable contacts, and the pail leaves the rack). Repeated baseline
trajectories are not bitwise identical, and the pail's subsequent path differs;
the geometry and initial poses match, and the support defect returns.

The extra bin is the scene's existing `plasticpail_a02`, defined at line 511 of
`assets/scenes/shelf_with_cleaning_products.usda`. The asset is unchanged from
Verified main. The bottle task's destination is `squarepail_a01`; the extra bin
is included for contact tracking but has no role in its success predicate.
Its presence predates the Newton port. The scene is shared by five tasks; no
claim is made about the original designer's reason for its placement.

## Contact routing and recording

The two-environment cube audit passed with both contact changes enabled:
resting support force is 0.99987 and 0.99938 times object weight, world-space
force direction is correct, lifting the cube in one environment clears only
that environment's support contact, two resets remain finite, and both pad
plus seven additional robot-body channels route and clear correctly. Exported
HDF5 channels match 50 live steps per environment. The injected overlap probes
exercise routing and recording; their large transient forces are not physical
force calibration or evidence of a successful grasp.

## Evidence and failed probes

All five final passive phases completed 20 seconds, with finite recorded poses
and velocities. Results are sampled at the 15 Hz control rate; contact audits
sample live solver contacts every five control steps. This does not establish
bounds on every substep. The first short paired harness reset ordinary scene
state rather than a frozen snapshot and detected a 39 µm starting-pose mismatch;
its failed run is preserved, then corrected by restoring a cloned full state.
The initial rolling adapter failure is likewise retained with exit status one.
Neither failed probe contributes a passing result.

The local full unit invocation from the parent repository initially failed
because this isolated diagnostic venv did not have RoboLab on its module path;
running from RoboLab-Verified passed all 76 cases after the added lifecycle cases.
The GPU-host unit suite passed 74 cases before those two test-only additions.
The direct native CPU capacity test uses two worlds and 250 steps, confirms
active dimension-six contacts and finite states, and matches trajectories from
construction-time dimension-six allocation exactly.

## Manipulation and final checks

The paired open-loop replays share one constructed scene and recorded starting
state, replay exactly the same recorded actions, then hold the last command for
three seconds. These are manipulation regressions, not fresh policy runs or a
benchmark success-rate measurement. Both baseline and candidate remain finite.

| Target | Baseline maximum lift | Candidate maximum lift | Baseline / candidate lifted frames with both pads >0.1 N |
| --- | ---: | ---: | ---: |
| Bottle | 0.2914 m | 0.2904 m | 142 / 134 |
| Cube | 0.1305 m | 0.1305 m | 46 / 46 |

Both replays release the object (neither has two-pad contact in its final ten
frames). The bottle's maximum linear speed over the final two seconds is
2.6e-6 m/s. The cube still has small settling motion in the bowl: maximum linear
speed 0.0070 m/s and angular speed 0.149 rad/s over those two seconds. Object/bowl
contact remains dynamic/dynamic; this intervention does not add rolling friction
to that pair. Final placement poses differ between the paired replays.

- **238 simulator/integration tests passed**, zero skips/failures, in 124.88 s;
  both process exit and JUnit report are retained. This repository integration
  suite includes ordinary simulator/PhysX checks; the explicit Newton physics
  evidence comes from the separate experiments above.
- The native allocation regression also passed on **CUDA in two worlds**,
  250 steps, active dimension-six contacts, with zero trajectory difference
  between resized and construction-time allocation.
- A four-second **two-environment cleaning-products scene** completed with both
  contact changes active, finite object states, and no sampled active
  immovable/immovable contacts. This exercises actual scenery-pair removal in
  replicated worlds, in addition to the simpler two-world cube contact audit.
- The **76 Newton unit cases pass locally**, including the final preset defaults
  and adapter application/restoration cases. The earlier GPU suite contains 74
  cases; the two additions cover runtime lifecycle with rolling enabled.

## Preset and remaining scope

The experimental Newton preset now defaults to
`filter_immovable_contacts=True` and `support_rolling_friction=0.001`.
For a control experiment, explicitly set both keys to `False` and `None`
respectively in `cfg.newton_contact_settings` before `managed_env`.
PhysX remains the overall default backend. The resolved settings are retained
on the environment config. A 1 mm rolling-friction length is a scene-tested
experimental setting, not a universal calibration.

These results close the cleaning-products scene's observed shelf/pail defect
and substantially reduce the sideways jug's sustained rocking. They do not
certify all benchmark scenes, long runs, arbitrary robot assets, force/inertia
calibration, or policy success-rate parity. The user requested a broader scene
follow-up while these tests ran. That audit is separate and still ongoing;
the historical cooking-scene tiny-onion-mesh compilation failure has reproduced.
No change to that asset/import issue is included in this contact-fix report.

## Retained evidence

Completed GPU evidence is saved locally at
`output/newton-settling-20261012/settling-validated.tar.gz` in the
`state-of-robot-aug-2026` workspace. SHA-256: `6df7608350a5737c5fbef4f5f0f4d27d95a71eaf0ba7dd6c341b84ad0e0c394a`.
It includes exact executed harnesses, package versions, completed and failed
probe logs/exits, models/geometry fingerprints, trajectories, live contacts,
HDF5 routing evidence, videos and JUnit reports. The archive manifest records
tested implementation `eb45cb7`; the promoted defaults select exactly the
explicit settings exercised in those GPU tests.

The same workspace's `validated_*` folders retain the controlled CPU probes,
source hashes, native-precision contact parameters, warnings and saved states.
`before-after.mp4` is a normal-speed comparison of the original and improved
full-scene contacts. `preview/paired_full_v3/contact-decay.png` plots the measured
two-second moving spans. The GPU node remains active for the separately requested
broader scene audit; it is to be stopped and retained after that work and download.
