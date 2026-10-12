# Newton passive-contact follow-up — October 12, 2026

The sideways-jug rocking reproduces on a fresh RTX PRO 5000 and in standalone
CPU MuJoCo. PhysX settles without entering sleep. A separate
full-scene experiment finds unexpected internal solver motion of the kinematic
shelf. Excluding contacts between immovable scenery eliminates that motion and
the pail stays supported in the control. Fresh-import geometry also varies, so
this is a strong implementation lead, not a fully isolated causal proof or a
validated general backend patch. The benchmark's physics settings remain unchanged.

This follows the [initial GPU validation](newton-gpu-2026-10-12.md). The new
experiments hold the initial arm target, with no policy. Earlier policy successes
were not rerun and are not included in this session's test count.

## Environment and scope

- Tested RoboLab-Verified `11ad4b231cf2a8fc45ce5db36ef49f3b019a76ac`.
- Fresh Vast instance `55521108`, separate host from unavailable `55449995`.
  RTX PRO 5000 Blackwell 48 GB, driver 595.71.05, 160 GB persistent disk.
  Quoted running cost approximately $0.6815/hour including allocated disk,
  excluding transfer charges.
- Same pinned physics packages as the first validation: Isaac Sim 6.1.0.0,
  Isaac Lab source `4ed09d414e93f35b8573db8850f8b27572779176`, Newton 1.5.2,
  MuJoCo/MuJoCo-Warp 3.11.0, Warp 1.16.0, Torch 2.11.0+cu128, Python 3.12.12.
- Scene: `OneBottleInSquarePailTask`, seed 0, one environment. The arm holds its
  initial joint target; success/target-lost terminations are disabled for these
  diagnostics. Full scenes record all three cameras where indicated.
- Newton baseline uses four substeps (effective timestep about 2.083 ms),
  100 solver iterations, 50 line-search iterations, and 4.5 ms contact response.

## Automated checks

- **61 Newton unit tests passed**.
- **238 simulator/integration tests passed** in 339.56 seconds, with zero
  failures or skips. Both the process exit and JUnit report are checked.
- Four full-scene comparisons completed 20 simulated seconds each: Newton
  baseline, PhysX baseline, PhysX with explicit zero sleep threshold, and Newton
  with 20 ms contacts.
- Seven isolated-jug comparisons completed 30 seconds each, and two isolated-pail
  controls completed 15 seconds each. A separate four-second support audit is
  described below, followed by an eight-second immovable-contact control. The
  313-test offline suite from the first validation was not repeated on this host.

Completion of a passive experiment means it reached its requested duration with
finite recorded states. It is not a declaration that its settling behavior is
correct. Short 0.2-second fixture probes are excluded from the main experiments.

## Sideways jug: full scene

The jug is `utilityjug_a02`, distinct from the jug knocked over by the policy in
the first validation. Starting root poses match between Newton and PhysX.

| 20-second run | Horizontal position span, seconds 15–20 | Angular-speed RMS, seconds 15–20 |
| --- | ---: | ---: |
| Newton baseline | 11.414 mm | 0.294774 rad/s |
| PhysX | 0.000 mm at sampled precision | 0.000596 rad/s |
| PhysX, explicit zero sleep threshold | 0.000 mm at sampled precision | 0.000596 rad/s |
| Newton, 20 ms contact response | 7.485 mm | 0.201463 rad/s |

Newton's horizontal span is 31.567 mm during seconds 2–8. Its rocking decays,
but remains visible through 20 seconds. Kinetic plus gravitational potential
energy decreases overall by about 0.487 mJ. This quantity excludes energy stored
in compliant contacts, so it is not a complete energy-conservation audit; it does
not provide evidence of a runaway gain in the jug's energy.

Both PhysX runs report `sleep_threshold=0.0` for both observed objects. All 301
sleep-state samples per object report awake. The identical PhysX trajectories
therefore do not support sleeping as the explanation for this difference.

The imported rigid-body properties are not identical across engines. Jug mass
is 0.191737 kg in Newton versus 0.234434 kg in PhysX; pail mass is 0.399636 kg
versus 0.504924 kg. COM and inertia also differ. These comparisons are not a
mass/inertia-matched solver benchmark, and neither engine is physical ground truth.

## Isolated jug: one change at a time

The isolated fixture retains the effective exported collision meshes, initial
pose, mass, COM and inertia. It removes other movable bodies and the robot, fixes
the support at its authored pose, and starts at zero velocity. Each run lasts
30 seconds. CPU and GPU controls use the same exported model. Exported decimal
precision and fixed supports make this a reduced reproduction, not an exact
replay of the full scene.

| Variant | Horizontal span, seconds 25–30 | Angular-speed RMS, seconds 25–30 |
| --- | ---: | ---: |
| GPU baseline | 8.873 mm | 0.281742 rad/s |
| CPU MuJoCo baseline | 10.699 mm | 0.302457 rad/s |
| Half timestep | 10.133 mm | 0.301071 rad/s |
| Torsional friction, `condim=4` | <0.001 mm | 0.000023 rad/s |
| Rolling friction, `condim=6`, coefficient 0.0001 | 9.201 mm | 0.291358 rad/s |
| Softer contact response, 20 ms | 0.021 mm | 0.008425 rad/s |
| Enclosing hull per material/contact group | 16.309 mm | 0.378480 rad/s |

Actual saved solver contacts confirm dimensions 3, 4 and 6 in the corresponding
runs. The torsional variant settles after approximately 21 seconds; the rolling
variant does not do so within 30 seconds. These single trajectories are sensitive
to their contact path and are insufficient to select a friction default.

Newton's unsuffixed mesh is the first convex-decomposition piece, not a duplicate
original hull. The geometry control constructs enclosing hulls from all pieces
while retaining inertia; deleting unsuffixed meshes would be an invalid control.
The jug has 169 collision geoms in the baseline. Simplification did not eliminate
its rocking. Halving the timestep also did not eliminate it. CPU reproduction
shows that the behavior is not specific to the GPU solver or robot control loop.

MuJoCo's [contact definition](https://mujoco.readthedocs.io/en/stable/computation/#contact)
distinguishes torsional (`condim=4`) from rolling (`condim=6`) friction. Its
[solver-parameter documentation](https://mujoco.readthedocs.io/en/stable/modeling.html#solver-parameters)
defines larger positive `solref` time constants as softer contacts. The parameter
changes above are diagnostic interventions, not calibrated material estimates.

## Background pail and full-scene sensitivity

The pail is `plasticpail_a02`. In the new baseline it moves about 0.398 m overall
and ends at root height 0.056608 m, supported by the `table_fixture` collider.
It does not repeat the earlier run's fall to root height -0.626 m. This variation
is retained as a result, rather than presenting the old outcome as deterministic.
PhysX settles near its initial position, at root height 0.111700 m.

At reset, recomputed CPU geometry at the full scene's GPU state finds the closest
pail/rack candidate at a positive 17.336 mm gap, and the jug/table candidate near
zero gap. These samples show no large initial overlap of those pairs. They are
not a complete audit of every collider or live GPU contact generation.

In isolation with a fixed rack, the pail settles under both GPU MuJoCo-Warp and
CPU MuJoCo at root heights 0.118070 and 0.118072 m respectively. Thus its full-scene
excursion does not reproduce as a pail/rack pair alone. The isolated GPU pail has
small residual angular motion (0.010469 rad/s RMS, seconds 10–15) despite only
0.0034 mm horizontal span.

The full-scene 20 ms contact-response run keeps the pail on its shelf for the
full 20 seconds, ending at root height 0.118297 m. It still moves: during seconds
15–20 its horizontal span is 3.596 mm and angular-speed RMS is 0.224592 rad/s.
The sideways jug also keeps rocking (7.485 mm horizontal span, 0.201463 rad/s),
despite the corresponding isolated soft-contact case nearly settling. The
isolated result therefore does not validate a full-scene repair. No softer
contact default is proposed from this evidence.

A four-second full-scene support audit samples the actual MuJoCo-Warp free-joint
state at every control step. The table position remains constant at sampled
precision (maximum sampled linear speed 4.90e-7 m/s). The rack solver state
shows a 1.731 mm position offset and up to 0.830803 m/s linear speed, mostly
upward. These are internal solver coordinates, not a measurement of rendered
shelf motion. The follow-up confirms both supports carry Newton kinematic flags.

The final eight-second control (`scene_immovable_filter_v3`) removes only
immovable/immovable candidate contacts. It checks every collision-mask/mobility
group to preserve all dynamic-pair permissions, and refreshes MuJoCo-Warp's
cached rigid-body pair tables. It removes 687 cached candidate pairs over 92
immovable geoms; all 24 live-contact samples report zero active
immovable/immovable contacts. The same 4.5 ms contact response is retained.

The rack's maximum sampled internal speed becomes 2.88e-6 m/s, with zero sampled
position drift. The pail stays on the shelf through eight seconds, ending at
root height 0.117916 m; its horizontal span during seconds 2–8 is 0.051 mm and
angular-speed RMS is 0.007377 rad/s. The jug continues rocking (26.716 mm span,
0.524275 rad/s RMS over seconds 2–8). This separates the shelf/pail lead from the
jug's reduced-model rocking.

**Fresh imports are not identical frozen models.** A post-download geometry audit
finds that the table's five geoms and rack's 84 geoms match across four full-scene
imports, ignoring piece names/order. The main baseline and final filtered jug
also match (169 pieces). The pail decomposition varies from 276 to 281 pieces;
the main baseline has 278 and the filtered run 280. Robot geometry/order also
varies. Thus the shelf result is compelling, but the pail comparison remains
confounded by fresh-import geometry. A next implementation check should reuse
one frozen full model for paired filter-on/filter-off runs, then validate
multiple environments and grasp/contact recording before promoting a generic
backend fix. The filter is currently diagnostic code outside the repository.

An earlier mask-only attempt (`scene_immovable_filter_v2`) did not update the
cached rigid-body pair tables and is **not a valid filtering intervention**.
Its completed trajectory is preserved and explicitly excluded from evidence that
filtering was applied. The final control verifies actual contacts, not just masks.

## Evidence quality and remaining work

- Saved full poses, COM velocities, mass/inertia, energy traces, sample times,
  actual process exits and completion markers. Isolated contact forces come from
  live solver state; full-scene contact logs are explicitly labeled as CPU
  geometry recomputed at GPU state and must not be treated as GPU force records.
- The diagnostic's initial PhysX kinetic-energy calculation applied an extra
  COM-frame rotation. Postprocessing removes that rotation using the documented
  [PhysX inertia convention](https://docs.omniverse.nvidia.com/kit/docs/omni_physics/107.3/extensions/runtime/source/omni.physics.tensors/docs/api/python.html#omni.physics.tensors.impl.api.RigidBodyView.get_inertias).
  Original traces are preserved beside corrected traces; the largest energy
  correction is 4.822 microjoules. Pose, velocity and sleep results are unaffected.
- Failed harness attempts are retained: an argparse flag conflict, an XML-save
  API incompatible with an MjSpec-compiled model, and an undersized isolated-pail
  GPU contact capacity, plus an attempted `.data` read on an unwrapped static
  scene asset. Corrected runs completed. An earlier full-scene attempt
  exited 139 after about 4.5 simulated seconds while a repeated faulthandler dump
  was active. The 20-second retry completed after disabling periodic dumps after
  reset; the cause of that crash is not established.
- No production solver/import code, task success rules, material coefficients or
  preset defaults were changed. The scripts and corrected calculations are
  diagnostic tooling. Contact-model calibration, mass/inertia parity and the
  remaining full-scene discrepancy need separate work before changing defaults.

## Reproduction and retained artifacts

The external evidence bundle is `newton-rocking-20261012`, kept outside Git.
It includes pinned package records, the exact diagnostic scripts, launcher
commands, JUnit reports, failed and successful logs, model exports, trajectories,
contact samples, plots and videos. Use fresh output directories for every run.

```bash
python -m pytest --confcutdir=tests/newton tests/newton -q
python -m pytest tests --ignore=tests/newton -q
python passive_scene.py --headless --physics newton_mjwarp --seconds 20 \
  --record-video --output fresh_scene
python passive_scene.py --headless --physics physx --seconds 20 \
  --record-video --output fresh_physx
python isolated_mjwarp.py --source fresh_scene/model-effective.xml \
  --variant baseline --seconds 30 --output fresh_isolated
```

Downloaded archive: `newton-rocking-results.tar.gz`, 49,142,619 bytes. SHA256 was
checked locally against the remote archive:

```text
5c8f809e7a136a81330b6cd5d646c447305c4614da591730cc4097f77e31cbd0
```

The local post-download `construction-geometry-comparison.json` is adjacent to
this archive; it was generated afterward and is not inside it. The 20-second
`newton-vs-physx-passive.mp4` comparison shows Newton on the left and PhysX on the
right, at simulation playback speed.

Instance `55521108` was stopped after download and SHA256 verification. Vast
reports `actual_status=exited`, `intended_status=stopped`; its 160 GB disk and
installed environment are retained for follow-up tests, with disk charges
continuing. Instance `55449995` was not modified.
