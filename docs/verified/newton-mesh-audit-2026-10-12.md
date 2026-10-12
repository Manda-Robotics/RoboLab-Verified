# Newton mesh import and wider scene audit — October 12, 2026

Follow-up to [the settling fixes](newton-settling-2026-10-12.md). Testing uses
Vast instance 55558470 (RTX PRO 5000, 48 GB) and the same pinned Isaac Sim
6.1.0.0 / Isaac Lab `4ed09d414e93f35b8573db8850f8b27572779176` /
Newton 1.5.2 / MuJoCo and MJWarp 3.11.0 stack. The starting repository revision
is `c718aca`; the final evidence bundle contains the exact tested patch.

The [October 13 compiled shelf follow-up](newton-shelf-2026-10-13.md) resolves the
rack startup cost recorded below: 44 collision pieces, 18–19 second startup and
under 8 GiB peak RSS in the three tested shelf tasks. The detailed-rack figures
below remain the historical baseline.

## Findings and fixes

The cooking and tools scenes fail during MuJoCo compilation on very small,
three-dimensional decomposition pieces. Captured onion and hammer tetrahedra
have maximum extents of 19.55 and 31.10 micrometres. The onion hull volume is
1.55e-16 cubic metres. LEGACY, EXACT and CONVEX mesh inertia modes all reject
that hull; SHELL compiles it. This does not mean the onion or hammer should have
hollow-body dynamics: Newton already supplies explicit body mass, centre of mass
and inertia, independently of the mesh's inertia preprocessing.

The new scoped adapter retries only the specific `mesh volume is too small`
error, and only when every geom using that mesh belongs to a positive-mass body
with explicit inertia, with compiler-forced geom inertia disabled. It changes
only the rejected mesh's inertia preprocessing, restores the input spec after
compilation, and propagates other failures. Actual captured hull tests preserve
mass, centre of mass and inertia; reconstructed world-space collision vertices
differ by at most 5.70e-13 metres. Ordinary meshes retain their original compiler
path. No source USD, body inertia, friction or visual geometry is edited.

The pitcher fails earlier: CoACD produces a three-vertex fragment, about
98.63 micrometres across, that Newton rejects as planar. An attempted removal
was rejected because containment in a retained hull could not be established.
The final adapter instead gives only newly generated, flat CoACD fragments no
larger than 0.1 mm an additional thickness of 1 micrometre, in scaled physical
coordinates. It makes a closed convex prism, moving each generated vertex at
most 0.5 micrometres before float32 rounding. Body properties and shape
transforms/materials remain unchanged. Larger planar shapes, lines and points
are not repaired automatically. This is an explicit, bounded collision-geometry
approximation, not merely an inertia change.

The shelf follow-up exposed a startup cost in retrying one rejected mesh at a
time. After the first confirmed tiny-volume error, the adapter now probes other
inline meshes no larger than 0.1 mm in isolation with the same MuJoCo compiler.
Each probe copies vertices, faces, scale, reference transform, hull limit and
inertia mode. It selects SHELL only after the compiler reports that exact
mesh-volume error and all actual users pass the explicit-inertia guard; size
alone never selects the fallback. Valid small meshes retain their original mode,
other errors return to the normal full-model path, and all modes are restored.
An eight-fragment regression needs only two full-model compiles. This batches
the same checks without discarding fragments or approximating solid geometry.

Both adapters are enabled by `cfg.newton_contact_settings["mesh_compatibility"]`
(default `True`). Set it to `False` to reproduce the original import failures.
They require the pinned versions, run before replication, reject nested scopes
and restore patched functions on every exit. PhysX does not enter this context.

## Runtime checks

The passive audit holds the robot's initial targets for 20 simulated seconds,
records three cameras, object poses/velocities, live contact samples and the
configured success predicate, and checks finite state. These are initialization
and passive-contact checks, not fresh policy runs, manipulation-success tests
or calibrated-force validation. The benchmark's success/rest rules are not
modified; the diagnostic disables automatic success termination so it can keep
observing the full interval.

Prototype runs completed 20 seconds each for cooking, tools and pitcher. The
cooking and tools runs had maximum object XY spans over the final three seconds
of 0.905 and 0.822 mm, respectively. The pitcher's corresponding span was
0.0158 mm. None produced passive confirmed task success.
The cooking PhysX control also completed (10 seconds). Videos show objects
supported by their authored table/containers. This is not matched trajectory or
policy-success-rate parity.

The bowl-stacking process initially crashed after about six seconds. The audit
was copying all GPU contact/constraint data into a CPU MuJoCo arena with
`mujoco_warp.get_data_into`. Reading the needed contact arrays directly from the
GPU instead completed two independent 20-second runs. The first run's worst
late object XY span was 7.54e-6 mm. This supports a diagnostic-copy-path issue;
the native crash has not been reduced enough to claim a proven upstream root
cause. No bowl asset or physical parameter was changed.

The earlier unattended suite also completed sauce bottles, white mugs in a bin,
and unstacking cubes (10 seconds each), plus the recorded banana-action replay.
The historical sauce-label import error did not reproduce, and the unstacking
scene did not produce passive confirmed success. Those checks do not certify
all cameras or all task variations.

## Large shelf import and memory

`TakeMugsOffOfShelfTask` first hit the 15-minute diagnostic timeout. A longer
retry was killed during model finalization after about 711 seconds of scene
creation. The container recorded one OOM kill and peak memory of 193,686,454,272
bytes against a 194,220,392,448-byte limit. Thus the longer retry is not evidence
of a settling/contact instability. A subsequent profiler stopped safely at
24.04 GiB resident memory, 532 seconds into initialization. Its stack identifies
`ModelBuilder.add_shape_collision_filter_pair`: Newton eagerly accumulates
same-body shape exclusions while adding CoACD pieces.

The cached import contains 36,778 shapes, of which 35,907 belong to `rack_l04`.
That body alone would generate about 645 million same-body pairs. The rack's
four source meshes have 4, 732, 56 and 24 geometrically connected components.
They contain detailed frame, fastener and shelf geometry. Their authored PhysX
`maxConvexHulls=256` is not directly carried into Newton's per-component CoACD
path. Other tasks using this rack are not covered by the passing scene checks.

The scoped memory adapter omits the redundant stored same-body pairs
and Newton's explicit candidate table, which native MuJoCo contacts do not use.
Explicit cross-body filters and body-to-shape bookkeeping remain intact.
Both of Newton's collision-mask compiler paths still receive the
same logical self-exclusions through a lazy view. A first prototype omitted
that step; a 262-shape regression detected a changed cross-body permission, so
that prototype was rejected. The revised implementation matches compiled permissions
below and above the 256-shape compiler cutoff, and produces bitwise-identical
500-step native MuJoCo CPU trajectories on a multigeom contact fixture.

The adapter is restricted to Newton 1.5.2 and native MuJoCo contacts, guards
against Newton's collision pipeline, and restores all hooks on exit. It changes
storage, not geometry, inertia or material values. These permission tests establish
parity with the pinned converter, not independent correctness of every possible
partial shape exclusion: its large-model coloring fallback can itself permit
some selectively excluded shape pairs.

The corrected shelf run (`mugs_storage_128`) completed all 20 simulated seconds
with finite poses, velocities and sensor forces, no active immovable/immovable
response contacts, and no passive confirmed success. Its maximum object XY span
in the final three seconds was 0.01292 mm. The mugs remain supported on the
shelf in the recorded cameras. Initialization confirmed 238 tiny-volume errors
in isolated probes, then compiled all 36,716 meshes in only two full-model
attempts.

Resource cost remains substantial. This run used cached outputs of the exact
native CoACD decomposition, took about 10 minutes to reach simulation and
12.55 minutes in total, and sampled a peak of 91.61 GiB resident host memory
under a 128 GiB guard. Earlier 64 and 96 GiB guarded attempts stopped during
MuJoCo-Warp GPU model construction, before simulation. MuJoCo-Warp 3.11 still
allocates quadratic native pair tables for this roughly 36,700-geom scene;
the new adapter removes Newton's redundant tables, not those native tables.
This is a passing functional check, not a startup-performance fix or a claim
that this scene fits a modest-memory host. Other tasks using this rack still
need their own runtime checks.

## Integrated validation

The mesh-fix revision `2b4dfb7` passed 91 Newton unit cases locally and on the
GPU host. Added tests exercise real MuJoCo compilation, unchanged geometry and inertial properties, refusal
when mass could be inferred from the mesh, unrelated-error propagation,
physical-unit thickness bounds, generated-only scope, version checks and
restoration after errors.

The repository implementation, through the normal `managed_env` launcher,
completed 20 seconds each on `CookingPickPastaToolTask`,
`ToolsPickingDrillTask`, `PickUpBluePitcherTask` and `RubiksCubeTask` (all exit 0).
Those four runs validate the earlier mesh-fix revision; the final checks below
exercise the added memory adapter and batched compiler path together.

The final implementation passes:

- 98 Newton tests on the remote host; locally 95 pass and three native Newton
  cases are skipped because Newton is absent from the local test venv.
- All 313 Verified offline regressions locally.
- All 238 simulator/integration cases on the GPU host, with zero skips,
  failures or errors (110.30 seconds). This shared suite includes the default
  PhysX empty-episode check; Newton execution is exercised separately below.
- The two-environment Newton contact audit: support force/weight ratios of
  1.00013 and 1.00015, per-object and per-environment routing, both pad and
  extra-body HDF5 values matching 50 live steps per environment, force clearing
  and repeated reset. Injected-penetration probes test routing, not force calibration.
- Five 20-second Newton passive scenes through `managed_env`, with finite
  state/contact data and no passive confirmed success:

| Latest run | Task | Largest XY span in final 3 s |
| --- | --- | ---: |
| `cube_storage_integrated` | RubiksCubeTask | 0.05970 mm |
| `mugs_storage_128` | TakeMugsOffOfShelfTask | 0.01292 mm |
| `cooking_current` | CookingPickPastaToolTask | 0.92265 mm |
| `tools_current` | ToolsPickingDrillTask | 0.93010 mm |
| `pitcher_current` | PickUpBluePitcherTask | 0.01429 mm |

The current-default bottle and cube recorded-action replays also exit 0. The
bottle lifts 0.2894 m, with both pads engaged in 135 lifted frames; the cube
lifts 0.1305 m, with both pads engaged in 46 lifted frames. Both pad force
vectors are exactly zero throughout each replay's final ten frames after release. These replay stored
actions through the current runtime, not a fresh policy server. They use fresh
imports, whose geometry hashes differ from the earlier paired experiments;
they are not a controlled old/new trajectory comparison or success-rate parity.

This completes the sampled integration/regression round. Full benchmark-wide
scene coverage, long-duration and large-batch stability, calibrated physical
parameters and fresh-policy success rates remain unestablished. PhysX remains
the default and Newton remains experimental.

## Evidence

The remote experiment directory is `/workspace/newton-settling`. The local
copy is under `state-of-robot-aug-2026/output/newton-settling-20261012`.
Keep failed attempts alongside successful ones: `unattended-20261012`,
`cooking_compile_capture`, `scene_ToolsPickingDrillTask`, `bowl_direct`,
`bowl_direct_repeat`, `cooking_tiny_fix`, `tools_tiny_fix`, `pitcher_capture`,
`pitcher_mesh_fix` (rejected deletion), and `pitcher_thin_fix`.
`*_integrated` runs use the repository implementation through `managed_env`,
without the prototype compatibility contexts. The evidence bundle includes
logs, per-test exit status, the diagnostic scripts, saved models and videos.

The completed mesh evidence bundle is `mesh-audit-validated.tar.gz` (446,783,305
bytes), SHA-256
`552cbd30d75617c5735ae57a9522655bde0d42193bdade48f30e87da8e1b3e80`.
The local download was verified, and its source hashes match mesh-fix commit
`2b4dfb7`. Later memory experiments are recorded separately.

The final memory and regression bundles were downloaded and SHA-256 verified:

| Bundle | Bytes | SHA-256 |
| --- | ---: | --- |
| `memory-audit-final.tar.gz` | 810924335 | `432ed8feb7d4fdfecdddeb8a097b91870d562ea297d981ee66b648f38a29b635` |
| `shelf-final-evidence.tar.gz` | 411301044 | `6b968231385e95b7b024fd696339d92de20f58ea05da173a26e4f7573491de4a` |
| `complete-integration-evidence.tar.gz` | 3993890 | `64de74429c25c7948c5b0123f5f6c3b7bb596f2b33b0a09dc3d781187de1225b` |
| `final-scenes-evidence.tar.gz` | 118172963 | `79201ce418ae9974cda59a1262164d8e5a50b869fa4a2bdc7654559f7de9be92` |

The integration manifest's 19 Python source hashes match the local implementation
and tests. Two additional AppleDouble metadata entries are not Python sources.
The local offline report is `offline_storage_final.xml`; earlier attempts using
incomplete local environments failed collection and are retained separately.
At the user's request, instance 55558470 is left running after collection;
there is no automatic stop or destroy action in the test queues.
