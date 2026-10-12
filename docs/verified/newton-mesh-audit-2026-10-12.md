# Newton mesh import and wider scene audit — October 12, 2026

Follow-up to [the settling fixes](newton-settling-2026-10-12.md). Testing uses
Vast instance 55558470 (RTX PRO 5000, 48 GB) and the same pinned Isaac Sim
6.1.0.0 / Isaac Lab `4ed09d414e93f35b8573db8850f8b27572779176` /
Newton 1.5.2 / MuJoCo and MJWarp 3.11.0 stack. The starting repository revision
is `c718aca`; the final evidence bundle contains the exact tested patch.

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

## Remaining shelf-of-mugs failure

`TakeMugsOffOfShelfTask` first hit the 15-minute diagnostic timeout. A longer
retry was killed during model finalization after about 711 seconds of scene
creation. The container recorded one OOM kill and peak memory of 193,686,454,272
bytes against a 194,220,392,448-byte limit. Thus the longer retry is not evidence
of a settling/contact instability. A subsequent profiler is capped at 24 GiB
resident memory and records the Python stack rather than exhausting the node.

## Integrated validation

All 91 Newton unit cases pass locally and on the GPU host. Added tests exercise
real MuJoCo compilation, unchanged geometry and inertial properties, refusal
when mass could be inferred from the mesh, unrelated-error propagation,
physical-unit thickness bounds, generated-only scope, version checks and
restoration after errors.

The repository implementation, through the normal `managed_env` launcher,
completed 20 seconds each on `CookingPickPastaToolTask`,
`ToolsPickingDrillTask`, `PickUpBluePitcherTask` and `RubiksCubeTask` (all exit 0).
The earlier 238 simulator/integration tests belong to the preceding contact-fix
revision and were not rerun for this mesh-only change.

A separate CPU diagnostic omitting Newton's explicit candidate-pair table
produced exactly identical 500-step box-drop/contact trajectories with native
MuJoCo contacts. This optimization remains an experiment; it is not part of the
mesh fix. The bounded mug profiler and a conditional follow-up will determine
whether that table is responsible for the large allocation.

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
