# Experimental Newton backend

The `newton` branch starts at main commit `82725b5`. Select Newton with
`--physics newton_mjwarp` in any shared policy runner. The initial preset
supports DROID tasks and retains Isaac RTX cameras. PhysX remains the default.
Other robot assets fail with an explicit unsupported-robot error.

Current 6.1 status: stationary and loaded-gripper smokes pass, but the first
pi0.5 cube grasp encounters a MuJoCo-Warp CUDA collision-kernel error. This
branch is not ready for comparable benchmark runs; see the validation section
at the end for the evidence and diagnostic commands.

## New VAST instance

The current experimental target is **Isaac Sim 6.1.0.0 + Isaac Lab
release/3.0.0 at `4ed09d414e93f35b8573db8850f8b27572779176`**.
That source revision selects Newton 1.5.2, MuJoCo/MuJoCo-Warp 3.11 and Warp
1.16.0. This is not the published Lab beta2 wheel. The source workspace has
separately versioned packages (for example `isaaclab==17.0.1`); use the git SHA,
not that internal extension version, to identify the Lab release.

After committing and pushing this branch to your fork, clone/check out `newton`
on an x86_64 Linux GPU server. In the RoboLab checkout:

```bash
# Fresh environment only; reuse an existing Python 3.12 .venv for an upgrade.
uv venv --python 3.12 .venv
.venv/bin/python scripts/install_newton61.py \
  --isaaclab-root /workspace/repos/IsaacLab-newton61 --install
export OMNI_KIT_ACCEPT_EULA=Y
.venv/bin/python examples/run_newton_smoke.py \
  --headless --physics newton_mjwarp --num-envs 2 --steps 60 \
  --output output/newton_smoke
```

The installer clones/pins Lab when the target directory does not exist. An
existing checkout must be clean and already at the pinned SHA. It reads the
workspace's dependencies and overrides, resolves them together with RoboLab,
then installs the resulting plan. This avoids accidentally downgrading Newton
to the older version requested by Isaac Sim's wheel metadata. Each invocation
saves its inputs, resolved requirements, git SHA and before/after package lists
in a git-ignored `newton61-install-*` directory.

Omit `--install` to resolve only. On a small existing instance, add
`--replace-isaacsim` to uninstall the old Sim packages **after** resolution and
before downloading replacements. This changes that venv in place; assets,
recordings and separate policy-server environments remain untouched.
Do not run `uv sync --extra isaac60` afterward: it reinstalls the old beta2
stack. The old `newton` wheel extra has been removed in favor of this installer.
Run simulation commands with `.venv/bin/python` to avoid an implicit re-sync.

On 6.1, omit the old `--newton-convex-decomposition-threshold` and
`--newton-contact-timeconst` options (smoke equivalents: `--decomposition-threshold`
and `--contact-timeconst`). Lab now honors authored approximations natively;
the beta2-only monkeypatch remains available for old environments but is
explicitly rejected on the new stack. First validate native collision behavior
before reintroducing any solver tuning. The robot overlay and gripper action
remain unchanged for this first version comparison.

The compatibility layer retains RoboLab's `--headless` / `--enable_cameras`
flags even though upstream no longer registers them. It also maps RTX mode
settings from the former `sim.render` field to each camera renderer's
`global_settings`, preserving other authored camera properties. The removed
`--rendering-type performance/balanced/quality` presets are rejected on 6.1
rather than silently ignored; omit that option for the native renderer defaults.

The results below describe the **earlier 6.0/beta2 experiments**, unless a section
explicitly says 6.1. They do not establish success-rate parity for the new stack.

The smoke test requires no policy server. It asserts finite robot state/rewards,
three nonconstant RGB camera images, filtered XYZ contact-force dimensions,
agreement between individual and batched contact queries, and at least one
detected contact. It prints `ROBOLAB_SMOKE_PASS` and writes `smoke_result.json`
only after passing. Verify both the process exit status and the pass marker.
Use a fresh output path for each test attempt.

Once the existing pi0.5 server is available on port 8000:

```bash
.venv/bin/python policies/pi0_family/run.py \
  --policy pi05 --headless --physics newton_mjwarp \
  --task RubiksCubeThenBananaTask --num-envs 1 --num-runs 1 \
  --remote-host localhost --remote-port 8000 \
  --video-mode sensor --output-folder-name pi05_newton_pilot
```

Repeat with `--physics physx --output-folder-name pi05_physx_pilot` for a
baseline. For the five-task smoke study, pass the same task names and policy
settings as the previous run, adding `--physics newton_mjwarp` to the actual
RoboLab policy command. Study wrappers do not automatically forward this flag.
The output backend marker prevents resuming PhysX episodes as Newton results.
Per-task `env_cfg.json` records the backend, full solver settings and asset path.

## Implementation and limits

`managed_env` configures physics before environment construction and keeps
Isaac Lab's `launch_simulation` context open through episode stepping and
cleanup. Existing policy scripts still own AppLauncher/Kit. Programmatic callers
can use `robolab.core.environments.physics.managed_env` after launching Kit:

```python
with managed_env(task_name, physics="newton_mjwarp", launcher_args=args,
                 device="cuda:0", num_envs=1) as (env, cfg):
    env.reset()
    # step, evaluate, and export here
```

The preset keeps task timestep/decimation, uses CUDA graphs, four substeps,
implicitfast integration, 100 solver iterations, 50 line-search iterations,
4096 constraint rows and 2048 contact slots. These are conservative feasibility
settings, not a throughput-tuned configuration. The earlier baseline stack was Isaac Lab
3.0.0b2.post1, Newton 1.2.1, MuJoCo 3.8.0 and MuJoCo-Warp 3.8.0.3.

`franka_robotiq_2f_85_newton.usda` is a small sublayer over the current Isaac Sim
6 rendering asset. It supplies missing Robotiq masses, estimates inertia and COM
from uniform local collision bounding boxes, fixes panda_link8's zero inertia,
and replaces zero principal-axis quaternions with identity. Panda tensors are
also re-estimated: the source's positive tensors are implausibly small for its
kilogram-scale links and were associated with unstable Newton rollouts.
Robotiq collision APIs are additionally authored on leaf meshes because Newton
does not recognize the source's parent-Xform collider declarations. Meshes,
transforms, joints and cameras stay unchanged; the PhysX asset stays unchanged.

The Newton-only gripper preset uses Kp=10 N m/rad and Kd=2 N m s/rad on
`finger_joint`, plus genuine MuJoCo passive damping (`mjc:damping=0.05`) on
its five mimic followers. Exported MJCF confirms those passive damping values.
The overlay now expresses the same five coupling relationships as native
MuJoCo equalities, preserving the authored gearing/offset, with
`solref=(0.006, 1)` and `solimp=(0.99, 0.99, 0.001, 0.5, 2)`. Newton's
PhysX-mimic conversion otherwise loses the authored coupling stiffness.
The policy retains its original binary 0=open/1=closed commands; an action term
slews the target at the physics-update rate, capped at pi/2 rad/s (0.5 s for the
full 45-degree stroke). Reset resynchronizes this target to the reset joint pose.
PhysX gains and actions are unchanged. These values are experimentally tuned,
not a calibrated Robotiq force/speed model.

Mass estimates come from the legacy [Robotiq CAD description](https://github.com/ros-industrial-attic/robotiq/blob/kinetic-devel/robotiq_2f_85_gripper_visualization/urdf/robotiq_arg2f_85_model_macro.xacro)
and [base-link description](https://github.com/ros-industrial-attic/robotiq/blob/kinetic-devel/robotiq_2f_85_gripper_visualization/urdf/robotiq_arg2f.xacro).
They are not calibrated hardware measurements. Using collision bounds avoids
assuming the URDF tensor axes match this USD's differently authored link frames.
This overlay differs from the temporary August feasibility asset. All estimated
inertias still require physical calibration; a stable rollout is not evidence
of hardware accuracy.

On VAST 50526640 (RTX PRO 5000, driver 595.71.05), the updated preset passed
60 stationary-target steps in two environments with CUDA graphs, all three
RTX cameras, and object/table contact forces in both environments. OpenPI
pi0.5 independently returned finite 15-by-8 action chunks after warm-up.
Before the gripper fix, the first policy-driven episode failed the
nonfinite-joint guard at step 141 (RubiksCubeThenBananaTask). With the updated
gripper preset, pi0.5 completed 900-step cube-then-banana, cube-and-banana and
white-mug episodes without nonfinite recorded states; all three timed out
without task success. Peak sampled gripper speeds were 3.29, 2.26 and 2.52 rad/s.
This preset is **not yet validated for comparable policy evaluation**.
The follow-up `RubiksCubeTask` test, selected because pi0.5 scored 10/10 in
both saved PhysX references, failed all three Newton rollouts at 600 steps.
Seed 0 was explicit; recorded initial joint positions match exactly and
cube/bowl/table positions match to sub-micrometre precision. The robot remains finite
but fails the cube-grasp subtask. This is a current-integration regression,
not evidence of solver equivalence.
The evaluator bypasses the PhysX-only Kit timeline gate under Newton and
terminates on nonfinite robot joints rather than exporting a misleading score.

The five-task feasibility run also exposed these remaining blockers:

- `SauceBottlesCrateTask` fails initialization because its scene enables
  collision on a planar decorative label. The diagnostic alternative
  `use_mujoco_contacts=False` initializes but lets bottles fall through their
  support; do not use that setting as an automatic fallback.
- `UnstackRubiksCubeTask` reports success even in a policy-free stationary-arm
  control, and its scene is obscured in RTX cameras in a fresh process. Its
  short policy-run success flag is not a validated manipulation result.
- Recorded end-effector velocities are zero despite changing positions.
  EE speed/jerk/SPARC metrics from these recordings are invalid.
- The native equality update reduced the isolated loaded-block peak mimic
  residual from 0.366 to 0.0215 rad; real grasps and force calibration still
  require validation. This is not a policy-success result.

Newton's Isaac Lab integration is beta. Full grasps, mimic-joint behavior,
actuator limits, reset/export, multi-task cleanup and long episodes still need
validation on the new server. Same-seed trajectories across different solvers
need not agree; compare task success, grasp/slip behavior, object poses and
contact events. Neither speedup nor improved physical accuracy is established.

## Local checks and regeneration

### Gripper-only diagnostic

`examples/run_newton_gripper_probe.py` holds the arm targets and performs a
three-phase open/close/open test without pi0.5. It logs joints and all gripper
link contact sensors at the physics-update rate and exports a video. Run with
`--gain-mode inherited`, `damping`, `soft`, or `gentle`; `--ramp-seconds 0.5`
tests target smoothing without changing the production action configuration.
Use a fresh `--output` directory for every invocation.

The 2026-09-11 probe reproduced oscillation without reported gripper contacts.
The inherited drive reached 397.7 rad/s. Reduced gains and a control-rate ramp
alone still reached 17.2 rad/s; adding actual passive follower damping and the
physics-rate limiter reduced this to 1.67 rad/s. A kinematically held 50-mm block
test stayed finite, peaked at 2.30 rad/s and 19.15 N maximum absolute net-force
component, and returned to zero contact after release. It is a contact stability
test, not a successful free-object grasp or calibrated force measurement. Mimic
residuals under load were substantial (0.366 rad peak). The follow-up native
equality test (`--mimic-timeconst 0.006`) reduced the peak to 0.0215 rad and the
closed-hold residual to 0.0138 rad, with clean release and no nonfinite states.
The corresponding free-space closed-hold residual was 0.000164 rad. These
probes used the previous convex-hull import path; new collision meshes must be
validated separately. The optional MuJoCo CPU control currently fails
contact-buffer initialization in the pinned stack.

Reproduce the current runtime action with explicit diagnostic settings:

```bash
.venv/bin/python examples/run_newton_gripper_probe.py --headless \
  --gain-mode gentle --passive-damping 0.05 --rate-limit \
  --mimic-timeconst 0.006 --contact-block --output output/gripper_contact_fresh
```

Omit `--contact-block` for the corresponding free-space test. The probe
deliberately disables the production preset before applying its CLI settings;
its default inherited-drive mode remains a diagnostic baseline.

### CPU regression checks

These tests do not import Isaac Sim's parent conftest:

```bash
uv run --no-project --with usd-core --with numpy --with pytest \
  python -m pytest --confcutdir=tests/newton tests/newton -q
uv run --no-project --with usd-core --with numpy \
  python scripts/build_newton_asset.py
```

The generator creates the portable overlay from the checked-in source asset.
Tests validate composition, untouched geometry/relationships, positive masses,
physically admissible principal inertias, backend isolation and cleanup.
The action limiter tests also require Torch (otherwise they are skipped); run
all checks (including the Torch action tests) using the RoboLab environment on the server:

```bash
.venv/bin/python -m pytest --confcutdir=tests/newton tests/newton -q
```

Full Isaac Sim tests remain `pytest tests/` on the CUDA host.

Shared policy runners accept `--seed 0` to override the environment seed.
Omitting it preserves existing backend defaults. Use this explicitly when
comparing runs; repeated rollouts with one seed are not independent seeds.

The stationary smoke additionally rejects nonfinite or runaway object
positions (default maximum displacement from reset: 1 m), saves camera PNGs,
and reports early termination under held targets. The diagnostic
`--newton-contacts` flag tests the alternate collision pipeline only; it does
not change policy defaults. A smoke pass checks only its stated assertions,
not scene correctness, task validity, visibility or calibrated contact physics.

Reference: [Isaac Lab MJWarp configuration](https://github.com/isaac-sim/IsaacLab/blob/release/3.0.0-beta2/docs/source/overview/core-concepts/physical-backends/newton/mjwarp-solver.rst).

## Generic collision import (opt-in)

This does **not** substitute a hand-built bowl collider or adjust object heights.
Isaac Lab beta2's `newton_replicate.py` imports with
`skip_mesh_approximation=True`, then convexifies every mesh. The scoped
`newton_meshes.py` compatibility context restores Newton's authored-approximation
path and skips that redundant blanket pass. Meshes requesting convex
decomposition use CoACD with normalized tolerance 0.05, 150 MCTS iterations,
depth 3 and seed 0 instead of Newton 1.2's coarse defaults. Other approximation
methods keep upstream behavior. This does not make unsupported SDF or invalid
meshes automatically compatible with MuJoCo.

Decompositions are cached by geometry, settings and CoACD version under
`${XDG_CACHE_HOME:-~/.cache}/robolab/newton-convex-v1/`. Shared asset geometry
reuses its cached result. Source USDs and installed packages remain untouched.
The shim is version-guarded to Newton 1.2.1, process-global, scoped to one
simulation lifetime, and restored on exceptions; concurrent simulations in
one process are unsupported. The reported threshold is not a metre-valued
penetration tolerance or PhysX `errorPercentage`. Hull counts and cache hits
are logged; first-time decomposition and many contact pieces can be expensive.

Enable explicitly for a policy evaluation:

```bash
.venv/bin/python policies/pi0_family/run.py --headless --policy pi05 \
  --physics newton_mjwarp --newton-convex-decomposition-threshold 0.05 \
  --newton-contact-timeconst 0.006 \
  --task RubiksCubeTask --seed 0 --num-envs 1 --num-runs 1 \
  --remote-host localhost --remote-port 8000 --video-mode sensor \
  --output-folder-name pi05_newton_generic_fresh
```

The equivalent no-policy diagnostic uses `--decomposition-threshold 0.05`.
Add `--drop-object rubiks_cube --into-object bowl` to test the bowl cavity.
It flushes teleported dynamic state before stepping, disables success
termination for the drop, verifies that the drop pose persists, and requires
destination contact at the end. Recorded MuJoCo contact distances are computed
at the actual GPU state (environment 0); this is a geometric audit, not a
separate CPU dynamics rollout. Sampling every ten control steps can miss
transient penetration peaks.

The optional `--newton-contact-timeconst 0.006` applies scene-wide default
contact stiffness/damping to each USD import builder (authored per-shape
overrides still take precedence). Assigning `ke/kd` only on Isaac Lab's
`default_shape_cfg` did not reach the compiled MJCF in this cloner path.
The smoke equivalent is `--contact-timeconst 0.006` and checks that the
requested value appears in the compiled scene. Values below twice the solver
timestep are rejected. This is explicit experimental solver tuning, not a
claim of calibrated material compliance.

On 2026-09-11 the 60-step cube drop passed with cube/bowl contact, no cube/table
contact, and sampled penetration below 1 mm. A broader 60-step
`WhiteMugsInBinTask` run remained finite with all ten movable objects supported,
but some cube/table distances were about -4.5 mm with the default contact
response. A stability smoke pass is therefore not a contact-fidelity pass.
Applying the verified 0.006-second contact response reduced its maximum
sampled penetration from 4.58 to 0.513 mm, with all objects supported. Its
RTX view is nevertheless obscured, so this is a contact regression fixture,
not a valid camera-conditioned policy benchmark yet. Scaled-asset rendering
remains to be investigated.

The combined mesh/contact/equality loaded-gripper probe remained finite over
1,080 physics samples, peaked at 1.695 rad/s and 0.0380 rad mimic residual,
and reopened after contact. This is less compliant than the original
0.366-rad loaded baseline, but still not a calibrated grasp-force test.

The first pi0.5 `RubiksCubeTask` episode with both generic opt-ins reports
1/1 success in 125 steps (8.33 s). Video/HDF5 show a real cube grasp and about
71 mm lift followed by transport to the bowl. It still contacts the table
and disturbs the bowl; an apparent wrong-object contact event needs auditing,
and the recording ends without post-release settling. This is progress over
the previous 0/3, not evidence of 10/10 parity or complete contact correctness.
No source scene/object was altered to achieve that result.

`WhiteMugInCenterOfTableTask` is unsuitable as a clean whole-scene regression:
its source `objects_around_table.usda` authors the bowl and banana at z≈-123 m
with downward velocities. They are not repaired or removed by this migration.
The smoke now rejects reset objects farther than 10 m from their environment
origin (configurable) to flag such invalid tabletop fixtures explicitly.

## Isaac Sim 6.1 source-stack validation (2026-09-11)

The in-place upgrade on VAST 50526640 installed Sim 6.1.0.0 and the pinned
Lab source revision above, with Newton 1.5.2, MuJoCo/MuJoCo-Warp 3.11.0,
Warp 1.16.0 and Torch 2.11.0+cu128. The separate OpenPI server environment,
assets and previous results were preserved. The remote Newton unit suite
passed all 28 tests; locally, 21 passed and the Torch-only module was skipped.

`newton61_native_cube_v2` passed the 60-step, one-environment no-policy
`RubiksCubeTask` smoke with all three RTX cameras, finite states, and expected
cube/table and bowl/table contacts. At ten-control-step sampling intervals,
maximum penetration was 0.3814 mm for cube/table and 0.3911 mm for bowl/table.
Those are sampled geometric distances, not a bound on transient penetration.
The native importer produced 230 cube, 96 bowl and five table collision pieces.
The exported MJCF retained the robot's 0.006-second coupling equalities and
native 0.02-second scene contact response. No beta2 mesh/contact override or
object-specific geometry/height fix was applied. First launch required several
minutes of scene-specialized CUDA compilation.

`newton61_native_gripper_v1` completed a 135-control-step loaded 50-mm block
probe (1,080 recorded physics samples). All states were finite, peak sampled
gripper speed was 1.354 rad/s, peak mimic residual 0.03022 rad and maximum
absolute net contact-force component 33.37 N. It reopened to about 0.00343 rad.
The closed hold was stable; its nonzero target error is expected because the
block prevents full closure. Arm target deviation reached 0.05148 rad.
These are stability diagnostics, not calibrated grasp-force or policy scores.

The first pi0.5 `RubiksCubeTask` run crashed after 32 completed control steps
with CUDA illegal memory access. A synchronous diagnostic located the fault
inside Newton graph replay. Repeating with `--newton-disable-cuda-graph`
(same physical settings, eager execution) also crashed, after 37 steps,
inside MuJoCo-Warp's `collision_convex.ccd_kernel`. Thus disabling graphs is
**not** a workaround, and the initial traceback in the gripper action was an
asynchronous symptom. These incomplete runs are not valid policy-score samples.
The upgraded stack is installed and smoke-tested, but full policy compatibility
is not established. The graph opt-out remains an explicit diagnostic only.

A targeted CUDA memory-check run was started afterward. Its final result
could not be retrieved because the instance was subsequently reported stopped
by Vast; it was left stopped. Inspect `/workspace/logs/newton61-ccd-memcheck.log`
after resuming the instance. The locally preserved results and installation
record are in the study checkout's `artifacts/newton-50526640/isaac61/`.

Upstream references: [pinned dependency workspace](https://github.com/isaac-sim/IsaacLab/blob/4ed09d414e93f35b8573db8850f8b27572779176/pyproject.toml)
and [Newton integration changelog](https://github.com/isaac-sim/IsaacLab/blob/4ed09d414e93f35b8573db8850f8b27572779176/source/isaaclab_newton/docs/CHANGELOG.rst).
