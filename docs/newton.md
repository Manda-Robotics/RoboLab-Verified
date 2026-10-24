# Experimental Newton backend

For the official Isaac Lab 3.0 GA / Isaac Sim 6.2 installation and current local
validation, see [New VAST instance](#new-vast-instance) below. Earlier experiment
sections retain the versions and results tested at the time.

For this merged RoboLab-Verified branch, see the [integration status and GPU
validation gates](verified/newton.md). The experiments below were reported by
the source RoboLab Newton branch; they do not validate the merged Verified build.

The `newton` branch starts at main commit `82725b5`. Select Newton with
`--physics newton_mjwarp` in any shared policy runner. The initial preset
supports DROID tasks and retains Isaac RTX cameras. PhysX remains the default.
Other robot assets fail with an explicit unsupported-robot error.

Earlier 6.1 validation: the collision crash was isolated to candidate-buffer
overflow. Guarded diagnostic runs with larger buffers complete, and a centered
scripted bottle grasp/lift/release passes with simplified fingertip pads and
robot gravity compensation. The tested pad/contact/capacity settings are now
the experimental Newton preset. This branch is not ready for
comparable benchmark runs; policy success-rate parity is unproven.

On 2026-09-15, three fresh live pi0.5/MJWarp seed-0 episodes (cube into bowl,
banana onto plate, bottle into square pail) succeeded and retained their success
conditions for five further seconds. All 31 captured policy requests passed
observation, image-resize and consumed-action conversion checks (maximum action
discrepancy below 1.2e-7). These are single-episode capability checks, not matched
success-rate estimates or proof of hardware-accurate physics.

Those runs established the settings now promoted from the diagnostic harness:
single convex hulls for fingertip
pads, cached object hulls, contact time constant 0.0045 s, impedance endpoints
0.99/0.999, gripper Kp=10/Kd=0.5, target slew pi rad/s, 65536 contact candidates,
131072 constraints and CUDA graphs disabled with a fail-before-kernel capacity
guard. The native CoACD disk cache remains a local startup optimization, not
a different decomposition method. The recording/analysis harness and recordings
are not distributed with RoboLab. At that stage no shell-inertia fallback was
applied: a cooking scene failed on a tiny red-onion hull, and the bottle task
had a non-target pail leave the scene. The October Verified follow-ups address
these failures; see the [settling report](verified/newton-settling-2026-10-12.md)
and [mesh-import audit](verified/newton-mesh-audit-2026-10-12.md).

After promotion, an ordinary `policies/pi0_family/run.py --policy pi05 --physics
newton_mjwarp` cube episode also succeeded at step 174 (11.6 s), without the study
harness or its disk cache. This validates the integrated launch path, not exact
trajectory parity with the earlier run; the policy server was not reseeded.

### Bagel defaults

Newton automatically uses the clean `bagel_00` and `bagel_06` collision assets
in the bagel/plate, breakfast-table and test-plate scenes, including renamed
instances such as `bagel_01`, `bagel_02` and `bagel_07` in the breakfast scene.
Each asset has 64 connected convex sectors and sliding friction 5. Visuals,
scene placement and the original Newton mass, center of mass and inertia are
retained. PhysX continues to load the original scenes and materials.

These Newton scenes default to **elliptic** friction cones. Cone type is a
solver-wide setting, so it affects every contact in that scene, not just the
bagels. Other Newton scenes retain pyramidal cones. An explicit
`cfg.newton_contact_settings["cone"]` can select either `"elliptic"` or
`"pyramidal"`; explicit `--friction` requests override the bagel material too.
The resolved scene, asset defaults and contact settings are saved in the
environment configuration. Use a fresh output directory for the changed preset.

Friction 5 is an empirical benchmark choice, not a measured material coefficient.
The GPU replay checks showed much shallower penetration, but the three friction-5
fixed-action trials did not carry the bagel; fresh-policy grasp/release coverage
and equivalent dynamic validation of `bagel_06` remain outstanding. The builder
`scripts/build_clean_bagels.py` uses `bagel_clean_reference_inertia.json` to
regenerate the assets without requiring the original diagnostic MJB file.

### Contact preset configuration

For programmatic experiments, set `cfg.newton_contact_settings` before
`configure_physics`/`managed_env`; omitted keys retain the preset values:

```python
cfg.newton_contact_settings = {
    "enable_multiccd": True,
    "compact_keyboard_colliders": True,
    "compact_shelf_colliders": True,
    "compact_contact_storage": True,
    "mesh_compatibility": True,
    "single_hull_fingertips": True,
    "filter_immovable_contacts": True,
    "support_rolling_friction": 0.001,  # rolling-friction length in metres
    "timeconst": 0.0045,
    "impedance": (0.99, 0.999),
    "contact_capacity": 65536,
    "constraint_capacity": 131072,
    "gripper_stiffness": 10.0,
    "gripper_damping": 0.5,
    "gripper_target_velocity": 3.141592653589793,
}
```

The Verified port now defaults to scenery filtering and support rolling
resistance; see the [October 12 validation](verified/newton-settling-2026-10-12.md).
Set these two options to `False` and `None` for the original-contact control.

The mesh compatibility option retries tiny-mesh inertia preprocessing only when
all users have explicit body inertia. It also gives generated flat CoACD
fragments <= 0.1 mm a 1 micrometre thickness; it does not modify source USD or
body inertial properties. Set `mesh_compatibility=False` for the original import
path. See the mesh-import audit for the measured approximation and limitations.

The compact storage option avoids storing redundant same-body exclusions and
an unused Newton candidate table when native MuJoCo generates contacts. Both
collision-mask compiler paths still receive the logical exclusions lazily.
Explicit cross-body filters are preserved; use with Newton's own collision
pipeline is rejected. Set `compact_contact_storage=False` for the original
allocation and initialization path. On the GPU path, the redundant CPU staging
contact solve is deferred to MJWarp; initial kinematics and equalities are still
computed, and contact flags are restored before uploading. CPU solvers and
runtime GPU contacts are unchanged. MuJoCo-Warp's own quadratic pair tables remain, so collision
geometry complexity still matters. The detailed shelf previously needed about
ten minutes and 92 GiB host RAM. The compiled shelf asset below brings this to
18–19 seconds and under 8 GiB in the three tested shelf tasks; see the
[compiled shelf audit](verified/newton-shelf-2026-10-13.md).

The resolved dictionary is retained on the environment config. Contact response
is applied to CPU and GPU model geom arrays before the first MJWarp step;
equality parameters are unchanged. The two Robotiq fingertip pad collision
meshes use a single convex hull. The `rack_l04` shelf uses the compiled collision
asset described below. The HOT3D keyboard uses 110 compiled component hulls.
Other object assets retain native authored approximation;
original visuals are retained.

The `enable_multiccd` option enables native multi-point contact manifolds for
convex meshes before MuJoCo compilation and GPU upload. Newton 1.5.2 otherwise
disables them. This prevents flat contact patches from behaving like a single
pivot: the three-cube stack's top-cube motion at 17–20 seconds fell from a
20.63 mm XY span to 0.20 mm, with unchanged geometry, mass and friction. The
pinned Isaac Lab configuration lacks this field, so a scoped constructor bridge
sets it only for native-contact GPU solvers. Newton handles the corresponding
zero-margin requirement. Set `enable_multiccd=False` for the earlier
single-contact diagnostic control; do not mix results from those presets.

The `compact_shelf_colliders` option replaces rack_l04's detailed CoACD output
with a compiled 44-piece USD: four post envelopes, four feet, eight braces,
four continuous decks and 24 rails/brackets. It preserves deck bounds and heights,
body mass/COM/inertia, transforms, materials and explicit collision filters.
Fastener collisions are omitted and the approximately 1 mm deck seams are closed
at the validated scale of 0.2. Original rendering is retained. Source and compiled
asset fingerprints prevent silent reuse after geometry changes. See the
[asset build and audit instructions](../assets/objects/vomp/rack_l04/README.md).
Set `compact_shelf_colliders=False` to restore the expensive original import.

The `compact_keyboard_colliders` option replaces the HOT3D keyboard's 5,503
CoACD pieces with 110 component hulls. The source consists of separate, convex
keycaps and base pieces; this keeps their exterior surfaces and inter-key gaps,
closing only open keycap bottoms. Source and compiled geometry fingerprints
are checked before import. Mass, inertia, visuals and filters retain the native
import path. Set the option to `False` for native decomposition. See the
[build and geometry checks](../assets/objects/hot3d/README.md).

The scoped adapters require the pinned stack and one active simulation runtime
per process. They restore patched methods on exit. Eager execution is intentional:
the capacity guard synchronously checks candidate counts before each CCD launch
and raises on overflow rather than truncating contacts or risking an illegal
GPU memory access. Enabling CUDA graphs with this preset is rejected. This costs
performance, and the larger buffers cost GPU memory. The
[October 13 Verified audit](verified/newton-wider-2026-10-13.md) measures cube and
shelf scaling through 16 worlds with these defaults; separate 32-world probes
pass with explicit smaller buffers after the defaults exhaust GPU memory.
Those tests exclude policy inference; a separate four-world cube policy run
passes. They do not establish a calibrated hardware contact model or broad
policy success rates. Revalidate the private APIs before upgrading Newton.

## New VAST instance

The current source target is **Isaac Sim 6.2.0.0 + Isaac Lab
[v3.0.0 GA](https://github.com/isaac-sim/IsaacLab/releases/tag/v3.0.0),
commit `e3f7d9ff1f5d7f8280de3ece9000b7117f66bcf4`**. The installer clones the
release tag and verifies this immutable SHA. It pins Newton **1.6.1**,
MuJoCo/MuJoCo-Warp **3.12.0**, Warp **1.17.0**, and PyTorch **2.11.0+cu130**
(torchvision **0.26.0+cu130**, CUDA **13.0**). MuJoCo patch versions stay exact
because our contact adapters use private APIs.

The source workspace packages have independent version numbers (`isaaclab==28.0.0`
for this tag). Use the release tag and SHA to identify the Lab release. The
GA stack passed the CUDA, simulator integration, rendered smoke and recording
checks below on 2026-10-24. Fresh-policy evaluation, a loaded bagel release and
full benchmark coverage remain unvalidated on GA. Earlier EA/6.1 results in this
guide are historical evidence, not GA validation.

On an x86_64 Linux GPU server with a CUDA-13-compatible NVIDIA driver, use a
fresh Python 3.12 environment and a new Lab checkout:

```bash
uv venv --python 3.12 .venv-ga
.venv-ga/bin/python scripts/install_newton61.py \
  --isaaclab-root /workspace/repos/IsaacLab-v3.0.0 --install
export OMNI_KIT_ACCEPT_EULA=Y
.venv-ga/bin/python examples/run_newton_smoke.py \
  --headless --physics newton_mjwarp --num-envs 2 --steps 60 \
  --output output/newton_ga_smoke
```

The installer retains its historical filename `install_newton61.py` for existing
scripts, but now selects GA/6.2. An existing Lab checkout must be clean and at
the pinned SHA; the installer never rewrites it. It reads the release workspace
requirements and overrides, resolves them with RoboLab, and installs that plan.
Each invocation saves inputs, resolved requirements, tag, SHA and before/after
package lists in a git-ignored `newton61-install-*` directory.

GA uses `usd-exchange` 3.0.1, which supplies `pxr`. Do not install RoboLab's offline
`[test]` extra or `usd-core` into the simulator venv: these install a second USD
provider over the same files. The installer includes `pytest` separately and
rejects venvs containing `usd-core`. The separate macOS/CPU test venv can still
use `usd-core`.

Omit `--install` to resolve only. `--replace-isaacsim` remains available for a
compatible venv and removes old Sim packages only after resolution. A separate
GA venv is preferred; assets, recordings and policy-server environments are not
modified. Do not run `uv sync --extra isaac60` afterward: that extra is retained
for beta2 experiments. Run `.venv-ga/bin/python` directly to avoid an implicit
re-sync.

### EA to GA compatibility

Local validation on 2026-10-24: the complete Linux/Python-3.12 installation
resolves (304 packages); 157 Newton CPU checks and 358 offline checks pass.
Four CUDA-only checks are skipped on macOS. Probes against the actual GA source
also match upstream RGB/depth results and exercise partial pose/camera resets and
solver configuration. Full RoboLab environment imports require Kit (`omni`),
which is unavailable on this Mac.

GPU validation on 2026-10-24 used an RTX PRO 4000 Blackwell (24 GB), driver
595.58.03 and Python 3.12.12 with the exact GA stack above. PyTorch reports CUDA
13.0 and a working CUDA device. Results:

| Check | Result |
| --- | --- |
| Newton suite | 161 passed, zero skipped, including all four formerly skipped CUDA checks |
| Simulator integration suite | 241 passed, zero skipped |
| Cube/banana Newton smoke | Two environments, 60 steps; finite state, support contacts, contact-query parity and three RGB cameras passed |
| Shelf Newton smoke | One environment, 60 steps; finite state, support contacts and three RGB cameras passed |
| Bagel recorded-action diagnostic | 345 steps / 23 seconds; source actions, finite data, HDF5/JSON agreement, native contacts and both 345-frame videos passed |
| Newton reset/recording integration | Slice, device-tensor and all-environment object/camera resets; two batches of two environments with staggered termination, finite data and JSON/HDF5 parity passed |

The initial Newton suite had three asset-check failures because it started before
LFS hydration completed (158 passed). After verifying all 2,275 asset files, only
those three checks were rerun; all passed. The four CUDA checks cover sphere and
box candidate-growth parity, MULTICCD flat-mesh settling, and deferred CPU staging
without changing the GPU trajectory.

A final live Newton check exercises the GA reset selectors and two successive
evaluation batches. Synthetic timeouts stop the two environments after 4 and 8
steps respectively. All four recordings contain exactly those actions/states,
retain failure labels, and receive no extra frames from later frozen steps;
owned RGB observation buffers also remain unchanged across simulation steps.
The first diagnostic attempt compared world-space positions with recorded
relative positions and stopped at its audit; correcting that test comparison
required no runtime change. These are lifecycle checks, not policy episodes.

The initial rendered smoke exposed a crash in the timed `faulthandler` watchdog:
a native debugger located it in `PyUnstable_InterpreterFrame_GetLine` while the
watchdog walked Python frames. The smoke example now retains fatal-error
reporting but omits the periodic all-thread stack walk; wall-clock limits are
external. The repaired cube and shelf runs completed. No contact laws or solver
settings were changed for this repair.

The shelf scene took 60 seconds to construct, including 5.4 seconds finalizing
the Newton model. Peak process RSS was 10.6 GiB and sampled GPU memory was 4.0 GiB;
the complete cold/warm-mixed rendered smoke took 232 seconds. These are single-run
measurements, not throughput benchmarks.

The bagel diagnostic retains the production clean colliders, friction 5,
elliptic cone, 120 Hz simulation and four substeps. Both sets of 64 authored
convex hulls survive compilation within 0.000085 mm vertex round-trip error;
mass, COM and inertia match the earlier EA clean assets. Sampled native
finger/bagel overlap peaked at 0.60 mm and contact ended at 16 seconds, without
contact or constraint overflow. Sampled overview and wrist frames show the
bagel being pushed and tilted, then left on the table as the empty gripper rises.
This matches the earlier friction-5 trials: **it does not exercise release from
a held grasp**. The replay is unscored, reuses recorded actions with diagnostic
termination suppression, and is neither a fresh policy episode nor a claim of
deterministic replay. It establishes runtime/recording compatibility, not task
success or towing elimination. Fresh-policy and full-task GA coverage remain
outstanding.

Raw logs, failed attempts, debugger output, recordings, geometry checks and
manual review are retained outside the repository in the workspace artifact
`output/isaaclab-ga-gpu-20261024`; they are not required runtime files.

The migration follows the release's **Upgrading from Isaac Lab 3.0 Early Access
to 3.0 GA** checklist. RoboLab keeps single-frame NHWC RGB observations (uint8
for policies), metric depth and WXYZ recordings. The camera adapter replaces the
removed generic image term and owns its returned buffers. Custom reset and
recorder callbacks accept slices as well as device tensors, including partial
resets and frozen episodes. Argument registration prefers `add_launcher_args`;
the Newton environment lifetime already uses `launch_simulation`. Legacy
AppLauncher calls only bootstrap the Kit services RoboLab still requires.

RoboLab authors its own combined property configs, whose deprecated aliases are
still supported in GA; it does not mutate the new upstream fragment lists.
Its task registry does not use the removed Hydra decorators. No custom in-place
noise, SB3 action wrappers, TorchScript loader, legacy visualizer fields or Gym
RecordVideo path require migration. Robot joint selections already retain their
selected dimension. These source checks do not replace GPU integration tests.

Newton 1.6.1 no longer builds explicit same-body filter pairs in `add_shape`, so
that EA workaround is bypassed. The remaining native-contact storage and compiled
shelf/keyboard adapters retain exact version guards and parity checks. MJWarp
3.12 bounds convex CCD candidate iteration by the contact-buffer capacity: our
candidate-growth adapter now launches bounded slices of the candidate arrays,
covering every pair while retaining the original output budget and overflow
checks. Contact coefficients, assets, timing, and the clean-bagel elliptic/5
preset are unchanged by the migration.

On the GA source stack, omit the old `--newton-convex-decomposition-threshold` and
`--newton-contact-timeconst` options (smoke equivalents: `--decomposition-threshold`
and `--contact-timeconst`). Lab now honors authored approximations natively;
the beta2-only monkeypatch remains available for old environments but is
explicitly rejected on the new stack. First validate native collision behavior
before changing solver tuning. The older version-comparison results below are
historical; the current preset includes the subsequently validated contact adapters.

The compatibility layer retains RoboLab's `--headless` / `--enable_cameras`
flags even though upstream no longer registers them. It also maps RTX mode
settings from the former `sim.render` field to each camera renderer's
`global_settings`, preserving other authored camera properties. The removed
`--rendering-type performance/balanced/quality` presets are rejected on the source stack
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
.venv-ea/bin/python policies/pi0_family/run.py \
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

The original beta2 preset kept task timestep/decimation and used CUDA graphs,
four substeps, implicitfast integration, 100 solver iterations, 50 line-search
iterations, 4096 constraint rows and 2048 contact slots. The current preset
disables graphs and uses the larger guarded capacities listed at the top of
this guide. Neither configuration is throughput-tuned. The earlier baseline stack was Isaac Lab
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

When the robot spawn configuration requests `rigid_props.disable_gravity=True`,
the preset selects `franka_robotiq_2f_85_newton_nogravity.usda`, which layers
`mjc:gravcomp=1` onto the 18 robot rigid bodies. Newton does not otherwise
translate PhysX's gravity-disable setting. Scene objects retain gravity; false
or unspecified settings select the uncompensated Newton overlay. Both overlays
are generated by `scripts/build_newton_asset.py`. The native GPU model has been
checked to receive robot-only compensation without a runtime override.

Newton contact queries use object-specific sensors rather than assuming batched
sensor columns follow the requested filter order. This prevents contacts with a
bowl from being attributed to a cube (or vice versa); the PhysX batch path is
unchanged. Trajectory jerk integration accepts both NumPy's newer `trapezoid`
API and the older `trapz` API used by the Isaac Sim 5 stack.

The Newton-only gripper preset uses Kp=10 N m/rad and Kd=0.5 N m s/rad on
`finger_joint`, plus genuine MuJoCo passive damping (`mjc:damping=0.05`) on
its five mimic followers. Exported MJCF confirms those passive damping values.
The overlay now expresses the same five coupling relationships as native
MuJoCo equalities, preserving the authored gearing/offset, with
`solref=(0.006, 1)` and `solimp=(0.99, 0.99, 0.001, 0.5, 2)`. Newton's
PhysX-mimic conversion otherwise loses the authored coupling stiffness.
The policy retains its original binary 0=open/1=closed commands; an action term
slews the target at the physics-update rate, capped at pi rad/s (0.25 s for the
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
