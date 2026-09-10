# Experimental Newton backend

The `newton` branch starts at main commit `82725b5`. Select Newton with
`--physics newton_mjwarp` in any shared policy runner. The initial preset
supports DROID tasks and retains Isaac RTX cameras. PhysX remains the default.
Other robot assets fail with an explicit unsupported-robot error.

## New VAST instance

After committing and pushing this branch to your fork, clone/check out `newton`
on the server. Run the normal Isaac Sim 6 bootstrap, then in the RoboLab checkout:

```bash
uv sync --python 3.12 --extra isaac60 --extra newton --extra test
export OMNI_KIT_ACCEPT_EULA=Y
.venv/bin/python examples/run_newton_smoke.py \
  --headless --physics newton_mjwarp --num-envs 2 --steps 60 \
  --output output/newton_smoke
```

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

The preset keeps task timestep/decimation, uses CUDA graphs, one substep,
implicitfast integration, 100 solver iterations, 50 line-search iterations,
4096 constraint rows and 2048 contact slots. These are conservative feasibility
settings, not a throughput-tuned configuration. The pinned stack is Isaac Lab
3.0.0b2.post1, Newton 1.2.1, MuJoCo 3.8.0 and MuJoCo-Warp 3.8.0.3.

`franka_robotiq_2f_85_newton.usda` is a small sublayer over the current Isaac Sim
6 rendering asset. It supplies missing Robotiq masses, estimates inertia and COM
from uniform local collision bounding boxes, fixes panda_link8's zero inertia,
and replaces zero principal-axis quaternions with identity. Existing positive
Panda inertias are retained. It changes no meshes, joints, collision shapes or
camera definitions; the PhysX asset stays unchanged.

Mass estimates come from the legacy [Robotiq CAD description](https://github.com/ros-industrial-attic/robotiq/blob/kinetic-devel/robotiq_2f_85_gripper_visualization/urdf/robotiq_arg2f_85_model_macro.xacro)
and [base-link description](https://github.com/ros-industrial-attic/robotiq/blob/kinetic-devel/robotiq_2f_85_gripper_visualization/urdf/robotiq_arg2f.xacro).
They are not calibrated hardware measurements. Using collision bounds avoids
assuming the URDF tensor axes match this USD's differently authored link frames.
This overlay differs from the temporary August feasibility asset and needs a
fresh GPU test. Existing Panda inertias also merit a later physical calibration.

Newton's Isaac Lab integration is beta. Full grasps, mimic-joint behavior,
actuator limits, reset/export, multi-task cleanup and long episodes still need
validation on the new server. Same-seed trajectories across different solvers
need not agree; compare task success, grasp/slip behavior, object poses and
contact events. Neither speedup nor improved physical accuracy is established.

## Local checks and regeneration

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
Full Isaac Sim tests remain `pytest tests/` on the CUDA host.

Reference: [Isaac Lab MJWarp configuration](https://github.com/isaac-sim/IsaacLab/blob/release/3.0.0-beta2/docs/source/overview/core-concepts/physical-backends/newton/mjwarp-solver.rst).
