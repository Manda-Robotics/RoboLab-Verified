# Isaac Sim 6 support

RoboLab supports Isaac Sim 6.0.1 with the Isaac Lab 3.0 beta 2 patch 1 stack.
This is the recommended stack for RTX PRO 5000/6000 and other recent GPUs.

## Install

Isaac Sim 6 requires Python 3.12. It cannot share an environment with an
Isaac Sim 5 stack.

```bash
uv venv --python 3.12
source .venv/bin/activate
uv sync --extra isaac60
uv run pytest tests/
```

For the container image:

```bash
./docker/build_docker.sh --isaac60
```

The default Docker base is `nvcr.io/nvidia/isaac-lab:3.0.0-beta2`.

## Compatibility design

RoboLab keeps its existing external contracts while adapting the simulator
boundary:

- Configs, observations, Cartesian actions, `WorldState`, and HDF5 scene states
  remain `(w, x, y, z)`. Isaac Lab 3's internal `(x, y, z, w)` values are
  converted centrally. Camera extrinsics remain the documented ROS XYZW
  exception.
- Isaac Lab 3 `ProxyArray` state is exposed to RoboLab as zero-copy torch views.
- State writers use Isaac Lab 3's indexed APIs, with fallbacks for Isaac Lab 2.
- `sim.physics`/`isaaclab_physx.physics.PhysxCfg` and the new `FrameView` API are
  used on Isaac Lab 3.
- Removed `omni.isaac.*` utilities have Isaac Lab or USD replacements on the
  Isaac Sim 6 path.

Old recordings can be read on Isaac Sim 6 because their quaternions are
translated on restore. Exact trajectory reproduction is still only expected
on the simulator stack that created the recording because PhysX behavior
changes between releases.

## Validation and known risk

Run at least one rendered smoke test and one contact-rich task on the target
driver/GPU before producing benchmark results:

```bash
uv run python examples/run_kinova_jointpos.py --headless --num-steps 180
uv run python examples/run_abs_ik_demo.py --headless
```

Isaac Lab 3.0 is still a beta line. NVIDIA's 3.0 beta release notes list a
known Robotiq 2F-85 issue, so the DROID and Kinova grippers need particular
attention in smoke testing. Keep Isaac Sim 5.1 available for result comparison
until the task suite passes on your hardware.
