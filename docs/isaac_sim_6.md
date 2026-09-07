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

- Configs, observations, Cartesian actions, `WorldState`, and all HDF5
  quaternion fields remain `(w, x, y, z)`. Isaac Lab 3's internal
  `(x, y, z, w)` values are converted centrally. Camera extrinsics retain ROS
  camera axes while using the same stable WXYZ component storage.
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

## Isaac Sim 6 breaking-change audit

The [6.0 breaking-change list](https://docs.isaacsim.omniverse.nvidia.com/latest/overview/release_notes.html#breaking-changes-and-deprecations)
has been checked against RoboLab:

- The removed `omni.isaac.*` shims are not used on the Isaac Sim 6 path.
  Isaac Sim 5 compatibility branches use the renamed `isaacsim.*` modules.
- The Isaac Sim 6 path avoids the deprecated `isaacsim.core.api`,
  `isaacsim.core.prims`, and `isaacsim.core.utils` modules. It uses Isaac Lab
  simulation/view APIs or `isaacsim.core.experimental` instead. The deprecated
  imports that remain in the source are lazy Isaac Sim 5-only fallbacks.
- RoboLab task cameras use Isaac Lab's camera abstraction. Standalone screenshot
  utilities use the replacement `RtxCamera`/`CameraSensor` API from
  `isaacsim.sensors.experimental.rtx`; they do not use RTX Camera prim JSON
  configurations.
- Contact and frame sensors are Isaac Lab sensors, not the deprecated direct
  `isaacsim.sensors.physics`, `isaacsim.sensors.physx`, or
  `isaacsim.sensors.rtx` APIs.
- RoboLab has no ROS 2 OmniGraph nodes, Cortex, wheeled-robot, manipulator,
  Lula/motion-generation, MobilityGen, domain-randomization, merge-mesh,
  Replicator Agent, ML archive, app selector, benchmark examples, Scene Blox,
  or asset-browser dependency. The `PushGraph` embedded in the tools-picking
  scene is an `omni.anim.curve.core.AnimCurve`, not a ROS 2 node.

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

## Validation in RoboLab Verified

Checked on an A40 with both stacks installed side by side (`.venv-51`: Python 3.11,
`--extra isaac51`; `.venv-60`: Python 3.12, `--extra isaac60`) from one tree, with an open-loop
replay of the same recording on both (`examples/run_recorded.py --env-config current --num_envs 4
--max-steps 450`, GR00T `GrabAFruitTask` env 1):

| check | Isaac Sim 5.1 | Isaac Sim 6.0 |
|---|---|---|
| offline suite (no simulator) | passed | passed |
| `run_empty --friction 0.5`: request applied to 3 objects and both pads | yes, PhysX readback 0.5/0.5 | yes, PhysX readback 0.5/0.5 on every object shape (256 per object) and both pads (64 each), identical to 5.1; the readback needed two fixes for the Isaac Lab 3 backend (facade `isinstance`, warp arrays) |
| replay reproduces the recorded contact event at the recorded time (all 4 envs) | yes | yes |
| recorded quaternions stay WXYZ (end-effector and object poses) | yes | yes |
| Robotiq 2F-85 in the wrist camera | whole | whole, no detached fragments |
| replay speed, 4 envs with video | ~1.3 s/step | ~0.8 s/step |

`run_recorded.py` accepts `--max-steps`, `--no-video` and `--output-dir` so a reproducer can replay
only the stretch up to an event. Exact trajectory reproduction still holds only on the stack that
made the recording; a published number states its stack.
