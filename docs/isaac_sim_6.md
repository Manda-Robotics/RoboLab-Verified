# Isaac Sim 6 support

The source installation now targets **Isaac Sim 6.1.0.0 + Isaac Lab 3.0 EA**,
pinned to `ae37b028ea415c91ea2bc32609efcd759ed2b974`. The
[EA migration record](verified/isaaclab-ea-2026-10-18.md) records local checks and the successful October 19 GPU/runtime checks;
the policy towing study is still running. The older Isaac Sim 6.0.1 / Lab beta2
wheel stack remains available as `isaac60` for reproducing existing experiments.

Experimental DROID Newton support is described in [newton.md](newton.md).

## Install EA

Use Linux x86_64 and a fresh Python 3.12 environment. Keep the old simulator and
policy-server environments separate:

```bash
uv venv --python 3.12 .venv-ea
.venv-ea/bin/python scripts/install_newton61.py \
  --isaaclab-root ../IsaacLab-v3.0.0-EA --install
source .venv-ea/bin/activate
python -m pytest offline_tests
```

See the [source installation instructions](newton.md#new-vast-instance) for
resolution-only mode, the USD provider constraint and GPU smoke commands. Use
`python` from the activated EA venv, not an implicit `uv sync` that would select
a different stack. The installer supplies `pytest` and uses `usd-exchange` for
USD bindings; do not add the offline `[test]` extra to this simulator environment.

## Legacy 6.0 stack

```bash
UV_PROJECT_ENVIRONMENT=.venv-60 uv sync --python 3.12 --extra isaac60
./docker/build_docker.sh --isaac60
```

The existing Docker recipe still defaults to
`nvcr.io/nvidia/isaac-lab:3.0.0-beta2`; it is a legacy workflow and does not build
the pinned EA source environment. No EA Docker image has been validated here.

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
.venv-ea/bin/python examples/run_kinova_jointpos.py --headless --num-steps 180
.venv-ea/bin/python examples/run_abs_ik_demo.py --headless
```

Isaac Lab 3.0 is now Early Access. Earlier beta releases listed a Robotiq
2F-85 issue; recheck both DROID and Kinova grippers when validating EA. Keep
the previous environment available for result comparison until the task suite
passes on your hardware.

## Earlier 6.0 validation in RoboLab Verified

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
