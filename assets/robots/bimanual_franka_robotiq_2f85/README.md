# Bimanual Franka Panda + Robotiq 2F-85

Two DROID arms (Franka Panda + Robotiq 2F-85) mounted 0.60 m apart on a shared torso,
as a **single articulation**. This is the first bimanual embodiment in RoboLab.

## Provenance

Generated — do not edit by hand. Rebuild with:

```bash
pip install usd-core            # pure-Python pxr; Isaac Sim is not needed
python assets/robots/_utils/build_bimanual_franka.py
```

The builder copies `/panda` from `../franka_robotiq_2f_85_flattened.usd` (the DROID
asset, same licences) twice, renames every rigid body and joint with a `left_arm_` /
`right_arm_` prefix, remaps all relationship targets and shader connections, strips the
copies' articulation roots and world joints, and mounts both on `/robot/torso` with fixed
joints. Mesh geometry stays instanced, so both arms share one set of prototypes and the
file is no larger than the single-arm one.

```
/robot                     articulation root
/robot/torso               mount plate (visual only), fixed to the world by root_joint
/robot/left_arm            at (0, +0.30, 0): left_arm_panda_link0 … left_arm_base_link, left_arm_left_inner_finger …
/robot/right_arm           at (0, −0.30, 0): right_arm_…
/robot/left_mount_joint    fixed, torso → left_arm_panda_link0
/robot/right_mount_joint   fixed, torso → right_arm_panda_link0
```

## Simulation configuration

Per-arm dynamics are DROID's: the same home pose, actuator gains, gripper drive and
μ = 2.0 finger-pad material, so results on this robot are comparable arm-for-arm with the
single-arm benchmark. Self-collision is disabled, as on DROID and Kinova.

The stale `isaac:physics:robotJoints` / `robotLinks` metadata carried by the source
(pointing at the original Panda hand that the Robotiq replaced) is dropped.

Mounting table: `assets/fixtures/bimanual_franka_table.usda`, `franka_table.usd` widened
1.8× in Y to span both bases.

## RoboLab wiring

- Robot cfg: `robolab/robots/bimanual_franka.py`
- Registration: `robolab/registrations/bimanual_franka/`
- Two-hand tasks: `robolab/tasks/bimanual/`
- Smoke test: `examples/run_bimanual_jointpos.py`
