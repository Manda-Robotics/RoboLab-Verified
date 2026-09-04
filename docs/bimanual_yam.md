# Bimanual YAM in RoboLab Verified

Two I2RT YAM arms side by side at the edge of the task table, one overhead camera and one
wrist camera per arm. This is the station most bimanual training data is collected on (Ai2's
MolmoAct 2 research kit, I2RT's teleoperation stations), and the one rig in this repository
with a released policy that drives it: Ai2's `MolmoAct2-BimanualYAM` checkpoint.

Everything below runs from a fresh clone. Isaac Sim 5.1 and Isaac Lab are needed for the
simulator steps (Linux with an NVIDIA GPU; see the top-level [README](../README.md#installation)).
The checks under "Without Isaac" run on any machine with three packages.

## Without Isaac: the asset and the client contract

```bash
git lfs pull                                         # the USD and STL files are LFS objects (a clone with git-lfs installed already did this)
pip install pytest numpy usd-core                    # all the no-Isaac checks need
python -m pytest offline_tests/test_bimanual_yam_asset.py offline_tests/test_molmoact2_yam_client.py   # 17 tests
python assets/robots/_utils/build_bimanual_yam.py    # rebuilds the USD from I2RT's URDF, a few seconds
```

Tried from a fresh clone on 2026-09-03: clone 93 s, LFS 5 s, 17 tests and the rebuild in
under a minute.

The asset tests compare every link's position and orientation in the USD with the URDF's
forward kinematics, pin the finger travel and the wrist-camera frame, and check the collision
and friction materials. The client test runs a full request/response round trip against a
fake inference server.

## See the rig move (no checkpoint)

```bash
python examples/run_bimanual_yam_jointpos.py --headless
python policies/bimanual/run.py --robot yam --task YamPutEverythingInBoxTask --headless
```

The first is the smoke test: both arms sweep to Ai2's rest pose and back, both grippers open
and close, and the script asserts that the wrist cameras follow the grippers, that the fingers
reach full travel, and that the simulated flange lands within a few millimetres of the URDF
forward kinematics. It writes the three policy camera views and the viewport video. The second
drives the rig through the ordinary evaluation harness with a scripted client, so the
recorders, event tracker and dashboard all see a YAM run.

## Run the released policy

MolmoAct 2 is served from Ai2's repository (about 16 GB of VRAM in bf16, CUDA 12.8):

```bash
git clone https://github.com/allenai/molmoact2 && cd molmoact2 && uv sync
uv run hf download allenai/MolmoAct2-BimanualYAM
uv run python examples/yam/host_server_yam.py --host 0.0.0.0 --port 8202 --dtype bfloat16
```

Then, from RoboLab Verified:

```bash
# the parity task: Ai2's own box-packing layout with RoboLab stand-in objects
python policies/molmoact2/run.py --task YamPutEverythingInBoxTask --task-dirs bimanual \
    --num-envs 4 --num-runs 2 --headless --server http://localhost:8202

# any benchmark task on the same rig
python policies/molmoact2/run.py --task BananaInBowlTask --task-dirs benchmark --num-envs 4 --headless
```

Results land in `output/<run>/` like every other run and show up in the dashboard.

## What to expect

| task set | result (2026-09-03) |
|---|---|
| parity task, 67 s cap | 5 / 8 (the same checkpoint in Ai2's ManiSkill harness: 4 / 8) |
| RoboLab-120, 76 of 120 tasks, 4 episodes each | 5 / 304, both on bananas-into-bin tasks |

The checkpoint is a fine-tune on Ai2's own tasks and objects; the benchmark is outside that
distribution. Over the 304 benchmark episodes the failures split into four modes, counted from
the event logs: never engages an object (81), touches the target but never closes on it (93),
grasps the wrong object (61), grasps and carries the right object but never places it (64).
Details and per-task rows: [`policies/molmoact2/README.md`](../policies/molmoact2/README.md).

## Bring your own bimanual policy

The rig speaks one contract, and the MolmoAct 2 client is a complete example of it
(`policies/molmoact2/yam_client.py`, about 150 lines). A policy client subclasses
`robolab.eval.base_client.InferenceClient` and implements four hooks:

| hook | what it gets / returns |
|---|---|
| `_extract_observation(raw_obs, env_id)` | images `top_cam`, `left_wrist_cam`, `right_wrist_cam` (640×360 RGB, one per env) and proprio `left_arm_joint_pos` (6), `left_gripper_pos` (1, 1 = open), `right_arm_joint_pos`, `right_gripper_pos` |
| `_pack_request` / `_query_server` / `_unpack_response` | whatever your server speaks; return a chunk of actions |
| `_postprocess_chunk(chunk)` | map to the env action, 16 floats: `[left arm 6, left fingers 2, right arm 6, right fingers 2]`, arm targets in radians (absolute), finger targets in metres (0 closed, −0.04695 open) |

`open_loop_horizon` sets how many actions of a chunk play before the next request. Control runs
at 30 Hz. A relative-IK action space is not offered for this rig; joint position is what the
released checkpoints emit.

The quickest start is to copy [`policies/yam_template/`](../policies/yam_template/): a client
written against the table above (its stand-in model wobbles the elbows and cycles the grippers)
and the runner that registers the env and hands it to the harness. Replace `_query_server`
with your model, keep `_postprocess_chunk`, run:

```bash
python policies/yam_template/run.py --task YamPutEverythingInBoxTask --task-dirs bimanual --num-envs 2 --headless
```

`offline_tests/test_yam_template_client.py` checks the template against the contract without Isaac.

## The rig itself

- Config: `robolab/robots/bimanual_yam.py`. Bases 0.48 m apart at x = 0 on the mount plane,
  facing +x; gravity compensated with Ai2's PD gains; start pose all joints at zero, grippers open.
- Cameras: overhead `--top-cam ai2_desk` (D435 on a desk mount, 0.15 m ahead of the bases, 0.80 m
  up, 80° down; the MolmoAct 2 kit) or `i2rt_gantry` (D405 on the station crossbar); wrist cameras
  on I2RT's D405 bracket extrinsics, riding the gripper bodies. All 640×360, the training format.
- Asset: `assets/robots/bimanual_yam/bimanual_yam.usd`, built by
  `assets/robots/_utils/build_bimanual_yam.py` from `assets/robots/yam_i2rt_v1/` (I2RT's URDF and
  meshes, MIT). White upper-arm and forearm shells, black joints and gripper, as on the real arm.
- Observation and action layouts: `BimanualYamProprioceptionObservationCfg`,
  `BimanualYamJointPositionActionCfg` in the config file.
