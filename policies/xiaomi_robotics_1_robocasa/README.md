# Xiaomi-Robotics-1 RoboCasa

This directory adapts
[`XiaomiRobotics/Xiaomi-Robotics-1-RoboCasa`](https://huggingface.co/XiaomiRobotics/Xiaomi-Robotics-1-RoboCasa)
for zero-shot evaluation on RoboLab's DROID embodiment. This is a
**cross-benchmark transfer experiment**, not a DROID-trained checkpoint or an
official Xiaomi/RoboLab baseline.

The released RoboCasa evaluator uses a Panda arm on an Omron mobile base, but
XR-1 predicts only the first seven controller dimensions (relative arm pose and
gripper); the evaluator leaves mobile-base and torso commands at zero. This
adapter mirrors that fixed-base policy behavior on RoboLab's Franka arm.

## Architecture

The integration is deliberately split into two processes:

1. `server.py` runs in Xiaomi's pinned PyTorch/Transformers environment. It
   loads both `AutoModel` and `AutoProcessor`, builds the official no-CoT
   prompt, and decodes the checkpoint's normalization statistics.
2. `client.py` runs inside RoboLab/Isaac Sim. It sends three RGB images, seven
   Franka joints, one emulated Panda gripper joint, and the instruction over a
   numeric-only msgpack/ZMQ protocol.

Keeping preprocessing on the server prevents `transformers==4.57.1`, custom
Hugging Face code, and FlashAttention from changing the Isaac Sim environment.

## Model server

Use a separate Python environment following Xiaomi's published requirements:
PyTorch 2.8.0, Transformers 4.57.1, FlashAttention 2, and a CUDA GPU with BF16
support. The server additionally needs `numpy`, `Pillow`, `pyzmq`, `msgpack`,
and `msgpack-numpy`.

From the RoboLab repository root:

```shell
python -m policies.xiaomi_robotics_1_robocasa.server \
  --model-path XiaomiRobotics/Xiaomi-Robotics-1-RoboCasa \
  --host 127.0.0.1 \
  --port 10086
```

The checkpoint requires `trust_remote_code=True`; the server enables it. For
reproducible and safer execution, download/inspect the checkpoint first or pass
an audited Hugging Face commit with `--revision`.

When the server runs on another machine, bind it to a private interface and
restrict TCP port 10086 at the network boundary. The protocol has no built-in
authentication or encryption.

## RoboLab client

Run a one-episode smoke test before scaling out:

```shell
uv run python policies/xiaomi_robotics_1_robocasa/run.py \
  --task GrabAFruitTask \
  --remote-host 127.0.0.1 \
  --remote-port 10086 \
  --headless \
  --video-mode all
```

Suggested first tasks are `GrabAFruitTask`, `PickDrillTask`, and
`BananaInBowlTask`. Start with `--num-envs 1`; one model server processes
requests serially, so multiple RoboLab environments improve simulator
throughput but not model inference throughput.

## Compatibility contract

| XR-1 RoboCasa contract | RoboLab adapter |
|---|---|
| `robot0_agentview_left` | `over_shoulder_left_camera` |
| `robot0_agentview_right` | `over_shoulder_right_camera` |
| `robot0_eye_in_hand` | `wrist_cam` |
| 256x256 image, 0.95 center crop | central square resize, then 0.95 crop and bilinear resize |
| state: 7 Panda joints + first gripper joint | 7 Franka joints + `0.04 * (1 - closedness)` |
| 20 Hz controller | RoboLab 120 Hz physics, decimation 6 |
| normalized XYZ action, ±0.05 m | pre-scaled for `DroidRelIKActionCfg(scale=0.5)` |
| normalized rotation, ±0.5 rad | pre-scaled for `DroidRelIKActionCfg(scale=0.5)` |
| gripper: negative open, positive close | binary `0` open, `1` close |
| action chunk 10 | execute 10, then replan |

The translation and rotation limits are CLI-configurable. Treat changes as
adapter calibration, and record them with benchmark results.

## Known domain gaps

- RoboCasa's base cameras are attached to a parked mobile manipulator; RoboLab's
  cameras are fixed over-the-shoulder views.
- RoboCasa uses a Panda gripper while RoboLab uses a Robotiq 2F-85.
- Kitchen-task post-training does not make this a DROID policy. Report results
  as `XR-1-RoboCasa zero-shot cross-benchmark`.
