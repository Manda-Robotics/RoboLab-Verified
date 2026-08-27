# MolmoAct2-DROID

This adapter evaluates [`allenai/MolmoAct2-DROID`](https://huggingface.co/allenai/MolmoAct2-DROID), an Apache-2.0 checkpoint trained on the filtered DROID Franka mixture with absolute joint-position control.

The model runs in Ai2's separate environment and RoboLab talks to its official FastAPI server. The adapter does not perform Cartesian or rotation-frame conversion.

## Start the server

Follow the installation instructions in [`allenai/molmoact2`](https://github.com/allenai/molmoact2), then start the DROID server from that repository:

```shell
uv sync
uv run hf download allenai/MolmoAct2-DROID
uv run python examples/droid/host_server_droid.py \
  --host 0.0.0.0 \
  --port 8000 \
  --dtype bfloat16
```

The upstream server uses `trust_remote_code=True`. Pin and audit a checkpoint revision for reproducible benchmark runs. BF16 normally fits under 16 GB VRAM; FP32 needs substantially more memory.

## Run RoboLab

```shell
uv run python policies/molmoact2_droid/run.py \
  --task BananaInBowlTask \
  --remote-host 127.0.0.1 \
  --remote-port 8000 \
  --headless \
  --video-mode all
```

## Compatibility contract

| MolmoAct2-DROID | RoboLab |
|---|---|
| exterior RGB, HWC `uint8` | `over_shoulder_left_camera` |
| wrist RGB, HWC `uint8` | `wrist_cam` |
| state `[q1..q7, gripper]` | seven Franka joints + RoboLab closedness (`0=open`, `1=closed`) |
| absolute joint-pose actions | `DroidJointPositionActionCfg` |
| 15-step action chunks | execute 15, then replan |
| continuous gripper position | binary at `0.5` for RoboLab's gripper action |

Image resizing, language normalization, DROID statistics (`franka_droid`), and continuous flow inference stay in the upstream server.
