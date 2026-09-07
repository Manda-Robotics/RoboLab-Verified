# G0.5-DROID

This adapter runs RoboLab's DROID joint-position environments against
[OpenGalaxea/G05](https://huggingface.co/OpenGalaxea/G05). It follows the
official [`GalaxeaVLA`](https://github.com/OpenGalaxea/GalaxeaVLA) DROID
WebSocket protocol and does not alter any shared RoboLab controller code.

## Server

Set up the upstream repository and request access to the gated checkpoint as
described by Galaxea. Download the model bundle into its expected local layout:

```bash
huggingface-cli download OpenGalaxea/G05 \
  --repo-type model \
  --local-dir checkpoints
```

From the GalaxeaVLA repository, launch its DROID experiment server:

```bash
CHECKPOINT_DIR=checkpoints/g05-droid \
POLICY_PORT=8000 \
POLICY_DEVICE=cuda:0 \
bash experiments/droid/start_server.sh \
  model.model_arch.discrete_action=true \
  model.model_arch.continuous_action=false
```

The checkpoint is published under Galaxea's non-commercial research license;
review its terms before using it in a benchmark or publishing derived results.

## RoboLab evaluation

```bash
python policies/g05_droid/run.py \
  --remote-host POLICY_SERVER_HOST \
  --remote-port 8000 \
  --task BananaInBowlTask \
  --num-runs 3 \
  --video-mode all
```

`--remote-uri ws://HOST:PORT` is also supported. With `--num-envs N`, the
adapter creates an independent WebSocket for each environment because the
upstream server's action-chunk cache is connection-local.

## Compatibility choices

| Boundary | Adapter behavior |
|---|---|
| Environment/control | DROID absolute 7-DoF joint position plus binary gripper |
| Cameras | RoboLab left exterior + wrist, converted from HWC to CHW `uint8` |
| Third image slot | Zero-filled right-wrist image, matching Galaxea's DROID client |
| Robot state | Seven joints plus gripper; gripper polarity is inverted for G0.5 |
| Action cadence | One server response per simulator step; `{}` advances a cached server chunk |
| Parallel evaluation | One stateful connection per RoboLab environment |

G0.5 is already a DROID policy, so no RoboCasa/VLABench coordinate-frame
conversion is involved.
