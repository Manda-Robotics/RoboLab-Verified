# MolmoAct 2 on the bimanual YAM rig

[MolmoAct 2](https://allenai.org/blog/molmoact2) (Ai2, May 2026) ships a bimanual checkpoint,
[`allenai/MolmoAct2-BimanualYAM`](https://huggingface.co/allenai/MolmoAct2-BimanualYAM),
fine-tuned on 720 h of teleoperation on two I2RT YAM arms. `robolab/robots/bimanual_yam.py`
is that rig; this client speaks the checkpoint's contract unchanged.

## Serve the checkpoint

Ai2's server, from the [allenai/molmoact2](https://github.com/allenai/molmoact2) repo (bf16
needs about 16 GB of VRAM, CUDA 12.8):

```bash
git clone https://github.com/allenai/molmoact2 && cd molmoact2 && uv sync
uv run hf download allenai/MolmoAct2-BimanualYAM
uv run python examples/yam/host_server_yam.py --host 0.0.0.0 --port 8202 --dtype bfloat16
```

## Run

```bash
# the parity task (Ai2's box-packing layout with RoboLab stand-in objects)
python policies/molmoact2/run.py --task YamPutEverythingInBoxTask --task-dirs bimanual \
    --num-envs 4 --num-runs 2 --headless --server http://localhost:8202

# the benchmark tasks, on the same rig
python policies/molmoact2/run.py --task BananaInBowlTask --task-dirs benchmark --num-envs 4 --headless
```

`--top-cam ai2_desk` (default, the MolmoAct 2 research kit: D435 on a desk mount) or
`--top-cam i2rt_gantry` (I2RT's teleoperation station: D405 on the gantry crossbar).

## Contract

| | |
|---|---|
| images | `top_cam`, `left_wrist_cam`, `right_wrist_cam`, 640×360 RGB, sent as `[top, left, right]` |
| state | 14 floats: left joints 1–6, left gripper, right joints 1–6, right gripper; gripper 1 = open |
| actions | 30 × 14 absolute joint targets at 30 Hz per request; the client plays the whole chunk, as Ai2's harness does |
| env action | 16-dim `[left arm 6, left fingers 2, right arm 6, right fingers 2]`; the gripper value fills both finger joints (0 closed, −0.04695 m open) |
| wire | json_numpy over HTTP `POST /act`, `normalization_tag = yam_dual_molmoact2` |

`offline_tests/test_molmoact2_yam_client.py` exercises the mapping and a full round trip
against a fake server; `offline_tests/test_bimanual_yam_asset.py` pins the asset the contract
assumes.

## Results (2026-09-03, 4 envs)

| task | success |
|---|---|
| YamPutEverythingInBoxTask, 67 s cap | 5 / 8 (Ai2's ManiSkill harness, same checkpoint: 4 / 8) |
| RoboLab-120, 76 of 120 tasks × 4 episodes (2026-09-03 sweep, priority order, stopped when the pod budget ran out) | 5 / 304; BananasInBinOneMore 3/4, BananasInBinThreeTotal 2/4, every other task 0/4. Modes over the 304 episodes, from the event logs: never engages 81, touches the target without closing 93, wrong object 61, grasps and carries but never places 64 |
| BananaInBowlTask | 0 / 4: reaches the banana and hovers, never commands a grasp |
| BBQSauceInBinTask | 0 / 4: coherent grasp and lift of the wrong object |
| FoodPacking2CansTask | 0 / 4: grasps the cans, then commands release; 40–120 drops per episode |

The checkpoint is a fine-tune on Ai2's 34 tasks; the benchmark tasks are outside that
distribution. Treat the numbers as a baseline for a released checkpoint, not as the rig's ceiling.
