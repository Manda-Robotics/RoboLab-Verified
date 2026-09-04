# Built-in Robots

This is the canonical list of robot embodiments that ship with RoboLab. For how to *use* a robot
(registration wiring, defining your own robot, contact grippers, wrist cameras), see
[`docs/robots.md`](../../docs/robots.md).

| | Robot | Embodiment | Action spaces | Cameras |
|---|-------|------------|---------------|---------|
| <img src="../../docs/images/robots/droid.png" width="480"> | **DROID**<br>(Franka + Robotiq 2F-85)<br>`droid.py` | `single-arm` `fixed-base` `parallel-jaw` | joint position, absolute EE IK, relative EE IK | wrist |
| <img src="../../docs/images/robots/franka.png" width="480"> | **Franka Panda**<br>`franka.py`, `franka_high_pd.py` | `single-arm` `fixed-base` `parallel-jaw` | joint position, absolute EE IK, relative EE IK | — |
| <img src="../../docs/images/robots/kinova_gen3.png" width="480"> | **Kinova Gen3**<br>(Gen3 7-DoF + Robotiq 2F-85)<br>`kinova_gen3.py` | `single-arm` `fixed-base` `parallel-jaw` | joint position | wrist |
| | **Bimanual YAM**<br>(2× I2RT YAM + linear gripper)<br>`bimanual_yam.py` | `bimanual` `fixed-base` `parallel-jaw` | joint position (16-dim) | top + 2× wrist |
| | **Dual Franka**<br>(2× Franka + Robotiq 2F-85)<br>`bimanual_franka.py` | `bimanual` `fixed-base` `parallel-jaw` | joint position (16-dim) | 2× wrist |
| | **Bimanual ViperX / ALOHA**<br>`aloha.py`, `bimanual_station.py` | `bimanual` `fixed-base` / `mobile` `parallel-jaw` | joint position (14-dim) | 2× wrist + high |

Gripper convention for all binary gripper actions: a scalar per gripper, `> 0.5` closes, `≤ 0.5` opens.
Quaternions are `(w, x, y, z)`; absolute IK targets are expressed in the robot root frame, translations in meters.

---

## DROID (Franka + Robotiq 2F-85)

`tags: single-arm · fixed-base · parallel-jaw · wrist-cam · gravity-disabled · high-PD · benchmark-default`

The default benchmark embodiment: a Franka Panda arm with a Robotiq 2F-85 gripper, matching the
[DROID](https://droid-dataset.github.io/) platform. High PD gains (400/80) with gravity disabled on the
arm, plus a 720p wrist camera whose intrinsics are calibrated to match pi05 / DreamZero training data.

| Action config | Layout | Dim |
|---------------|--------|-----|
| `DroidJointPositionActionCfg` | 7 arm joint targets + binary gripper | 8 |
| `DroidIKActionCfg` | absolute EE pose `(x, y, z, qw, qx, qy, qz)` + binary gripper | 8 |
| `DroidRelIKActionCfg` | relative EE pose `(dx, dy, dz, droll, dpitch, dyaw)` + binary gripper | 7 |

- **Config classes:** `DroidCfg` (robot + wrist camera + EE frame transformers)
- **Proprioception:** `ProprioceptionObservationCfg` — arm joint positions, gripper open fraction,
  EE pose (both the gripper mount flange `ee_*` and the rotated control frame `eef_*`)
- **Contact gripper:** `{"gripper": ["gripper_left", "gripper_right"], ...}` — one sensor per finger pad, `gripper` = either pad
- **Friction bodies:** `["left_inner_finger", "right_inner_finger"]` — the targets of a `--friction` override ([physics](../../docs/physics.md#friction))
- **Registrations:** `robolab/registrations/droid/` (jointpos, abs-IK, rel-IK, lighting/background variations)

```python
from robolab.robots.droid import DroidCfg, DroidJointPositionActionCfg, contact_gripper
```

## Franka Panda

`tags: single-arm · fixed-base · parallel-jaw`

A stock Franka Panda with its factory finger gripper. Two articulation variants share the same action
configs: `franka.py` with standard PD gains (80/4), and `franka_high_pd.py` with high gains (400/80)
and gravity disabled (better target tracking for policy control).

| Action config | Layout | Dim |
|---------------|--------|-----|
| `FrankaJointPositionActionCfg` | 7 arm joint targets + binary gripper | 8 |
| `FrankaIKActionCfg` | absolute EE pose `(x, y, z, qw, qx, qy, qz)` + binary gripper | 8 |
| `FrankaRelIKActionCfg` | relative EE pose `(dx, dy, dz, droll, dpitch, dyaw)` + binary gripper | 7 |

- **Config classes:** `FrankaCfg` (one per variant file; action configs in `franka_definitions.py`)
- **Proprioception:** EE frame pose and finger joint positions (`franka_definitions.py`)
- **Contact gripper:** `{"gripper": ...panda_leftfinger}`

```python
from robolab.robots.franka import FrankaCfg                      # standard PD
from robolab.robots.franka_high_pd import FrankaCfg              # high PD, gravity disabled
from robolab.robots.franka_definitions import FrankaJointPositionActionCfg, contact_gripper
```

## Kinova Gen3 (Gen3 7-DoF + Robotiq 2F-85)

`tags: single-arm · fixed-base · parallel-jaw · wrist-cam · gravity-disabled`

A Kinova Gen3 7-DoF arm with a Robotiq 2F-85 gripper, welded to the world at the base
(`fix_root_link=True`, gravity disabled on the arm). The USD is vendored under
`assets/robots/kinova_gen3_robotiq_2f85/` and derived from Kinova `ros2_kortex` and PickNik
`ros2_robotiq_gripper` at pinned revisions — see the README and license files in that folder.

The gripper's six joints are driven together from one binary action rather than through PhysX mimic
constraints; the signed per-joint targets live in `GRIPPER_JOINT_COMMANDS`.

| Action config | Layout | Dim |
|---------------|--------|-----|
| `KinovaJointPositionActionCfg` | 7 arm joint targets + binary gripper | 8 |

- **Config classes:** `KinovaGen3Cfg` (robot + wrist camera + EE frame transformer),
  `KinovaWristCameraCfg` (exposes the wrist camera to image observations)
- **Proprioception:** `KinovaProprioceptionObservationCfg` — arm joint positions, gripper open
  fraction, EE pose at `robotiq_85_base_link`
- **Contact gripper:** `{"gripper": ...robotiq_85_.*_finger_tip_link}`
- **Registrations:** `robolab/registrations/kinova/` (jointpos)

```python
from robolab.robots.kinova_gen3 import (
    KinovaGen3Cfg,
    KinovaJointPositionActionCfg,
    KinovaProprioceptionObservationCfg,
    contact_gripper,
)
```

Actuator gains are simulation defaults, not measured from hardware — this is a functional
simulation model rather than a calibrated digital twin.

---

Also in this folder: `delta_actions.py`, a helper that converts a target EE pose into a relative
(delta) pose action — used by trajectory replay, not an action space itself.

The robot stills above are rendered in an empty scene at each robot's reset posture.

## Bimanual YAM (2× I2RT YAM + linear gripper)

`tags: bimanual · fixed-base · parallel-jaw · top-cam + 2× wrist-cam · released policy`

Two I2RT YAM arms 0.48 m apart at the table edge as one articulation, the side-by-side station
most bimanual training data is collected on (Ai2's MolmoAct 2 kit, I2RT's teleoperation
stations). 16-dim joint-position action (`[6 arm, 2 finger] × 2`, fingers in metres, 0 closed
and −0.04695 open), per-arm end-effector recording, an overhead camera (`--top-cam ai2_desk`,
D435 on a desk mount, or `i2rt_gantry`, D405 on the station crossbar) and wrist cameras on
I2RT's D405 bracket, all at 640×360. Gravity is compensated (as on the real arm) and the PD
gains are Ai2's. Driven by Ai2's released MolmoAct 2 bimanual checkpoint through
`policies/molmoact2/run.py`; results in [`README_bimanual.md`](README_bimanual.md).

- **Config classes:** `BimanualYamCfg`
- **Asset:** `assets/robots/bimanual_yam/bimanual_yam.usd`, built from I2RT's URDF
  (`assets/robots/yam_i2rt_v1/`, MIT) by `python assets/robots/_utils/build_bimanual_yam.py`
  (needs only `usd-core` and `numpy`; no Isaac importer)
- **Registrations:** `robolab/registrations/bimanual_yam/`
- **Guide:** [`docs/bimanual_yam.md`](../../docs/bimanual_yam.md)
- **Smoke test:** `python examples/run_bimanual_yam_jointpos.py --headless`; scripted run through the harness: `python policies/bimanual/run.py --robot yam --task YamPutEverythingInBoxTask --headless`
- **Parity task:** `YamPutEverythingInBoxTask` (`--task-dirs bimanual`), Ai2's box-packing layout

## Dual Franka (2× Franka + Robotiq 2F-85)

`tags: bimanual · fixed-base · parallel-jaw · 2× wrist-cam · scripted-only`

Two DROID arms on one table fixture as a single articulation: 16-dim joint-position action
(`[7 arm, 1 gripper] × 2`), per-arm end-effector recording (`left_ee_pose` / `right_ee_pose`),
tracking wrist cameras, and the unchanged benchmark predicates (`gripper_name="gripper"` means
*either* hand; a list means both). Verified with the scripted lift 6 of 6 clean. No released
checkpoint drives two arms; `policies/bimanual/run.py` runs the scripted client.

- **Config classes:** `BimanualFrankaCfg`
- **Asset:** `assets/robots/bimanual_franka_robotiq_2f85/` (rebuild with
  `python assets/robots/_utils/build_bimanual_franka.py`, needs only `usd-core`)
- **Registrations:** `robolab/registrations/bimanual_franka/`
- **Smoke-test tasks:** `robolab/tasks/bimanual/` (`--task-dirs bimanual`; outside the benchmark set)

## Bimanual ViperX (ALOHA)

`tags: bimanual · fixed-base or mobile · parallel-jaw · 2× wrist-cam · no working policy · asset not shipped`

> The ALOHA 2 asset (`assets/robots/aloha2/`) and the MJCF import scripts are not in this repository yet; the config is here so the observation and action layouts are documented, but the rig does not load from a clean clone.

Two ViperX 300 arms in the opposing (`aloha.py`), station and mobile (`bimanual_station.py`)
configurations, 14-dim action. The rig runs — arms build, wrist cameras track, per-arm metrics
record — but **no released policy works on it**: the π0.5 base checkpoint scored 0 of 6
(coherent reach, twitchy, no grasp). An ALOHA number is a statement about the checkpoint, not
the rig; see [`README_bimanual.md`](README_bimanual.md) before placing one next to a Franka
number.

- **Config classes:** `AlohaCfg`, `AlohaGymMatchCfg`, `MobileAlohaCfg`, `BimanualStationCfg`
- **Registrations:** `robolab/registrations/aloha/`
- **Run:** `python policies/bimanual/run.py --robot aloha --task AlohaTransferCubeTask --headless`
