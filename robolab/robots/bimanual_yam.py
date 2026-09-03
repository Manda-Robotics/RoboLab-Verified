# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Bimanual YAM: two I2RT YAM arms (linear_4310 grippers, D405 wrist cameras) side by side.

This is the rig Ai2's MolmoAct 2 bimanual checkpoint was trained on, laid out as in Ai2's
own ManiSkill evaluation of that checkpoint (``sim_eval/robots/bimanual_yam.py``):

- bases 0.48 m apart on the mount plane, both facing +x (the workspace);
- ``left_*`` is the arm at +y (the robot's left when facing the workspace), ``right_*`` at -y;
- top camera 0.15 m in front of the bases, 0.80 m up, pitched 80 deg down (D435, 69.4 deg HFOV);
- wrist cameras on I2RT's D405 bracket: flange -> camera (-0.07, 0, -0.077), optical axis
  canted 25 deg toward the fingertips (from I2RT's station README; the asset carries the
  frame under ``{side}_gripper``, and the cfg rides the body by pose);
- gravity compensated (Ai2: ``gravcomp=1``), so ``disable_gravity=True`` with low PD gains.

Joint-space contract (what the policy sees and commands), per arm: six joints in radians,
gripper in [0, 1] with **1 = open, 0 = closed** (I2RT convention). The asset's finger joints
run from 0 (closed) to -0.04695 m (open); the observation and the action term convert.

Assets: ``assets/robots/bimanual_yam/bimanual_yam.usd`` built by
``assets/robots/_utils/build_bimanual_yam.py`` from I2RT's URDF.
"""

import math
import os

import isaaclab.sim as sim_utils
import numpy as np
import torch
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg
from isaaclab.envs import ManagerBasedRLEnv, mdp
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import TiledCameraCfg
from isaaclab.utils import configclass
from isaaclab.utils.math import subtract_frame_transforms

from robolab.constants import ROBOTS_DIR
from robolab.core.environments.scene_fixture import FRANKA_TABLE_FIXTURE
from robolab.core.sensors.body_attached_camera import BodyAttachedTiledCameraCfg
from robolab.robots.droid import _to_torch

ARMS = ("left", "right")
ARM_JOINT_NAMES = {side: [f"{side}_joint{i}" for i in range(1, 7)] for side in ARMS}
FINGER_JOINT_NAMES = {side: [f"{side}_joint7", f"{side}_joint8"] for side in ARMS}
EE_BODY_NAME = {side: f"{side}_gripper" for side in ARMS}     # the flange (joint6 output)
FINGER_TRAVEL_M = 0.04695       # joint7/8 range: 0 closed .. -0.04695 open
ARM_SPACING_M = 0.48
# Where the rig sits on the mount plane: the DROID Franka's mount point, bases split to +/-y.
# The home-pose fingertips reach x = 0.17 and the nearest object in a benchmark scene sits at
# x = 0.21-0.34 (median 0.34), i.e. Ai2's own 0.35 m workspace distance. Moving the rig forward
# spawns the arms inside the nearest objects.
RIG_X_M = 0.0

# Start pose: every arm joint at zero (Ai2's sim "home" and their real launcher's
# ``start_joints``), grippers OPEN as on the real rig (``start_joints[6] = 1.0``). Ai2's sim
# starts them closed; one of our eight parity episodes stalled at the start pose with closed
# fingers, so the real-robot start is used.
_HOME_ARM = {f"joint{i}": 0.0 for i in range(1, 7)}
_HOME_FINGER_M = -FINGER_TRAVEL_M

# ENV_REGEX_NS is doubled so .format(side=...) leaves it for IsaacLab to substitute per env.
_GRIPPER_PRIM = "{{ENV_REGEX_NS}}/robot/{side}_arm/{side}_gripper"

########################################################
# Cameras (MolmoAct 2 training format: 640x360 RGB)
########################################################

_IMG_W, _IMG_H = 640, 360
_APERTURE_MM = 20.955          # Isaac's default horizontal aperture; focal length sets the FOV


def _pinhole(hfov_deg: float) -> sim_utils.PinholeCameraCfg:
    focal = _APERTURE_MM / (2.0 * math.tan(math.radians(hfov_deg) / 2.0))
    return sim_utils.PinholeCameraCfg(
        focal_length=focal,
        focus_distance=400.0,
        horizontal_aperture=_APERTURE_MM,
        vertical_aperture=_APERTURE_MM * _IMG_H / _IMG_W,
        clipping_range=(0.01, 20.0),
    )


def _quat_from_columns(right, down, forward) -> tuple[float, float, float, float]:
    """(w, x, y, z) of the rotation whose columns are the ROS optical axes (x right, y down,
    z forward) expressed in the parent frame."""
    m = np.array([right, down, forward], dtype=float).T
    w = math.sqrt(max(0.0, 1.0 + m[0, 0] + m[1, 1] + m[2, 2])) / 2.0
    x = math.copysign(math.sqrt(max(0.0, 1.0 + m[0, 0] - m[1, 1] - m[2, 2])) / 2.0, m[2, 1] - m[1, 2])
    y = math.copysign(math.sqrt(max(0.0, 1.0 - m[0, 0] + m[1, 1] - m[2, 2])) / 2.0, m[0, 2] - m[2, 0])
    z = math.copysign(math.sqrt(max(0.0, 1.0 - m[0, 0] - m[1, 1] + m[2, 2])) / 2.0, m[1, 0] - m[0, 1])
    return (w, x, y, z)


# Top camera: 0.15 m ahead of the bases, 0.80 m up, optical axis 80 deg below horizontal
# toward +x. Image right = -y (the robot's right), image down = toward the robot.
_TOP_PITCH_DEG = 80.0
_TOP_FORWARD = (math.cos(math.radians(_TOP_PITCH_DEG)), 0.0, -math.sin(math.radians(_TOP_PITCH_DEG)))
_TOP_RIGHT = (0.0, -1.0, 0.0)
_TOP_DOWN = tuple(np.cross(_TOP_FORWARD, _TOP_RIGHT).tolist())
_TOP_CAM = TiledCameraCfg(
    prim_path="{ENV_REGEX_NS}/robot/torso/top_cam",
    height=_IMG_H,
    width=_IMG_W,
    data_types=["rgb"],
    spawn=_pinhole(69.4),          # RealSense D435 colour, as in Ai2's sim
    offset=TiledCameraCfg.OffsetCfg(
        pos=(0.15, 0.0, 0.80),
        rot=_quat_from_columns(_TOP_RIGHT, _TOP_DOWN, _TOP_FORWARD),
        convention="ros",
    ),
)

# I2RT gantry station (YAM Cell / Teleoperation Station, ``yam_station_linear_4310_d405``):
# a third D405 on the crossbar, on the midline between the arms, 0.954 m above the base plane
# and 0.166 m *behind* the bases, optical axis 60 deg below horizontal, meeting the table
# 0.384 m in front of the arms. Same image geometry as the wrists (D405, 87 deg HFOV).
_GANTRY_PITCH_DEG = 60.0
_GANTRY_FORWARD = (math.cos(math.radians(_GANTRY_PITCH_DEG)), 0.0, -math.sin(math.radians(_GANTRY_PITCH_DEG)))
_TOP_CAM_GANTRY = TiledCameraCfg(
    prim_path="{ENV_REGEX_NS}/robot/torso/top_cam",
    height=_IMG_H,
    width=_IMG_W,
    data_types=["rgb"],
    spawn=_pinhole(87.0),
    offset=TiledCameraCfg.OffsetCfg(
        pos=(-0.166, 0.0, 0.954),
        rot=_quat_from_columns(_TOP_RIGHT, tuple(np.cross(_GANTRY_FORWARD, _TOP_RIGHT).tolist()), _GANTRY_FORWARD),
        convention="ros",
    ),
)

TOP_CAMERA_VARIANTS = {
    "ai2_desk": _TOP_CAM,          # MolmoAct 2 kit: D435 on a desk mount (default)
    "i2rt_gantry": _TOP_CAM_GANTRY,  # I2RT station: D405 on the gantry crossbar
}


def set_top_camera(variant: str) -> None:
    """Pick the overhead camera before registering envs (class-level cfg field)."""
    if variant not in TOP_CAMERA_VARIANTS:
        raise ValueError(f"top camera must be one of {sorted(TOP_CAMERA_VARIANTS)}, got {variant!r}")
    BimanualYamCfg.top_cam = TOP_CAMERA_VARIANTS[variant]
    BimanualYamCamerasCfg.top_cam = TOP_CAMERA_VARIANTS[variant]


# Wrist cameras: I2RT station README, flange -> camera, ROS optical (+z forward).
_WRIST_OFFSET_POS = (-0.0704, 0.0, -0.077)
_WRIST_OFFSET_ROT = (0.1531, 0.6903, 0.6903, 0.1530)


def _wrist_cam(side: str) -> BodyAttachedTiledCameraCfg:
    # Rides the gripper body by pose (robolab/core/sensors/body_attached_camera.py): a
    # hierarchy-parented camera on an imported arm never moved with it.
    return BodyAttachedTiledCameraCfg(
        prim_path=f"{{ENV_REGEX_NS}}/cam_{side}_wrist",
        articulation_prim_path_expr="{ENV_REGEX_NS}/robot",
        body_name=EE_BODY_NAME[side],
        offset=TiledCameraCfg.OffsetCfg(pos=_WRIST_OFFSET_POS, rot=_WRIST_OFFSET_ROT, convention="ros"),
        height=_IMG_H,
        width=_IMG_W,
        data_types=["rgb"],
        spawn=_pinhole(87.0),      # RealSense D405
    )


_CAM_LEFT_WRIST = _wrist_cam("left")
_CAM_RIGHT_WRIST = _wrist_cam("right")

########################################################
# Articulation
########################################################

@configclass
class BimanualYamCfg:
    """Fixed-base bimanual YAM articulation plus its three cameras."""

    robot = ArticulationCfg(
        prim_path="{ENV_REGEX_NS}/robot",
        spawn=sim_utils.UsdFileCfg(
            usd_path=os.path.join(ROBOTS_DIR, "bimanual_yam", "bimanual_yam.usd"),
            activate_contact_sensors=True,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=True,           # = Ai2's gravcomp=1
                max_depenetration_velocity=5.0,
            ),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=False,
                solver_position_iteration_count=64,
                solver_velocity_iteration_count=0,
            ),
            # Colours live in the asset: white upper-arm and forearm shells, black joints and
            # gripper, from Ai2's rig photo.
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(RIG_X_M, 0.0, 0.0),
            rot=(1.0, 0.0, 0.0, 0.0),
            joint_pos={
                **{f"{side}_{j}": v for side in ARMS for j, v in _HOME_ARM.items()},
                **{j: _HOME_FINGER_M for side in ARMS for j in FINGER_JOINT_NAMES[side]},
            },
        ),
        soft_joint_pos_limit_factor=1.0,
        # Gains = Ai2's ManiSkill config, which mirrors I2RT's MJCF actuator classes
        # (dm4340 on joints 1-3, joint 4 softer, dm4310 on joints 5-6).
        actuators={
            "shoulders": ImplicitActuatorCfg(
                joint_names_expr=[".*_joint[1-3]"],
                effort_limit=28.0, velocity_limit=3.14, stiffness=40.0, damping=2.5,
            ),
            "elbow": ImplicitActuatorCfg(
                joint_names_expr=[".*_joint4"],
                effort_limit=10.0, velocity_limit=3.14, stiffness=20.0, damping=0.5,
            ),
            "wrists": ImplicitActuatorCfg(
                joint_names_expr=[".*_joint[5-6]"],
                effort_limit=10.0, velocity_limit=3.14, stiffness=10.0, damping=1.0,
            ),
            "fingers": ImplicitActuatorCfg(
                joint_names_expr=[".*_joint[7-8]"],
                effort_limit=40.0, velocity_limit=1.0, stiffness=2000.0, damping=40.0,
            ),
        },
    )

    top_cam = _TOP_CAM
    left_wrist_cam = _CAM_LEFT_WRIST
    right_wrist_cam = _CAM_RIGHT_WRIST


# Labels assigned after the class body so configclass does not turn them into fields.
BimanualYamCfg.table_fixture = FRANKA_TABLE_FIXTURE       # the mount table under the profile
BimanualYamCfg.ee_recorder_bodies = {
    "left_ee_pose": EE_BODY_NAME["left"],
    "right_ee_pose": EE_BODY_NAME["right"],
}
# P79 friction override targets: the four finger tips. The asset binds a 3.0 / 2.5 pad material
# on exactly these bodies (build_bimanual_yam.py::FINGER_FRICTION), so `--friction` lands there.
BimanualYamCfg.friction_bodies = [f"{side}_tip_{finger}" for side in ARMS for finger in ("left", "right")]


@configclass
class BimanualYamCamerasCfg:
    """Exposes the three policy cameras to ``generate_image_obs_from_cameras`` under the
    names MolmoAct 2 expects (``top_cam``, ``left_wrist_cam``, ``right_wrist_cam``)."""

    top_cam = _TOP_CAM
    left_wrist_cam = _CAM_LEFT_WRIST
    right_wrist_cam = _CAM_RIGHT_WRIST


########################################################
# Contact gripper
########################################################

# Links are siblings under the arm Xform (flat hierarchy), so the finger body is
# {ENV_REGEX_NS}/robot/{side}_arm/{side}_tip_left, not a child of the gripper.
contact_gripper = {
    "left": "{ENV_REGEX_NS}/robot/left_arm/left_tip_left",
    "right": "{ENV_REGEX_NS}/robot/right_arm/right_tip_left",
    "gripper": ["left", "right"],
}

########################################################
# Observations
########################################################


def _joint_pos(env, names, asset_name="robot"):
    robot = env.scene[asset_name]
    idx = [robot.data.joint_names.index(n) for n in names]
    return _to_torch(robot.data.joint_pos)[:, idx]


def _gripper_open(env, side, asset_name="robot"):
    """1 = open, 0 = closed (I2RT / MolmoAct 2 convention), from the first finger joint."""
    q = _joint_pos(env, FINGER_JOINT_NAMES[side][:1], asset_name)
    return torch.clamp(-q / FINGER_TRAVEL_M, 0.0, 1.0)


def _ee_pose(env, side, asset_name="robot"):
    robot = env.scene[asset_name]
    body_idx = robot.data.body_names.index(EE_BODY_NAME[side])
    return subtract_frame_transforms(
        _to_torch(robot.data.root_pos_w), _to_torch(robot.data.root_quat_w),
        _to_torch(robot.data.body_pos_w)[:, body_idx, :], _to_torch(robot.data.body_quat_w)[:, body_idx, :],
    )


def left_arm_joint_pos(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")):
    return _joint_pos(env, ARM_JOINT_NAMES["left"], asset_cfg.name)


def right_arm_joint_pos(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")):
    return _joint_pos(env, ARM_JOINT_NAMES["right"], asset_cfg.name)


def left_gripper_pos(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")):
    return _gripper_open(env, "left", asset_cfg.name)


def right_gripper_pos(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")):
    return _gripper_open(env, "right", asset_cfg.name)


def left_ee_pos(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")):
    return _ee_pose(env, "left", asset_cfg.name)[0]


def left_ee_quat(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")):
    return _ee_pose(env, "left", asset_cfg.name)[1]


def right_ee_pos(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")):
    return _ee_pose(env, "right", asset_cfg.name)[0]


def right_ee_quat(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")):
    return _ee_pose(env, "right", asset_cfg.name)[1]


@configclass
class BimanualYamProprioceptionObservationCfg(ObsGroup):
    left_arm_joint_pos = ObsTerm(func=left_arm_joint_pos)
    left_gripper_pos = ObsTerm(func=left_gripper_pos)
    left_ee_pos = ObsTerm(func=left_ee_pos)
    left_ee_quat = ObsTerm(func=left_ee_quat)
    right_arm_joint_pos = ObsTerm(func=right_arm_joint_pos)
    right_gripper_pos = ObsTerm(func=right_gripper_pos)
    right_ee_pos = ObsTerm(func=right_ee_pos)
    right_ee_quat = ObsTerm(func=right_ee_quat)

    def __post_init__(self) -> None:
        self.enable_corruption = False
        self.concatenate_terms = False


########################################################
# Actions: 16-dim [left arm 6, left fingers 2, right arm 6, right fingers 2]
########################################################
# Finger targets are joint positions in metres (0 closed .. -0.04695 open). The MolmoAct 2
# client converts the policy's [0, 1] gripper into both finger slots; see
# policies/molmoact2/yam_client.py.


@configclass
class BimanualYamJointPositionActionCfg:
    left_arm = mdp.JointPositionActionCfg(
        asset_name="robot", joint_names=ARM_JOINT_NAMES["left"], preserve_order=True, use_default_offset=False)
    left_fingers = mdp.JointPositionActionCfg(
        asset_name="robot", joint_names=FINGER_JOINT_NAMES["left"], preserve_order=True, use_default_offset=False)
    right_arm = mdp.JointPositionActionCfg(
        asset_name="robot", joint_names=ARM_JOINT_NAMES["right"], preserve_order=True, use_default_offset=False)
    right_fingers = mdp.JointPositionActionCfg(
        asset_name="robot", joint_names=FINGER_JOINT_NAMES["right"], preserve_order=True, use_default_offset=False)
