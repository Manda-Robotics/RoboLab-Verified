# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Bimanual Franka Panda: two DROID arms (Franka + Robotiq 2F-85) on a shared torso.

Both arms live in ONE articulation. RoboLab's recorders, ground-truth export and
metrics all address a single articulation named ``robot``, so two separate arm
articulations would only half-record; and because IsaacLab resolves joints and
bodies by name, every body and joint carries a ``left_arm_`` / ``right_arm_`` prefix
(``left_arm_panda_joint1``, ``right_arm_base_link``, ...). The asset is generated
from the single-arm DROID file by ``assets/robots/_utils/build_bimanual_franka.py``.

Per-arm behaviour (home pose, gains, gripper mapping, wrist-camera intrinsics) is
copied from :mod:`robolab.robots.droid` so a bimanual result is comparable to the
single-arm benchmark, arm for arm.

Action layout is always **left then right**:

    joint position  (16)  [left arm 7, left gripper 1, right arm 7, right gripper 1]
    relative IK     (14)  [left dpos 3, left drot 3, left gripper 1, right ...]

Gripper commands follow the RoboLab convention: 0 opens, 1 closes.
"""

import os

import isaaclab.envs.mdp as mdp
import isaaclab.sim as sim_utils
import numpy as np
import torch
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg
from isaaclab.controllers.differential_ik_cfg import DifferentialIKControllerCfg
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.envs.mdp.actions.actions_cfg import DifferentialInverseKinematicsActionCfg
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import TiledCameraCfg
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import FrameTransformerCfg, OffsetCfg
from isaaclab.utils import configclass
from isaaclab.utils.math import subtract_frame_transforms

from robolab.constants import ASSET_DIR, ROBOTS_DIR
from robolab.core.environments.scene_fixture import TableFixtureCfg
from robolab.robots.droid import (
    EEF_OFFSET_POS,
    EEF_OFFSET_ROT,
    BinaryJointPositionZeroToOneActionCfg,
    _to_torch,
)

ARMS = ("left_arm", "right_arm")
ARM_JOINT_NAMES = {arm: [f"{arm}_panda_joint{i}" for i in range(1, 8)] for arm in ARMS}
GRIPPER_JOINT_NAME = {arm: f"{arm}_finger_joint" for arm in ARMS}
EE_BODY_NAME = {arm: f"{arm}_base_link" for arm in ARMS}   # Robotiq 2F-85 gripper base
GRIPPER_CLOSED_RAD = np.pi / 4                            # DROID: finger_joint at 45 deg
# Prim subtree of each arm's Robotiq gripper inside the articulation. ENV_REGEX_NS is
# doubled so .format(arm=...) leaves it for IsaacLab to substitute per env.
_GRIPPER_PRIM = "{{ENV_REGEX_NS}}/robot/{arm}/Gripper/Robotiq_2F_85"

# Home pose and gains are DROID's (robolab/robots/droid.py), applied to both arms.
# Every Robotiq joint, listed explicitly: with two arms, the overlapping regexes
# DROID uses (".*_finger_joint" vs ".*_left_inner.*") match the same joint twice,
# which IsaacLab rejects. Only finger_joint is driven; the rest are passive.
_ROBOTIQ_JOINTS = (
    "finger_joint",
    "right_outer_knuckle_joint",
    "left_inner_finger_joint",
    "left_inner_finger_knuckle_joint",
    "right_inner_finger_joint",
    "right_inner_finger_knuckle_joint",
)
_HOME_JOINT_POS = {
    "panda_joint1": 0.0,
    "panda_joint2": -1 / 5 * np.pi,
    "panda_joint3": 0.0,
    "panda_joint4": -4 / 5 * np.pi,
    "panda_joint5": 0.0,
    "panda_joint6": 3 / 5 * np.pi,
    "panda_joint7": 0.0,
}


def _wrist_cam(arm: str) -> TiledCameraCfg:
    # Same sensor as the DROID wrist camera (policy-calibrated focal length 2.8),
    # mounted on this arm's gripper base.
    return TiledCameraCfg(
        prim_path=f"{_GRIPPER_PRIM.format(arm=arm)}/{arm}_base_link/wrist_cam",
        height=720,
        width=1280,
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=2.8,
            focus_distance=28.0,
            horizontal_aperture=5.376,
            vertical_aperture=3.024,
        ),
        offset=TiledCameraCfg.OffsetCfg(
            pos=(0.011, -0.031, -0.074), rot=(-0.420, 0.570, 0.576, -0.409), convention="opengl"
        ),
    )


_LEFT_WRIST_CAM = _wrist_cam("left_arm")
_RIGHT_WRIST_CAM = _wrist_cam("right_arm")


@configclass
class BimanualFrankaCfg:
    """Fixed-base bimanual Franka articulation plus both wrist cameras."""

    robot = ArticulationCfg(
        prim_path="{ENV_REGEX_NS}/robot",
        spawn=sim_utils.UsdFileCfg(
            usd_path=os.path.join(
                ROBOTS_DIR, "bimanual_franka_robotiq_2f85", "bimanual_franka_robotiq_2f85.usd"
            ),
            activate_contact_sensors=True,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=True,
                max_depenetration_velocity=5.0,
            ),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                # Arms are 0.60 m apart and never cross in the benchmark workspace;
                # matches DROID/Kinova and keeps the solver cheap.
                enabled_self_collisions=False,
                solver_position_iteration_count=64,
                solver_velocity_iteration_count=0,
            ),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.0, 0.0, 0.0),
            rot=(1.0, 0.0, 0.0, 0.0),
            joint_pos={
                **{f"{arm}_{joint}": value
                   for arm in ARMS for joint, value in _HOME_JOINT_POS.items()},
                **{f"{arm}_{joint}": 0.0 for arm in ARMS for joint in _ROBOTIQ_JOINTS},
            },
        ),
        soft_joint_pos_limit_factor=1.0,
        actuators={
            "shoulders": ImplicitActuatorCfg(
                joint_names_expr=[".*_arm_panda_joint[1-4]"],
                effort_limit=87.0,
                velocity_limit=2.175,
                stiffness=400.0,
                damping=80.0,
            ),
            "forearms": ImplicitActuatorCfg(
                joint_names_expr=[".*_arm_panda_joint[5-7]"],
                effort_limit=12.0,
                velocity_limit=2.61,
                stiffness=400.0,
                damping=80.0,
            ),
            "grippers": ImplicitActuatorCfg(
                joint_names_expr=[".*_arm_finger_joint"],
                stiffness=None,
                damping=None,
                velocity_limit=5.0,
            ),
        },
    )

    left_wrist_cam = _LEFT_WRIST_CAM
    right_wrist_cam = _RIGHT_WRIST_CAM

    frames = FrameTransformerCfg(
        prim_path="{ENV_REGEX_NS}/robot/torso",
        debug_vis=False,
        target_frames=[
            FrameTransformerCfg.FrameCfg(
                prim_path=f"{_GRIPPER_PRIM.format(arm=arm)}/{arm}_base_link",
                name=f"{arm}_eef_frame",
                offset=OffsetCfg(pos=EEF_OFFSET_POS, rot=EEF_OFFSET_ROT),
            )
            for arm in ARMS
        ],
    )


# Labels assigned after the class body so configclass does not turn them into fields.
BimanualFrankaCfg.table_fixture = TableFixtureCfg(
    usd_path=os.path.join(ASSET_DIR, "fixtures", "bimanual_franka_table.usda"),
    pos=(-0.087, 0.0, 0.0),
    rot=(6.123233995736766e-17, 0.0, 0.0, 1.0),
)
# One HDF5 channel per arm (docs/robots.md#end-effector-pose-recording).
BimanualFrankaCfg.ee_recorder_bodies = {
    "left_ee_pose": EE_BODY_NAME["left_arm"],
    "right_ee_pose": EE_BODY_NAME["right_arm"],
}
# P79 friction override targets: both pads of both Robotiq hands (declared from the
# asset builder's link names; not yet read back on a GPU -- see docs/physics.md).
BimanualFrankaCfg.friction_bodies = [f"{arm}_{side}_inner_finger" for arm in ("left_arm", "right_arm") for side in ("left", "right")]


@configclass
class BimanualWristCamerasCfg:
    """Exposes both wrist cameras to ``generate_image_obs_from_cameras``.

    The scene sensors come from :class:`BimanualFrankaCfg` so the robot exists before
    they are spawned; this wrapper only contributes the observation names.
    """

    left_wrist_cam = _LEFT_WRIST_CAM
    right_wrist_cam = _RIGHT_WRIST_CAM


########################################################
# Contact gripper
########################################################

# One finger per gripper: IsaacLab's filtered contact sensor needs exactly one prim
# per env (see robolab/robots/droid.py). "gripper" is the group, so every benchmark
# task's default gripper_name="gripper" means *either hand* (docs/task_conditionals.md).
contact_gripper = {
    "left": f"{_GRIPPER_PRIM.format(arm='left_arm')}/left_arm_left_inner_finger",
    "right": f"{_GRIPPER_PRIM.format(arm='right_arm')}/right_arm_left_inner_finger",
    "gripper": ["left", "right"],
}

########################################################
# Observations
########################################################


def _arm_joint_pos(env: ManagerBasedRLEnv, arm: str, asset_name: str = "robot"):
    robot = env.scene[asset_name]
    indices = [robot.data.joint_names.index(name) for name in ARM_JOINT_NAMES[arm]]
    return _to_torch(robot.data.joint_pos)[:, indices]


def _gripper_pos(env: ManagerBasedRLEnv, arm: str, asset_name: str = "robot"):
    """0 = open, 1 = closed, like DROID's ``gripper_pos``."""
    robot = env.scene[asset_name]
    index = robot.data.joint_names.index(GRIPPER_JOINT_NAME[arm])
    return _to_torch(robot.data.joint_pos)[:, index : index + 1] / GRIPPER_CLOSED_RAD


def _ee_pose(env: ManagerBasedRLEnv, arm: str, asset_name: str = "robot"):
    """Gripper base pose in the robot-root frame (docs/frames.md)."""
    robot = env.scene[asset_name]
    body_idx = robot.data.body_names.index(EE_BODY_NAME[arm])
    return subtract_frame_transforms(
        _to_torch(robot.data.root_pos_w),
        _to_torch(robot.data.root_quat_w),
        _to_torch(robot.data.body_pos_w)[:, body_idx, :],
        _to_torch(robot.data.body_quat_w)[:, body_idx, :],
    )


def left_arm_joint_pos(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")):
    return _arm_joint_pos(env, "left_arm", asset_cfg.name)


def right_arm_joint_pos(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")):
    return _arm_joint_pos(env, "right_arm", asset_cfg.name)


def left_gripper_pos(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")):
    return _gripper_pos(env, "left_arm", asset_cfg.name)


def right_gripper_pos(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")):
    return _gripper_pos(env, "right_arm", asset_cfg.name)


def left_ee_pos(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")):
    return _ee_pose(env, "left_arm", asset_cfg.name)[0]


def left_ee_quat(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")):
    return _ee_pose(env, "left_arm", asset_cfg.name)[1]


def right_ee_pos(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")):
    return _ee_pose(env, "right_arm", asset_cfg.name)[0]


def right_ee_quat(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")):
    return _ee_pose(env, "right_arm", asset_cfg.name)[1]


@configclass
class BimanualProprioceptionObservationCfg(ObsGroup):
    left_arm_joint_pos = ObsTerm(func=left_arm_joint_pos)
    left_gripper_pos = ObsTerm(func=left_gripper_pos, clip=(0.0, 1.0))
    left_ee_pos = ObsTerm(func=left_ee_pos)
    left_ee_quat = ObsTerm(func=left_ee_quat)
    right_arm_joint_pos = ObsTerm(func=right_arm_joint_pos)
    right_gripper_pos = ObsTerm(func=right_gripper_pos, clip=(0.0, 1.0))
    right_ee_pos = ObsTerm(func=right_ee_pos)
    right_ee_quat = ObsTerm(func=right_ee_quat)

    def __post_init__(self) -> None:
        self.enable_corruption = False
        self.concatenate_terms = False


########################################################
# Actions
########################################################


def _gripper_action(arm: str) -> BinaryJointPositionZeroToOneActionCfg:
    joint = GRIPPER_JOINT_NAME[arm]
    return BinaryJointPositionZeroToOneActionCfg(
        asset_name="robot",
        joint_names=[joint],
        open_command_expr={joint: 0.0},
        close_command_expr={joint: GRIPPER_CLOSED_RAD},
    )


@configclass
class BimanualFrankaJointPositionActionCfg:
    """16-dim joint-space control: left arm, left gripper, right arm, right gripper.

    Term order is declaration order, which IsaacLab preserves in the action vector.
    """

    left_arm = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=ARM_JOINT_NAMES["left_arm"],
        preserve_order=True,
        use_default_offset=False,
    )
    left_gripper = _gripper_action("left_arm")
    right_arm = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=ARM_JOINT_NAMES["right_arm"],
        preserve_order=True,
        use_default_offset=False,
    )
    right_gripper = _gripper_action("right_arm")


def _rel_ik_action(arm: str) -> DifferentialInverseKinematicsActionCfg:
    # Mirrors DroidRelIKActionCfg: deltas on robot-root axes, tracking the gripper
    # base, scale 0.5. Each arm gets its own controller over its own seven joints.
    return DifferentialInverseKinematicsActionCfg(
        asset_name="robot",
        joint_names=ARM_JOINT_NAMES[arm],
        body_name=EE_BODY_NAME[arm],
        controller=DifferentialIKControllerCfg(
            command_type="pose", use_relative_mode=True, ik_method="dls"
        ),
        scale=0.5,
        body_offset=DifferentialInverseKinematicsActionCfg.OffsetCfg(pos=[0.0, 0.0, 0.0]),
    )


@configclass
class BimanualFrankaRelIKActionCfg:
    """14-dim relative end-effector control: left (6 + gripper), right (6 + gripper)."""

    left_arm = _rel_ik_action("left_arm")
    left_gripper = _gripper_action("left_arm")
    right_arm = _rel_ik_action("right_arm")
    right_gripper = _gripper_action("right_arm")
