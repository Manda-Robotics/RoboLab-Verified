# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Bimanual ViperX (ALOHA) rig.

LIMITED USE -- read this before running anything on it.

The rig itself is verified: the arms build, the wrist cameras track the bodies they
are attached to, and per-arm EE channels are recorded (see P67, without which a
bimanual demo silently yields no metrics at all).

What is NOT verified is any policy on it. The released pi05 checkpoint scores **0/6**
on the opposing (training) rig with working wrist cameras: coherent reach, twitchy
motion, no grasp. The checkpoint is out of distribution for this embodiment and
fine-tuning is the only path we found. Treat ALOHA numbers as a statement about the
checkpoint, not about the rig or the benchmark, and do not put them next to the
Franka numbers without that caveat.

The dual-Franka rig (robolab/robots/bimanual_franka.py) is verified 6/6 clean at
max 0.18 rad/step and is the bimanual rig to reach for by default.
"""

import copy
import os

import isaaclab.envs.mdp as mdp
import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import TiledCameraCfg

from robolab.core.sensors.body_attached_camera import BodyAttachedTiledCameraCfg
from isaaclab.utils import configclass
from isaaclab.utils.math import subtract_frame_transforms

from robolab.constants import ROBOTS_DIR
from robolab.robots.droid import _to_torch

ARMS = ("left", "right")
ARM_JOINT_NAMES = {
    side: [f"{side}_{j}" for j in
           ("waist", "shoulder", "elbow", "forearm_roll", "wrist_angle", "wrist_rotate")]
    for side in ARMS
}
FINGER_JOINT_NAMES = {side: [f"{side}_left_finger", f"{side}_right_finger"] for side in ARMS}
EE_BODY_NAME = {side: f"{side}_gripper_link" for side in ARMS}
FINGER_TRAVEL_M = 0.041          # ALOHA 2 finger joint range; 0 closed, 0.041 open

# Menagerie keyframe "neutral_pose" — the ALOHA ready pose the pi checkpoints start from.
_NEUTRAL = {"waist": 0.0, "shoulder": -0.96, "elbow": 1.16,
            "forearm_roll": 0.0, "wrist_angle": -0.3, "wrist_rotate": 0.0}

# D405, but at the REAL pipeline's format, not the menagerie sim one: openpi's
# aloha_real env feeds 640x480 (4:3), while the MJCF authors 1280x720 (16:9).
# pi letterboxes to square 224x224, so aspect ratio changes what the policy sees:
# a 16:9 frame arrives with a vertically narrower FOV and different padding than
# every training image. Horizontal intrinsics keep the MJCF calibration
# (focal 1.93 mm over 3.896 mm); the vertical aperture is widened to 4:3.
# 640x480 is also ~3.4x cheaper to render than 720p.
_D405 = dict(
    height=480, width=640, data_types=["rgb"],
    spawn=sim_utils.PinholeCameraCfg(
        focal_length=1.93, focus_distance=28.0,
        horizontal_aperture=3.896, vertical_aperture=2.922,
    ),
)

# Frame-mounted cameras, in the aloha rig frame (MJCF scene.xml), attached to the
# torso (at (0, -0.019, 0.01)) so they follow the rig's placement. MuJoCo cameras
# look down -z with +y up — the same convention as "opengl" offsets.
_CAM_HIGH = TiledCameraCfg(
    prim_path="{ENV_REGEX_NS}/robot/torso/cam_high",
    offset=TiledCameraCfg.OffsetCfg(
        pos=(0.0, -0.284794, 1.01524), rot=(0.976332, 0.216277, 0.0, 0.0), convention="opengl"),
    **_D405,
)
_CAM_LOW = TiledCameraCfg(
    prim_path="{ENV_REGEX_NS}/robot/torso/cam_low",
    offset=TiledCameraCfg.OffsetCfg(
        pos=(0.0, -0.358167, 0.0216055), rot=(0.672659, 0.739953, 0.0, 0.0), convention="opengl"),
    **_D405,
)


def _wrist_cam(side: str) -> TiledCameraCfg:
    # The MJCF authors this camera in the `{side}_gripper_base` frame
    # (pos (0, -0.0824748, -0.0095955), euler x = 2.70525955 rad). But
    # `gripper_base` hangs on a *fixed* joint, which PhysX merges into its
    # parent link — the prim never receives pose updates, and a camera parked
    # there films the MJCF zero pose forever: the aloha_pi05_sweep* runs
    # recorded both wrist cams frozen at (0.569, ±0.014, 0.53), pointing at
    # the room, while the grippers moved at (·, ±0.317, 0.305). Attach to
    # `{side}_gripper_link` instead — a real DOF body (the EE recorder rides
    # it) — with the same physical pose re-expressed in that frame, i.e.
    # T(gripper_link → gripper_base) ∘ T(cam), taken from the authored USD.
    # ...and even so the prim never moved (2026-08-24 probe: camera at the USD
    # default pose for 60 steps while gripper_link travelled 30 cm). The camera
    # therefore rides the body by pose: a top-level env prim whose world pose is
    # set from the PhysX view every update (BodyAttachedTiledCamera).
    return BodyAttachedTiledCameraCfg(
        prim_path=f"{{ENV_REGEX_NS}}/cam_{side}_wrist",
        articulation_prim_path_expr="{ENV_REGEX_NS}/robot",
        body_name=f"{side}_gripper_link",
        offset=TiledCameraCfg.OffsetCfg(
            pos=(0.025339, -0.000066, 0.082467),
            rot=(0.596065, 0.380403, -0.379928, -0.596368), convention="opengl"),
        **_D405,
    )


_CAM_LEFT_WRIST = _wrist_cam("left")
_CAM_RIGHT_WRIST = _wrist_cam("right")

# gym-aloha's "top" camera (assets/bimanual_viperx_transfer_cube.xml): 0.6 m to the
# side of the workspace centre, 0.8 m up, aimed at the table, fovy 78 deg. This is
# the ONLY view pi0_aloha_sim ever saw; matching it (pose AND the 78-deg lens, vs
# the D405's 58) is what the reference-parity experiment requires.
# gym-aloha's table sits at (0, 0.6, 0) and the camera at (0, 0.6, 0.8) — i.e.
# DIRECTLY overhead, looking straight down (fovy 78). An opengl-identity camera
# looks along -z with image-up = +y, so arms (at rig +-x) render left/right as in
# the reference. Torso is at rig (0, -0.019, 0.01).
_CAM_GYM_TOP = TiledCameraCfg(
    prim_path="{ENV_REGEX_NS}/robot/torso/cam_high",
    offset=TiledCameraCfg.OffsetCfg(
        pos=(0.0, 0.019, 0.79), rot=(1.0, 0.0, 0.0, 0.0), convention="opengl"),
    height=480, width=640, data_types=["rgb"],
    spawn=sim_utils.PinholeCameraCfg(
        focal_length=1.0, focus_distance=28.0,
        horizontal_aperture=2.1596, vertical_aperture=1.6197,
    ),
)


@configclass
class GymMatchCamerasCfg:
    """Reference-parity camera set: gym-aloha's top view only (as cam_high)."""

    cam_high = _CAM_GYM_TOP


_ALOHA_ROBOT = ArticulationCfg(
        prim_path="{ENV_REGEX_NS}/robot",
        spawn=sim_utils.UsdFileCfg(
            usd_path=os.path.join(ROBOTS_DIR, "aloha2", "aloha2_bimanual.usd"),
            activate_contact_sensors=True,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=True,
                max_depenetration_velocity=5.0,
            ),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=False,
                solver_position_iteration_count=64,
                solver_velocity_iteration_count=0,
            ),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            # Rig yawed +90 deg: arms along the table's Y axis, workspace at (0.55, 0).
            pos=(0.55, 0.0, 0.0),
            rot=(0.70710678, 0.0, 0.0, 0.70710678),
            joint_pos={
                **{f"{side}_{joint}": value
                   for side in ARMS for joint, value in _NEUTRAL.items()},
                **{name: 0.0084 for side in ARMS for name in FINGER_JOINT_NAMES[side]},
            },
        ),
        soft_joint_pos_limit_factor=1.0,
        # The MJCF importer drops the <position> actuator gains (they live in an
        # included actuators file), authoring drives with stiffness 0 — position
        # targets then do nothing and the arms hang on joint damping. Author the
        # menagerie actuator classes' kp (stiffness) and the joints' damping here.
        actuators={
            "waists": ImplicitActuatorCfg(
                joint_names_expr=[".*_waist"], stiffness=43.0, damping=5.76, effort_limit=35.0),
            "shoulders": ImplicitActuatorCfg(
                joint_names_expr=[".*_shoulder"], stiffness=265.0, damping=20.0, effort_limit=144.0),
            "elbows": ImplicitActuatorCfg(
                joint_names_expr=[".*_elbow"], stiffness=227.0, damping=18.49, effort_limit=59.0),
            "forearm_rolls": ImplicitActuatorCfg(
                joint_names_expr=[".*_forearm_roll"], stiffness=78.0, damping=6.78, effort_limit=22.0),
            "wrist_angles": ImplicitActuatorCfg(
                joint_names_expr=[".*_wrist_angle"], stiffness=37.0, damping=6.28, effort_limit=35.0),
            "wrist_rotates": ImplicitActuatorCfg(
                joint_names_expr=[".*_wrist_rotate"], stiffness=10.4, damping=1.2, effort_limit=35.0),
            "fingers": ImplicitActuatorCfg(
                joint_names_expr=[".*_finger"], stiffness=2000.0, damping=124.0, effort_limit=35.0),
        },
)


@configclass
class AlohaCfg:
    """Fixed-base ALOHA 2 rig, placed so its workspace centre is the task area."""

    robot = _ALOHA_ROBOT
    cam_high = _CAM_HIGH
    cam_low = _CAM_LOW
    cam_left_wrist = _CAM_LEFT_WRIST
    cam_right_wrist = _CAM_RIGHT_WRIST


def _gym_grey_robot() -> ArticulationCfg:
    """The ALOHA articulation with gym-aloha's flat grey arms.

    The asset's own material is the real ALOHA's black; gym-aloha's MJCF has no
    textures and renders mid-grey. Isaac Lab binds ``visual_material`` on the
    spawned root, which only takes effect because ``finalize_aloha_usd.py``
    de-instances the link geometry (instance proxies ignore outside bindings —
    look probe, 2026-08-24)."""
    robot = copy.deepcopy(_ALOHA_ROBOT)
    # gym-aloha mounts the arm bases ON the table top (vx300s bodies at z=0, the
    # tabletop's upper face at z=0); the ALOHA 2 asset carries a 2 cm mount plate
    # (BASE_Z = 0.02). The checkpoint's reach is learned relative to its base, so
    # on our rig it stopped ~2 cm short of the cube (gym_match_v5: touches, never
    # grasps). Drop the rig by that plate height for parity.
    # (gym-aloha's ALOHA 1 finger link sits 7.3 cm from gripper_link, the ALOHA 2
    # asset's 6.1 cm; lowering the rig a further 1.2 cm for that — gym_match_v10 —
    # did NOT lower the fingertips: the policy servoes on the image and ended at
    # the same world height. 0/6 vs v8's 1/6, so only the plate is compensated.)
    x, y, z = robot.init_state.pos
    robot.init_state.pos = (x, y, z - 0.02)
    robot.spawn.visual_material = sim_utils.PreviewSurfaceCfg(
        diffuse_color=(0.30, 0.30, 0.31), roughness=0.7, metallic=0.0)
    robot.spawn.visual_material_path = "gym_grey"
    return robot


@configclass
class AlohaGymMatchCfg:
    """ALOHA rig whose ONLY spawned camera is gym-aloha's overhead ``top``.

    The camera-set cfg alone is not enough: the sensor at a given prim path is
    spawned by the ROBOT cfg, so a variant that wants a different cam_high must
    carry it here — otherwise AlohaCfg's D405 silently wins the prim path."""

    robot = _gym_grey_robot()
    cam_high = _CAM_GYM_TOP


# The rig stands on the task table itself; no pedestal fixture.
AlohaCfg.table_fixture = None
AlohaCfg.ee_recorder_bodies = {
    "left_ee_pose": EE_BODY_NAME["left"],
    "right_ee_pose": EE_BODY_NAME["right"],
}
# P79 friction override targets (ViperX finger links; declared, not yet read back on a GPU).
AlohaCfg.friction_bodies = [f"{arm}_{side}_finger_link" for arm in ("left", "right") for side in ("left", "right")]
AlohaGymMatchCfg.table_fixture = None
AlohaGymMatchCfg.ee_recorder_bodies = dict(AlohaCfg.ee_recorder_bodies)
AlohaGymMatchCfg.friction_bodies = list(AlohaCfg.friction_bodies)


@configclass
class AlohaCamerasCfg:
    """Expose the rig cameras to image observation generation (names = openpi keys)."""

    cam_high = _CAM_HIGH
    cam_low = _CAM_LOW
    cam_left_wrist = _CAM_LEFT_WRIST
    cam_right_wrist = _CAM_RIGHT_WRIST


contact_gripper = {
    "left": "{ENV_REGEX_NS}/robot/left_base_link/left_left_finger_link",
    "right": "{ENV_REGEX_NS}/robot/right_base_link/right_left_finger_link",
    "gripper": ["left", "right"],
}

########################################################
# Observations
########################################################


def _joint_pos(env, names, asset_name="robot"):
    robot = env.scene[asset_name]
    idx = [robot.data.joint_names.index(n) for n in names]
    return _to_torch(robot.data.joint_pos)[:, idx]


def left_arm_joint_pos(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")):
    return _joint_pos(env, ARM_JOINT_NAMES["left"], asset_cfg.name)


def right_arm_joint_pos(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")):
    return _joint_pos(env, ARM_JOINT_NAMES["right"], asset_cfg.name)


def left_gripper_pos(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")):
    """Normalised 0 (closed) .. 1 (open), the real ALOHA driver's convention."""
    return _joint_pos(env, FINGER_JOINT_NAMES["left"][:1], asset_cfg.name) / FINGER_TRAVEL_M


def right_gripper_pos(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")):
    return _joint_pos(env, FINGER_JOINT_NAMES["right"][:1], asset_cfg.name) / FINGER_TRAVEL_M


def _ee_pose(env, side, asset_name="robot"):
    robot = env.scene[asset_name]
    body_idx = robot.data.body_names.index(EE_BODY_NAME[side])
    return subtract_frame_transforms(
        _to_torch(robot.data.root_pos_w),
        _to_torch(robot.data.root_quat_w),
        _to_torch(robot.data.body_pos_w)[:, body_idx, :],
        _to_torch(robot.data.body_quat_w)[:, body_idx, :],
    )


def left_ee_pos(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")):
    return _ee_pose(env, "left", asset_cfg.name)[0]


def left_ee_quat(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")):
    return _ee_pose(env, "left", asset_cfg.name)[1]


def right_ee_pos(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")):
    return _ee_pose(env, "right", asset_cfg.name)[0]


def right_ee_quat(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")):
    return _ee_pose(env, "right", asset_cfg.name)[1]


@configclass
class AlohaProprioceptionObservationCfg(ObsGroup):
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


@configclass
class AlohaJointPositionActionCfg:
    """16-dim: left arm 6, left fingers 2 (metres), right arm 6, right fingers 2."""

    left_arm = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=ARM_JOINT_NAMES["left"],
        preserve_order=True,
        use_default_offset=False,
    )
    left_fingers = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=FINGER_JOINT_NAMES["left"],
        preserve_order=True,
        use_default_offset=False,
    )
    right_arm = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=ARM_JOINT_NAMES["right"],
        preserve_order=True,
        use_default_offset=False,
    )
    right_fingers = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=FINGER_JOINT_NAMES["right"],
        preserve_order=True,
        use_default_offset=False,
    )
