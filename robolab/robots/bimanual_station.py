# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Bimanual workstation: two ViperX 300 arms side by side, leaning into the table.

The industry bimanual archetype — both arms on a shared mount at the near edge,
facing the workspace, pitched ~15 deg into it. This is the geometry of Dyna-style
stations, YAM/I2RT rigs, and the Mobile ALOHA front end, in contrast to the
opposite-facing stationary ALOHA 2 (`robolab/robots/aloha.py`) that the released
pi checkpoints were trained on.

Same arms, gripper, cameras, gains, observations and 16-dim action space as the
ALOHA 2 config — everything is imported from ``robolab.robots.aloha``; only the
asset (base placement baked into ``bimanual_station_viperx.usd``), the rig's world
placement, and the camera rigging differ. The arm model is swappable at the asset
level (`finalize_aloha_usd.py --layout station`): ViperX today, YAM/ARX/UR5e as
their descriptions are imported.
"""

import copy
import math
import os

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg
from isaaclab.sensors import TiledCameraCfg
from isaaclab.utils import configclass

from robolab.constants import ASSET_DIR, ROBOTS_DIR
from robolab.core.environments.scene_fixture import TableFixtureCfg
from robolab.robots.aloha import (  # noqa: F401  (re-exported for clients)
    ARM_JOINT_NAMES,
    ARMS,
    FINGER_JOINT_NAMES,
    FINGER_TRAVEL_M,
    AlohaJointPositionActionCfg,
    AlohaProprioceptionObservationCfg,
    _CAM_LEFT_WRIST,
    _CAM_RIGHT_WRIST,
    _D405,
    _NEUTRAL,
    contact_gripper,
)
from robolab.robots.aloha import AlohaCfg as _AlohaCfg
from robolab.robots.aloha import EE_BODY_NAME

# Overhead camera on the station frame: centred above the near edge, looking
# forward-down over the workspace (rig frame; the rig is placed by init_state).
_CAM_HIGH_STATION = TiledCameraCfg(
    prim_path="{ENV_REGEX_NS}/robot/torso/cam_high",
    offset=TiledCameraCfg.OffsetCfg(
        pos=(0.0, -0.42, 0.95), rot=(0.939693, 0.342020, 0.0, 0.0), convention="opengl"),
    **_D405,
)


# Derive the articulation from the ALOHA cfg (configclass fields are instance-level,
# and nothing extra may live in the class body — every member becomes a scene field).
_BASE = _AlohaCfg()


def _station_robot(usd_name: str) -> ArticulationCfg:
    # configclass has no .replace() in this Isaac Lab (it crashed the first
    # gym_match v6 attempt) — deepcopy and assign.
    robot = copy.deepcopy(_BASE.robot)
    robot.spawn.usd_path = os.path.join(ROBOTS_DIR, "aloha2", usd_name)
    robot.init_state = ArticulationCfg.InitialStateCfg(
        # Rig yawed -90 deg: bases land on the table's near edge at
        # (0.13, -/+spacing/2), facing +x toward the workspace centre at (0.55, 0).
        pos=(0.55, 0.0, 0.0),
        rot=(0.70710678, 0.0, 0.0, -0.70710678),
        joint_pos=dict(_BASE.robot.init_state.joint_pos),
    )
    return robot


_STATION_ROBOT = _station_robot("bimanual_station_viperx.usd")

# ---- Mobile ALOHA ------------------------------------------------------------
# The "Mobile Trossen" platform of the pi0 / pi0.5 pretraining mix: two ViperX 300
# on the front plate of a Tracer cart, 40 cm apart, mounted flat and facing
# forward, one forward camera on the mast between the arms plus the two wrist
# cameras (no cam_low). Numbers we could not source are marked as assumptions:
#   * base plate height: the paper gives a 65 cm min reach height and 100 cm
#     forward extension; that puts the plate ~0.70 m off the floor, i.e. about
#     15 cm BELOW an 85 cm table (ASSUMPTION: MOBILE_BASE_BELOW_TABLE_M).
#   * mast camera: "mounted between the arms, facing forward" — ASSUMPTION:
#     0.55 m above the plate, 0.15 m behind the base line, pitched 35 deg down,
#     Logitech C922x (~70 deg horizontal FOV at 4:3, 640x480).
MOBILE_SPACING_M = 0.40
MOBILE_BASE_BELOW_TABLE_M = 0.15
_MOBILE_CAM_PITCH_DEG = 35.0
_c, _s = math.cos(math.radians(_MOBILE_CAM_PITCH_DEG / 2)), math.sin(math.radians(_MOBILE_CAM_PITCH_DEG / 2))
# opengl camera looks along -z; in the rig frame the arms face +y (before the rig's
# -90 deg yaw). Rotate the default down-looking (-z) camera: first tilt so it looks
# along +y (rot x by +90 deg), then pitch down by the mast angle.
_MOBILE_CAM_HIGH = TiledCameraCfg(
    prim_path="{ENV_REGEX_NS}/robot/torso/cam_high",
    offset=TiledCameraCfg.OffsetCfg(
        pos=(0.0, -0.57, 0.55),
        rot=(math.cos(math.radians((90 - _MOBILE_CAM_PITCH_DEG) / 2)),
             math.sin(math.radians((90 - _MOBILE_CAM_PITCH_DEG) / 2)), 0.0, 0.0),
        convention="opengl"),
    height=480, width=640, data_types=["rgb"],
    spawn=sim_utils.PinholeCameraCfg(
        # C922x: ~70 deg horizontal FOV -> focal 1.0 with aperture 2*tan(35deg)=1.40
        focal_length=1.0, focus_distance=28.0,
        horizontal_aperture=1.4004, vertical_aperture=1.0503,
    ),
)


def _mobile_robot() -> ArticulationCfg:
    robot = _station_robot("mobile_aloha_viperx.usd")
    x, y, z = robot.init_state.pos
    robot.init_state.pos = (x, y, z - MOBILE_BASE_BELOW_TABLE_M)
    return robot


@configclass
class MobileAlohaCfg:
    """Mobile-ALOHA front end parked at the table's near edge (see module notes)."""

    robot = _mobile_robot()
    cam_high = _MOBILE_CAM_HIGH
    cam_left_wrist = _CAM_LEFT_WRIST
    cam_right_wrist = _CAM_RIGHT_WRIST


# The cart body under the plates: top face = the plate (15 cm below the table top),
# bottom on the canonical ground (-0.697); 0.40 m deep so it stops short of the
# table's near edge. Without it the arms looked floor-mounted (The reviewer, 2026-08-25),
# and the real mast camera sees the cart top at the bottom of its frame.
MobileAlohaCfg.table_fixture = TableFixtureCfg(
    usd_path=os.path.join(ASSET_DIR, "fixtures", "mobile_cart.usda"),
    pos=(0.0, 0.0, -MOBILE_BASE_BELOW_TABLE_M),
)
MobileAlohaCfg.ee_recorder_bodies = {
    "left_ee_pose": EE_BODY_NAME["left"],
    "right_ee_pose": EE_BODY_NAME["right"],
}
MobileAlohaCfg.friction_bodies = [f"{arm}_{side}_finger_link" for arm in ("left", "right") for side in ("left", "right")]  # P79


@configclass
class MobileCamerasCfg:
    """Mobile ALOHA's camera set: forward mast camera + both wrists."""

    cam_high = _MOBILE_CAM_HIGH
    cam_left_wrist = _CAM_LEFT_WRIST
    cam_right_wrist = _CAM_RIGHT_WRIST


@configclass
class BimanualStationCfg:
    """Side-by-side ViperX station; workspace centre at the benchmark task area."""

    robot = _STATION_ROBOT

    cam_high = _CAM_HIGH_STATION
    cam_left_wrist = _CAM_LEFT_WRIST
    cam_right_wrist = _CAM_RIGHT_WRIST


BimanualStationCfg.table_fixture = None
BimanualStationCfg.ee_recorder_bodies = {
    "left_ee_pose": EE_BODY_NAME["left"],
    "right_ee_pose": EE_BODY_NAME["right"],
}
BimanualStationCfg.friction_bodies = [f"{arm}_{side}_finger_link" for arm in ("left", "right") for side in ("left", "right")]  # P79


@configclass
class StationCamerasCfg:
    """cam_high + both wrists (no cam_low: stations rarely have a worm's-eye view)."""

    cam_high = _CAM_HIGH_STATION
    cam_left_wrist = _CAM_LEFT_WRIST
    cam_right_wrist = _CAM_RIGHT_WRIST
