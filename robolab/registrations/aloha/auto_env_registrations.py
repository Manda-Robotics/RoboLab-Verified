# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""RoboLab registrations for the ALOHA 2 rig (env postfix ``Aloha``).

Cameras are the rig's own four (cam_high, cam_low, both wrists) plus the standard
mirrored viewport for review video. Benchmark tasks register unchanged: the generic
``"gripper"`` group means either hand.
"""

from robolab.constants import DEFAULT_TASK_SUBFOLDERS, TASK_DIR


def auto_register_aloha_envs(task_dirs=None, task=None, variant: str = "opposing"):
    """variant="opposing" is the stationary ALOHA 2 cell; "station" is the
    side-by-side forward-leaning workstation (env postfix ``BimanualStation``);
    "gym_match" is the reference-parity setup for pi0_aloha_sim: the opposing rig
    seen only through gym-aloha's top camera, against a plain dark backdrop."""
    from robolab.core.environments.factory import auto_discover_and_create_cfgs
    from robolab.core.observations.observation_utils import (
        generate_image_obs_from_cameras,
        generate_obs_cfg,
    )
    from robolab.robots.aloha import (
        AlohaCamerasCfg,
        AlohaCfg,
        AlohaGymMatchCfg,
        GymMatchCamerasCfg,
        AlohaJointPositionActionCfg,
        AlohaProprioceptionObservationCfg,
        contact_gripper,
    )
    from robolab.robots.bimanual_station import (
        BimanualStationCfg, MobileAlohaCfg, MobileCamerasCfg, StationCamerasCfg,
    )

    robot_cfg, cameras_cfg, postfix = {
        "opposing": (AlohaCfg, AlohaCamerasCfg, "Aloha"),
        "station": (BimanualStationCfg, StationCamerasCfg, "BimanualStation"),
        "mobile": (MobileAlohaCfg, MobileCamerasCfg, "MobileAloha"),
        "gym_match": (AlohaGymMatchCfg, GymMatchCamerasCfg, "AlohaGymMatch"),
    }[variant]
    from robolab.variations.backgrounds import (
        DarkVoidBackgroundCfg,
        HomeOfficeBackgroundCfg,
    )
    from robolab.variations.camera import EgocentricMirroredWideAngleHighCameraCfg
    from robolab.variations.lighting import GymMatchLightCfg, SphereLightCfg

    if task_dirs is None:
        task_dirs = DEFAULT_TASK_SUBFOLDERS

    ViewportCameraCfg = generate_image_obs_from_cameras(
        [EgocentricMirroredWideAngleHighCameraCfg]
    )
    ImageObsCfg = generate_image_obs_from_cameras([cameras_cfg])
    ObservationCfg = generate_obs_cfg(
        {
            "image_obs": ImageObsCfg(),
            "proprio_obs": AlohaProprioceptionObservationCfg(),
            "viewport_cam": ViewportCameraCfg(),
        }
    )

    auto_discover_and_create_cfgs(
        task_dir=TASK_DIR,
        task_subdirs=task_dirs,
        tasks=task,
        pattern="*.py",
        env_postfix=postfix,
        observations_cfg=ObservationCfg(),
        actions_cfg=AlohaJointPositionActionCfg(),
        robot_cfg=robot_cfg,
        # The rig cameras are attached through AlohaCfg; only free-standing extras here.
        camera_cfg=[EgocentricMirroredWideAngleHighCameraCfg],
        # gym_match: gym-aloha's three directional lights instead of the sphere
        # light whose gradient the pi0_aloha_sim checkpoint has never seen.
        lighting_cfg=(GymMatchLightCfg if variant == "gym_match" else SphereLightCfg),
        background_cfg=(DarkVoidBackgroundCfg if variant == "gym_match"
                        else HomeOfficeBackgroundCfg),
        contact_gripper=contact_gripper,
        dt=1 / 120,
        render_interval=8,
        decimation=8,
        seed=1,
    )
