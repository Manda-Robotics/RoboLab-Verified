# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""RoboLab registrations for the bimanual YAM (two I2RT arms, one articulation).

Env postfix ``BimanualYam``, 16-dim joint-position actions. ``top_cam`` selects the
overhead camera: ``"ai2_desk"`` (MolmoAct 2 kit, D435) or ``"i2rt_gantry"`` (I2RT station, D405). Cameras are the three the
MolmoAct 2 checkpoint was trained on (top, left wrist, right wrist, 640x360), plus the
wide mirrored viewport for recording. Control runs at 30 Hz (``dt=1/120, decimation=4``),
the checkpoint's native rate, so a 30-step chunk is one second of motion.

Task discovery defaults to the ``bimanual`` folder; pass ``task_dirs=["benchmark"]`` to run
the single-arm benchmark on this robot (its generic ``"gripper"`` checks mean *either hand*).
"""

from robolab.constants import TASK_DIR

BIMANUAL_TASK_SUBFOLDERS = ["bimanual"]


def auto_register_bimanual_yam_envs(task_dirs=None, task=None, top_cam: str = "ai2_desk"):
    from robolab.core.environments.factory import auto_discover_and_create_cfgs
    from robolab.core.observations.observation_utils import (
        generate_image_obs_from_cameras,
        generate_obs_cfg,
    )
    from robolab.robots.bimanual_yam import (
        BimanualYamCamerasCfg,
        BimanualYamCfg,
        BimanualYamJointPositionActionCfg,
        BimanualYamProprioceptionObservationCfg,
        contact_gripper,
        set_top_camera,
    )
    set_top_camera(top_cam)
    from robolab.variations.backgrounds import HomeOfficeBackgroundCfg
    from robolab.variations.camera import EgocentricMirroredWideAngleHighCameraCfg
    from robolab.variations.lighting import SphereLightCfg

    if task_dirs is None:
        task_dirs = BIMANUAL_TASK_SUBFOLDERS

    ViewportCameraCfg = generate_image_obs_from_cameras([EgocentricMirroredWideAngleHighCameraCfg])
    ImageObsCfg = generate_image_obs_from_cameras([BimanualYamCamerasCfg])
    ObservationCfg = generate_obs_cfg(
        {
            "image_obs": ImageObsCfg(),
            "proprio_obs": BimanualYamProprioceptionObservationCfg(),
            "viewport_cam": ViewportCameraCfg(),
        }
    )

    auto_discover_and_create_cfgs(
        task_dir=TASK_DIR,
        task_subdirs=task_dirs,
        tasks=task,
        pattern="*.py",
        env_postfix="BimanualYam",
        observations_cfg=ObservationCfg(),
        actions_cfg=BimanualYamJointPositionActionCfg(),
        robot_cfg=BimanualYamCfg,
        # The three policy cameras are attached through BimanualYamCfg (they need the
        # robot to exist); only the viewport camera is listed here.
        camera_cfg=[EgocentricMirroredWideAngleHighCameraCfg],
        lighting_cfg=SphereLightCfg,
        background_cfg=HomeOfficeBackgroundCfg,
        contact_gripper=contact_gripper,
        dt=1 / 120,
        render_interval=8,
        decimation=4,
        seed=1,
    )
