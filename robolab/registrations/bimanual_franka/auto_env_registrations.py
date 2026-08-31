# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""RoboLab registrations for the bimanual Franka (two DROID arms, one articulation).

Two action spaces are offered, selected by ``action_space``:

- ``"jointpos"`` → env postfix ``BimanualFrankaJointPosition``, 16-dim actions
- ``"rel_ik"``   → env postfix ``BimanualFrankaRelIK``, 14-dim actions

Both see the same cameras: the left over-shoulder view plus one wrist camera per
arm, and the wide mirrored viewport for recording. Task discovery defaults to the
``bimanual`` folder, which holds the tasks written for two hands; pass
``task_dirs=["benchmark"]`` to run the single-arm benchmark on this robot (its
generic ``"gripper"`` checks then mean *either hand*).
"""

from robolab.constants import TASK_DIR

BIMANUAL_TASK_SUBFOLDERS = ["bimanual"]


def auto_register_bimanual_franka_envs(
    task_dirs=None,
    task=None,
    action_space: str = "jointpos",
):
    from robolab.core.environments.factory import auto_discover_and_create_cfgs
    from robolab.core.observations.observation_utils import (
        generate_image_obs_from_cameras,
        generate_obs_cfg,
    )
    from robolab.robots.bimanual_franka import (
        BimanualFrankaCfg,
        BimanualFrankaJointPositionActionCfg,
        BimanualFrankaRelIKActionCfg,
        BimanualProprioceptionObservationCfg,
        BimanualWristCamerasCfg,
        contact_gripper,
    )
    from robolab.variations.backgrounds import HomeOfficeBackgroundCfg
    from robolab.variations.camera import (
        EgocentricMirroredWideAngleHighCameraCfg,
        OverShoulderLeftCameraCfg,
    )
    from robolab.variations.lighting import SphereLightCfg

    if task_dirs is None:
        task_dirs = BIMANUAL_TASK_SUBFOLDERS

    actions = {
        "jointpos": (BimanualFrankaJointPositionActionCfg, "BimanualFrankaJointPosition"),
        "rel_ik": (BimanualFrankaRelIKActionCfg, "BimanualFrankaRelIK"),
    }
    if action_space not in actions:
        raise ValueError(f"action_space must be one of {sorted(actions)}, got {action_space!r}")
    actions_cfg_cls, env_postfix = actions[action_space]

    ViewportCameraCfg = generate_image_obs_from_cameras(
        [EgocentricMirroredWideAngleHighCameraCfg]
    )
    ImageObsCfg = generate_image_obs_from_cameras(
        [OverShoulderLeftCameraCfg, BimanualWristCamerasCfg]
    )
    ObservationCfg = generate_obs_cfg(
        {
            "image_obs": ImageObsCfg(),
            "proprio_obs": BimanualProprioceptionObservationCfg(),
            "viewport_cam": ViewportCameraCfg(),
        }
    )

    auto_discover_and_create_cfgs(
        task_dir=TASK_DIR,
        task_subdirs=task_dirs,
        tasks=task,
        pattern="*.py",
        env_postfix=env_postfix,
        observations_cfg=ObservationCfg(),
        actions_cfg=actions_cfg_cls(),
        robot_cfg=BimanualFrankaCfg,
        # Wrist cameras are attached through BimanualFrankaCfg; listing their
        # wrapper here would spawn them before the gripper links exist.
        camera_cfg=[
            OverShoulderLeftCameraCfg,
            EgocentricMirroredWideAngleHighCameraCfg,
        ],
        lighting_cfg=SphereLightCfg,
        background_cfg=HomeOfficeBackgroundCfg,
        contact_gripper=contact_gripper,
        dt=1 / 120,
        render_interval=8,
        decimation=8,
        seed=1,
    )
