# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Bimanual observations retain RoboLab's WXYZ contract across IsaacLab versions."""
from types import SimpleNamespace

import numpy as np
import torch

from robolab.core.utils.isaaclab_compat import quat_wxyz_to_isaaclab
from robolab.robots import bimanual_franka as robot_cfg


def test_both_arm_poses_use_robot_root_and_export_wxyz():
    c = 2 ** -0.5
    data = SimpleNamespace(
        body_names=["left_arm_base_link", "right_arm_base_link"],
        root_pos_w=torch.tensor([[2., 3., 4.]]),
        root_quat_w=quat_wxyz_to_isaaclab(torch.tensor([[c, 0., 0., c]])),
        body_pos_w=torch.tensor([[[3., 3., 4.], [2., 4., 4.]]]),
        body_quat_w=quat_wxyz_to_isaaclab(torch.tensor([[[1., 0., 0., 0.], [0., 1., 0., 0.]]])),
    )
    env = SimpleNamespace(scene={"robot": SimpleNamespace(data=data)})
    np.testing.assert_allclose(robot_cfg.left_ee_pos(env), [[0., -1., 0.]], atol=1e-6)
    np.testing.assert_allclose(robot_cfg.right_ee_pos(env), [[1., 0., 0.]], atol=1e-6)
    np.testing.assert_allclose(robot_cfg.left_ee_quat(env), [[c, 0., 0., -c]], atol=1e-6)
    np.testing.assert_allclose(robot_cfg.right_ee_quat(env), [[0., c, -c, 0.]], atol=1e-6)
