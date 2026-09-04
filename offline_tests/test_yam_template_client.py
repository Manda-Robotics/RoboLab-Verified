# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The copy-me client for a third-party bimanual YAM policy speaks the documented contract."""
import numpy as np

from policies.yam_template.client import FINGER_TRAVEL_M, TemplateYamClient


def _obs(n=2):
    img = np.zeros((n, 360, 640, 3), np.uint8)
    return {"image_obs": {"top_cam": img, "left_wrist_cam": img, "right_wrist_cam": img},
            "proprio_obs": {"left_arm_joint_pos": np.zeros((n, 6), np.float32),
                            "left_gripper_pos": np.ones((n, 1), np.float32),
                            "right_arm_joint_pos": np.zeros((n, 6), np.float32),
                            "right_gripper_pos": np.ones((n, 1), np.float32)}}


def test_template_emits_the_rig_action_from_the_documented_observation():
    c = TemplateYamClient()
    acts = np.stack([c.infer(_obs(), "wobble", env_id=1)["action"] for _ in range(60)])
    assert acts.shape == (60, 16)
    for base in (6, 14):                                     # finger pairs move together, inside the joint range
        assert np.array_equal(acts[:, base], acts[:, base + 1])
        assert acts[:, base].min() >= -FINGER_TRAVEL_M - 1e-6 and acts[:, base].max() <= 0.0
    assert np.abs(acts[:, 2]).max() > 0.05 and np.abs(acts[:, 10]).max() > 0.05   # the elbows actually move
