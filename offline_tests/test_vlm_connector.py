# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The VLM pointing connector: translation only, and it must not need the package.

RoboLab Verified carries the connector, not the controller -- `vlm-pinpoint` is an
optional extra. So this module has to import, and its coordinate maths has to be
testable, with the package absent (it is absent here).
"""
import numpy as np
import pytest

from policies.vlm_pinpoint.connector import (
    FLANGE_TO_FINGERTIP_M,
    RoboLabPointingClient,
    fingertip_position,
    quat_rotate,
    to_harness_observation,
)


def test_module_imports_without_the_optional_package():
    import importlib.util
    assert importlib.util.find_spec("vlm_pinpoint") is None, "test is meaningless if it is installed"


def test_identity_rotation_leaves_a_vector_alone():
    assert np.allclose(quat_rotate((1, 0, 0, 0), (1, 2, 3)), (1, 2, 3))


def test_half_turn_about_z_flips_x_and_y():
    assert np.allclose(quat_rotate((0, 0, 0, 1), (1, 0, 0)), (-1, 0, 0), atol=1e-9)


def test_fingertip_sits_ahead_of_the_flange_along_local_z():
    tip = fingertip_position((0, 0, 0), (1, 0, 0, 0))
    assert np.allclose(tip, (0, 0, FLANGE_TO_FINGERTIP_M))


def test_fingertip_follows_the_hand_orientation():
    # rotated 90 deg about y: local +z now points along world +x
    q = (np.cos(np.pi / 4), 0.0, np.sin(np.pi / 4), 0.0)
    tip = fingertip_position((1, 0, 0), q)
    assert np.allclose(tip, (1 + FLANGE_TO_FINGERTIP_M, 0, 0), atol=1e-9)


def test_observation_is_unbatched_for_the_requested_env():
    obs = {"policy": {"scene_rgb": np.zeros((2, 4, 4, 3)),
                      "ee_pos": np.array([[0.0, 0, 0], [1.0, 0, 0]]),
                      "ee_quat": np.array([[1.0, 0, 0, 0], [1.0, 0, 0, 0]])}}
    out = to_harness_observation(obs, env_id=1)
    assert out["scene"].shape == (4, 4, 3)
    assert np.allclose(out["tip_pos"], (1.0, 0, FLANGE_TO_FINGERTIP_M))


def test_missing_channels_become_none_not_an_exception():
    out = to_harness_observation({"policy": {}}, env_id=0)
    assert out["depth"] is None and out["tip_pos"] is None


def test_using_the_client_without_the_package_says_what_to_install():
    with pytest.raises(ImportError, match="vlm-pinpoint"):
        RoboLabPointingClient("pick up the banana").get_action({"policy": {}})
