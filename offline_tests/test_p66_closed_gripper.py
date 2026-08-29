# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""P66: GRIPPER_FULLY_CLOSED must mean fully closed.

`gripper_fully_closed` defaults to 75 % of the open->closed span, so a gripper
stalled on an object it is holding reads as "fully closed". Anchor, measured from
`isaac60_robolab120_pi05/BlackItemsInBinTask/run_0.hdf5`: in env0 the driving
`finger_joint` peaks at **0.830** of the nominal pi/4 span -- the smartphone
wedged between the fingers blocks it -- and 69 of that episode's 74 events are
GRIPPER_FULLY_CLOSED. At the event threshold of 0.98 it emits none.

The reviewer 2026-08-26: "I've seen this bug a couple times where the gripper actually
wasn't fully closed, so this is also true."
"""
import math
import types

import torch

import robolab.constants
from robolab.core.task.conditionals import gripper_fully_closed

SPAN = math.pi / 4
BLACKITEMS_ENV0_PEAK = 0.830          # fraction of SPAN, measured from the HDF5


class _World:
    def __init__(self, normalised):
        self.env = types.SimpleNamespace(num_envs=1, device="cpu")
        self._pos = torch.tensor([[normalised * SPAN]])

    def get_joint_positions(self, robot_name, env_id=None):
        return self._pos[env_id] if env_id is not None else self._pos

    def get_joint_names(self, robot_name):
        return ["finger_joint"]

    def resolve_contact_bodies(self, gripper_name):
        return ["gripper"]


def _closed(normalised, threshold):
    import robolab.core.task.conditionals as C
    C.get_world = lambda env: _World(normalised)
    env = types.SimpleNamespace(cfg=types.SimpleNamespace(gripper_closure_cfg=None))
    return bool(gripper_fully_closed(env, env_id=None, closed_threshold=threshold).item())


def test_the_wedged_gripper_is_not_fully_closed():
    assert _closed(BLACKITEMS_ENV0_PEAK, 0.75) is True, "the old threshold is what fired 69 times"
    assert _closed(BLACKITEMS_ENV0_PEAK, robolab.constants.GRIPPER_CLOSED_EVENT_THRESHOLD) is False


def test_a_genuinely_closed_gripper_still_fires():
    assert _closed(1.0, robolab.constants.GRIPPER_CLOSED_EVENT_THRESHOLD) is True


def test_the_predicate_default_is_untouched():
    """gripper_slightly_closed shares this function at 0.30; the default must stay 0.75."""
    import inspect
    sig = inspect.signature(gripper_fully_closed)
    assert sig.parameters["closed_threshold"].default == 0.75


def test_the_event_threshold_is_strict():
    assert robolab.constants.GRIPPER_CLOSED_EVENT_THRESHOLD >= 0.95
