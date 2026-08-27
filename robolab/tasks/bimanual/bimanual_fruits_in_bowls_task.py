# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""A banana and a bowl on each side of the table; each banana goes in its own bowl.

Either hand may do either placement — the checks use the default ``"gripper"``
group, which on the bimanual robot means left or right — but the layout rewards
doing both at once, one arm per side.
"""

from dataclasses import dataclass

import isaaclab.envs.mdp as mdp
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass

from robolab.core.scenes.utils import import_scene
from robolab.core.task.conditionals import (
    object_groups_in_containers,
    pick_and_place_grouped,
)
from robolab.core.task.task import Task

GROUPS = [
    {"object": "banana_left", "container": "bowl_left", "require_gripper_detached": True},
    {"object": "banana_right", "container": "bowl_right", "require_gripper_detached": True},
]


@configclass
class BimanualFruitsInBowlsTerminations:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    success = DoneTerm(func=object_groups_in_containers, params={"groups": GROUPS})


@dataclass
class BimanualFruitsInBowlsTask(Task):
    contact_object_list = ["banana_left", "banana_right", "bowl_left", "bowl_right", "table"]
    scene = import_scene("bimanual_fruits_bowls.usda", contact_object_list)
    terminations = BimanualFruitsInBowlsTerminations
    instruction = {
        "default": "Put each banana in the bowl on its own side",
        "vague": "Put the fruit in the bowls",
        "specific": "Place the left banana in the left bowl and the right banana in the right bowl",
    }
    episode_length_s: int = 60
    attributes = ["bimanual", "spatial"]
    subtasks = [
        pick_and_place_grouped(groups=GROUPS, logical="all", score=1.0),
    ]
