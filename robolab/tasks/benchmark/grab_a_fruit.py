# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass
from functools import partial

import isaaclab.envs.mdp as mdp
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass

from robolab.core.scenes.utils import import_scene
from robolab.core.task.conditionals import object_above, object_grabbed, object_picked_up
from robolab.core.task.subtask import Subtask
from robolab.core.task.task import Task


@configclass
class Terminations:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    success_banana = DoneTerm(func=object_picked_up, params={"object": "banana", "surface": "table", "distance": 0.05})
    success_banana_01 = DoneTerm(func=object_picked_up, params={"object": "banana_01", "surface": "table", "distance": 0.05})
    success_apple = DoneTerm(func=object_picked_up, params={"object": "apple_01", "surface": "table", "distance": 0.05})
    success_orange = DoneTerm(func=object_picked_up, params={"object": "orange2", "surface": "table", "distance": 0.05})


@dataclass
class GrabAFruitTask(Task):
    """Task: Grab a fruit."""
    contact_object_list = [
        "table", "bowl", "banana", "bagel_07", "coffee_can", "banana_01",
        "yogurt_cup", "coffee_pot", "ceramic_mug", "pitcher", "fork_big",
        "spoon_big", "apple_01", "orange2", "milk_carton",
        "orange_juice_carton", "bagel_01", "bagel_02", "plate_small",
        "plate_large"
    ]
    scene = import_scene("breakfast_table.usda", contact_object_list)
    terminations = Terminations
    instruction = {
        "default": "Pick up a fruit",
        "vague": "Grab a fruit",
        "specific": "Reach for and grasp any one of the fruits on the table and lift it off the surface",
    }
    episode_length_s: int = 30
    attributes = ['semantics']
    # P65 (H-R9-T4): same defect as GrabABagel -- success is a 50 mm lift
    # (object_picked_up), the ladder was contact only, so env1 of
    # isaac60_robolab120_pi05 logs "Completed subtask 'grab_a_fruit' 1/1" at
    # 6.20 s on an episode whose success is False. Two rungs per fruit, the
    # second one being the success predicate.
    subtasks = [
        Subtask(
            name="grab_a_fruit",
            conditions={
                "banana": [
                    partial(object_grabbed, object="banana"),
                    partial(object_picked_up, object="banana", surface="table", distance=0.05),
                ],
                "banana_01": [
                    partial(object_grabbed, object="banana_01"),
                    partial(object_picked_up, object="banana_01", surface="table", distance=0.05),
                ],
                "apple_01": [
                    partial(object_grabbed, object="apple_01"),
                    partial(object_picked_up, object="apple_01", surface="table", distance=0.05),
                ],
                "orange2": [
                    partial(object_grabbed, object="orange2"),
                    partial(object_picked_up, object="orange2", surface="table", distance=0.05),
                ],
            },
            logical="any",
            score=1.0
        )
    ]
