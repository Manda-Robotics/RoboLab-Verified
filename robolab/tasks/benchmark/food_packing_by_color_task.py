# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass

import isaaclab.envs.mdp as mdp
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass

from robolab.core.scenes.utils import import_scene
from robolab.core.task.conditionals import object_groups_in_containers, object_in_container, pick_and_place
from robolab.core.task.task import Task


@configclass
class FoodPackingByColorTerminations:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    success = DoneTerm(
        func=object_groups_in_containers,
        params={
            "groups": [
                {"object": ["coffee_can"], "container": "bin_a06", "logical": "all", "require_contact_with": False, "require_gripper_detached": True},
                {"object": ["mustard", "sugar_box"], "container": "bin_b03", "logical": "all", "require_contact_with": False, "require_gripper_detached": True},
            ]
        }
    )

@dataclass
class FoodPackingByColorTask(Task):
    contact_object_list = ["bin_a06", "cheez_it", "chocolate_pudding", "coffee_can", "mustard", "spam_can", "sugar_box", "tomato_soup_can", "bin_b03", "table"]
    scene = import_scene("food_packing.usda", contact_object_list)
    terminations = FoodPackingByColorTerminations
    instruction = {
        "default": "Pack yellow objects in right container and blue object in the left container",
        "vague": "Sort things by color",
        "specific": "Pick up the yellow can and place it in the right container, then pick up the blue object and place it in the left container",
    }
    episode_length_s: int = 120
    attributes = ['spatial', 'sorting', 'color']
    # Same placement as `success` above: yellow (mustard, sugar_box) -> bin_b03,
    # blue (coffee_can) -> bin_a06. The ladder used to send mustard to bin_a06 and
    # coffee_can to bin_b03 and never mentioned sugar_box, so a policy could hold
    # subtask score 1.0 while being structurally unable to succeed (VERIFIED_PLAN
    # H-R8-2; caught by scripts/find_task_definition_conflicts.py).
    subtasks = [
        pick_and_place(
            object=["mustard", "sugar_box"],
            container="bin_b03",
            logical="all",
            score=0.5
        ),
        pick_and_place(
            object=["coffee_can"],
            container="bin_a06",
            logical="all",
            score=0.5
        )
    ]
