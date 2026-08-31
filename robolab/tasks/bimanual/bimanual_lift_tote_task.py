# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Lift a 0.6 m tote off the table with both hands.

The tote spans the table with one end in front of each arm; no single Robotiq
gripper can lift it level. Success needs *both* grippers in contact
(``gripper_name=["left", "right"]`` — a list means all of them, see
docs/task_conditionals.md#gripper-names) while the tote is clear of the table.
"""

from dataclasses import dataclass
from functools import partial

import isaaclab.envs.mdp as mdp
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass

from robolab.core.scenes.utils import import_scene
from robolab.core.task.conditionals import object_grabbed, object_picked_up
from robolab.core.task.subtask import Subtask
from robolab.core.task.task import Task

BOTH_HANDS = ["left", "right"]
LIFT_HEIGHT_M = 0.08


@configclass
class BimanualLiftToteTerminations:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    success = DoneTerm(
        func=object_picked_up,
        params={"object": "tote", "surface": "table", "distance": LIFT_HEIGHT_M,
                "gripper_name": BOTH_HANDS},
    )


@dataclass
class BimanualLiftToteTask(Task):
    contact_object_list = ["tote", "table"]
    scene = import_scene("bimanual_lift_tote.usda", contact_object_list)
    terminations = BimanualLiftToteTerminations
    instruction = {
        "default": "Lift the tote off the table with both hands",
        "vague": "Pick up the container",
        "specific": "Grab the grey tote by both ends, one hand on each, and lift it straight up off the table",
    }
    episode_length_s: int = 40
    attributes = ["bimanual"]

    # Keyed by object so the event tracker knows the tote is the target
    # (a bare partial would make it read the keyword name "conditions" instead).
    subtasks = [
        Subtask(
            name="bimanual_lift",
            conditions={
                "tote": [
                    (partial(object_grabbed, object="tote", gripper_name=BOTH_HANDS), 0.5),
                    (partial(object_picked_up, object="tote", surface="table",
                             distance=LIFT_HEIGHT_M, gripper_name=BOTH_HANDS), 1.0),
                ]
            },
            logical="all",
            score=1.0,
        )
    ]
