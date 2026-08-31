# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""ALOHA transfer-cube: the pi0_aloha_sim checkpoint's literal training task.

Harness control experiment: a policy trained on this task in MuJoCo sim should
produce recognizable reach-grasp-handover behaviour on a correct ALOHA rig even
across the sim-to-sim gap. Success = the cube held by the right gripper (the
gym-aloha convention: right arm picks up, hands to left... scored simply here as
lifted by either hand; the point is behavioural coherence, not the score).
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


@configclass
class AlohaTransferCubeTerminations:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    success = DoneTerm(
        func=object_picked_up,
        params={"object": "red_block", "surface": "table", "distance": 0.08},
    )


@dataclass
class AlohaTransferCubeTask(Task):
    contact_object_list = ["red_block", "table"]
    scene = import_scene("aloha_transfer_cube.usda", contact_object_list)
    terminations = AlohaTransferCubeTerminations
    instruction = {
        "default": "Transfer cube",
        "vague": "Pick up the cube",
        "specific": "Pick up the red cube and transfer it to the other gripper",
    }
    episode_length_s: int = 30
    attributes = ["bimanual"]
    subtasks = [
        Subtask(
            name="grab_cube",
            conditions={"red_block": [
                (partial(object_grabbed, object="red_block"), 0.5),
                (partial(object_picked_up, object="red_block", surface="table", distance=0.08), 1.0),
            ]},
            logical="all",
            score=1.0,
        )
    ]
