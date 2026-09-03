# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Parity task for the bimanual YAM: Ai2's ``BimanualYAMPutEverythingInBox-v1`` in RoboLab.

Two objects sit 0.35 m in front of the arms, one on each side (block left, orange right, in
place of Ai2's lego duplo and tennis ball); an open-top box sits 0.50 m ahead on the midline.
Success = both objects inside the box. The point of this task is a like-for-like comparison
with the MolmoAct 2 result in Ai2's ManiSkill harness (4/8 at 40 s, 4/8 at 67 s): same
checkpoint, same layout, our simulator. It is a smoke test, not a benchmark task.
"""

from dataclasses import dataclass

import isaaclab.envs.mdp as mdp
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass

from robolab.core.scenes.utils import import_scene
from robolab.core.task.conditionals import object_in_container, pick_and_place
from robolab.core.task.task import Task

OBJECTS = ["block", "orange"]


@configclass
class YamPutEverythingInBoxTerminations:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    success = DoneTerm(
        func=object_in_container,
        params={"object": OBJECTS, "container": "box", "logical": "all", "tolerance": 0.0,
                "require_contact_with": True, "require_gripper_detached": True,
                "gripper_name": "gripper"},
    )


@dataclass
class YamPutEverythingInBoxTask(Task):
    contact_object_list = ["block", "orange", "box", "table"]
    scene = import_scene("yam_put_everything_in_box.usda", contact_object_list)
    terminations = YamPutEverythingInBoxTerminations
    instruction = {
        "default": "put everything into the box",
        "vague": "Clear the table into the box",
        "specific": "Pick up the red block and the orange and place both of them inside the box",
    }
    episode_length_s: int = 67          # Ai2's longer cap (2000 steps at 30 Hz)
    attributes = ["bimanual", "parity"]

    subtasks = [
        pick_and_place(object=OBJECTS, container="box", logical="all", score=1.0),
    ]
