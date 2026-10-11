# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Guard the eval runner's argparse wiring against AppLauncher collisions.

The per-policy runners build their parser as::

    add_common_eval_args(parser)
    add_app_launcher_args(parser)

IsaacLab 2.3's ``add_app_launcher_args`` raises ``ValueError`` if the parser
already declares a field it owns (e.g. ``rendering_mode``). This never shows up
in the other tests because they drive ``AppLauncher`` directly rather than
through the runner's parser — so it's asserted explicitly here.
"""

import argparse

from isaaclab.app import AppLauncher

from robolab.eval.runner import add_common_eval_args
from robolab.core.utils.app_launcher import add_app_launcher_args


def test_common_eval_args_do_not_collide_with_app_launcher():
    parser = argparse.ArgumentParser()
    add_common_eval_args(parser)
    # Exercise the same compatibility wrapper as the policy entry points.
    add_app_launcher_args(parser)
    native_parser = argparse.ArgumentParser()
    AppLauncher.add_app_launcher_args(native_parser)
    native_dests = {action.dest for action in native_parser._actions}

    dests = {action.dest for action in parser._actions}
    # Preserve every native option, including rendering_mode on older Lab.
    # Lab 3 removed that option; RoboLab's stable launch flags still coexist.
    assert native_dests <= dests
    assert "renderer" in dests
    assert "rendering_type" in dests
    launch = parser.parse_args(["--headless", "--enable_cameras"])
    assert launch.headless and launch.enable_cameras
    assert "enable_gt_state" in dests
    assert parser.parse_args([]).physics == "physx"
    assert parser.parse_args(["--physics", "newton_mjwarp"]).physics == "newton_mjwarp"
