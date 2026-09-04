# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Template runner for a bimanual YAM policy: the MolmoAct 2 runner with the client swapped.

Copy policies/yam_template/, point ``make_client`` at your client, run:

    python policies/yam_template/run.py --task YamPutEverythingInBoxTask --task-dirs bimanual --num-envs 2 --headless
"""
import argparse
import sys
import traceback

import cv2  # noqa: F401 -- must import this before isaaclab. Do not remove
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="A bimanual YAM policy through RoboLab (template).")
parser.add_argument("--top-cam", "--top_cam", choices=["ai2_desk", "i2rt_gantry"], default="ai2_desk",
                    help="Overhead camera placement: Ai2 desk kit (MolmoAct 2 default) or I2RT gantry station.")
parser.add_argument("--enable-verbose", "--enable_verbose", action="store_true")
parser.add_argument("--enable-debug", "--enable_debug", action="store_true")
parser.add_argument("--record-image-data", "--record_image_data", action="store_true")

from robolab.eval.runner import add_common_eval_args, run_evaluation  # noqa: E402

add_common_eval_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli, _ = parser.parse_known_args()
args_cli.enable_cameras = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import robolab.constants  # noqa: E402
from policies.yam_template.client import TemplateYamClient  # noqa: E402
from robolab.registrations.bimanual_yam.auto_env_registrations import auto_register_bimanual_yam_envs  # noqa: E402

robolab.constants.ENABLE_SUBTASK_PROGRESS_CHECKING = args_cli.enable_subtask
robolab.constants.RECORD_IMAGE_DATA = args_cli.record_image_data
robolab.constants.VERBOSE = args_cli.enable_verbose
robolab.constants.DEBUG = args_cli.enable_debug

auto_register_bimanual_yam_envs(task_dirs=args_cli.task_dirs, task=args_cli.task, top_cam=args_cli.top_cam)


def make_client(args: argparse.Namespace) -> TemplateYamClient:
    return TemplateYamClient()          # construct your client here (server address, chunk length, ...)


def main() -> None:
    run_evaluation(args_cli, policy="yam_template", client_factory=make_client)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        sys.exit(1)
    finally:
        simulation_app.close()
