# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Evaluate MolmoAct 2 (bimanual YAM checkpoint) on the bimanual YAM rig.

Serve the checkpoint with Ai2's server first (allenai/molmoact2 repo):

    uv run python examples/yam/host_server_yam.py --host 0.0.0.0 --port 8202 --dtype bfloat16

then, from RoboLab:

    python policies/molmoact2/run.py --task YamPutEverythingInBoxTask --task-dirs bimanual \
        --num-envs 2 --headless --server http://localhost:8202
    python policies/molmoact2/run.py --task BananaInBowlTask --task-dirs benchmark ...
"""
import argparse
import sys
import traceback

import cv2  # noqa: F401 -- must import this before isaaclab. Do not remove
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="MolmoAct 2 on the bimanual YAM.")
parser.add_argument("--server", type=str, default="http://localhost:8202",
                    help="MolmoAct 2 YAM inference server (host_server_yam.py); '/act' is appended.")
parser.add_argument("--open-loop-horizon", "--open_loop_horizon", type=int, default=None,
                    help="Actions executed per 30-step chunk before replanning (default 30 = whole chunk).")
parser.add_argument("--num-steps", "--num_steps", type=int, default=10, help="Flow solver steps (server default 10).")
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
from policies.molmoact2.yam_client import MolmoAct2YamClient  # noqa: E402
from robolab.registrations.bimanual_yam.auto_env_registrations import auto_register_bimanual_yam_envs  # noqa: E402

robolab.constants.ENABLE_SUBTASK_PROGRESS_CHECKING = args_cli.enable_subtask
robolab.constants.RECORD_IMAGE_DATA = args_cli.record_image_data
robolab.constants.VERBOSE = args_cli.enable_verbose
robolab.constants.DEBUG = args_cli.enable_debug

auto_register_bimanual_yam_envs(task_dirs=args_cli.task_dirs, task=args_cli.task, top_cam=args_cli.top_cam)


def make_client(args: argparse.Namespace) -> MolmoAct2YamClient:
    return MolmoAct2YamClient(server=args.server, open_loop_horizon=args.open_loop_horizon,
                              num_steps=args.num_steps)


def main() -> None:
    run_evaluation(args_cli, policy="molmoact2_bimanual_yam", client_factory=make_client)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        sys.exit(1)
    finally:
        simulation_app.close()
