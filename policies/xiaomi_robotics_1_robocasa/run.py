# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Evaluate Xiaomi-Robotics-1-RoboCasa on RoboLab tasks."""

import argparse
import sys
import traceback

import cv2  # noqa: F401 -- must import this before isaaclab. Do not remove
from isaaclab.app import AppLauncher

POLICY = "xiaomi_robotics_1_robocasa"

parser = argparse.ArgumentParser(description="Evaluate XR-1-RoboCasa through a remote model server.")
parser.add_argument("--remote-host", "--remote_host", default="localhost")
parser.add_argument("--remote-port", "--remote_port", default=10086, type=int)
parser.add_argument("--server-timeout-seconds", type=float, default=300.0)
parser.add_argument(
    "--open-loop-horizon", type=int, default=10, help="Actions consumed per query (official RoboCasa baseline: 10)."
)
parser.add_argument("--image-size", type=int, default=256)
parser.add_argument("--crop-ratio", type=float, default=0.95)
parser.add_argument(
    "--translation-delta-m",
    type=float,
    default=0.05,
    help="Physical translation represented by a normalized RoboCasa action of 1.",
)
parser.add_argument(
    "--rotation-delta-rad",
    type=float,
    default=0.5,
    help="Physical rotation represented by a normalized RoboCasa action of 1.",
)
parser.add_argument(
    "--gripper-open-position",
    type=float,
    default=0.04,
    help="Panda gripper joint value sent to XR-1 when RoboLab is fully open.",
)
parser.add_argument(
    "--gripper-closed-position",
    type=float,
    default=0.0,
    help="Panda gripper joint value sent to XR-1 when RoboLab is fully closed.",
)
parser.add_argument("--request-seed", type=int, default=42)

from robolab.eval.runner import add_common_eval_args, run_evaluation  # noqa: E402

add_common_eval_args(parser)
AppLauncher.add_app_launcher_args(parser)

args_cli = parser.parse_args()
args_cli.enable_cameras = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import robolab.constants  # noqa: E402
from policies.xiaomi_robotics_1_robocasa.client import XR1RoboCasaClient  # noqa: E402
from robolab.registrations.droid.auto_env_registrations_rel_ik import (  # noqa: E402
    auto_register_droid_rel_ik_envs,
)
from robolab.registrations.droid.camera_presets import WRIST_LEFT_RIGHT  # noqa: E402

robolab.constants.ENABLE_SUBTASK_PROGRESS_CHECKING = args_cli.enable_subtask

# RoboCasa's reference control rate is 20 Hz. RoboLab simulates at 120 Hz, so
# six physics steps are applied for each policy action.
auto_register_droid_rel_ik_envs(
    task_dirs=args_cli.task_dirs,
    task=args_cli.task,
    cameras=WRIST_LEFT_RIGHT,
    dt=1 / 120,
    render_interval=6,
    decimation=6,
)


def make_client(args: argparse.Namespace) -> XR1RoboCasaClient:
    return XR1RoboCasaClient(
        remote_host=args.remote_host,
        remote_port=args.remote_port,
        timeout_seconds=args.server_timeout_seconds,
        open_loop_horizon=args.open_loop_horizon,
        image_size=args.image_size,
        crop_ratio=args.crop_ratio,
        translation_delta_m=args.translation_delta_m,
        rotation_delta_rad=args.rotation_delta_rad,
        gripper_open_position=args.gripper_open_position,
        gripper_closed_position=args.gripper_closed_position,
        request_seed=args.request_seed,
    )


def main() -> None:
    run_evaluation(args_cli, policy=POLICY, client_factory=make_client)
    simulation_app.close()


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"\033[96m[RoboLab] Terminated with error: {exc}\033[0m")
        traceback.print_exc()
        simulation_app.close()
        sys.exit(1)
