# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Run a bimanual rig.

The fork ships two bimanual embodiments and, until now, no way to run either: the
single-arm runners register `droid` envs and drive them with an 8-dim DROID policy.
This registers the bimanual envs and drives them with the scripted client, which is
what makes the rigs verifiable at runtime at all.

    # dual Franka — the default, and the rig to reach for
    python policies/bimanual/run.py --task BimanualLiftToteTask --num-envs 2 --headless

    # bimanual YAM — the rig with a released policy (policies/molmoact2/); this is the
    # no-checkpoint way to see it move
    python policies/bimanual/run.py --robot yam --task YamPutEverythingInBoxTask --headless

    # ALOHA / ViperX — config only in this repo (asset not shipped)
    python policies/bimanual/run.py --robot aloha --task AlohaTransferCubeTask --headless

There is no `--policy` beyond `scripted`: no released checkpoint drives two arms.
Numbers out of this runner are a statement that the stack turns, nothing more, and
the banner below says so on every run.
"""
import argparse
import sys
import traceback

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Run a bimanual rig with the scripted client.")
parser.add_argument("--robot", choices=["bimanual_franka", "yam", "aloha"], default="bimanual_franka",
                    help="Which bimanual rig. Dual Franka (default, scripted lift 6/6), the bimanual "
                         "YAM (the rig with a released policy, see policies/molmoact2/), or ALOHA "
                         "(config only in this repo; its asset is not shipped).")
parser.add_argument("--top-cam", "--top_cam", choices=["ai2_desk", "i2rt_gantry"], default="ai2_desk",
                    help="YAM only: overhead camera placement (Ai2 desk kit or I2RT gantry station).")
parser.add_argument("--action-space", "--action_space", choices=["jointpos", "rel_ik"],
                    default="jointpos", help="16-dim absolute joint targets, or 14-dim relative IK.")
parser.add_argument("--aloha-variant", "--aloha_variant", default="opposing",
                    help="ALOHA arm layout. Only read when --robot aloha.")
parser.add_argument("--amplitude-rad", "--amplitude_rad", type=float, default=0.12,
                    help="Elbow excursion of the scripted motion. Small on purpose: a big "
                         "swing trips the collision flags and the smoke test then reads "
                         "like a failing evaluation.")
# These three are NOT in add_common_eval_args -- each runner declares its own, and the
# constants block below reads all three. Omitting them cost a pod launch: the run got as
# far as building Isaac and then died on AttributeError: 'Namespace' has no attribute
# 'record_image_data'.
parser.add_argument("--enable-verbose", "--enable_verbose", action="store_true",
                    help="Verbose output (default: False).")
parser.add_argument("--enable-debug", "--enable_debug", action="store_true",
                    help="Debug output (default: False).")
parser.add_argument("--record-image-data", "--record_image_data", action="store_true",
                    help="Enable proprio image data recording (default: False).")

from robolab.eval.runner import add_common_eval_args, run_evaluation  # noqa: E402

add_common_eval_args(parser)
AppLauncher.add_app_launcher_args(parser)

args_cli, _ = parser.parse_known_args()
args_cli.enable_cameras = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import robolab.constants  # noqa: E402
from robolab.constants import DEFAULT_TASK_SUBFOLDERS  # noqa: E402

from policies.bimanual.client import ScriptedBimanualClient  # noqa: E402

robolab.constants.ENABLE_SUBTASK_PROGRESS_CHECKING = args_cli.enable_subtask
robolab.constants.RECORD_IMAGE_DATA = args_cli.record_image_data
robolab.constants.VERBOSE = args_cli.enable_verbose
robolab.constants.DEBUG = args_cli.enable_debug

BANNER = """\033[93m
================================================================
  SMOKE TEST, NOT AN EVALUATION.
  The scripted client is a fixed sinusoid, not a policy. It shows
  the rig builds, both arms actuate, both wrist cameras render and
  the event pipeline runs. Success rates from this run mean nothing
  and must not be reported next to policy numbers.
================================================================\033[0m"""


def bimanual_task_dirs():
    """Forward --task-dirs only if the user actually asked for it.

    `add_common_eval_args` defaults it to ["benchmark"], the single-arm task set.
    Passing that through would have the two-arm rigs discover one-arm tasks and find
    no bimanual task at all. Leaving it None lets each registration use its own
    default -- ["bimanual"] for the Franka. Running the single-arm benchmark on two
    arms is still available, by asking for it: --task-dirs benchmark.
    """
    if list(args_cli.task_dirs) == list(DEFAULT_TASK_SUBFOLDERS):
        return None
    return args_cli.task_dirs


def register() -> None:
    task_dirs = bimanual_task_dirs()
    if args_cli.robot == "yam":
        from robolab.registrations.bimanual_yam.auto_env_registrations import auto_register_bimanual_yam_envs
        if args_cli.action_space != "jointpos":
            raise SystemExit("the bimanual YAM has a joint-position action space only")
        auto_register_bimanual_yam_envs(task_dirs=task_dirs, task=args_cli.task, top_cam=args_cli.top_cam)
    elif args_cli.robot == "aloha":
        from robolab.registrations.aloha.auto_env_registrations import auto_register_aloha_envs
        print("\033[93m[RoboLab] ALOHA/ViperX: rig verified, no policy verified on it. "
              "See robolab/robots/aloha.py.\033[0m")
        auto_register_aloha_envs(task_dirs=task_dirs, task=args_cli.task,
                                 variant=args_cli.aloha_variant)
    else:
        from robolab.registrations.bimanual_franka.auto_env_registrations import (
            auto_register_bimanual_franka_envs,
        )
        auto_register_bimanual_franka_envs(task_dirs=task_dirs, task=args_cli.task,
                                           action_space=args_cli.action_space)


def make_client(args: argparse.Namespace) -> ScriptedBimanualClient:
    finger_travel = None
    if args.robot == "yam":
        from robolab.robots.bimanual_yam import FINGER_TRAVEL_M
        finger_travel = FINGER_TRAVEL_M
    return ScriptedBimanualClient(action_space=args.action_space,
                                  amplitude_rad=args.amplitude_rad,
                                  control_hz=30.0 if args.robot == "yam" else 15.0,
                                  finger_travel_m=finger_travel)


def main() -> None:
    print(BANNER)
    register()
    run_evaluation(args_cli, policy=f"scripted-{args_cli.robot}", client_factory=make_client)
    simulation_app.close()


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\033[96m[RoboLab] Terminated with error: {e}\033[0m")
        traceback.print_exc()
        simulation_app.close()
        sys.exit(1)
