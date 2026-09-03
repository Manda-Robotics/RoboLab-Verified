# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# isort: skip_file
"""Rendered joint-position smoke test for the bimanual YAM.

Both arms move from the home pose to Ai2's "rest" pose (joint2 = pi/4, joint3 = pi/2) and
back on a sinusoid, out of phase; both grippers open and close. Asserts: 16-dim action,
arm tracking, finger travel, and that each wrist camera stays a fixed distance from its
gripper while the arm moves (the failure that blinded the ALOHA sweeps). Writes the
viewport video, a policy-view strip (top | left wrist | right wrist, what MolmoAct 2
sees) and first-frame PNGs of the three policy cameras.
"""
import argparse
import math
import os
import sys
import traceback

import cv2  # noqa: F401  must be imported before isaaclab
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--num-steps", type=int, default=240)
parser.add_argument("--task", type=str, default="YamPutEverythingInBoxTask")
parser.add_argument("--output", type=str, default=None)
parser.add_argument("--top-cam", choices=["ai2_desk", "i2rt_gantry"], default="ai2_desk")
AppLauncher.add_app_launcher_args(parser)
args_cli, _ = parser.parse_known_args()
args_cli.enable_cameras = True
simulation_app = AppLauncher(args_cli).app

import numpy as np  # noqa: E402
import torch  # noqa: E402
from robolab.constants import PACKAGE_DIR  # noqa: E402
from robolab.core.environments.factory import get_envs  # noqa: E402
from robolab.core.environments.runtime import create_env  # noqa: E402
from robolab.core.observations.observation_utils import unpack_image_obs, unpack_viewport_cams  # noqa: E402
from robolab.core.utils.video_utils import VideoWriter  # noqa: E402
from robolab.robots.bimanual_yam import (  # noqa: E402
    ARM_JOINT_NAMES, ARMS, EE_BODY_NAME, FINGER_JOINT_NAMES, FINGER_TRAVEL_M,
)
from robolab.registrations.bimanual_yam.auto_env_registrations import auto_register_bimanual_yam_envs  # noqa: E402

REST = torch.tensor([0.0, math.pi / 4, math.pi / 2, 0.0, 0.0, 0.0])


def label(img, text):
    out = np.ascontiguousarray(img.copy())
    cv2.putText(out, text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2, cv2.LINE_AA)
    return out


def main() -> None:
    auto_register_bimanual_yam_envs(task=args_cli.task, top_cam=args_cli.top_cam)
    env_name = get_envs(task=args_cli.task)[0]
    env, env_cfg = create_env(env_name, device=args_cli.device, num_envs=1, use_fabric=True)
    out_dir = args_cli.output or os.path.join(PACKAGE_DIR, "output", "bimanual_yam_smoke")
    os.makedirs(out_dir, exist_ok=True)
    fps = 1 / (env_cfg.sim.render_interval * env_cfg.sim.dt)
    video = VideoWriter(os.path.join(out_dir, "viewport.mp4"), fps)
    strip = VideoWriter(os.path.join(out_dir, "policy_views.mp4"), fps)
    try:
        obs, _ = env.reset()
        robot = env.scene["robot"]
        names = robot.data.joint_names
        print(f"Environment: {env_name}")
        print(f"Bodies ({len(robot.data.body_names)}): {robot.data.body_names}")
        print(f"Joints ({len(names)}): {names}")
        print(f"Action dimension: {env.action_manager.total_action_dim}")
        print(f"Control dt: {env_cfg.sim.dt * env_cfg.decimation:.4f} s")
        if env.action_manager.total_action_dim != 16:
            raise RuntimeError("expected 16-dim actions: 6 + 2 + 6 + 2")
        arm_idx = {s: [names.index(j) for j in ARM_JOINT_NAMES[s]] for s in ARMS}
        fin_idx = {s: [names.index(j) for j in FINGER_JOINT_NAMES[s]] for s in ARMS}
        body_idx = {s: robot.data.body_names.index(EE_BODY_NAME[s]) for s in ARMS}
        cams = {s: env.scene[f"{s}_wrist_cam"] for s in ARMS}

        # First-frame policy cameras.
        imgs = unpack_image_obs(obs)
        for k in ("top_cam", "left_wrist_cam", "right_wrist_cam"):
            if k not in imgs:
                raise RuntimeError(f"policy camera {k} missing from image_obs: {sorted(imgs)}")
            cv2.imwrite(os.path.join(out_dir, f"{k}_t0.png"), cv2.cvtColor(imgs[k], cv2.COLOR_RGB2BGR))
            print(f"{k}: {imgs[k].shape}")

        cam_dist = {s: [] for s in ARMS}
        open_err = {}
        closed_err = {}
        tracking = {}
        n = args_cli.num_steps
        for step in range(n):
            phase = 2.0 * math.pi * step / n
            parts = []
            for k, s in enumerate(ARMS):
                w = 0.5 * (1.0 - math.cos(phase + (0.0 if k == 0 else math.pi / 2)))   # 0 -> 1 -> 0
                t = (1.0 - w) * torch.zeros(6) + w * REST
                grip_open = 1.0 if math.sin(phase) > 0.0 else 0.0            # 1 = open
                finger = -FINGER_TRAVEL_M * grip_open
                parts += [t, torch.tensor([finger, finger])]
                tracking[s] = t
            action = torch.cat(parts).unsqueeze(0).to(env.device)
            obs, _, _, _, _ = env.step(action)

            for s in ARMS:
                # Wrist camera must ride the gripper: constant camera-to-gripper distance.
                cam_pos = cams[s].data.pos_w[0]
                grip_pos = robot.data.body_pos_w[0, body_idx[s]]
                cam_dist[s].append(torch.norm(cam_pos - grip_pos).item())
                q7 = robot.data.joint_pos[0, fin_idx[s][0]].item()
                if step == n // 4:                      # told to open since step 0
                    open_err[s] = abs(q7 - (-FINGER_TRAVEL_M))
                if step == 3 * n // 4:                  # told to close since n/2
                    closed_err[s] = abs(q7 - 0.0)

            imgs = unpack_image_obs(obs)
            frame = unpack_viewport_cams(obs).get("combined_image")
            if frame is not None:
                video.write(frame)
            strip.write(cv2.hconcat([label(imgs["top_cam"], "TOP"), label(imgs["left_wrist_cam"], "LEFT WRIST"),
                                     label(imgs["right_wrist_cam"], "RIGHT WRIST")]))

        ok = True
        for s in ARMS:
            d = np.array(cam_dist[s])
            print(f"{s}: wrist cam to gripper distance mean {d.mean():.4f} m, spread {d.max() - d.min():.5f} m "
                  f"(expected {math.sqrt(0.0704**2 + 0.077**2):.4f} m, spread ~0)")
            if d.max() - d.min() > 0.003:
                ok = False
                print(f"  FAIL: {s} wrist camera is not following the gripper")
            print(f"{s}: finger open error {open_err[s] * 1000:.2f} mm, closed error {closed_err[s] * 1000:.2f} mm")
            if open_err[s] > 0.005 or closed_err[s] > 0.005:
                ok = False
                print(f"  FAIL: {s} fingers did not reach the commanded travel")
            err = torch.max(torch.abs(robot.data.joint_pos[0, arm_idx[s]].cpu() - tracking[s])).item()
            print(f"{s}: final arm tracking error {err:.4f} rad")
            if err > 0.05:
                ok = False
                print(f"  FAIL: {s} arm tracking error too large")
        gp = {s: robot.data.body_pos_w[0, body_idx[s]].cpu().numpy().round(3).tolist() for s in ARMS}
        print(f"gripper world positions at end: {gp}")
        print(f"Saved: {out_dir}")
        print("BIMANUAL_YAM_SMOKE_OK" if ok else "BIMANUAL_YAM_SMOKE_FAILED")
        if not ok:
            sys.exit(2)
    finally:
        video.release()
        strip.release()
        env.close()
        simulation_app.close()


if __name__ == "__main__":
    try:
        main()
    except SystemExit:
        raise
    except Exception as error:
        print(f"Terminated with error: {error}")
        traceback.print_exc()
        simulation_app.close()
        sys.exit(1)
