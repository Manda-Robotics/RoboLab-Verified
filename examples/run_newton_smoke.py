"""DROID Newton/RTX smoke test without a policy server. Run on a CUDA host."""

import argparse
import json
import traceback

import cv2  # noqa: F401 -- load OpenCV before Kit
from isaaclab.app import AppLauncher


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--physics", choices=["physx", "newton_mjwarp"], default="newton_mjwarp")
    parser.add_argument("--task", default="RubiksCubeThenBananaTask")
    parser.add_argument("--steps", type=int, default=60)
    parser.add_argument("--num-envs", type=int, default=2)
    parser.add_argument("--output", default="output/newton_smoke")
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    if args.steps < 1 or args.num_envs < 1:
        parser.error("--steps and --num-envs must be positive")
    args.enable_cameras = True
    launcher = AppLauncher(args)

    import torch
    from robolab.constants import set_output_dir
    from robolab.core.environments.factory import get_envs
    from robolab.core.environments.physics import ensure_output_backend, managed_env
    from robolab.core.sensors.camera_utils import get_cameras
    from robolab.core.sensors.contact_sensor_utils import get_contact_sensors
    from robolab.core.utils.isaaclab_compat import as_torch
    from robolab.core.world.world_state import get_world
    from robolab.registrations.droid.auto_env_registrations_jointpos import auto_register_droid_envs

    ensure_output_backend(args.output, args.physics)
    set_output_dir(args.output)
    auto_register_droid_envs(task=[args.task])
    task = get_envs(task=args.task)[0]
    with managed_env(task, physics=args.physics, launcher_args=args,
                     device=args.device, num_envs=args.num_envs, seed=0) as (env, cfg):
        env.reset()
        for _ in range(args.steps):
            _, reward, _, _, _ = env.step(torch.zeros(
                (env.num_envs, env.action_manager.total_action_dim), device=env.device
            ))
            assert torch.isfinite(as_torch(env.scene["robot"].data.joint_pos)).all()
            assert torch.isfinite(as_torch(reward)).all()
        cameras = get_cameras(env.scene)
        assert len(cameras) == 3, list(cameras)
        for camera in cameras.values():
            rgb = as_torch(camera.data.output["rgb"])
            assert rgb.shape[0] == args.num_envs and rgb.shape[-1] == 3
            assert torch.isfinite(rgb).all() and rgb.max() > rgb.min()
        world = get_world(env)
        contacts = {}
        for name, sensor in get_contact_sensors(env.scene).items():
            forces = as_torch(sensor.data.force_matrix_w)
            assert forces.ndim == 4 and forces.shape[0] == args.num_envs and forces.shape[-1] == 3
            assert torch.isfinite(forces).all()
            if name.endswith("__all_objs"):
                continue
            a, b = name.split("__", 1)
            batch = world.in_contact(a, b, env_id=None)
            assert batch.shape == (args.num_envs,)
            assert batch.tolist() == [world.in_contact(a, b, env_id=i) for i in range(args.num_envs)]
            assert world.get_contact_force(a, b).shape == (args.num_envs, 3)
            contacts[name] = batch.tolist()
        assert contacts and any(any(v) for v in contacts.values()), contacts
        result = {"physics": args.physics, "task": task, "steps": args.steps,
                  "num_envs": args.num_envs, "contacts": contacts, "cameras": list(cameras)}
        from pathlib import Path
        Path(args.output, "smoke_result.json").write_text(json.dumps(result, indent=2) + "\n")
        print("ROBOLAB_SMOKE_PASS=" + json.dumps(result), flush=True)
    # Kit close can exit(0); only close after every assertion and export passed.
    launcher.app.close()


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        raise SystemExit(1)
