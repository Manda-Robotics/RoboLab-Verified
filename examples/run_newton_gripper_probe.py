"""Isolate DROID gripper dynamics: hold arm, open/close/hold, record at 120 Hz.

Run each gain setting in a separate process and a fresh output directory.
Does not change the production Newton preset or require a policy server.
"""

import argparse
import json
import math
import os
from pathlib import Path
import sys
import traceback

import cv2  # noqa: F401 -- before Kit
from isaaclab.app import AppLauncher


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gain-mode", choices=["inherited", "damping", "soft", "gentle"], default="inherited")
    parser.add_argument("--output", required=True)
    parser.add_argument("--phase-seconds", type=float, default=3.0)
    parser.add_argument("--mujoco-cpu", action="store_true", help="Diagnostic control: use MuJoCo CPU with the same imported model")
    parser.add_argument("--ramp-seconds", type=float, default=0.0, help="Ramp the target over this duration instead of applying a step")
    parser.add_argument("--follower-damping", type=float, default=0.0, help="Optional velocity damping on the five passive mimic joints")
    parser.add_argument("--passive-damping", type=float, default=0.0, help="Author MuJoCo passive joint damping in a diagnostic overlay")
    parser.add_argument("--substeps", type=int, default=4)
    parser.add_argument("--rate-limit", action="store_true", help="Use the runtime physics-rate binary target limiter")
    parser.add_argument("--contact-block", action="store_true", help="Close on a kinematically held 50-mm block")
    parser.add_argument("--mimic-timeconst", type=float, default=0.0,
                        help="Diagnostic: author explicit MuJoCo equality constraints with this time constant")
    parser.add_argument("--newton-convex-decomposition-threshold", type=float, default=None)
    parser.add_argument("--newton-contact-timeconst", type=float, default=None)
    from robolab.core.utils.app_launcher import add_app_launcher_args
    add_app_launcher_args(parser)
    args = parser.parse_args()
    out = Path(args.output)
    if out.exists() and any(out.iterdir()):
        parser.error("Use a fresh output directory")
    if args.phase_seconds <= 0:
        parser.error("--phase-seconds must be positive")
    if args.ramp_seconds < 0 or args.ramp_seconds > args.phase_seconds:
        parser.error("--ramp-seconds must be between zero and --phase-seconds")
    if min(args.follower_damping, args.passive_damping) < 0 or args.substeps < 1:
        parser.error("Damping must be nonnegative and substeps positive")
    if args.rate_limit and args.ramp_seconds:
        parser.error("Use either --rate-limit or --ramp-seconds")
    if not math.isfinite(args.mimic_timeconst) or args.mimic_timeconst < 0:
        parser.error("--mimic-timeconst must be finite and nonnegative")
    args.enable_cameras = True
    launcher = AppLauncher(args)

    import numpy as np
    import torch
    from isaaclab.sensors import ContactSensorCfg
    from robolab.constants import set_output_dir
    from robolab.core.environments.config import parse_env_cfg
    from robolab.core.environments.factory import get_envs
    from robolab.core.environments.physics import ensure_output_backend, managed_env
    import robolab.core.environments.physics as physics_module
    from robolab.core.utils.isaaclab_compat import as_torch
    from robolab.core.utils.video_utils import VideoWriter
    from robolab.registrations.droid.auto_env_registrations_jointpos import auto_register_droid_envs

    ensure_output_backend(out, "newton_mjwarp")
    original_configure = physics_module.configure_physics
    def configure_probe(cfg, backend="physx"):
        original_configure(cfg, backend)
        cfg.sim.physics.num_substeps = args.substeps
        cfg.sim.physics.solver_cfg.save_to_mjcf = str((out / "model.xml").resolve())
        if args.mujoco_cpu:
            cfg.sim.physics.solver_cfg.use_mujoco_cpu = True
            cfg.sim.physics.use_cuda_graph = False
        # Explicitly override even zero, isolating each diagnostic from the runtime preset.
        from pxr import Sdf, Usd
        overlay = Usd.Stage.CreateInMemory()
        overlay.GetRootLayer().subLayerPaths = [cfg.scene.robot.spawn.usd_path]
        overlay.SetDefaultPrim(overlay.GetPrimAtPath("/panda"))
        for name in ("left_inner_finger_joint", "left_inner_finger_knuckle_joint",
                     "right_outer_knuckle_joint", "right_inner_finger_joint", "right_inner_finger_knuckle_joint"):
            prim = overlay.OverridePrim(f"/panda/Gripper/Robotiq_2F_85/Joints/{name}")
            prim.CreateAttribute("mjc:damping", Sdf.ValueTypeNames.Float).Set(args.passive_damping)
            if not args.mimic_timeconst:
                prim.RemoveAppliedSchema("MjcEqualityJointAPI")
            if args.mimic_timeconst:
                # PhysX mimic conversion creates an equality with default softness.
                # Native equality authoring preserves the relationship and permits
                # setting its solver parameters; Newton skips the duplicate mimic.
                prim.AddAppliedSchema("MjcEqualityJointAPI")
                prim.CreateRelationship("mjc:target").SetTargets([
                    "/panda/Gripper/Robotiq_2F_85/Joints/finger_joint"])
                sign = 1.0 if name in ("right_outer_knuckle_joint", "right_inner_finger_joint") else -1.0
                prim.CreateAttribute("mjc:coef0", Sdf.ValueTypeNames.Double).Set(0.0)
                prim.CreateAttribute("mjc:coef1", Sdf.ValueTypeNames.Double).Set(sign)
                prim.CreateAttribute("mjc:solref", Sdf.ValueTypeNames.DoubleArray).Set([args.mimic_timeconst, 1.0])
                prim.CreateAttribute("mjc:solimp", Sdf.ValueTypeNames.DoubleArray).Set([0.99, 0.99, 0.001, 0.5, 2.0])
        path = (out / "robot.usda").resolve()
        overlay.GetRootLayer().Export(str(path))
        cfg.scene.robot.spawn.usd_path = str(path)
    physics_module.configure_physics = configure_probe
    set_output_dir(str(out))
    auto_register_droid_envs(task=["RubiksCubeThenBananaTask"])
    cfg = parse_env_cfg(get_envs(task="RubiksCubeThenBananaTask")[0], device=args.device, num_envs=1, seed=0)
    cfg.newton_gripper_stabilization = False
    gripper = cfg.scene.robot.actuators["gripper"]
    if args.rate_limit:
        from robolab.robots.newton_gripper import RateLimitedGripperActionCfg
        cfg.actions.finger_joint = RateLimitedGripperActionCfg(
            asset_name="robot", joint_names=["finger_joint"],
            open_command_expr={"finger_joint": 0.0}, close_command_expr={"finger_joint": float(np.pi/4)})
    if args.contact_block:
        import isaaclab.sim as sim_utils
        from isaaclab.assets import RigidObjectCfg
        cfg.scene.probe_block = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/probe_block",
            spawn=sim_utils.CuboidCfg(size=(0.025, 0.05, 0.025),
                rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True, disable_gravity=True),
                collision_props=sim_utils.CollisionPropertiesCfg(),
                mass_props=sim_utils.MassPropertiesCfg(mass=0.1),
                activate_contact_sensors=True,
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.1, 0.6, 0.9))),
            init_state=RigidObjectCfg.InitialStateCfg(pos=(3.0, 0.0, 1.0)))
    if args.ramp_seconds:
        from isaaclab.envs import mdp
        cfg.actions.finger_joint = mdp.JointPositionActionCfg(
            asset_name="robot", joint_names=["finger_joint"], scale=float(np.pi/4), use_default_offset=False)
    if args.gain_mode == "damping":
        gripper.damping = 2.0  # N m s / rad; change damping only
    elif args.gain_mode == "soft":
        gripper.stiffness, gripper.damping = 100.0, 2.0
    elif args.gain_mode == "gentle":
        gripper.stiffness, gripper.damping = 10.0, 2.0
    if args.follower_damping:
        from isaaclab.actuators import ImplicitActuatorCfg
        cfg.scene.robot.actuators["mimic_damping"] = ImplicitActuatorCfg(
            joint_names_expr=["left_inner_finger_joint", "left_inner_finger_knuckle_joint",
                              "right_outer_knuckle_joint", "right_inner_finger_joint", "right_inner_finger_knuckle_joint"],
            stiffness=0.0, damping=args.follower_damping)
    # Probe every gripper link, including contacts not seen by the benchmark's
    # left-finger-only sensor. No object interaction is intended in this pose.
    links = ["base_link"] + [f"{side}_{part}" for side in ("left", "right")
                             for part in ("outer_knuckle", "outer_finger", "inner_finger", "inner_knuckle")]
    for i, link in enumerate(links):
        setattr(cfg.scene, f"probe_contact_{i}", ContactSensorCfg(
            prim_path=f"{{ENV_REGEX_NS}}/robot/Gripper/Robotiq_2F_85/{link}", update_period=0.0))

    samples, controls = [], []
    error = None
    writer = None
    with managed_env(cfg, physics="newton_mjwarp", launcher_args=args) as (env, cfg):
        env.reset()
        robot = env.scene["robot"]
        if args.contact_block:
            from robolab.core.utils.isaaclab_compat import quat_isaaclab_to_wxyz
            body_index = robot.body_names.index("base_link")
            pos = as_torch(robot.data.body_pos_w)[:, body_index].clone()
            quat = as_torch(robot.data.body_quat_w)[:, body_index].clone()
            wxyz = quat_isaaclab_to_wxyz(quat)
            offset = torch.tensor([[0.14, 0.0, 0.0]], device=env.device)
            vector = wxyz[:, 1:]
            uv = torch.cross(vector, offset, dim=-1)
            rotated = offset + 2 * (wxyz[:, :1] * uv + torch.cross(vector, uv, dim=-1))
            pose = torch.cat((pos + rotated, quat), dim=-1)
            env.scene["probe_block"].write_root_pose_to_sim_index(root_pose=pose)
        names = list(robot.joint_names)
        arm_ids, _ = robot.find_joints("panda_joint.*", preserve_order=True)
        finger_ids, _ = robot.find_joints("finger_joint")
        action = torch.zeros((1, env.action_manager.total_action_dim), device=env.device)
        arm_start = as_torch(robot.data.joint_pos)[:, arm_ids].clone()
        action[:, :7] = arm_start
        dt = cfg.sim.dt * cfg.decimation
        n = round(args.phase_seconds / dt)
        if n < 1:
            raise ValueError("Phase must span at least one control step")
        writer = VideoWriter(str(out / "gripper.mp4"), 1 / dt, crf=20, max_width=1280)
        original_update = env.scene.update

        def sampled_update(*a, **kw):
            original_update(*a, **kw)
            q = as_torch(robot.data.joint_pos)[0].detach().cpu().numpy().copy()
            qd = as_torch(robot.data.joint_vel)[0].detach().cpu().numpy().copy()
            forces = [float(as_torch(env.scene[f"probe_contact_{i}"].data.net_forces_w).abs().max())
                      for i in range(len(links))]
            samples.append((q, qd, forces))
            if not np.isfinite(q).all() or not np.isfinite(qd).all():
                raise RuntimeError(f"Nonfinite joint state at physics sample {len(samples)}")

        env.scene.update = sampled_update
        try:
            # Three seconds each: hold open, close and hold, reopen and hold.
            for step in range(3 * n):
                command = float(n <= step < 2 * n)
                if args.ramp_seconds:
                    if n <= step < 2 * n:
                        command = min(1.0, (step-n+1)*dt/args.ramp_seconds)
                    elif step >= 2 * n:
                        command = max(0.0, 1.0-(step-2*n+1)*dt/args.ramp_seconds)
                action[0, -1] = command
                controls.append((step * dt, command))
                env.step(action)
                views = [as_torch(env.scene[name].data.output["rgb"])[0].cpu().numpy()
                         for name in ("over_shoulder_left_camera", "wrist_cam")]
                frame = np.concatenate([cv2.resize(v, (640, 360)) for v in views], axis=1)
                cv2.putText(frame, f"{args.gain_mode} | t={(step+1)*dt:.2f}s | {'CLOSE' if command else 'OPEN'}",
                            (10, 28), cv2.FONT_HERSHEY_SIMPLEX, .65, (255, 255, 255), 2)
                writer.write(frame)
                if step % 15 == 0:
                    print(f"GRIPPER_PROBE step={step} command={command} q={samples[-1][0].tolist()}", flush=True)
        except Exception as exc:
            error = str(exc)
            traceback.print_exc()
        finally:
            env.scene.update = original_update
            writer.release()
            q = np.asarray([s[0] for s in samples])
            qd = np.asarray([s[1] for s in samples])
            forces = np.asarray([s[2] for s in samples])
            np.savez_compressed(out / "trace.npz", q=q, qd=qd, contact_forces=forces,
                                controls=np.asarray(controls), joint_names=np.asarray(names), physics_dt=cfg.sim.dt)
            result = {"gain_mode": args.gain_mode, "ramp_seconds": args.ramp_seconds, "mujoco_cpu": args.mujoco_cpu, "error": error, "physics_samples": len(samples),
                      "follower_damping": args.follower_damping,
                      "passive_damping": args.passive_damping,
                      "rate_limit": args.rate_limit, "contact_block": args.contact_block,
                      "mimic_timeconst": args.mimic_timeconst,
                      "requested_control_steps": 3*n, "attempted_control_steps": len(controls),
                      "phase_seconds": n*dt, "physics_dt": cfg.sim.dt, "substeps": cfg.sim.physics.num_substeps,
                      "joint_names": names, "gripper_link_names": links,
                      "arm_max_deviation_rad": float(np.nanmax(np.abs(q[:, arm_ids]-arm_start.cpu().numpy()))),
                      "max_abs_joint_velocity": float(np.nanmax(np.abs(qd))),
                      "max_gripper_contact_force_component": float(np.nanmax(forces)),
                      "finger_joint_index": finger_ids[0],
                      "configured_stiffness": gripper.stiffness, "configured_damping": gripper.damping}
            (out / "result.json").write_text(json.dumps(result, indent=2) + "\n")
            print("GRIPPER_PROBE_RESULT=" + json.dumps(result), flush=True)
    if error:
        raise RuntimeError(error)
    launcher.app.close()


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(1)
