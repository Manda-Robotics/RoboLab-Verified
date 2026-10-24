"""DROID Newton/RTX smoke test without a policy server. Run on a CUDA host."""

import argparse
import faulthandler
import json
import math
import os
from pathlib import Path
import sys
import traceback

import cv2  # noqa: F401 -- load OpenCV before Kit
from isaaclab.app import AppLauncher


def main():
    # Timed all-thread stack walks can crash CPython 3.12's frame walker on
    # the GA stack (PyUnstable_InterpreterFrame_GetLine). Keep fatal reporting;
    # enforce wall-clock limits externally instead of starting a watchdog thread.
    faulthandler.enable()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--physics", choices=["physx", "newton_mjwarp"], default="newton_mjwarp")
    parser.add_argument("--task", default="RubiksCubeThenBananaTask")
    parser.add_argument("--steps", type=int, default=60)
    parser.add_argument("--num-envs", type=int, default=2)
    parser.add_argument("--output", default="output/newton_smoke")
    parser.add_argument("--newton-contacts", action="store_true", help="Diagnostic: use Newton collision detection instead of MuJoCo contacts")
    parser.add_argument("--decomposition-threshold", type=float, default=None,
                        help="Diagnostic: normalized CoACD tolerance for all convex-decomposed meshes")
    parser.add_argument("--contact-timeconst", type=float, default=None,
                        help="Diagnostic: scene-wide default contact response time (seconds)")
    parser.add_argument("--drop-object", help="Diagnostic: drop this object above --into-object")
    parser.add_argument("--into-object", help="Diagnostic: destination object for a vertical drop")
    parser.add_argument("--max-object-displacement", type=float, default=1.0,
                        help="Fail if any rigid object moves this far from its reset position (meters)")
    parser.add_argument("--max-initial-object-radius", type=float, default=10.0,
                        help="Reject tabletop smoke scenes with reset objects farther than this from their environment origin (meters)")
    from robolab.core.utils.app_launcher import add_app_launcher_args
    add_app_launcher_args(parser)
    args = parser.parse_args()
    if (args.steps < 1 or args.num_envs < 1 or not math.isfinite(args.max_object_displacement)
            or args.max_object_displacement <= 0
            or not math.isfinite(args.max_initial_object_radius) or args.max_initial_object_radius <= 0):
        parser.error("--steps, --num-envs and --max-object-displacement must be positive")
    if bool(args.drop_object) != bool(args.into_object):
        parser.error("Use --drop-object and --into-object together")
    if args.contact_timeconst is not None and (not math.isfinite(args.contact_timeconst) or args.contact_timeconst <= 0):
        parser.error("--contact-timeconst must be finite and positive")
    if Path(args.output).exists() and any(Path(args.output).iterdir()):
        parser.error("Use a fresh output directory; a previous pass marker must not survive a failed rerun")
    args.enable_cameras = True
    print("ROBOLAB_SMOKE_LAUNCH_BEGIN", flush=True)
    launcher = AppLauncher(args)
    print("ROBOLAB_SMOKE_LAUNCH_DONE", flush=True)

    import torch
    from robolab.constants import set_output_dir
    from robolab.core.environments.factory import get_envs
    from robolab.core.environments.physics import ensure_output_backend, managed_env
    import robolab.core.environments.physics as physics_module
    from robolab.core.sensors.camera_utils import get_cameras
    from robolab.core.sensors.contact_sensor_utils import get_contact_sensors
    from robolab.core.utils.isaaclab_compat import as_torch, write_root_pose, write_root_velocity
    from robolab.core.world.world_state import get_world
    from robolab.registrations.droid.auto_env_registrations_jointpos import auto_register_droid_envs

    if args.physics == "newton_mjwarp" or args.newton_contacts or args.decomposition_threshold is not None or args.contact_timeconst is not None:
        if args.physics != "newton_mjwarp":
            parser.error("Newton contact/mesh diagnostics require --physics newton_mjwarp")
        original_configure = physics_module.configure_physics
        def configure_contact_diagnostic(cfg, backend):
            original_configure(cfg, backend)
            if args.newton_contacts:
                cfg.sim.physics.solver_cfg.use_mujoco_contacts = False
            if args.drop_object:
                # A successful drop is a physics diagnostic, not a policy
                # episode. Avoid the evaluator's post-success zero actions.
                cfg.terminations.success = None
            cfg.newton_convex_decomposition_threshold = args.decomposition_threshold
            cfg.sim.physics.solver_cfg.save_to_mjcf = str(Path(args.output, "model.xml").resolve())
            if args.contact_timeconst is not None:
                if args.contact_timeconst < 2 * cfg.sim.dt / cfg.sim.physics.num_substeps:
                    raise ValueError("Contact time constant must be at least twice the solver timestep")
                cfg.newton_contact_timeconst = args.contact_timeconst
        physics_module.configure_physics = configure_contact_diagnostic

    ensure_output_backend(args.output, args.physics)
    set_output_dir(args.output)
    auto_register_droid_envs(task=[args.task])
    task = get_envs(task=args.task)[0]
    with managed_env(task, physics=args.physics, launcher_args=args,
                     device=args.device, num_envs=args.num_envs, seed=0) as (env, cfg):
        if args.contact_timeconst is not None:
            import xml.etree.ElementTree as ET
            model = ET.parse(Path(args.output, "model.xml")).getroot()
            imported_times = [float(g.get("solref", "0.02 1").split()[0]) for g in model.iter("geom")
                              if "/scene/" in g.get("name", "")]
            assert any(math.isclose(t, args.contact_timeconst, rel_tol=1e-4) for t in imported_times), (
                "Requested contact response did not reach the compiled scene model")
            print("ROBOLAB_COMPILED_CONTACT_TIMECONSTS=" + json.dumps(sorted(set(imported_times))), flush=True)
        print("ROBOLAB_SMOKE_RESET_BEGIN", flush=True)
        env.reset()
        print("ROBOLAB_SMOKE_RESET_DONE", flush=True)
        if args.drop_object:
            body = env.scene[args.drop_object]
            destination = env.scene[args.into_object]
            pos = as_torch(destination.data.root_pos_w).clone()
            pos[:, 2] += 0.12
            pose = torch.cat((pos, as_torch(body.data.root_quat_w)), dim=-1)
            write_root_pose(body, pose)
            write_root_velocity(body, torch.zeros((env.num_envs, 6), device=env.device))
            # Dynamic free-joint writes need FK before MJWarp's first step;
            # otherwise its body-pose input can still contain the reset pose.
            env.sim.forward()
        contact_samples = []
        def geometric_contacts(step):
            # Evaluate native MuJoCo geometry at the actual GPU state. This is
            # a geometric penetration audit, NOT a CPU dynamics rollout and
            # NOT a replacement for the GPU contact-force sensor checks.
            if args.physics != "newton_mjwarp" or args.newton_contacts:
                return
            import mujoco
            from isaaclab_newton.physics import NewtonManager
            solver = NewtonManager._solver
            model = solver.mj_model
            data = mujoco.MjData(model)
            data.qpos[:] = solver.mjw_data.qpos.numpy()[0]
            data.qvel[:] = solver.mjw_data.qvel.numpy()[0]
            mujoco.mj_forward(model, data)
            pairs = {}
            for contact in data.contact[:data.ncon]:
                names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, int(i)) or str(i) for i in contact.geom]
                def group(name):
                    for token in ("/scene/", "/Robotiq_2F_85/", "/robot/"):
                        if token in name:
                            return name.split(token, 1)[1].split("/")[0]
                    return name
                key = " / ".join(sorted(group(name) for name in names))
                pairs[key] = min(pairs.get(key, 0.0), float(contact.dist))
            row = {"step": step, "environment": 0, "ncon": int(data.ncon),
                   "minimum_distance_by_pair_m": pairs}
            contact_samples.append(row)
            print("ROBOLAB_GEOMETRIC_CONTACTS=" + json.dumps(row), flush=True)
        def report_state(step):
            state = {name: as_torch(body.data.root_pos_w).tolist()
                     for name, body in env.scene.rigid_objects.items()}
            forces = {name: as_torch(sensor.data.force_matrix_w).abs().max().item()
                      for name, sensor in get_contact_sensors(env.scene).items()}
            print("ROBOLAB_SMOKE_STATE=" + json.dumps({"step": step, "positions": state,
                                                      "forces": forces}), flush=True)
        report_state(0)
        initial_object_positions = {name: as_torch(body.data.root_pos_w).clone()
                                    for name, body in env.scene.rigid_objects.items()}
        for name, position in initial_object_positions.items():
            distance = torch.linalg.vector_norm(position - as_torch(env.scene.env_origins), dim=-1).max().item()
            assert math.isfinite(distance) and distance < args.max_initial_object_radius, (
                f"Object {name} starts {distance:.3f} m from the environment origin; "
                "invalid tabletop fixture, inspect the authored scene/reset before testing physics")
        # DROID uses absolute joint targets: zeros would drive the arm into a
        # different (partly out-of-limit) pose, not hold it stationary.
        action = torch.zeros((env.num_envs, env.action_manager.total_action_dim), device=env.device)
        joint_ids, _ = env.scene["robot"].find_joints("panda_joint.*", preserve_order=True)
        action[:, :len(joint_ids)] = as_torch(env.scene["robot"].data.joint_pos)[:, joint_ids]
        first_termination_step = None
        for step in range(args.steps):
            _, reward, _, _, _ = env.step(action)
            assert torch.isfinite(as_torch(env.scene["robot"].data.joint_pos)).all(), f"Nonfinite joints at step {step + 1}"
            assert torch.isfinite(as_torch(env.scene["robot"].data.joint_vel)).all(), f"Nonfinite joint velocities at step {step + 1}"
            assert torch.isfinite(as_torch(reward)).all()
            for name, body in env.scene.rigid_objects.items():
                position = as_torch(body.data.root_pos_w)
                assert torch.isfinite(position).all(), f"Nonfinite object {name} at step {step + 1}"
                displacement = torch.linalg.vector_norm(position - initial_object_positions[name], dim=-1).max().item()
                assert displacement <= args.max_object_displacement, (
                    f"Object {name} moved {displacement:.3f} m at stationary-control step {step + 1}; "
                    "possible missing support collision")
            if args.drop_object and step == 0:
                position = as_torch(env.scene[args.drop_object].data.root_pos_w)
                horizontal_shift = torch.linalg.vector_norm(
                    position[:, :2] - initial_object_positions[args.drop_object][:, :2], dim=-1).max().item()
                assert horizontal_shift < 0.02, "Drop pose did not persist into physics; diagnostic is invalid"
            if env.all_terminated and first_termination_step is None:
                first_termination_step = step + 1
            if (step + 1) % 10 == 0:
                print(f"ROBOLAB_SMOKE_STEP={step + 1}", flush=True)
                report_state(step + 1)
                geometric_contacts(step + 1)
        cameras = get_cameras(env.scene)
        assert len(cameras) == 3, list(cameras)
        for name, camera in cameras.items():
            rgb = as_torch(camera.data.output["rgb"])
            assert rgb.shape[0] == args.num_envs and rgb.shape[-1] == 3
            assert torch.isfinite(rgb).all() and rgb.max() > rgb.min()
            cv2.imwrite(os.path.join(args.output, name + ".png"),
                        cv2.cvtColor(rgb[0].cpu().numpy(), cv2.COLOR_RGB2BGR))
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
        if args.drop_object:
            assert world.in_contact(args.drop_object, args.into_object, env_id=None).all(), (
                "Dropped object did not retain destination contact; inspect geometry and the diagnostic")
        result = {"physics": args.physics, "task": task, "steps": args.steps,
                  "num_envs": args.num_envs, "contacts": contacts, "cameras": list(cameras),
                  "newton_contacts": args.newton_contacts,
                  "decomposition_threshold": args.decomposition_threshold,
                  "contact_timeconst": args.contact_timeconst,
                  "drop_object": args.drop_object, "into_object": args.into_object,
                  "geometric_contacts": contact_samples,
                  "max_object_displacement_m": args.max_object_displacement,
                  "max_initial_object_radius_m": args.max_initial_object_radius,
                  "stationary_first_termination_step": first_termination_step}
        Path(args.output, "smoke_result.json").write_text(json.dumps(result, indent=2) + "\n")
        print("ROBOLAB_SMOKE_PASS=" + json.dumps(result), flush=True)
    # Kit close can exit(0); only close after every assertion and export passed.
    launcher.app.close()


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        # Kit's shutdown hooks can override SystemExit with exit(0).
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(1)
