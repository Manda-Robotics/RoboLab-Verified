# SPDX-License-Identifier: Apache-2.0
"""Recorded integration probe for public physics controls (no policy server)."""
import argparse
import json
import traceback
import os
import sys
from pathlib import Path
import cv2
from isaaclab.app import AppLauncher
from robolab.core.utils.app_launcher import add_app_launcher_args
from robolab.eval.runner import add_common_eval_args

p = argparse.ArgumentParser()
add_common_eval_args(p)
add_app_launcher_args(p)
p.add_argument('--probe-output', type=Path, required=True)
p.add_argument('--probe-steps', type=int, default=30)
a = p.parse_args()
a.enable_cameras = True
a.probe_output.mkdir(parents=True, exist_ok=False)
launcher = AppLauncher(a)


def main():
    import numpy as np
    import torch
    import robolab.constants
    from robolab.constants import set_output_dir
    from robolab.core.environments.physics import managed_env
    from robolab.core.environments.factory import get_envs
    from robolab.core.environments.config import parse_env_cfg
    from robolab.core.physics.friction import read_applied, check_applied
    from robolab.core.sensors.camera_utils import get_cameras
    from robolab.core.utils.isaaclab_compat import as_torch
    from robolab.registrations.droid.auto_env_registrations_jointpos import auto_register_droid_envs
    robolab.constants.FRICTION = a.friction
    set_output_dir(str(a.probe_output))
    auto_register_droid_envs(task=a.task)
    cfg = parse_env_cfg(get_envs(task=a.task)[0], device=a.device, num_envs=a.num_envs, seed=0)
    for name, term in vars(cfg.terminations).items():
        if term is not None and hasattr(term, 'func') and not getattr(term, 'time_out', False):
            setattr(cfg.terminations, name, None)
    report = {'physics': a.physics, 'friction': a.friction, 'num_envs': a.num_envs, 'checks': {}}
    with managed_env(cfg, physics=a.physics, launcher_args=a) as (env, cfg):
        report['timing'] = getattr(cfg, 'physics_timing', {'dt': cfg.sim.dt, 'decimation': cfg.decimation})
        report['contact_settings'] = getattr(cfg, 'newton_contact_settings', None)
        report['applied_startup'] = read_applied(env)
        assert not check_applied(report['applied_startup'])
        report['checks']['startup_readback'] = True
        env.reset()
        joints, _ = env.scene['robot'].find_joints('panda_joint.*', preserve_order=True)
        action = torch.zeros((a.num_envs, env.action_manager.total_action_dim), device=env.device)
        action[:, :len(joints)] = as_torch(env.scene['robot'].data.joint_pos)[:, joints]
        env.recorder_manager.set_hdf5_file('controls.hdf5')
        for env_id in range(a.num_envs):
            env.recorder_manager.set_episode_index(env_id, env_ids=[env_id])
        for step in range(a.probe_steps):
            env.step(action)
            assert torch.isfinite(as_torch(env.scene['robot'].data.joint_pos)).all()
            for obj in env.scene.rigid_objects.values():
                assert torch.isfinite(as_torch(obj.data.root_pos_w)).all()
        report['applied_after_steps'] = read_applied(env)
        assert not check_applied(report['applied_after_steps'])
        report['checks']['step_readback_and_finite_states'] = True
        report['elapsed_sim_s'] = a.probe_steps * env.step_dt
        report['cameras'] = {}
        for name, cam in get_cameras(env.scene).items():
            data = as_torch(cam.data.output['rgb']).cpu().numpy()
            assert np.isfinite(data).all() and data.std() > 1
            report['cameras'][name] = list(data.shape)
            cv2.imwrite(str(a.probe_output/(name+'.png')), cv2.cvtColor(data[0, ..., :3], cv2.COLOR_RGB2BGR))
        report['checks']['cameras'] = bool(report['cameras'])
        env.reset()
        report['applied_after_reset'] = read_applied(env)
        assert not check_applied(report['applied_after_reset'])
        report['checks']['reset_readback'] = True
        if a.physics == 'newton_mjwarp':
            from isaaclab_newton.physics import NewtonManager
            solver = NewtonManager._solver
            report['native_timestep_s'] = solver.mjw_model.opt.timestep.numpy().reshape(-1).tolist()
            assert np.allclose(report['native_timestep_s'], cfg.sim.dt / cfg.sim.physics.num_substeps)
            report['checks']['native_timestep'] = True
            report['geoms'] = solver.mj_model.ngeom
        (a.probe_output/'result.json').write_text(json.dumps(report, indent=2)+'\n')
    print('CONTROLS_PROBE_PASS', a.probe_output, flush=True)


try:
    main()
except Exception:
    (a.probe_output/'error.txt').write_text(traceback.format_exc())
    traceback.print_exc()
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(1)  # Kit shutdown can otherwise replace a failed run's exit code with zero.
else:
    launcher.app.close()
