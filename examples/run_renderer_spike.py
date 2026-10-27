"""Experimental single-scene renderer comparison; never a scored policy run.

Install each renderer in a separate fresh venv with scripts/install_newton61.py.
Use identical --fixture JSON for each process. Captures restore frozen physical
states and camera calibration, with no physics stepping during image accumulation.
"""
import time
START = time.perf_counter()
import argparse
import hashlib
import importlib.metadata as metadata
import json
from pathlib import Path
import resource
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--backend', choices=['isaacrtx', 'ovrtx', 'newton-warp'], required=True)
    parser.add_argument('--task', default='BagelsOnPlateTask')
    parser.add_argument('--fixture', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--frames', type=int, default=32)
    from robolab.core.utils.app_launcher import add_app_launcher_args
    add_app_launcher_args(parser)
    args = parser.parse_args()
    if args.frames < 2 or args.output.exists():
        parser.error('Use at least two frames and a fresh output directory')
    args.output.mkdir(parents=True)
    args.headless = True
    args.enable_cameras = True
    versions = {d.metadata['Name']: d.version for d in metadata.distributions()}
    if args.backend != 'isaacrtx':
        assert not any(n.lower().replace('_', '-').startswith(('isaacsim', 'isaac-sim', 'kit-sdk')) for n in versions), 'Kit-less test requires an environment without Isaac Sim/Kit'
    report = dict(backend=args.backend, task=args.task, versions=versions, timings_s={}, captures={}, fixture_sha256=hashlib.sha256(args.fixture.read_bytes()).hexdigest(), script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    def stamp(key):
        report['timings_s'][key] = time.perf_counter()-START
        (args.output/'progress.json').write_text(json.dumps(report, indent=2)+'\n')
        print('SPIKE_PHASE', key, report['timings_s'][key], flush=True)
    stamp('argument_imports')
    import cv2
    # Kit must establish its USD runtime before any RoboLab/pxr imports.
    # OVRTX deliberately never constructs AppLauncher.
    launcher = None
    if args.backend == 'isaacrtx':
        from isaaclab.app import AppLauncher
        launcher = AppLauncher(args)
        stamp('application_ready')
    import numpy as np
    import torch
    from isaaclab.sensors import CameraCfg
    from robolab.constants import set_output_dir
    from robolab.core.environments.config import parse_env_cfg
    from robolab.core.environments.factory import get_envs
    from robolab.core.environments.physics import managed_env
    from robolab.core.sensors.camera_utils import get_cameras
    from robolab.core.utils.isaaclab_compat import as_torch, scene_state_from_isaaclab, scene_state_to_isaaclab, quat_wxyz_to_isaaclab, quat_isaaclab_to_wxyz
    from robolab.registrations.droid.auto_env_registrations_jointpos import auto_register_droid_envs
    set_output_dir(str(args.output/'runtime'))
    auto_register_droid_envs(task=[args.task])
    cfg = parse_env_cfg(get_envs(task=args.task)[0], device=args.device, num_envs=1, seed=0)
    cfg.experimental_kitless = args.backend != 'isaacrtx'
    cameras = []
    for name, camera in vars(cfg.scene).items():
        if not isinstance(camera, CameraCfg):
            continue
        # OVRTX receives camera transforms through the sensor's pose refresh.
        # With the default False, attached cameras can keep their initial view,
        # and pose readbacks are also stale. Enable it for both comparison arms.
        camera.update_latest_camera_pose = True
        if args.backend == 'ovrtx':
            from isaaclab_ov.renderers import OVRTXRendererCfg
            camera.renderer_cfg = OVRTXRendererCfg(async_rendering=False, enable_shadows=True, log_level='warn', log_file_path=str(args.output/'ovrtx.log'), temp_usd_dir=str(args.output/'usd'))
        elif args.backend == 'newton-warp':
            from isaaclab_newton.renderers import NewtonWarpRendererCfg
            camera.renderer_cfg = NewtonWarpRendererCfg(enable_shadows=True)
        else:
            from isaaclab_physx.renderers import IsaacRtxRendererCfg
            camera.renderer_cfg = IsaacRtxRendererCfg()
        cameras.append(name)
    assert len(cameras) == 3, cameras
    report['camera_names'] = cameras
    report['update_latest_camera_pose'] = True
    stamp('configured')
    fixtures = json.loads(args.fixture.read_text())
    def plain(value):
        if isinstance(value, dict): return {k:plain(v) for k,v in value.items()}
        return as_torch(value).detach().cpu().numpy().tolist()
    def overlay(target, source):
        for k,v in source.items():
            if k not in target: raise KeyError(k)
            if isinstance(v,dict): overlay(target[k],v)
            else: target[k] = torch.as_tensor(v, device=args.device, dtype=as_torch(target[k]).dtype)
    with managed_env(cfg, physics='newton_mjwarp', launcher_args=args) as (env,cfg):
        stamp('environment_ready')
        env.reset()
        stamp('reset_ready')
        action = torch.zeros((1,env.action_manager.total_action_dim),device=env.device)
        ids,_ = env.scene['robot'].find_joints('panda_joint.*',preserve_order=True)
        action[:,:len(ids)] = as_torch(env.scene['robot'].data.joint_pos)[:,ids]
        env.step(action)
        torch.cuda.synchronize()
        assert torch.isfinite(as_torch(env.scene['robot'].data.joint_pos)).all()
        stamp('first_newton_action')
        cams = get_cameras(env.scene)
        for label, fixture in fixtures['captures'].items():
            env.reset_eval_state()
            state = scene_state_from_isaaclab(env.scene.get_state(is_relative=True))
            overlay(state, fixture['state'])
            env.reset_to(scene_state_to_isaaclab(state),env_ids=None,is_relative=True)
            env.sim.forward()
            for name, pose in fixture['cameras'].items():
                cams[name].set_world_poses(
                    positions=torch.tensor(pose['position'],device=env.device)+as_torch(env.scene.env_origins),
                    orientations=quat_wxyz_to_isaaclab(torch.tensor(pose['orientation'],device=env.device)),
                    convention='ros')
            before = plain(scene_state_from_isaaclab(env.scene.get_state(is_relative=True)))
            def check_restore(actual, expected, path=''):
                for key, value in expected.items():
                    if isinstance(value, dict): check_restore(actual[key], value, path+'/'+key)
                    else:
                        assert np.allclose(actual[key], value, rtol=0, atol=2e-5), ('restore mismatch',path+'/'+key)
            check_restore(before,fixture['state'])
            folder=args.output/label;folder.mkdir()
            times=[]
            for i in range(args.frames):
                t=time.perf_counter()
                env.sim.render()
                for cam in cams.values():cam.update(0,force_recompute=True)
                torch.cuda.synchronize()
                times.append(time.perf_counter()-t)
                if i in (0,args.frames-1):
                    for name,cam in cams.items():
                        rgb=as_torch(cam.data.output['rgb'])[0].cpu().numpy()
                        assert np.isfinite(rgb).all() and rgb.max()>rgb.min(), name
                        cv2.imwrite(str(folder/f'{name}_{i:03d}.png'),cv2.cvtColor(rgb,cv2.COLOR_RGB2BGR))
            after=plain(scene_state_from_isaaclab(env.scene.get_state(is_relative=True)))
            assert before==after, 'Frozen image accumulation changed physical state'
            camera_data={name:dict(position=plain(as_torch(cam.data.pos_w)-as_torch(env.scene.env_origins)),orientation=plain(quat_isaaclab_to_wxyz(cam.data.quat_w_ros)),intrinsics=plain(cam.data.intrinsic_matrices),frame=plain(cam.frame)) for name,cam in cams.items()}
            entry=dict(state=after,cameras=camera_data,render_times_s=times,median_render_s=float(np.median(times[1:])))
            (folder/'capture.json').write_text(json.dumps(entry,indent=2)+'\n')
            report['captures'][label]=entry
            stamp('capture_'+label)
        report['max_rss_kib']=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        report['kit_app_loaded']='omni.kit.app' in sys.modules
        if args.backend!='isaacrtx':assert not report['kit_app_loaded']
        report['physics_timing']=cfg.physics_timing
        report['passed']=True
        (args.output/'result.json').write_text(json.dumps(report,indent=2)+'\n')
        print('SPIKE_PASS',flush=True)
    if launcher is not None:
        launcher.app.close()


if __name__=='__main__':
    main()
