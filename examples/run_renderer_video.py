"""Compare renderer motion and capture latency using identical recorded states.

This is state playback, not a new physics rollout or policy evaluation. Timed
passes exclude pose restoration, CPU readback, validation and video encoding.
Camera pose publication, rendering and GPU camera output delivery are included.
"""
import time
START = time.perf_counter()
import argparse
import copy
import hashlib
import importlib.metadata as metadata
import json
from pathlib import Path
import resource
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--backend', choices=['isaacrtx', 'ovrtx'], required=True)
    parser.add_argument('--fixture', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--repetitions', type=int, default=3)
    parser.add_argument('--warmup', type=int, default=32)
    parser.add_argument('--limit', type=int, default=0)
    from robolab.core.utils.app_launcher import add_app_launcher_args
    add_app_launcher_args(parser)
    args = parser.parse_args()
    if args.output.exists() or args.repetitions < 1 or args.warmup < 2:
        parser.error('Use a fresh output, positive repetitions and at least two warmup updates')
    args.output.mkdir(parents=True)
    args.headless = args.enable_cameras = True
    versions = {d.metadata['Name']: d.version for d in metadata.distributions()}
    if args.backend == 'ovrtx':
        assert not any(n.lower().replace('_', '-').startswith(('isaacsim', 'isaac-sim', 'kit-sdk')) for n in versions)
    report = dict(backend=args.backend, versions=versions, timings_s={}, passes=[],
                  fixture_sha256=hashlib.sha256(args.fixture.read_bytes()).hexdigest(),
                  script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    def stamp(name):
        report['timings_s'][name] = time.perf_counter()-START
        (args.output/'progress.json').write_text(json.dumps(report, indent=2)+'\n')
        print('VIDEO_PHASE', name, report['timings_s'][name], flush=True)
    import cv2
    launcher = None
    if args.backend == 'isaacrtx':
        from isaaclab.app import AppLauncher
        launcher = AppLauncher(args)
    import numpy as np
    import torch
    from isaaclab.sensors import CameraCfg, SensorBase
    from robolab.constants import set_output_dir
    from robolab.core.environments.config import parse_env_cfg
    from robolab.core.environments.factory import get_envs
    from robolab.core.environments.physics import managed_env
    from robolab.core.sensors.camera_utils import get_cameras
    from robolab.core.utils.isaaclab_compat import as_torch, scene_state_from_isaaclab, scene_state_to_isaaclab, quat_wxyz_to_isaaclab, quat_isaaclab_to_wxyz
    from robolab.registrations.droid.auto_env_registrations_jointpos import auto_register_droid_envs
    set_output_dir(str(args.output/'runtime'))
    auto_register_droid_envs(task=['BagelsOnPlateTask'])
    cfg = parse_env_cfg(get_envs(task='BagelsOnPlateTask')[0], device=args.device, num_envs=1, seed=0)
    cfg.experimental_kitless = args.backend == 'ovrtx'
    for camera in vars(cfg.scene).values():
        if not isinstance(camera, CameraCfg):
            continue
        camera.update_latest_camera_pose = True
        if args.backend == 'ovrtx':
            from isaaclab_ov.renderers import OVRTXRendererCfg
            camera.renderer_cfg = OVRTXRendererCfg(async_rendering=False, enable_shadows=True,
                log_level='warn', log_file_path=str(args.output/'ovrtx.log'))
        else:
            from isaaclab_physx.renderers import IsaacRtxRendererCfg
            camera.renderer_cfg = IsaacRtxRendererCfg()
    fixture = json.loads(args.fixture.read_text())
    dt = fixture['frame_dt_s']
    source_states = fixture['frames'][:args.limit or None]
    report.update(frame_dt_s=dt, frame_count=len(source_states), repetitions=args.repetitions,
                  update_latest_camera_pose=True, capture_api='SensorBase.update_batch',
                  warmup_updates=args.warmup, discard_first_timing_frames=min(15, len(source_states)//2))
    def plain(value):
        if isinstance(value, dict): return {k:plain(v) for k,v in value.items()}
        return as_torch(value).detach().cpu().numpy().tolist()
    def overlay(target, source):
        for key,value in source.items():
            if isinstance(value,dict): overlay(target[key],value)
            else: target[key] = torch.as_tensor(value, device=args.device, dtype=as_torch(target[key]).dtype)
    def max_difference(actual, expected):
        diffs=[]
        for key,value in expected.items():
            if isinstance(value,dict): diffs.append(max_difference(actual[key],value))
            else: diffs.append(float(np.max(np.abs(np.asarray(actual[key])-np.asarray(value)))))
        return max(diffs, default=0.)
    def summarize(values):
        x=np.asarray(values)
        return dict(mean_s=float(x.mean()), median_s=float(np.median(x)), p95_s=float(np.percentile(x,95)),
                    camera_sets_per_s=float(1/x.mean()), camera_images_per_s=float(3/x.mean()))
    with managed_env(cfg, physics='newton_mjwarp', launcher_args=args) as (env,cfg):
        env.reset()
        action = torch.zeros((1,env.action_manager.total_action_dim),device=env.device)
        ids,_=env.scene['robot'].find_joints('panda_joint.*',preserve_order=True)
        action[:,:len(ids)] = as_torch(env.scene['robot'].data.joint_pos)[:,ids]
        env.step(action)
        torch.cuda.synchronize()
        stamp('environment_and_first_action_ready')
        cams=get_cameras(env.scene);assert len(cams)==3
        assert all(cam.supports_batch_update for cam in cams.values())
        template=scene_state_from_isaaclab(env.scene.get_state(is_relative=True))
        def prepare(source):
            state=copy.deepcopy(template);overlay(state,source)
            return scene_state_to_isaaclab(state)
        initial=prepare(fixture['initial']['state'])
        states=[prepare(s) for s in source_states]
        report['camera_resolutions']={n:[c.cfg.width,c.cfg.height] for n,c in cams.items()}
        report['physics_timing']=cfg.physics_timing
        step_count=env.sim.get_physics_step_count()
        def restore(state):
            # Unlike env.reset_to(), this does not reset sensors or denoiser history.
            env.scene.reset_to(state,is_relative=True)
            env.sim.forward()
            torch.cuda.synchronize()
        def capture():
            # Publish changed link transforms before reading attached-camera poses.
            env.sim.render_context.update_scene_state(env.sim.get_physics_step_count())
            env.sim.render()
            SensorBase.update_batch(list(cams.values()),dt)
            torch.cuda.synchronize()
        for pass_index in range(args.repetitions+1):
            recording=pass_index==args.repetitions
            label='video' if recording else f'timing_{pass_index}'
            folder=args.output/label;folder.mkdir()
            restore(initial)
            for name,pose in fixture['initial']['cameras'].items():
                cams[name].set_world_poses(positions=torch.tensor(pose['position'],device=env.device)+as_torch(env.scene.env_origins),
                    orientations=quat_wxyz_to_isaaclab(torch.tensor(pose['orientation'],device=env.device)),convention='ros')
            for _ in range(args.warmup): capture()
            previous={n:int(as_torch(c.frame)[0].item()) for n,c in cams.items()}
            encoders={}; encoder_logs=[]
            if recording:
                for name,cam in cams.items():
                    log=open(folder/(name+'-encode.log'),'wb');encoder_logs.append(log)
                    command=['ffmpeg','-y','-v','error','-f','rawvideo','-pix_fmt','rgb24','-s',f'{cam.cfg.width}x{cam.cfg.height}',
                             '-r',str(1/dt),'-i','pipe:0','-an','-c:v','libx264','-preset','fast','-crf','16','-pix_fmt','yuv420p','-threads','2',
                             '-movflags','+faststart',str(folder/(name+'.mp4'))]
                    encoders[name]=subprocess.Popen(command,stdin=subprocess.PIPE,stdout=subprocess.DEVNULL,stderr=log)
            rows=[]
            try:
                for index,state in enumerate(states):
                    start=time.perf_counter();restore(state);restore_s=time.perf_counter()-start
                    start=time.perf_counter();capture();render_s=time.perf_counter()-start
                    # Audit and CPU readback occur after the synchronized render timer.
                    start=time.perf_counter()
                    actual=plain(scene_state_from_isaaclab(env.scene.get_state(is_relative=True)))
                    difference=max_difference(actual,source_states[index]);assert difference<2e-5,(index,difference)
                    camera_data={}
                    for name,cam in cams.items():
                        frame=int(as_torch(cam.frame)[0].item());assert frame==previous[name]+1,(name,frame,previous[name])
                        previous[name]=frame
                        camera_data[name]=dict(position=plain(as_torch(cam.data.pos_w)-as_torch(env.scene.env_origins)),
                            orientation=plain(quat_isaaclab_to_wxyz(cam.data.quat_w_ros)),intrinsics=plain(cam.data.intrinsic_matrices),frame=frame)
                        if recording:
                            rgb=as_torch(cam.data.output['rgb'])[0].cpu().numpy()
                            assert rgb.dtype==np.uint8 and rgb.max()>rgb.min(),(name,index)
                            encoders[name].stdin.write(rgb.tobytes())
                            if index%60==0 or index==len(states)-1:
                                cv2.imwrite(str(folder/f'{name}_{index:04d}.png'),cv2.cvtColor(rgb,cv2.COLOR_RGB2BGR))
                    rows.append(dict(index=index,restore_s=restore_s,render_s=render_s,audit_and_video_submission_s=time.perf_counter()-start,
                                     max_state_error=difference,cameras=camera_data))
                    if index%60==0:print('VIDEO_FRAME',label,index,'render_s',render_s,flush=True)
                assert env.sim.get_physics_step_count()==step_count,'Playback advanced physics'
            finally:
                for encoder in encoders.values(): encoder.stdin.close()
                codes=[encoder.wait(timeout=120) for encoder in encoders.values()]
                for log in encoder_logs: log.close()
                assert all(code==0 for code in codes),codes
            (folder/'frames.json').write_text(json.dumps(rows,indent=2)+'\n')
            values=[row['render_s'] for row in rows]
            entry=dict(label=label,recording=recording,frames=len(rows),all_frames=summarize(values),
                       steady_state=summarize(values[report['discard_first_timing_frames']:]),
                       max_state_error=max(row['max_state_error'] for row in rows))
            report['passes'].append(entry);stamp(label+'_complete')
        report.update(max_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                      kit_app_loaded='omni.kit.app' in sys.modules,physics_steps_during_playback=env.sim.get_physics_step_count()-step_count,passed=True)
        if args.backend=='ovrtx':assert not report['kit_app_loaded']
        (args.output/'result.json').write_text(json.dumps(report,indent=2)+'\n')
        print('RENDERER_VIDEO_PASS',flush=True)
    if launcher is not None: launcher.app.close()


if __name__=='__main__':
    main()
