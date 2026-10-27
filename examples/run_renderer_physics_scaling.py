"""Measure the production Newton stepping/rendering/observation path across envs.

Replay recorded actions from a common initial state without teleporting between
steps. Preserve physical coefficients, default observation/recorder terms and
terminations. Fail on premature frozen environments instead of reporting idle
rollouts as useful throughput. Policy inference and video/HDF5 final export are
excluded; per-step RGB readback is reported separately. Results are unscored.
"""
import argparse
import copy
import hashlib
import importlib.metadata as metadata
import json
from pathlib import Path
import resource
import subprocess
import sys
import threading
import time

START = time.perf_counter()


def summarize(seconds, num_envs, camera_count=3):
    import numpy as np
    x = np.asarray(seconds, dtype=float)
    assert len(x) and np.isfinite(x).all() and (x > 0).all()
    mean = float(x.mean())
    return dict(samples=len(x), mean_s=mean, median_s=float(np.median(x)),
                p95_s=float(np.percentile(x, 95)), batches_per_s=1 / mean,
                environment_camera_sets_per_s=num_envs / mean,
                camera_images_per_s=camera_count * num_envs / mean)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--backend', choices=['isaacrtx', 'ovrtx'], required=True)
    parser.add_argument('--fixture', type=Path, required=True)
    parser.add_argument('--actions', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--num-envs', type=int, required=True)
    parser.add_argument('--frames', type=int, default=345)
    parser.add_argument('--repetitions', type=int, default=1)
    parser.add_argument('--warmup', type=int, default=32)
    parser.add_argument('--save-native-samples', action='store_true', help='Save lossless full-resolution samples after timing')
    parser.add_argument('--diagnostic-finite-every-step', action='store_true', help='Extra untimed state checks; exclude this diagnostic from throughput comparisons')
    parser.add_argument('--startup-trace-seconds', type=int, default=0, help='Optional startup-only stack diagnostics')
    from robolab.core.utils.app_launcher import add_app_launcher_args
    add_app_launcher_args(parser)
    args = parser.parse_args()
    if args.output.exists() or min(args.num_envs, args.frames, args.repetitions, args.warmup) < 1 or args.startup_trace_seconds < 0:
        parser.error('Use a fresh output and positive counts')
    args.output.mkdir(parents=True)
    args.headless = args.enable_cameras = True
    versions = {d.metadata['Name']: d.version for d in metadata.distributions()}
    if args.backend == 'ovrtx':
        assert not any(n.lower().replace('_', '-').startswith(('isaacsim', 'isaac-sim', 'kit-sdk')) for n in versions)
    report = dict(backend=args.backend, num_envs=args.num_envs, versions=versions,
                  fixture_sha256=hashlib.sha256(args.fixture.read_bytes()).hexdigest(),
                  script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  frames_per_pass=args.frames, repetitions=args.repetitions, save_native_samples=args.save_native_samples,
                  diagnostic_state_checks_every_step=args.diagnostic_finite_every_step, measurement_qualified=not args.diagnostic_finite_every_step,
                  warmup=args.warmup, passes=[], timings_s={}, passed=False,
                  measurement='production env.step with Newton physics, rendering, observations and default recorder terms; separate native RGB CPU readback; no policy inference',
                  memory_sampling='whole-device nvidia-smi at 1 Hz, including Vulkan and runtime; sampled peak, not exact peak')
    memory = []
    stop_monitor = threading.Event()
    def monitor():
        while not stop_monitor.is_set():
            try:
                r = subprocess.run(['nvidia-smi', '--query-gpu=memory.used,memory.total,utilization.gpu',
                                    '--format=csv,noheader,nounits'], capture_output=True, text=True, timeout=5, check=True)
                used, total, utilization = [float(v.strip()) for v in r.stdout.strip().splitlines()[0].split(',')]
                memory.append(dict(elapsed_s=time.perf_counter()-START, used_mib=used,
                                   total_mib=total, utilization_percent=utilization))
            except Exception as error:
                memory.append(dict(elapsed_s=time.perf_counter()-START, error=str(error)))
            stop_monitor.wait(1)
    thread = threading.Thread(target=monitor, daemon=True)
    thread.start()
    def stamp(name):
        report['phase'] = name
        report['timings_s'][name] = time.perf_counter()-START
        (args.output/'progress.json').write_text(json.dumps(report, indent=2)+'\n')
        print('SCALING_PHASE', name, flush=True)
    def persist_report():
        # Kit may terminate the process in app.close(), bypassing Python finally.
        # Save successful measurements before handing control to runtime shutdown.
        stop_monitor.set(); thread.join(timeout=6)
        report.update(max_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                      elapsed_s=time.perf_counter()-START)
        used = [m['used_mib'] for m in memory if 'used_mib' in m]
        report['sampled_peak_gpu_mib'] = max(used, default=None)
        (args.output/'gpu-memory.json').write_text(json.dumps(memory, indent=2)+'\n')
        (args.output/'result.json').write_text(json.dumps(report, indent=2)+'\n')
    launcher = None
    startup_trace = None
    if args.startup_trace_seconds:
        import faulthandler
        startup_trace = (args.output/'startup-stacks.log').open('w')
        faulthandler.dump_traceback_later(args.startup_trace_seconds, repeat=True, file=startup_trace)
    try:
        stamp('imports')
        if args.backend == 'isaacrtx':
            from isaaclab.app import AppLauncher
            launcher = AppLauncher(args)
        import cv2
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
        cfg = parse_env_cfg(get_envs(task='BagelsOnPlateTask')[0], device=args.device, num_envs=args.num_envs, seed=0)
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
        action_fixture = json.loads(args.actions.read_text())
        assert action_fixture['source_sha256'] == fixture['source_sha256']
        assert action_fixture['frame_dt_s'] == fixture['frame_dt_s']
        action_source = np.asarray(action_fixture['actions'], dtype=np.float32)
        assert len(action_source) >= args.frames and args.repetitions == 1
        report['actions_sha256'] = hashlib.sha256(args.actions.read_bytes()).hexdigest()
        dt = fixture['frame_dt_s']
        source = fixture['frames']
        offsets = [i*len(source)//args.num_envs for i in range(args.num_envs)]
        report.update(phase_offsets=offsets, fixture_frames=len(source), frame_dt_s=dt,
                      update_latest_camera_pose=True, capture_api='SensorBase.update_batch')
        def plain(value):
            if isinstance(value, dict): return {k: plain(v) for k, v in value.items()}
            return as_torch(value).detach().cpu().numpy().tolist()
        def batch_sources(sources):
            def collect(values):
                if isinstance(values[0], dict):
                    return {k: collect([v[k] for v in values]) for k in values[0]}
                return np.concatenate([np.asarray(v) for v in values], axis=0).tolist()
            return collect(sources)
        def overlay(target, data):
            for key, value in data.items():
                if isinstance(value, dict): overlay(target[key], value)
                else: target[key] = torch.as_tensor(value, device=args.device, dtype=as_torch(target[key]).dtype)
        def difference(actual, expected):
            values = []
            for key, value in expected.items():
                if isinstance(value, dict): values.append(difference(actual[key], value))
                else:
                    a, b = np.asarray(actual[key]), np.asarray(value)
                    assert a.shape == b.shape and np.isfinite(a).all()
                    values.append(float(np.max(np.abs(a-b))))
            return max(values, default=0.)
        stamp('constructing_environment')
        with managed_env(cfg, physics='newton_mjwarp', launcher_args=args) as (env, cfg):
            env.reset()
            stamp('environment_reset')
            action = torch.zeros((args.num_envs, env.action_manager.total_action_dim), device=env.device)
            ids, _ = env.scene['robot'].find_joints('panda_joint.*', preserve_order=True)
            action[:, :len(ids)] = as_torch(env.scene['robot'].data.joint_pos)[:, ids]
            env.step(action)
            torch.cuda.synchronize()
            stamp('first_action_complete')
            if startup_trace is not None:
                faulthandler.cancel_dump_traceback_later()
                startup_trace.close()
                startup_trace = None
            cams = get_cameras(env.scene)
            assert len(cams) == 3 and all(c.supports_batch_update for c in cams.values())
            template = scene_state_from_isaaclab(env.scene.get_state(is_relative=True))
            def prepare(data):
                state = copy.deepcopy(template)
                overlay(state, data)
                return scene_state_to_isaaclab(state)
            initial = prepare(batch_sources([fixture['initial']['state']]*args.num_envs))
            frame_sources = [batch_sources([source[(index+offset)%len(source)] for offset in offsets])
                             for index in range(args.frames+args.warmup)]
            states = [prepare(s) for s in frame_sources]
            report['camera_resolutions'] = {n: [c.cfg.width, c.cfg.height] for n, c in cams.items()}
            report['renderer_cfg_repr'] = {n: repr(c.cfg.renderer_cfg) for n, c in cams.items()}
            report['camera_data_types'] = {n: list(c.cfg.data_types) for n, c in cams.items()}
            if args.backend == 'isaacrtx':
                import carb
                settings = carb.settings.get_settings()
                report['isaacrtx_readback'] = {key: settings.get(key) for key in
                    ['/rtx/rendermode', '/rtx/post/aa/op', '/rtx/post/dlss/execMode', '/rtx/post/dlss/enabled', '/rtx/rtpt/cached/enabled', '/rtx/rtpt/lightcache/cached/enabled', '/isaaclab/render/deterministic', '/rtx/pathtracing/spp', '/rtx/pathtracing/totalSpp']}
            report['physics_timing'] = cfg.physics_timing
            report['contact_settings'] = cfg.newton_contact_settings
            report['env_spacing'] = cfg.scene.env_spacing
            import h5py
            source_actions = torch.as_tensor(action_source, device=env.device)
            assert source_actions.shape[1] == env.action_manager.total_action_dim
            report.update(phase_offsets=[0]*args.num_envs, capture_api='RobolabEnv.step, production lazy camera observations',
                          lazy_sensor_update=cfg.scene.lazy_sensor_update, decimation=cfg.decimation,
                          physics_handles_decimation=env._physics_handles_decimation,
                          recorder_terms=list(env.recorder_manager.active_terms),
                          termination_policy='unchanged; fail if an environment freezes during the timed horizon',
                          timed_steps=args.frames, warmup_physics_steps=args.warmup)
            def reset_rollout():
                env.reset_eval_state()
                env.reset()
                env.scene.reset_to(initial, is_relative=True)
                env.sim.forward()
                for name, pose in fixture['initial']['cameras'].items():
                    positions = torch.tensor(pose['position'], device=env.device).repeat(args.num_envs, 1)
                    orientations = torch.tensor(pose['orientation'], device=env.device).repeat(args.num_envs, 1)
                    cams[name].set_world_poses(positions=positions+as_torch(env.scene.env_origins),
                        orientations=quat_wxyz_to_isaaclab(orientations), convention='ros')
                env.sim.render_context.update_scene_state(env.sim.get_physics_step_count())
                env.sim.render()
                SensorBase.update_batch(list(cams.values()), dt)
                torch.cuda.synchronize()
                actual = plain(scene_state_from_isaaclab(env.scene.get_state(is_relative=True)))
                error = difference(actual, batch_sources([fixture['initial']['state']]*args.num_envs))
                assert error < 2e-5, error
                assert not env._frozen_envs.any() and not env._has_stepped
                return error
            reset_rollout()
            stamp('physics_warmup_start')
            for index in range(args.warmup):
                env.step(source_actions[index].reshape(1,-1).repeat(args.num_envs,1))
                torch.cuda.synchronize()
                assert not env._frozen_envs.any(), 'warmup environment terminated'
            report['initial_state_error'] = reset_rollout()
            env.recorder_manager.set_hdf5_file(str(args.output/'rollout.hdf5'))
            env.recorder_manager.clear()
            ids = list(range(args.num_envs))
            env.recorder_manager.reset(env_ids=ids)
            env.recorder_manager.record_post_reset(env_ids=ids)
            for env_id in ids:
                env.recorder_manager.set_episode_index(env_id, env_ids=[env_id])
            previous = {n: as_torch(c.frame).detach().cpu().numpy().copy() for n,c in cams.items()}
            start_count = env.sim.get_physics_step_count()
            start_control_count = int(env.common_step_counter)
            rows = []
            video_frames = {name: [] for name in cams}
            snapshots = []
            def export_videos():
                for name,images in video_frames.items():
                    # Kit's bundled OpenCV may lack video encoders. Use the system
                    # encoder after all timed steps; physics and capture are unchanged.
                    subprocess.run(['ffmpeg', '-v', 'error', '-nostdin', '-n',
                        '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-s', '640x360',
                        '-r', str(1/dt/3), '-i', 'pipe:0', '-an', '-c:v', 'libx264',
                        '-preset', 'fast', '-crf', '18', '-pix_fmt', 'yuv420p',
                        str(args.output/f'{name}-env000.mp4')],
                        input=np.ascontiguousarray(images).tobytes(), check=True)
            stamp('physics_timing_start')
            for index in range(args.frames):
                action = source_actions[index].reshape(1,-1).repeat(args.num_envs,1)
                assert not env._frozen_envs.any(), 'timed environment frozen'
                start = time.perf_counter()
                obs, reward, terminated, truncated, info = env.step(action)
                torch.cuda.synchronize()
                step_s = time.perf_counter()-start
                start = time.perf_counter()
                rgb_cpu = {name: as_torch(cam.data.output['rgb']).detach().cpu().numpy() for name,cam in cams.items()}
                torch.cuda.synchronize()
                readback_s = time.perf_counter()-start
                assert not env._frozen_envs.any(), 'timed environment terminated; preserve partial attempt'
                np.testing.assert_array_equal(as_torch(env.action_manager.action).cpu().numpy(), action.cpu().numpy())
                deltas = {}
                for name,cam in cams.items():
                    frame = as_torch(cam.frame).cpu().numpy().copy()
                    delta = frame-previous[name]
                    assert np.all(delta >= 1), (name, index, delta)
                    assert np.all(delta == delta[0]), (name,index,delta)
                    deltas[name] = int(delta[0]); previous[name] = frame
                    image = rgb_cpu[name]
                    assert image.shape == (args.num_envs,cam.cfg.height,cam.cfg.width,3) and image.dtype == np.uint8
                    if index % 3 == 0:
                        video_frames[name].append(cv2.resize(image[0], (640,360), interpolation=cv2.INTER_AREA))
                    if index in (0,args.frames//2,args.frames-1):
                        for env_id in sorted(set([0,args.num_envs-1])):
                            assert image[env_id].max() > image[env_id].min()
                            cv2.imwrite(str(args.output/f'{name}-step{index:03d}-env{env_id:03d}.png'),cv2.cvtColor(image[env_id],cv2.COLOR_RGB2BGR))
                state_error = None
                if args.diagnostic_finite_every_step or index in (0,args.frames//2,args.frames-1):
                    state = plain(scene_state_from_isaaclab(env.scene.get_state(is_relative=True)))
                    issues=[]
                    def find_nonfinite(value, path=''):
                        if isinstance(value,dict):
                            for key,child in value.items():find_nonfinite(child,path+'/'+key)
                        else:
                            array=np.asarray(value)
                            if not np.isfinite(array).all():
                                issues.append(dict(path=path,shape=list(array.shape),indices=np.argwhere(~np.isfinite(array)).tolist()))
                    find_nonfinite(state)
                    if index in (0,args.frames//2,args.frames-1) or issues:
                        snapshots.append(dict(index=index,state=state))
                        (args.output/'physical-snapshots.json').write_text(json.dumps(snapshots)+'\n')
                    if issues:
                        report['nonfinite_state']=dict(index=index,control_step=index+1,simulated_s=(index+1)*dt,issues=issues)
                        (args.output/'failure-state.json').write_text(json.dumps(dict(report=report['nonfinite_state'],state=state),indent=2)+'\n')
                        failure_row=dict(index=index,step_s=step_s,rgb_cpu_readback_s=readback_s,combined_s=step_s+readback_s,camera_frame_deltas=deltas,active_envs=args.num_envs,physics_step_count=env.sim.get_physics_step_count())
                        (args.output/'physics-frames-partial.json').write_text(json.dumps(rows+[failure_row],indent=2)+'\n')
                        try:
                            env.recorder_manager.export_episodes(env_ids=ids)
                            report['partial_hdf_exported']=True
                            export_videos()
                        except Exception as export_error:
                            report['partial_export_error']=repr(export_error)
                        raise FloatingPointError('Non-finite physics state; see failure-state.json')
                    # Same finite-state gate; failing states are now persisted first.
                    difference(state,state)
                rows.append(dict(index=index,step_s=step_s,rgb_cpu_readback_s=readback_s,
                                 combined_s=step_s+readback_s,camera_frame_deltas=deltas,active_envs=args.num_envs,
                                 physics_step_count=env.sim.get_physics_step_count()))
                if (index+1) % 30 == 0:
                    (args.output/'physics-progress.json').write_text(json.dumps(dict(completed=index+1,total=args.frames,
                        mean_step_s=float(np.mean([v['step_s'] for v in rows]))))+'\n')
                    print('PHYSICS_PROGRESS',index+1,args.frames,flush=True)
            stamp('physics_timing_complete')
            (args.output/'physics-frames.json').write_text(json.dumps(rows,indent=2)+'\n')
            (args.output/'physical-snapshots.json').write_text(json.dumps(snapshots)+'\n')
            calls_per_step = 1 if env._physics_handles_decimation else cfg.decimation
            expected_calls = args.frames*calls_per_step
            delta = env.sim.get_physics_step_count()-start_count
            assert delta == expected_calls, (delta,expected_calls)
            assert int(env.common_step_counter)-start_control_count == args.frames
            export_started = time.perf_counter()
            env.recorder_manager.export_episodes(env_ids=ids)
            report['recording_export_s'] = time.perf_counter()-export_started
            with h5py.File(args.output/'rollout.hdf5','r') as h:
                demos=list(h['data']); assert len(demos)==args.num_envs,(demos,args.num_envs)
                for name in demos:
                    group=h['data'][name]
                    np.testing.assert_array_equal(group['actions'][:], action_source[:args.frames])
                    assert int(group.attrs['num_samples']) == args.frames
                report['hdf5_action_counts_verified']={name:len(h['data'][name]['actions']) for name in demos}
            export_videos()
            report.update(kit_app_loaded='omni.kit.app' in sys.modules, passed=True,
                physics_steps_during_playback=delta, physics_calls_per_control_step=calls_per_step,
                control_steps=args.frames, simulated_seconds_per_environment=args.frames*dt,
                step_and_gpu_observations=summarize([v['step_s'] for v in rows],args.num_envs),
                step_and_cpu_rgb=summarize([v['combined_s'] for v in rows],args.num_envs),
                rgb_cpu_readback=summarize([v['rgb_cpu_readback_s'] for v in rows],args.num_envs),
                success_scored=False)
            if args.backend == 'ovrtx': assert not report['kit_app_loaded']
            persist_report()
            print('PHYSICS_RENDERER_SCALING_PASS', flush=True)
        if launcher is not None: launcher.app.close()
    except BaseException as error:
        report.update(passed=False, error_type=type(error).__name__, error=str(error))
        raise
    finally:
        if startup_trace is not None:
            faulthandler.cancel_dump_traceback_later()
            startup_trace.close()
        persist_report()


if __name__ == '__main__':
    main()
