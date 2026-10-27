"""Measure aggregate three-camera throughput across parallel RoboLab environments.

Recorded-state playback isolates rendering from policy inference and physics
stepping. Production scene, physics allocation and native camera quality remain
unchanged. Restore/kinematics, image readback and auditing are outside capture
latency; restore plus capture is also reported separately. Run each batch size
in a fresh process. Failed allocations are evidence, not permission to reduce
camera resolution or contact storage.
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
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--num-envs', type=int, required=True)
    parser.add_argument('--frames', type=int, default=345)
    parser.add_argument('--repetitions', type=int, default=1)
    parser.add_argument('--warmup', type=int, default=32)
    parser.add_argument('--save-native-samples', action='store_true', help='Save lossless full-resolution samples after timing')
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
                  warmup=args.warmup, passes=[], timings_s={}, passed=False,
                  measurement='synchronized batched capture, GPU outputs; no physics/policy/CPU image copy',
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
            step_count = env.sim.get_physics_step_count()
            def restore(state):
                env.scene.reset_to(state, is_relative=True)
                env.sim.forward()
                torch.cuda.synchronize()
            def capture():
                env.sim.render_context.update_scene_state(env.sim.get_physics_step_count())
                env.sim.render()
                SensorBase.update_batch(list(cams.values()), dt)
                torch.cuda.synchronize()
            native_samples = []
            for pass_index in range(args.repetitions):
                restore(initial)
                for name, pose in fixture['initial']['cameras'].items():
                    positions = torch.tensor(pose['position'], device=env.device).repeat(args.num_envs, 1)
                    orientations = torch.tensor(pose['orientation'], device=env.device).repeat(args.num_envs, 1)
                    cams[name].set_world_poses(positions=positions+as_torch(env.scene.env_origins),
                        orientations=quat_wxyz_to_isaaclab(orientations), convention='ros')
                for state in states[:args.warmup]: restore(state); capture()
                previous = {n: as_torch(c.frame).detach().cpu().numpy().copy() for n, c in cams.items()}
                assert all(v.shape == (args.num_envs,) for v in previous.values())
                rows = []
                stamp(f'timing_{pass_index}_start')
                for index in range(args.frames):
                    state_index = index+args.warmup
                    start = time.perf_counter(); restore(states[state_index]); restore_s = time.perf_counter()-start
                    start = time.perf_counter(); capture(); render_s = time.perf_counter()-start
                    row = dict(index=index, source_indices=[(state_index+o)%len(source) for o in offsets],
                               render_s=render_s, restore_s=restore_s)
                    for name, cam in cams.items():
                        frame = as_torch(cam.frame).detach().cpu().numpy()
                        assert np.array_equal(frame, previous[name]+1), (name, frame, previous[name])
                        previous[name] = frame.copy()
                    if index in (0, args.frames//2, args.frames-1):
                        error = difference(plain(scene_state_from_isaaclab(env.scene.get_state(is_relative=True))), frame_sources[state_index])
                        assert error < 2e-5, error
                        row.update(max_state_error=error, cameras={})
                        for name, cam in cams.items():
                            row['cameras'][name] = dict(position=plain(as_torch(cam.data.pos_w)-as_torch(env.scene.env_origins)),
                                orientation=plain(quat_isaaclab_to_wxyz(cam.data.quat_w_ros)), intrinsics=plain(cam.data.intrinsic_matrices))
                            rgb = as_torch(cam.data.output['rgb'])
                            assert tuple(rgb.shape) == (args.num_envs, cam.cfg.height, cam.cfg.width, 3), rgb.shape
                            if pass_index == 0 and index == args.frames//2:
                                for env_id in range(args.num_envs):
                                    image = rgb[env_id].cpu().numpy()
                                    assert image.dtype == np.uint8 and image.max() > image.min()
                                    if args.save_native_samples:
                                        native_samples.append((f'{name}-env{env_id:03d}.png', image.copy()))
                                    thumb = cv2.resize(image, (320, 180), interpolation=cv2.INTER_AREA)
                                    cv2.imwrite(str(args.output/f'{name}-env{env_id:03d}.png'), cv2.cvtColor(thumb, cv2.COLOR_RGB2BGR))
                    rows.append(row)
                assert env.sim.get_physics_step_count() == step_count
                (args.output/f'frames-{pass_index}.json').write_text(json.dumps(rows, indent=2)+'\n')
                report['passes'].append(dict(pass_index=pass_index,
                    capture=summarize([r['render_s'] for r in rows], args.num_envs),
                    restore_plus_capture=summarize([r['restore_s']+r['render_s'] for r in rows], args.num_envs)))
                stamp(f'timing_{pass_index}_complete')
            if native_samples:
                native_dir = args.output/'native'
                native_dir.mkdir()
                for name, image in native_samples:
                    assert cv2.imwrite(str(native_dir/name), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
                report['native_samples_saved'] = len(native_samples)
            report.update(kit_app_loaded='omni.kit.app' in sys.modules, physics_steps_during_playback=0, passed=True)
            if args.backend == 'ovrtx': assert not report['kit_app_loaded']
            persist_report()
            print('RENDERER_SCALING_PASS', flush=True)
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
