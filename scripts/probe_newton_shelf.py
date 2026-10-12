# SPDX-License-Identifier: Apache-2.0
"""Drop and slide eight probes onto the compiled shelf using MuJoCo-Warp.

Run on the pinned GPU stack. This isolates deck contact quality; the normal
Isaac Lab scene tests exercise the dynamic rack, objects, cameras and sensors.
"""
import argparse
import importlib.util
import json
from pathlib import Path

import mujoco
import numpy as np


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--cpu', action='store_true', help='CPU diagnostic; default tests MuJoCo-Warp on CUDA')
    args = p.parse_args()
    root = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location('shelf', root / 'robolab/core/environments/newton_shelf.py')
    shelf = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(shelf)
    manifest, parts = shelf.load_colliders(root / 'assets/objects/vomp/rack_l04/rack_l04_colliders.usda')
    model_spec = mujoco.MjSpec()
    model_spec.option.timestep = .001
    model_spec.option.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
    model_spec.option.cone = mujoco.mjtCone.mjCONE_ELLIPTIC
    model_spec.option.iterations = 100
    contact = dict(friction=[.5, .005, .001], solref=[.0045, 1], solimp=[.99, .999, .001, .5, 2], condim=6)
    for i, (v, f) in enumerate(parts.values()):
        model_spec.add_mesh(name=f'rack_{i}', uservert=(v*.2).ravel().tolist(), userface=f.ravel().tolist(), maxhullvert=64)
        model_spec.worldbody.add_geom(name=f'rack_{i}', type=mujoco.mjtGeom.mjGEOM_MESH, meshname=f'rack_{i}', **contact)
    decks = manifest['meshes']['M_Rack_L01_Shelf']['paths']
    expected = []
    for i, path in enumerate(decks):
        top = float(parts[path][0][:, 2].max()) * .2
        for kind, y in [('sphere', -.08), ('box', .08)]:
            name = f'{kind}_{i}'
            body = model_spec.worldbody.add_body(name=name, pos=[-.02, y, top+.04])
            body.add_freejoint()
            body.add_geom(type=mujoco.mjtGeom.mjGEOM_SPHERE if kind == 'sphere' else mujoco.mjtGeom.mjGEOM_BOX,
                          size=[.01, .01, .01], mass=.1, **contact)
            expected.append(dict(name=name, support_height_m=top, expected_center_height_m=top+.01))
    m = model_spec.compile()
    d = mujoco.MjData(m)
    # Tangential speed tests friction after a 30 mm drop; travel stays on deck.
    for i in range(1, 8, 2):
        d.qvel[6*i] = .15
    mujoco.mj_forward(m, d)
    if args.cpu:
        def step():
            for _ in range(10):
                mujoco.mj_step(m, d)
    else:
        import mujoco_warp as mjw
        import warp as wp
        wp.init()
        if not wp.get_device().is_cuda:
            raise RuntimeError('This probe requires a CUDA device; --cpu is diagnostic only')
        wm = mjw.put_model(m)
        wd = mjw.put_data(m, d, nconmax=1024, njmax=4096)
        # The small fixed model uses a graph only to batch ten steps. The full
        # RoboLab integration separately uses its guarded, non-graph preset.
        with wp.ScopedCapture() as capture:
            for _ in range(10):
                mjw.step(wm, wd)
        def step():
            wp.capture_launch(capture.graph)
            assert int(wd.nacon.numpy()[0]) <= wd.naconmax
            assert int(wd.nefc.numpy().max()) <= wd.njmax
            mjw.get_data_into(d, m, wd)
    rows = []
    for _ in range(200):
        step()
        assert np.isfinite(d.qpos).all() and np.isfinite(d.qvel).all()
        rows.append(d.qpos.copy().reshape(8, 7))
    states = np.asarray(rows)
    for i, row in enumerate(expected):
        row.update(final_height_m=float(states[-1, i, 2]),
                   height_error_m=float(abs(states[-1, i, 2] - row['expected_center_height_m'])),
                   late_position_span_m=float(np.linalg.norm(np.ptp(states[-50:, i, :3], axis=0))),
                   final_speed_m_s=float(np.linalg.norm(d.qvel[6*i:6*i+3])),
                   horizontal_travel_m=float(np.linalg.norm(states[-1, i, :2] - [-.02, -.08 if i % 2 == 0 else .08])))
        assert row['height_error_m'] < .0001, row
        assert row['late_position_span_m'] < .0001, row
        assert row['final_speed_m_s'] < .001, row
        if i % 2:
            assert .005 < row['horizontal_travel_m'] < .04, row
    report = dict(backend='MuJoCo CPU' if args.cpu else 'MuJoCo-Warp CUDA',
                  collision_asset_sha256=manifest['collision_asset_sha256'], seconds=2., probes=expected)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
