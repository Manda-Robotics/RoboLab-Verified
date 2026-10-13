# SPDX-License-Identifier: Apache-2.0
"""Native contact manifolds, adapter scope and a physical mesh stack."""
import importlib.util
import inspect
from pathlib import Path

import numpy as np
import pytest

spec = importlib.util.spec_from_file_location('multiccd_under_test', Path(__file__).resolve().parents[2]
    / 'robolab/core/environments/newton_multiccd.py')
multiccd = importlib.util.module_from_spec(spec)
spec.loader.exec_module(multiccd)


@pytest.mark.parametrize('cpu,native,fail', [(False, True, False), (False, True, True),
                                            (True, True, False), (False, False, False)])
def test_native_gpu_only_and_restoration(cpu, native, fail):
    seen = []
    class Solver:
        def __init__(self, model, *, use_mujoco_cpu=False, use_mujoco_contacts=True, enable_multiccd=False):
            seen.append((model, use_mujoco_cpu, use_mujoco_contacts, enable_multiccd))
            if fail:
                raise ValueError('construction failed')
    original = Solver.__init__
    try:
        with multiccd.native_multiccd(Solver):
            assert inspect.signature(Solver.__init__) == inspect.signature(original)
            Solver('model', use_mujoco_cpu=cpu, use_mujoco_contacts=native)
    except ValueError:
        assert fail
    assert seen == [('model', cpu, native, native and not cpu)]
    assert Solver.__init__ is original


def test_flat_mesh_stack_has_multiple_contacts_and_settles():
    newton = pytest.importorskip('newton')
    wp = pytest.importorskip('warp')
    mujoco = pytest.importorskip('mujoco')
    if not wp.is_cuda_available():
        pytest.skip('CUDA required for native MJWarp contact manifolds')
    from scipy.spatial import ConvexHull
    from newton.solvers import SolverMuJoCo
    vertices = np.array([[x,y,z] for x in [-.03,.03] for y in [-.03,.03] for z in [-.03,.03]], dtype=np.float32)
    faces = ConvexHull(vertices).simplices.copy()
    for face in faces:
        a,b,c = vertices[face]
        if np.dot(np.cross(b-a,c-a), (a+b+c)/3) < 0:
            face[1],face[2] = face[2],face[1]
    builder = newton.ModelBuilder()
    cfg = newton.ModelBuilder.ShapeConfig(density=925.926, mu=1.0, margin=0.)
    for i in range(3):
        body = builder.add_body(xform=wp.transform(wp.vec3(0,0,.03+.06*i),wp.quat_identity()))
        builder.add_shape_mesh(body, mesh=newton.Mesh(vertices, faces.flatten()), cfg=cfg)
    builder.add_ground_plane()
    model = builder.finalize(device='cuda:0')
    with multiccd.native_multiccd(SolverMuJoCo):
        solver = SolverMuJoCo(model, nconmax=1024, njmax=4096, solver='newton', iterations=100)
    assert not solver.mj_model.opt.disableflags & int(mujoco.mjtDisableBit.mjDSBL_MULTICCD)
    assert not int(solver.mjw_model.opt.disableflags) & int(mujoco.mjtDisableBit.mjDSBL_MULTICCD)
    assert np.all(solver.mj_model.geom_margin == 0.)
    state,out,control = model.state(),model.state(),model.control()
    samples = []
    for i in range(1200):
        solver.step(state,out,control,None,1/480)
        state,out = out,state
        if i >= 960:
            samples.append(state.body_q.numpy().copy())
    poses = np.asarray(samples)
    assert np.isfinite(poses).all()
    np.testing.assert_allclose(poses[-1,:,:3], [[0,0,.03],[0,0,.09],[0,0,.15]], atol=.002)
    assert np.max(np.ptp(poses[:,:,:2],axis=0)) < .0005
    count = int(solver.mjw_data.nacon.numpy()[0])
    pairs = solver.mjw_data.contact.geom.numpy()[:count]
    _,counts = np.unique(np.sort(pairs,axis=1),axis=0,return_counts=True)
    assert counts.max() >= 3  # a contact manifold, not a single pivot point
