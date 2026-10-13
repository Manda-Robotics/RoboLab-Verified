# SPDX-License-Identifier: Apache-2.0
"""CPU staging contact scope and real GPU dynamics parity."""
from contextlib import nullcontext
import importlib.util
from pathlib import Path
from types import SimpleNamespace as NS

import numpy as np
import pytest

spec = importlib.util.spec_from_file_location('initialization_under_test', Path(__file__).resolve().parents[2]
    / 'robolab/core/environments/newton_initialization.py')
initialization = importlib.util.module_from_spec(spec)
spec.loader.exec_module(initialization)


@pytest.mark.parametrize('cpu,native,fail', [(False, True, False), (False, True, True),
                                            (True, True, False), (False, False, False)])
def test_only_owned_gpu_staging_contacts_are_deferred_and_flags_restored(cpu, native, fail):
    seen = []
    owner_model = NS(opt=NS(disableflags=4))
    other_model = NS(opt=NS(disableflags=8))
    owner_data, other_data = object(), object()
    def forward(model, data):
        seen.append((model, data, model.opt.disableflags))
        if fail and model is owner_model and data is owner_data:
            raise ValueError('staging failed')
    mj = NS(mj_forward=forward, mjtDisableBit=NS(mjDSBL_CONTACT=16))
    class Solver:
        use_mujoco_cpu = cpu
        _use_mujoco_contacts = native
        mj_model = owner_model
        mj_data = owner_data
        def _convert_to_mjc(self, *, solver):
            assert solver == "newton"
            mj.mj_forward(other_model, other_data)
            mj.mj_forward(owner_model, other_data)
            mj.mj_forward(owner_model, owner_data)
            assert owner_model.opt.disableflags == 4  # before the GPU upload
    original = Solver._convert_to_mjc
    with initialization.defer_initial_cpu_contacts(Solver, mj):
        with pytest.raises(ValueError, match='staging failed') if fail else nullcontext():
            Solver()._convert_to_mjc(solver="newton")
        assert mj.mj_forward is forward
    assert Solver._convert_to_mjc is original
    assert owner_model.opt.disableflags == 4 and other_model.opt.disableflags == 8
    assert [row[2] for row in seen] == [8, 4, 20 if not cpu and native else 4]


def test_deferred_staging_preserves_gpu_contact_trajectory(tmp_path):
    newton = pytest.importorskip('newton')
    wp = pytest.importorskip('warp')
    mujoco = pytest.importorskip('mujoco')
    if not wp.is_cuda_available():
        pytest.skip('CUDA required for real MJWarp staging parity')
    from newton.solvers import SolverMuJoCo
    trajectories, forces = [], []
    for deferred in [False, True]:
        with initialization.defer_initial_cpu_contacts(SolverMuJoCo, mujoco) if deferred else nullcontext():
            builder = newton.ModelBuilder()
            # One initially contacting box plus a falling box, exercising both
            # loaded initial contacts and contacts generated later on the GPU.
            for x, z in [(0., .0295), (.15, .15)]:
                body = builder.add_body(xform=wp.transform(wp.vec3(x, 0., z), wp.quat_identity()))
                builder.add_shape_box(body, hx=.03, hy=.03, hz=.03)
            builder.add_ground_plane()
            model = builder.finalize(device='cuda:0')
            solver = SolverMuJoCo(model, use_mujoco_cpu=False, use_mujoco_contacts=True,
                                  nconmax=1024, njmax=1024, solver='newton')
            assert not solver.mj_model.opt.disableflags & int(mujoco.mjtDisableBit.mjDSBL_CONTACT)
            assert not int(solver.mjw_model.opt.disableflags) & int(mujoco.mjtDisableBit.mjDSBL_CONTACT)
            assert solver.mj_data.ncon == 0 if deferred else solver.mj_data.ncon > 0
            state, out, control = model.state(), model.state(), model.control()
            samples=[]
            for _ in range(240):
                solver.step(state, out, control, None, .002)
                state, out = out, state
                samples.append(state.body_q.numpy().copy())
            assert solver.mjw_data.nacon.numpy()[0] > 0
            assert np.isfinite(samples).all()
            np.testing.assert_allclose(np.asarray(samples)[-1,:,2], .03, atol=.001)
            trajectories.append(np.asarray(samples))
            forces.append(solver.mjw_data.qfrc_constraint.numpy().copy())
    np.testing.assert_allclose(*trajectories, atol=1e-6, rtol=1e-6)
    np.testing.assert_allclose(*forces, atol=1e-5, rtol=1e-5)
