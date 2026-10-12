# SPDX-License-Identifier: Apache-2.0
"""Collision-storage scope, filter preservation, and native solver parity."""
import importlib.util
import inspect
from pathlib import Path
import sys
from types import SimpleNamespace as NS
from contextlib import nullcontext

import numpy as np
import pytest

spec = importlib.util.spec_from_file_location('native_storage_under_test', Path(__file__).resolve().parents[2]
    / 'robolab/core/environments/newton_native_contacts.py')
native = importlib.util.module_from_spec(spec)
spec.loader.exec_module(native)


def test_storage_preserves_other_body_filters_and_restores(monkeypatch):
    class Builder:
        def __init__(self):
            self.body_shapes = {0: [], 1: []}
            self.filters = []
            self.count = 0
        def add_shape(self, *, body, fail=False):
            shape = self.count
            self.count += 1
            self.filters.extend((i, shape) for i in self.body_shapes[body])
            self.filters.extend((i, shape) for i in self.body_shapes[1-body])
            self.body_shapes[body].append(shape)
            if fail:
                raise ValueError('shape failure')
            return shape
        def _find_shape_contact_pairs(self, model, *, allow_filter_blocks):
            model.rebuilt = True
    class Solver:
        @staticmethod
        def _compile_newton_collision_masks(model, selected):
            return None
        @staticmethod
        def _color_collision_shapes(model, selected):
            return None
        def __init__(self, model, *, use_mujoco_contacts=True, solver="newton", nconmax=17):
            self.nconmax = nconmax
    class Pipeline:
        def __init__(self, model):
            self.model = model
    monkeypatch.setitem(sys.modules, 'newton', NS(ModelBuilder=Builder, CollisionPipeline=Pipeline))
    monkeypatch.setitem(sys.modules, 'newton.solvers', NS(SolverMuJoCo=Solver))
    monkeypatch.setattr(native.importlib.metadata, 'version', lambda name: '1.5.2')
    originals = Builder.add_shape, Builder._find_shape_contact_pairs, Solver.__init__, Pipeline.__init__
    with pytest.raises(ValueError, match='shape failure'):
        with native.native_contact_storage():
            assert inspect.signature(Solver.__init__) == inspect.signature(originals[2])
            b = Builder()
            for body in [0, 1, 1, 0]:
                b.add_shape(body=body)
            assert b.body_shapes == {0: [0, 3], 1: [1, 2]}
            assert b.filters == [(0, 1), (0, 2), (1, 3), (2, 3)]
            m = NS(shape_count=4)
            b._find_shape_contact_pairs(m, allow_filter_blocks=True)
            assert m.shape_contact_pairs is None and m.shape_contact_pair_count == 0
            assert Solver(m, solver="newton", nconmax=65536).nconmax == 65536
            with pytest.raises(RuntimeError, match='native MuJoCo'):
                Solver(m, use_mujoco_contacts=False)
            with pytest.raises(RuntimeError, match='collision pipeline'):
                Pipeline(m)
            Pipeline(NS())  # Models not finalized in this scope are unaffected.
            with pytest.raises(RuntimeError, match='nested'):
                with native.native_contact_storage():
                    pass
            b.add_shape(body=0, fail=True)
    assert b.body_shapes[0] == [0, 3, 4]
    assert (Builder.add_shape, Builder._find_shape_contact_pairs, Solver.__init__, Pipeline.__init__) == originals
    assert not native._active
    monkeypatch.setattr(native.importlib.metadata, 'version', lambda name: 'unsupported')
    with pytest.raises(RuntimeError, match='Re-audit'):
        with native.native_contact_storage():
            pass


def test_native_mujoco_trajectories_match_with_compact_storage(tmp_path):
    newton = pytest.importorskip('newton')
    wp = pytest.importorskip('warp')
    from newton.solvers import SolverMuJoCo
    wp.config.kernel_cache_dir = str(tmp_path/'warp-cache')
    trajectories = []
    for compact in [False, True]:
        with native.native_contact_storage() if compact else nullcontext():
            b = newton.ModelBuilder()
            first = b.add_body(xform=wp.transform(wp.vec3(0, 0, .1), wp.quat_identity()))
            b.add_shape_box(first, hx=.04, hy=.03, hz=.02)
            b.add_shape_box(first, xform=wp.transform(wp.vec3(.02, 0, 0), wp.quat_identity()), hx=.02, hy=.02, hz=.02)
            second = b.add_body(xform=wp.transform(wp.vec3(.2, 0, .1), wp.quat_identity()))
            b.add_shape_box(second, hx=.04, hy=.03, hz=.02)
            b.add_shape_box(second, hx=.02, hy=.02, hz=.02)
            b.add_shape_collision_filter_pair(0, 2)
            assert b.body_shapes[first] == [0, 1] and b.body_shapes[second] == [2, 3]
            assert (0, 2) in list(b._shape_collision_filter_pairs)
            b.add_ground_plane()
            model = b.finalize(device='cpu')
            solver = SolverMuJoCo(model, use_mujoco_cpu=True, use_mujoco_contacts=True, solver="newton")
            state, out, control = model.state(), model.state(), model.control()
            trajectory = []
            for _ in range(500):
                solver.step(state, out, control, None, .002)
                state, out = out, state
                trajectory.append(state.body_q.numpy().copy())
            assert solver.mj_data.ncon > 0
            assert abs(trajectory[-1][0, 2]-.02) < .001
            trajectories.append(np.asarray(trajectory))
    np.testing.assert_array_equal(*trajectories)


@pytest.mark.parametrize("parts", [8, 260])
def test_compiled_cross_body_permissions_match(parts):
    newton = pytest.importorskip('newton')
    wp = pytest.importorskip('warp')
    from newton.solvers import SolverMuJoCo
    permissions = []
    for compact in [False, True]:
        with native.native_contact_storage() if compact else nullcontext():
            b = newton.ModelBuilder()
            for i in range(2):
                body = b.add_body(xform=wp.transform(wp.vec3(i, 0, 1), wp.quat_identity()))
                for j in range(parts if i == 0 else 2):
                    b.add_shape_box(body, hx=.01, hy=.01, hz=.01)
            b.add_shape_collision_filter_pair(0, parts)
            model = b.finalize(device='cpu')
            solver = SolverMuJoCo(model, use_mujoco_cpu=True, use_mujoco_contacts=True)
            m = solver.mj_model
            a = np.arange(parts)[:, None]
            c = np.arange(parts, parts+2)[None, :]
            permissions.append(((m.geom_contype[a] & m.geom_conaffinity[c]) |
                                (m.geom_contype[c] & m.geom_conaffinity[a])) != 0)
    np.testing.assert_array_equal(*permissions)
