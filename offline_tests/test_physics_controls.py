# SPDX-License-Identifier: Apache-2.0
"""Physics controls preserve time semantics and verify actual material arrays."""
from argparse import ArgumentParser, Namespace
from types import SimpleNamespace as NS

import numpy as np
import pytest

from robolab.core.physics import newton_friction as F
from robolab.core.physics.timing import configure_timing, timing_provenance
from robolab.eval.runner import add_common_eval_args


class Array:
    def __init__(self, values):
        self.values = np.asarray(values).copy()
    def numpy(self):
        return self.values.copy()
    def assign(self, values):
        self.values[:] = values


def solver():
    geom = np.array([[2., .005, .001], [3., .02, .003], [1., .01, .002]])
    pairs = np.array([[2., 2., .005, .001, .001]])
    return NS(_use_mujoco_contacts=True,
              mj_model=NS(ngeom=3, npair=1, geom_friction=geom.copy(), pair_friction=pairs.copy(),
                          pair_geom1=np.array([0]), pair_geom2=np.array([1]), geom_priority=np.array([0, 0, 0])),
              mjw_model=NS(geom_friction=Array(np.stack([geom, geom])), pair_friction=Array(pairs[None])),
              model=NS(shape_material_mu=Array([2., 3., 1., 2., 3., 1.]), shape_label=[
                  f"/World/envs/env_{world}/{path}" for world in range(2)
                  for path in ("scene/cup/mesh", "robot/hand/finger/pad", "table_fixture/collider")]),
              mjc_geom_to_newton_shape=Array([[0, 1, 2], [3, 4, 5]]))


TARGETS = {"objects": {"cup": "/World/envs/env_[^/]+/scene/cup"},
           "gripper": {"finger": "/World/envs/env_[^/]+/robot/(?:.*/)?finger"}}
REQUEST = {"mode": "table", "objects": {"cup": {"static": .5, "dynamic": .35, "source": "class:test"}},
           "gripper": {"static": .8, "dynamic": .7}, "gripper_bodies": ["finger"]}


def test_material_override_updates_both_worlds_and_explicit_pairs_only_sliding():
    s = solver()
    before = s.mjw_model.geom_friction.numpy()
    F.apply(s, REQUEST, TARGETS)
    np.testing.assert_allclose(s.model.shape_material_mu.numpy(), [.35, .7, 1., .35, .7, 1.])
    after = s.mjw_model.geom_friction.numpy()
    np.testing.assert_array_equal(after[..., 1:], before[..., 1:])
    np.testing.assert_array_equal(after[:, 2], before[:, 2])
    np.testing.assert_allclose(s.mj_model.pair_friction, [[.7, .7, .005, .001, .001]])
    measured = F.read_solver(s, REQUEST, TARGETS)
    assert not F.check_applied(measured)
    assert measured['summary']['objects']['cup']['worlds'] == 2
    s.mjw_model.geom_friction.values[1, 0, 0] = .9
    assert F.check_applied(F.read_solver(s, REQUEST, TARGETS))


def test_explicit_pair_respects_geom_priority():
    s = solver()
    s.mj_model.geom_priority[0] = 1
    F.apply(s, REQUEST, TARGETS)
    np.testing.assert_allclose(s.mj_model.pair_friction[0, :2], [.35, .35])


def test_bad_target_fails_before_any_mutation():
    s = solver()
    before = s.mjw_model.geom_friction.numpy()
    with pytest.raises(RuntimeError, match='no compiled geoms'):
        F.apply(s, REQUEST, {**TARGETS, 'objects': {'cup': '/missing'}})
    np.testing.assert_array_equal(s.mjw_model.geom_friction.numpy(), before)
    # A similarly named object must not accidentally match.
    s.model.shape_label[0] = '/World/envs/env_0/scene/cupboard/mesh'
    with pytest.raises(RuntimeError, match='no compiled geoms'):
        F.target_geoms(s, TARGETS)


def test_missing_and_nonfinite_readback_never_pass():
    s = solver()
    F.apply(s, REQUEST, TARGETS)
    measured = F.read_solver(s, REQUEST, TARGETS)
    measured['objects']['cup'][0]['sliding'] = float('nan')
    assert F.check_applied(measured)
    measured['objects']['cup'] = []
    assert F.check_applied(measured)


def test_newton_verifier_uses_dynamic_mapping_and_real_measurements(tmp_path):
    import json
    from scripts.verify_patches import p79_friction_applied
    s = solver()
    F.apply(s, REQUEST, TARGETS)
    p = tmp_path/'Task'/'friction_applied.json'
    p.parent.mkdir()
    p.write_text(json.dumps(F.read_solver(s, REQUEST, TARGETS)))
    assert p79_friction_applied(str(tmp_path)).verdict == 'PASS'
    s.mjw_model.geom_friction.values[1, 1, 0] = .8
    p.write_text(json.dumps(F.read_solver(s, REQUEST, TARGETS)))
    assert p79_friction_applied(str(tmp_path)).verdict == 'FAIL'


def test_scoped_solver_hook_restores_and_upstream_does_not_mutate(monkeypatch):
    import sys
    calls = []
    class Solver:
        def __init__(self, solver=None):
            calls.append('init')
    original = Solver.__init__
    monkeypatch.setitem(sys.modules, 'newton.solvers', NS(SolverMuJoCo=Solver))
    monkeypatch.setattr(F, 'apply', lambda *args: calls.append('apply'))
    with F.friction_runtime({'mode': 'upstream'}, TARGETS):
        import inspect
        assert inspect.signature(Solver.__init__) == inspect.signature(original)
        Solver()
    assert calls == ['init']
    with pytest.raises(ValueError):
        with F.friction_runtime(REQUEST, TARGETS):
            Solver(solver="newton")
            raise ValueError('test exit')
    assert calls == ['init', 'init', 'apply']
    assert Solver.__init__ is original


@pytest.mark.parametrize('backend', ['physx', 'newton_mjwarp'])
@pytest.mark.parametrize('rate,decimation', [(60, 4), (120, 8), (240, 16)])
def test_rate_preserves_control_render_and_episode_duration(backend, rate, decimation):
    cfg = NS(sim=NS(dt=1/120, render_interval=8, physics=NS(num_substeps=4)),
             decimation=8, episode_length_s=60)
    configure_timing(cfg, backend, rate)
    assert cfg.decimation == decimation and cfg.sim.render_interval == decimation
    assert cfg.sim.dt * cfg.decimation == pytest.approx(1/15)
    assert cfg.episode_length_s / (cfg.sim.dt * cfg.decimation) == pytest.approx(900)
    measured = timing_provenance(cfg, backend, rate)
    assert measured['sim_rate_hz'] == rate
    assert measured['control_rate_hz'] == 15


@pytest.mark.parametrize('rate', [0, -1, float('nan'), float('inf'), 100, 1])
def test_invalid_or_nonintegral_rate_is_rejected_without_mutating_config(rate):
    cfg = NS(sim=NS(dt=1/120, render_interval=8), decimation=8)
    with pytest.raises(ValueError):
        configure_timing(cfg, 'physx', rate)
    assert cfg.sim.dt == 1/120 and cfg.decimation == 8


def test_newton_substeps_cannot_silently_apply_to_physx():
    with pytest.raises(ValueError, match='requires --physics'):
        configure_timing(NS(), 'physx', newton_substeps=2)


@pytest.mark.parametrize('flag,value', [('--sim-rate', 'nan'), ('--sim-rate', '0'),
                                      ('--newton-substeps', '0'), ('--newton-substeps', '1.5')])
def test_cli_rejects_invalid_values(flag, value):
    parser = ArgumentParser()
    add_common_eval_args(parser)
    with pytest.raises(SystemExit):
        parser.parse_args([flag, value])


def test_cli_defaults_and_requested_controls():
    parser = ArgumentParser()
    add_common_eval_args(parser)
    default = parser.parse_args([])
    assert default.sim_rate is None and default.newton_substeps is None
    args = parser.parse_args(['--physics', 'newton_mjwarp', '--sim-rate', '240', '--newton-substeps', '2', '--friction', '.5'])
    assert (args.sim_rate, args.newton_substeps, args.friction) == (240, 2, '.5')


def test_resume_cannot_mix_friction_or_timing_conditions(tmp_path):
    import importlib.util
    from pathlib import Path
    spec = importlib.util.spec_from_file_location("physics_controls_test", Path(__file__).resolve().parents[1] / "robolab/core/environments/physics.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    ensure_output_backend = module.ensure_output_backend
    parameters = {"friction": {"mode": "uniform", "uniform": .5}, "sim_rate_hz": 240, "newton_substeps": 2}
    ensure_output_backend(tmp_path, "newton_mjwarp", parameters)
    ensure_output_backend(tmp_path, "newton_mjwarp", parameters)
    for changed in ({**parameters, "sim_rate_hz": 120}, {**parameters, "newton_substeps": 4},
                    {**parameters, "friction": {"mode": "uniform", "uniform": .7}}):
        with pytest.raises(ValueError, match="different physics parameters"):
            ensure_output_backend(tmp_path, "newton_mjwarp", changed)
