# SPDX-FileCopyrightText: Copyright (c) 2026 Manda Robotics
# SPDX-License-Identifier: Apache-2.0
"""Collision-response regression cases, independent of Kit and CUDA."""
import importlib.util
from pathlib import Path
from types import SimpleNamespace as NS

import numpy as np
import pytest

spec = importlib.util.spec_from_file_location("collision_filter_under_test",
    Path(__file__).resolve().parents[2] / "robolab/core/environments/newton_collision_filter.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class Array:
    def __init__(self, values, dtype=None, device="test"):
        self.values = np.asarray(values).copy()
        self.shape = self.values.shape
        self.dtype = self.values.dtype if dtype is None else dtype
        self.device = device

    def numpy(self):
        return self.values.copy()

    def assign(self, values):
        self.values[:] = values


def fixture():
    # World, kinematic shelf, free object, fixed child of object, child of shelf.
    model = NS(nbody=5, body_parentid=np.array([0, 0, 0, 2, 1]),
               body_jntnum=np.array([0, 1, 1, 0, 0]),
               body_mocapid=np.full(5, -1), geom_bodyid=np.arange(5))
    mapping = np.array([[-1, 0, 1, -1, -1], [-1, 2, 3, -1, -1]])
    return model, mapping, np.array([2, 1, 2, 1])


def test_fixed_children_follow_parent_and_world_mapping():
    model, mapping, flags = fixture()
    np.testing.assert_array_equal(module.immovable_body_mask(model, mapping, flags, 2),
                                  [True, True, False, False, True])
    flags[2] = 1
    with pytest.raises(RuntimeError, match="identical body mobility"):
        module.immovable_body_mask(model, mapping, flags, 2)


def test_unmapped_jointed_body_stays_active_and_mocap_is_immovable():
    model, mapping, flags = fixture()
    mapping[:, 2] = -1
    assert not module.immovable_body_mask(model, mapping, flags, 2)[2]
    model.body_mocapid[2] = 0
    assert module.immovable_body_mask(model, mapping, flags, 2)[2]


def test_filter_preserves_dynamic_pairs_and_collision_sensors():
    model, mapping, flags = fixture()
    pairs = np.array([[0, 1], [1, 4], [1, 2], [2, 3], [0, 2]])
    original_ids = np.array([[-1, -1], [7, 3], [9, -1], [-2, -1], [-1, -1]])
    gpu = NS(nxn_geom_pair=Array(pairs), nxn_pairid=Array(original_ids),
             nxn_geom_pair_filtered=Array(pairs), nxn_pairid_filtered=Array(original_ids))
    solver = NS(_use_mujoco_contacts=True, mj_model=model, mjw_model=gpu,
                model=NS(body_flags=Array(flags)), mjc_body_to_newton=Array(mapping))
    report = module.filter_immovable_contacts(solver, NS(array=Array), 2)
    assert report["removed_contact_pairs"] == 2
    assert report["worlds"] == 2
    np.testing.assert_array_equal(gpu.nxn_pairid.numpy(),
                                  [[-2, -1], [-2, 3], [9, -1], [-2, -1], [-1, -1]])
    np.testing.assert_array_equal(gpu.nxn_geom_pair_filtered.numpy(), pairs[[1, 2, 4]])
    np.testing.assert_array_equal(gpu.nxn_pairid_filtered.numpy(), gpu.nxn_pairid.numpy()[[1, 2, 4]])
    assert module.filter_immovable_contacts(solver, NS(array=Array), 2)["removed_contact_pairs"] == 0


def test_native_newton_collision_pipeline_is_untouched():
    assert not module.filter_immovable_contacts(NS(_use_mujoco_contacts=False), None, 2)["applied"]


def test_invalid_mapping_fails_before_filtering():
    model, mapping, flags = fixture()
    mapping[0, 1] = 99
    with pytest.raises(RuntimeError, match="Invalid Newton body index"):
        module.immovable_body_mask(model, mapping, flags, 2)


def test_support_rolling_preserves_dynamic_pairs_and_sliding_friction():
    model, mapping, flags = fixture()
    model.geom_condim = np.full(5, 3)
    model.geom_friction = np.tile([.7, .005, .0001], (5, 1))
    model.npair = 2
    model.pair_geom1, model.pair_geom2 = np.array([1, 2]), np.array([2, 3])
    model.pair_dim = np.array([3, 4])
    model.pair_friction = np.tile([.5, .6, .004, .0002, .0003], (2, 1))
    fields = ('geom_condim', 'geom_friction', 'pair_dim', 'pair_friction')
    originals = {name: getattr(model, name).copy() for name in fields}
    gpu = NS(**{name: Array(getattr(model, name)[None]) for name in fields})
    solver = NS(_use_mujoco_contacts=True, mj_model=model, mjw_model=gpu,
                model=NS(body_flags=Array(flags)), mjc_body_to_newton=Array(mapping))
    report = module.override_support_rolling(solver, .001, None, 2)
    assert report['support_geoms'] == 3
    assert report['explicit_support_pairs'] == 1
    np.testing.assert_array_equal(model.geom_condim, [6, 6, 3, 3, 6])
    np.testing.assert_array_equal(model.geom_friction[:, :2], originals['geom_friction'][:, :2])
    np.testing.assert_array_equal(model.geom_friction[[2, 3]], originals['geom_friction'][[2, 3]])
    np.testing.assert_array_equal(model.pair_dim, [6, 4])
    np.testing.assert_array_equal(model.pair_friction[:, :3], originals['pair_friction'][:, :3])
    np.testing.assert_array_equal(model.pair_friction[1], originals['pair_friction'][1])
    assert (model.pair_friction[0, -2:] == .001).all()
    for name in fields:
        np.testing.assert_array_equal(getattr(gpu, name).numpy()[0], getattr(model, name))


def test_support_rolling_rejects_invalid_coefficient_and_collision_pipeline():
    for value in (0, -1, float('nan'), float('inf')):
        with pytest.raises(ValueError, match='finite and positive'):
            module.override_support_rolling(None, value, None, 2)
    with pytest.raises(ValueError, match='native MuJoCo'):
        module.override_support_rolling(NS(_use_mujoco_contacts=False), .001, None, 2)
