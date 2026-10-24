# SPDX-License-Identifier: Apache-2.0
"""Candidate growth must preserve every pair and the contact output budget."""
import importlib.util
from pathlib import Path
from types import SimpleNamespace as NS

import numpy as np
import pytest

spec = importlib.util.spec_from_file_location('candidate_storage_tested', Path(__file__).resolve().parents[1]
    / 'robolab/core/environments/newton_candidates.py')
C = importlib.util.module_from_spec(spec)
spec.loader.exec_module(C)


class Array:
    def __init__(self, n, dtype=int, device='cpu'):
        self.value = np.zeros(n, dtype=int)
        self.dtype, self.device, self.shape = dtype, device, (n,)
    def numpy(self):
        return self.value
    def zero_(self):
        self.value[:] = 0


@pytest.mark.parametrize('algorithm', ['sap_broadphase', 'nxn_broadphase'])
def test_overflow_retries_before_contacts_and_preserves_budget(algorithm):
    data = NS(naconmax=4, ncollision=Array(1), nacon=Array(1))
    data.nacon.value[0] = 3  # Existing contacts from an incremental sleeping pass.
    ctx = NS(**{k: Array(4) for k in ('collision_pair', 'collision_pairid', 'collision_worldid')})
    calls = []
    awake = object()
    def broadphase(model, view, context, awake_prev):
        assert awake_prev is awake
        calls.append(view.naconmax)
        view.ncollision.value[0] = 7
        n = min(view.naconmax, 7)
        context.collision_pair.value[:n] = np.arange(n)
    driver = NS(nxn_broadphase=broadphase, sap_broadphase=broadphase)
    launch = lambda *a, **kw: None
    wp = NS(empty=Array, launch=launch)
    with C.candidate_storage(wp, driver):
        getattr(driver, algorithm)(None, data, ctx, awake)
        assert calls == [4, 8]
        assert data.naconmax == 4 and data.nacon.value[0] == 3
        np.testing.assert_array_equal(ctx.collision_pair.value[:7], np.arange(7))
        assert data._robolab_candidate_storage['growth_events'] == 1
        data.ncollision.zero_()
        new_ctx = NS(**{k: Array(4) for k in vars(ctx)})
        getattr(driver, algorithm)(None, data, new_ctx, awake)
        assert calls == [4, 8, 8]
        assert new_ctx.collision_pair is ctx.collision_pair
    assert driver.nxn_broadphase is broadphase and driver.sap_broadphase is broadphase
    assert wp.launch is launch


def test_hard_limit_fails_closed_and_restores_hooks():
    def broadphase(model, data, ctx, awake):
        data.ncollision.value[0] = 33
    driver = NS(nxn_broadphase=broadphase, sap_broadphase=broadphase)
    original = lambda *a, **kw: None
    wp = NS(empty=lambda *a, **kw: pytest.fail('Must not allocate over limit'), launch=original)
    data = NS(naconmax=4, ncollision=Array(1))
    with pytest.raises(RuntimeError, match='bounded limit'):
        with C.candidate_storage(wp, driver):
            driver.nxn_broadphase(None, data, NS())
    assert wp.launch is original and driver.nxn_broadphase is broadphase


@pytest.mark.parametrize('key,dim,expected', [('primitive_narrowphase',4,8), ('ccd_hfield_kernel',4,8),
                                           ('_sdf_narrowphase',(3,4),(3,8)), ('ccd_kernel',2,2)])
def test_candidate_launch_covers_all_pairs_without_changing_contact_limit(key, dim, expected):
    labels = ['ncollision_in', 'collision_pair_in', 'naconmax_in']
    inputs = [Array(1), Array(8), 4]
    if key == 'ccd_kernel':
        labels.append('grid_stride_in'); inputs.append(2)
    kernel = NS(key=key, adj=NS(args=[NS(label=x) for x in labels]))
    kwargs = {'dim':dim, 'inputs':inputs}
    actual = C.candidate_launch_dimensions(kernel, kwargs)
    assert actual['dim'] == expected and actual['inputs'][2] == 4
    assert kwargs['dim'] == dim


def test_unknown_narrowphase_layout_fails_closed():
    kernel = NS(key='new_kernel', adj=NS(args=[NS(label=k) for k in
        ('ncollision_in', 'collision_pair_in', 'naconmax_in')]))
    with pytest.raises(RuntimeError, match='Unsupported candidate narrowphase'):
        C.candidate_launch_dimensions(kernel, {'dim':4,'inputs':[Array(1),Array(8),4]})


@pytest.mark.parametrize("count", [5, 8, 9])
def test_ccd_batches_cover_tail_pairs_and_preserve_output_budget(count):
    """MJWarp 3.12 clamps CCD candidates to naconmax; every tail pair must run."""
    labels = ["ncollision_in", "collision_pair_in", "collision_pairid_in",
              "collision_worldid_in", "naconmax_in", "grid_stride_in"]
    kernel = NS(key="ccd_kernel", adj=NS(args=[NS(label=x) for x in labels]))
    class DeviceArray(Array):
        def fill_(self, value):
            self.value.fill(value)
        def __getitem__(self, key):
            result = DeviceArray(len(self.value[key]))
            result.value = self.value[key]
            return result
    counter = DeviceArray(1); counter.fill_(count)
    arrays = [DeviceArray(16) for _ in range(3)]
    for i, arr in enumerate(arrays):
        arr.value[:] = 100 * i + np.arange(16)
    output = object()
    calls = []
    def launch(kernel, **kwargs):
        inputs = kwargs["inputs"]
        n = int(inputs[0].numpy()[0])
        assert n <= inputs[4] == 4
        assert kwargs["outputs"] == [output] and inputs[5] == 2
        calls.append([a.numpy()[:n].copy() for a in inputs[1:4]])
    driver = NS(nxn_broadphase=lambda *a: None, sap_broadphase=lambda *a: None)
    wp = NS(empty=DeviceArray, launch=launch)
    with C.candidate_storage(wp, driver):
        wp.launch(kernel, dim=2, inputs=[counter, *arrays, 4, 2], outputs=[output])
    for i, arr in enumerate(arrays):
        np.testing.assert_array_equal(np.concatenate([c[i] for c in calls]), arr.numpy()[:count])
    assert counter.numpy()[0] == count and wp.launch is launch
