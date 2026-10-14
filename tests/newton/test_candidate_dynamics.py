# SPDX-License-Identifier: Apache-2.0
"""Candidate growth preserves real primitive and convex contact results."""
import importlib.util
from pathlib import Path
import numpy as np
import pytest

@pytest.mark.parametrize("device", ["cpu", "cuda:0"])
@pytest.mark.parametrize("shape", ["sphere", "box"])
def test_candidate_growth_matches_large_contact_buffer(shape, device):
    mujoco = pytest.importorskip("mujoco")
    mjw = pytest.importorskip("mujoco_warp")
    wp = pytest.importorskip("warp")
    if device.startswith("cuda") and not wp.is_cuda_available():
        pytest.skip("CUDA required for GPU candidate parity")
    from mujoco_warp._src.types import BroadphaseType
    spec = importlib.util.spec_from_file_location("candidate_dynamics", Path(__file__).resolve().parents[2]
        / "robolab/core/environments/newton_candidates.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    bodies = "".join(f'<body pos="{i if i < 11 else 10.07} 0 1"><freejoint/><geom type="{shape}" size=".05 .05 .05"/></body>' for i in range(12))
    model = mujoco.MjModel.from_xml_string("<mujoco><worldbody>" + bodies + "</worldbody></mujoco>")
    with wp.ScopedDevice(device):
        gpu = mjw.put_model(model)
        gpu.opt.broadphase = BroadphaseType.NXN
        gpu.opt.broadphase_filter = 0
        def run(capacity, adapt):
            from contextlib import nullcontext
            d = mjw.put_data(model, mujoco.MjData(model), nworld=2, nconmax=capacity, njmax=128)
            with module.candidate_storage(wp) if adapt else nullcontext():
                mjw.forward(gpu, d)
            n = int(d.nacon.numpy()[0])
            assert n <= d.naconmax
            rows = sorted(zip(d.contact.worldid.numpy()[:n].tolist(), d.contact.geom.numpy()[:n].tolist(),
                              np.round(d.contact.dist.numpy()[:n], 6).tolist()))
            return rows, getattr(d, "_robolab_candidate_storage", None), d.naconmax
        reference, _, _ = run(128, False)
        actual, stats, capacity = run(16, True)
        assert reference == actual and len(actual) >= 2
        assert stats["peak_candidates"] > capacity and stats["growth_events"] == 1
        assert capacity == 32
