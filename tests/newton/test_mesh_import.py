"""The mesh compatibility shim is generic, cached, and exception-safe."""
import importlib.util
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest

spec = importlib.util.spec_from_file_location("mesh_import_under_test",
    Path(__file__).resolve().parents[2] / "robolab/core/environments/newton_meshes.py")
meshes = importlib.util.module_from_spec(spec)
spec.loader.exec_module(meshes)


def test_key_covers_geometry_settings_and_version():
    v = np.zeros((4, 3))
    f = np.array([[0, 1, 2]])
    key = meshes.decomposition_key(v, f, {"threshold": .05}, "1")
    assert key != meshes.decomposition_key(v + 1, f, {"threshold": .05}, "1")
    assert key != meshes.decomposition_key(v, f, {"threshold": .01}, "1")
    assert key != meshes.decomposition_key(v, f, {"threshold": .05}, "2")


@pytest.mark.parametrize("fail", [False, True])
def test_scope_cache_and_non_decomposed_meshes(monkeypatch, tmp_path, fail):
    vertices = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=float)
    faces = np.array([[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]])
    mesh = SimpleNamespace(vertices=vertices, indices=faces)
    calls = []
    coacd = ModuleType("coacd")
    def decompose(mesh, **kwargs):
        calls.append(kwargs)
        return [(vertices, faces)]
    coacd.run_coacd = decompose
    class Builder:
        def __init__(self):
            self.default_shape_cfg = SimpleNamespace(ke=2500.0, kd=100.0)
        def add_usd(self, *args, **kwargs):
            if kwargs.pop("fail_import", False):
                raise ValueError("import failed")
            return {**kwargs, "ke": self.default_shape_cfg.ke, "kd": self.default_shape_cfg.kd}
        def approximate_meshes(self, method="convex_hull", shape_indices=None, **kwargs):
            if method != "coacd":
                return kwargs
            assert kwargs.pop("raise_on_failure") is True
            return coacd.run_coacd(mesh, **kwargs)
    original = Builder.approximate_meshes
    original_add = Builder.add_usd
    newton = ModuleType("newton")
    newton.ModelBuilder = Builder
    monkeypatch.setitem(sys.modules, "newton", newton)
    monkeypatch.setitem(sys.modules, "coacd", coacd)
    monkeypatch.setattr(meshes.importlib.metadata, "version", lambda n: "1.2.1" if n == "newton" else "test")
    try:
        with meshes.mesh_decomposition_context(.05, tmp_path, contact_timeconst=.006) as reports:
            Builder().approximate_meshes("coacd")
            Builder().approximate_meshes("coacd")
            assert len(calls) == 1
            assert calls[0]["threshold"] == .05
            assert calls[0]["mcts_iterations"] == 150
            assert [r["cache_hit"] for r in reports] == [False, True]
            assert Builder().approximate_meshes("convex_hull", untouched=True) == {"untouched": True}
            b = Builder()
            result = b.add_usd("asset.usd", skip_mesh_approximation=True)
            assert result["skip_mesh_approximation"] is False
            assert result["ke"] == pytest.approx(1 / .006**2)
            assert result["kd"] == pytest.approx(2 / .006)
            assert b.default_shape_cfg.ke == 2500
            with pytest.raises(ValueError, match="import failed"):
                b.add_usd("bad.usd", fail_import=True)
            assert b.default_shape_cfg.ke == 2500
            assert b.default_shape_cfg.kd == 100
            assert b.approximate_meshes("convex_hull", keep_visual_shapes=True) == set()
            assert b.approximate_meshes("convex_hull", keep_visual_shapes=True) == {"keep_visual_shapes": True}
            with pytest.raises(RuntimeError, match="Concurrent/nested"):
                with meshes.mesh_decomposition_context(.05, tmp_path):
                    pass
            if fail:
                raise ValueError("initialization failed")
    except ValueError:
        assert fail
    assert coacd.run_coacd is decompose
    assert Builder.approximate_meshes is original
    assert Builder.add_usd is original_add
    assert not meshes._active


def test_rejects_invalid_parts():
    with pytest.raises(ValueError):
        meshes.validate_parts([])
