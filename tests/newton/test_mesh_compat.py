# SPDX-License-Identifier: Apache-2.0
"""Real MuJoCo compiler checks; no Kit or GPU is required."""
import importlib.util
from pathlib import Path
from types import SimpleNamespace as NS

import numpy as np
import pytest

spec = importlib.util.spec_from_file_location("mesh_compat_under_test", Path(__file__).resolve().parents[2]
    / "robolab/core/environments/newton_mesh_compat.py")
compat = importlib.util.module_from_spec(spec)
spec.loader.exec_module(compat)
mujoco = pytest.importorskip("mujoco")

V = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=float)
F = np.array([[0, 2, 1], [0, 1, 3], [0, 3, 2], [1, 2, 3]])


def mesh_spec(*, size=1e-5, explicit=True):
    spec = mujoco.MjSpec()
    spec.add_mesh(name="tiny", uservert=(V * size).ravel(), userface=F.ravel())
    body = spec.worldbody.add_body(mass=.12, inertia=[.003, .004, .005],
                                   ipos=[.01, .02, .03], explicitinertial=explicit)
    body.add_freejoint()
    body.add_geom(type=mujoco.mjtGeom.mjGEOM_MESH, meshname="tiny")
    return spec


def test_tiny_mesh_keeps_geometry_and_explicit_body_properties():
    spec = mesh_spec()
    original = mujoco.MjSpec.compile
    with pytest.raises(ValueError, match="volume is too small"):
        original(spec)
    mode = spec.meshes[0].inertia
    reports = []
    model = compat.compile_with_explicit_inertia(spec, original, mujoco, reports)
    assert spec.meshes[0].inertia == mode
    assert len(reports) == 1
    assert model.body_mass[1] == .12
    np.testing.assert_array_equal(model.body_inertia[1], [.003, .004, .005])
    np.testing.assert_array_equal(model.body_ipos[1], [.01, .02, .03])
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    world_vertices = model.mesh_vert @ data.geom_xmat[0].reshape(3, 3).T + data.geom_xpos[0]
    distances = np.linalg.norm(world_vertices[:, None] - (V * 1e-5)[None], axis=2)
    assert distances.min(axis=0).max() < 1e-10


@pytest.mark.parametrize("unsafe", ["inferred", "forced", "shared_inferred"])
def test_fallback_refuses_to_change_inferred_mass(unsafe):
    spec = mesh_spec(explicit=unsafe != "inferred")
    if unsafe == "forced":
        spec.compiler.inertiafromgeom = mujoco.mjtInertiaFromGeom.mjINERTIAFROMGEOM_TRUE
    if unsafe == "shared_inferred":
        body = spec.worldbody.add_body()
        body.add_freejoint()
        body.add_geom(type=mujoco.mjtGeom.mjGEOM_MESH, meshname="tiny")
    with pytest.raises(ValueError, match="volume is too small"):
        compat.compile_with_explicit_inertia(spec, mujoco.MjSpec.compile, mujoco, [])
    assert spec.meshes[0].inertia == mujoco.mjtMeshInertia.mjMESH_INERTIA_LEGACY


def test_ordinary_mesh_and_unrelated_errors_are_unchanged():
    spec = mesh_spec(size=.1)
    expected = spec.compile()
    reports = []
    actual = compat.compile_with_explicit_inertia(spec, mujoco.MjSpec.compile, mujoco, reports)
    assert not reports
    for name in ("mesh_vert", "mesh_face", "body_mass", "body_inertia", "geom_pos", "geom_quat"):
        np.testing.assert_array_equal(getattr(actual, name), getattr(expected, name))
    spec = mesh_spec()
    calls = []
    def fail_after_retry(spec):
        calls.append(1)
        if len(calls) == 1:
            return mujoco.MjSpec.compile(spec)
        raise ValueError("unrelated error")
    with pytest.raises(ValueError, match="unrelated error"):
        compat.compile_with_explicit_inertia(spec, fail_after_retry, mujoco, [])
    assert spec.meshes[0].inertia == mujoco.mjtMeshInertia.mjMESH_INERTIA_LEGACY


@pytest.mark.parametrize("scale", [[1, 1, 1], [.01, 10, .1]])
def test_fragment_repair_is_bounded_in_physical_coordinates(scale):
    world = np.array([[0, 0, 0], [2e-6, 0, 0], [0, 9e-5, 0]]) + [.1, .2, .3]
    vertices, faces, diameter = compat.thicken_generated_fragment(world / scale, scale)
    repaired = vertices * scale
    assert diameter < 1e-4
    assert np.linalg.norm(repaired[:, None] - world[None], axis=2).min(1).max() == pytest.approx(5e-7)
    assert len(faces) == 8
    assert np.ptp(repaired[:, 2]) == pytest.approx(1e-6)
    for face in faces:
        normal = np.cross(repaired[face[1]] - repaired[face[0]], repaired[face[2]] - repaired[face[0]])
        assert np.dot(normal, repaired[face].mean(0) - repaired.mean(0)) > 0


@pytest.mark.parametrize("v", [V * 1e-5, V[:3] * .1, np.zeros((3, 3)), V[:2], V * float('nan')])
def test_solid_large_or_invalid_fragments_are_not_repaired(v):
    assert compat.thicken_generated_fragment(v, [1, 1, 1]) is None


def test_generated_only_and_restored_after_exception():
    class Mesh:
        vertices = V[:3] * 1e-5
        def copy(self, **kwargs):
            return NS(**kwargs)
    existing = Mesh()
    class Builder:
        shape_source = [existing]
        shape_type = [1]
        shape_scale = [[1, 1, 1]]
        shape_label = ["authored"]
        def approximate_meshes(self, method):
            self.shape_source = [existing, Mesh()]
            self.shape_type = [1, 1]
            self.shape_scale = [[1, 1, 1], [1, 1, 1]]
            self.shape_label = ["authored", "generated"]
            return {1}
    original = Builder.approximate_meshes
    reports = []
    builder = Builder()
    with pytest.raises(RuntimeError, match="exit"):
        with compat.generated_fragment_context(Builder, 1, reports):
            assert builder.approximate_meshes("coacd") == {1}
            assert builder.shape_source[0] is existing
            assert len(builder.shape_source[1].vertices) == 6
            raise RuntimeError("exit")
    assert Builder.approximate_meshes is original
    assert [r['shape'] for r in reports] == ["generated"]


def test_runtime_scope_restores_compiler_and_builder_on_failure(monkeypatch):
    import sys
    class Builder:
        def approximate_meshes(self, *args, **kwargs):
            return set()
    monkeypatch.setitem(sys.modules, "newton", NS(ModelBuilder=Builder, GeoType=NS(CONVEX_MESH=1)))
    monkeypatch.setattr(compat.importlib.metadata, "version", lambda n: {"newton": "1.5.2", "mujoco": "3.11.0"}[n])
    compile_original = mujoco.MjSpec.compile
    approximate_original = Builder.approximate_meshes
    with pytest.raises(ValueError, match="simulation failure"):
        with compat.mesh_compatibility() as reports:
            assert mesh_spec().compile().body_mass[1] == .12
            assert len(reports) == 1
            with pytest.raises(RuntimeError, match="nested"):
                with compat.mesh_compatibility():
                    pass
            raise ValueError("simulation failure")
    assert mujoco.MjSpec.compile is compile_original
    assert Builder.approximate_meshes is approximate_original
    assert not compat._active
    monkeypatch.setattr(compat.importlib.metadata, "version", lambda n: "unsupported")
    with pytest.raises(RuntimeError, match="Re-audit"):
        with compat.mesh_compatibility():
            pass
