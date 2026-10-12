# SPDX-License-Identifier: Apache-2.0
"""Shelf geometry integrity and native importer preservation, without Kit."""
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace as NS

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


shelf = load("shelf_under_test", ROOT / "robolab/core/environments/newton_shelf.py")
ASSET = ROOT / "assets/objects/vomp/rack_l04/rack_l04_colliders.usda"


def test_asset_budget_source_and_manifest():
    manifest, parts = shelf.load_colliders(ASSET)
    source = ASSET.with_name("rack_l04.usd")
    assert hashlib.sha256(source.read_bytes()).hexdigest() == manifest["source_asset_sha256"]
    assert manifest == json.loads(ASSET.with_suffix(".json").read_text())
    assert len(parts) == manifest["part_count"] == 44
    assert sum(m["part_count"] for m in manifest["meshes"].values()) == 44
    assert {k: v["part_count"] for k, v in manifest["meshes"].items()} == {
        "M_Rack_L01_Frame": 4, "M_Rack_L01_FrameJoints": 12,
        "M_Rack_L01_Shelf": 4, "M_Rack_L01_ShelfElements": 24,
    }


def test_baked_parts_are_convex():
    from scipy.spatial import ConvexHull
    _, parts = shelf.load_colliders(ASSET)
    for vertices, faces in parts.values():
        h = ConvexHull(vertices)
        assert h.volume > 0
        # Every face is on a supporting plane of the point cloud, within the
        # source float precision. Reject a non-convex or malformed bake.
        center = vertices.mean(0)
        tri = vertices[faces]
        n = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
        good = np.linalg.norm(n, axis=1) > 1e-15
        n = n[good]; tri = tri[good]
        n /= np.linalg.norm(n, axis=1)[:, None]
        n[np.einsum('ij,ij->i', n, tri.mean(1) - center) < 0] *= -1
        distance = np.einsum('fvi,fi->fv', vertices[None] - tri[:, :1], n)
        assert distance.max() < 2e-6


def native_builder(monkeypatch):
    newton = pytest.importorskip("newton")
    build = load("shelf_build_under_test", ROOT / "scripts/build_newton_shelf.py")
    source = build.source_components(ASSET.with_name("rack_l04.usd"))
    coacd = ModuleType("coacd")
    coacd.Mesh = lambda v, f: NS(vertices=np.asarray(v, dtype=float), indices=np.asarray(f))
    def unexpected(*args, **kwargs):
        raise AssertionError("The baked shelf must never invoke native CoACD")
    coacd.run_coacd = unexpected
    monkeypatch.setitem(sys.modules, "coacd", coacd)
    builder = newton.ModelBuilder()
    body = builder.add_body(mass=20., label="rack_l04")
    other = builder.add_body(mass=1., label="probe")
    probe = builder.add_shape_sphere(other, radius=.01)
    ids = []
    for name, (v, f, _) in source.items():
        cfg = newton.ModelBuilder.ShapeConfig(density=7850., mu=.37, mu_torsional=.013, restitution=.21)
        ids.append(builder.add_shape_mesh(body, mesh=newton.Mesh(v, f), scale=(.2, .2, .2), cfg=cfg,
                   label=f"/World/scene/rack_l04/SM_Rack_L04_01/{name}"))
    builder.add_shape_collision_filter_pair(probe, ids[1])
    return newton, builder, coacd, ids, probe


def test_native_conversion_preserves_mass_materials_visuals_and_filters(monkeypatch):
    newton, b, coacd, ids, probe = native_builder(monkeypatch)
    properties = [np.array(x).copy() for x in (b.body_mass, b.body_com, b.body_inertia)]
    originals = {i: b.shape_source[i] for i in ids}
    old_method = newton.ModelBuilder.approximate_meshes
    old_decompose = coacd.run_coacd
    with shelf.shelf_colliders(newton.ModelBuilder, coacd, ASSET) as reports:
        assert b.approximate_meshes('coacd', ids, keep_visual_shapes=True) == set(ids)
        assert [r['parts'] for r in reports] == [4, 12, 4, 24]
        with pytest.raises(RuntimeError, match="nested"):
            with shelf.shelf_colliders(newton.ModelBuilder, coacd, ASSET):
                pass
    assert newton.ModelBuilder.approximate_meshes is old_method and coacd.run_coacd is old_decompose
    for before, after in zip(properties, (b.body_mass, b.body_com, b.body_inertia)):
        np.testing.assert_array_equal(before, np.asarray(after))
    collision = [i for i in b.body_shapes[0] if b.shape_flags[i] & newton.ShapeFlags.COLLIDE_SHAPES]
    assert len(collision) == 44
    for i in collision:
        assert b.shape_material_mu[i] == pytest.approx(.37)
        assert b.shape_material_mu_torsional[i] == pytest.approx(.013)
        assert b.shape_material_restitution[i] == pytest.approx(.21)
        assert list(b.shape_scale[i]) == pytest.approx([.2, .2, .2])
    for i, original in originals.items():
        visual = b.shape_label.index(b.shape_label[i] + '_visual')
        assert b.shape_source[visual] is original
    filters = set(b._shape_collision_filter_pairs)
    joints = [i for i in collision if 'FrameJoints' in b.shape_label[i]]
    assert len(joints) == 12
    assert all(tuple(sorted((probe, i))) in filters for i in joints)


@pytest.mark.parametrize('change', ['vertices', 'scale'])
def test_changed_geometry_fails_before_decomposition_and_restores(monkeypatch, change):
    newton, b, coacd, ids, _ = native_builder(monkeypatch)
    if change == 'vertices':
        b.shape_source[ids[0]].vertices[0, 0] += .001
    else:
        b.shape_scale[ids[0]] = (1., 1., 1.)
    original = newton.ModelBuilder.approximate_meshes
    with pytest.raises(RuntimeError, match='(geometry changed|scale exceeds)'):
        with shelf.shelf_colliders(newton.ModelBuilder, coacd, ASSET):
            b.approximate_meshes('coacd', ids)
    assert newton.ModelBuilder.approximate_meshes is original
    assert not shelf._active


def test_support_planes_and_openings():
    pytest.importorskip("newton")
    audit = load("shelf_audit_under_test", ROOT / "scripts/audit_newton_shelf.py")
    report = json.loads(json.dumps(audit.audit(ROOT)))
    assert report["part_count"] == 44
    assert report["max_vertices"] <= 64
    assert report["max_deck_envelope_error_m"] < 1e-7
    assert min(report["inter_deck_clearances_m"]) > .115


def test_changed_compiled_asset_is_rejected(tmp_path):
    import shutil
    asset = tmp_path / ASSET.name
    shutil.copyfile(ASSET, asset)
    shutil.copyfile(ASSET.with_suffix(".json"), asset.with_suffix(".json"))
    asset.write_bytes(asset.read_bytes() + b"\n# unexpected change\n")
    with pytest.raises(ValueError, match="changed shelf collision asset"):
        shelf.load_colliders(asset)
