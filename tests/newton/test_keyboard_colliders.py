# SPDX-License-Identifier: Apache-2.0
"""Keyboard contact boundary and native importer preservation, without Kit."""
import hashlib
import importlib.util
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


keyboard = load('keyboard_under_test', ROOT / 'robolab/core/environments/newton_keyboard.py')
build = load('keyboard_build', ROOT / 'scripts/build_newton_keyboard.py')
ASSET = ROOT / 'assets/objects/hot3d/keyboard_colliders.usda'


def test_compiled_geometry_matches_all_source_contact_surfaces():
    pytest.importorskip('newton')
    manifest, parts = keyboard.load_colliders(ASSET)
    source = ASSET.with_name('keyboard.usd')
    assert hashlib.sha256(source.read_bytes()).hexdigest() == manifest['source_asset_sha256']
    assert len(parts) == manifest['part_count'] == 110
    vertices, faces, components = build.source_components(source)
    entry = manifest['meshes'][build.MESH_NAME]
    assert entry['source_geometry_sha256'] == keyboard.geometry_key(vertices, faces)
    for (v, f), path in zip(components, entry['paths'], strict=True):
        compiled_v, compiled_f, inset = build.compile_component(v, f)
        np.testing.assert_array_equal(parts[path][0], compiled_v.astype(np.float32))
        np.testing.assert_array_equal(parts[path][1], compiled_f)
        assert inset < 2e-7
        assert len(compiled_v) <= 44


def native_builder(monkeypatch):
    newton = pytest.importorskip('newton')
    v, f, _ = build.source_components(ASSET.with_name('keyboard.usd'))
    coacd = ModuleType('coacd')
    coacd.Mesh = lambda v, f: NS(vertices=np.asarray(v, dtype=float), indices=np.asarray(f))
    def unexpected(*args, **kwargs):
        raise AssertionError('Compiled keyboard must not invoke CoACD')
    coacd.run_coacd = unexpected
    monkeypatch.setitem(sys.modules, 'coacd', coacd)
    b = newton.ModelBuilder()
    body = b.add_body(mass=.3, label='keyboard')
    probe = b.add_shape_sphere(b.add_body(mass=1., label='probe'), radius=.01)
    cfg = newton.ModelBuilder.ShapeConfig(density=800., mu=.37, restitution=.21)
    i = b.add_shape_mesh(body, mesh=newton.Mesh(v, f), cfg=cfg,
                         label='/World/envs/env_0/keyboard/obj_000028_Mesh')
    b.add_shape_collision_filter_pair(probe, i)
    return newton, b, coacd, i, probe


def test_native_conversion_preserves_mass_visuals_materials_filters_and_stacks(monkeypatch):
    newton, b, coacd, i, probe = native_builder(monkeypatch)
    shelf = load('shelf_stacked', ROOT / 'robolab/core/environments/newton_shelf.py')
    before = [np.array(x).copy() for x in (b.body_mass, b.body_com, b.body_inertia)]
    source = b.shape_source[i]
    original_method, original_decompose = newton.ModelBuilder.approximate_meshes, coacd.run_coacd
    with shelf.shelf_colliders(newton.ModelBuilder, coacd):
        with keyboard.keyboard_colliders(newton.ModelBuilder, coacd, ASSET) as reports:
            assert b.approximate_meshes('coacd', [i], keep_visual_shapes=True) == {i}
            assert reports[0]['parts'] == 110
            with pytest.raises(RuntimeError, match='nested'):
                with keyboard.keyboard_colliders(newton.ModelBuilder, coacd, ASSET):
                    pass
    assert newton.ModelBuilder.approximate_meshes is original_method and coacd.run_coacd is original_decompose
    for old, new in zip(before, (b.body_mass, b.body_com, b.body_inertia)):
        np.testing.assert_array_equal(old, new)
    collision = [j for j in b.body_shapes[0] if b.shape_flags[j] & newton.ShapeFlags.COLLIDE_SHAPES]
    assert len(collision) == 110
    filters = set(b._shape_collision_filter_pairs)
    for j in collision:
        assert b.shape_material_mu[j] == pytest.approx(.37)
        assert b.shape_material_restitution[j] == pytest.approx(.21)
        assert tuple(sorted((probe, j))) in filters
    assert b.shape_source[b.shape_label.index(b.shape_label[i]+'_visual')] is source


@pytest.mark.parametrize('change', ['vertices', 'scale'])
def test_changed_source_or_scale_fails_and_restores(monkeypatch, change):
    newton, b, coacd, i, _ = native_builder(monkeypatch)
    if change == 'vertices':
        b.shape_source[i].vertices[0, 0] += .001
    else:
        b.shape_scale[i] = (2., 2., 2.)
    original = newton.ModelBuilder.approximate_meshes
    with pytest.raises(RuntimeError, match='(geometry changed|scale exceeds)'):
        with keyboard.keyboard_colliders(newton.ModelBuilder, coacd, ASSET):
            b.approximate_meshes('coacd', [i])
    assert newton.ModelBuilder.approximate_meshes is original and not keyboard._active


def test_changed_compiled_asset_is_rejected(tmp_path):
    import shutil
    asset = tmp_path / ASSET.name
    shutil.copyfile(ASSET, asset)
    shutil.copyfile(ASSET.with_suffix('.json'), asset.with_suffix('.json'))
    asset.write_bytes(asset.read_bytes()+b'\n# changed\n')
    with pytest.raises(ValueError, match='changed keyboard collision asset'):
        keyboard.load_colliders(asset)
