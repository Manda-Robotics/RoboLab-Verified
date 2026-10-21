"""Newton bagel defaults and composed USD assets, without Isaac Sim/Kit."""
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace as NS

import numpy as np
import pytest
from pxr import Usd, UsdGeom, UsdPhysics

ROOT = Path(__file__).resolve().parents[2]


def load(name, relative):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


bagels = load('bagel_defaults_under_test', 'robolab/core/physics/newton_bagels.py')
physics = load('bagel_physics_under_test', 'robolab/core/environments/physics.py')
SCENES = {
    'bagel_plate_banana_bowl.usda': {'bagel_00': 'bagel_00', 'bagel_06': 'bagel_06'},
    'breakfast_table.usda': {'bagel_07': 'bagel_06', 'bagel_01': 'bagel_00', 'bagel_02': 'bagel_06'},
    'test_plate_banana_rubiks_cube.usda': {'bagel_00': 'bagel_00', 'bagel_06': 'bagel_06'},
}


def config(scene):
    return NS(
        sim=NS(physics='physx', dt=1 / 120), actions=NS(),
        scene=NS(scene=NS(spawn=NS(usd_path=str(scene))),
                 robot=NS(spawn=NS(usd_path=str(ROOT / 'assets/robots/franka_robotiq_2f_85_isaac60.usd')),
                          actuators={'gripper': NS()})),
    )


@pytest.fixture
def backend(monkeypatch):
    for name, attrs in [
        ('isaaclab_newton.physics', dict(NewtonCfg=NS, MJWarpSolverCfg=NS)),
        ('robolab.constants', dict(ROBOTS_DIR=str(ROOT / 'assets/robots'))),
        ('robolab.robots.newton_gripper', dict(RateLimitedGripperActionCfg=NS)),
    ]:
        module = ModuleType(name)
        module.__dict__.update(attrs)
        monkeypatch.setitem(sys.modules, name, module)


@pytest.mark.parametrize('scene', SCENES)
def test_newton_selects_complete_bagel_preset_and_physx_is_unchanged(backend, scene):
    source = ROOT / 'assets/scenes' / scene
    physx, newton = config(source), config(source)
    physics.configure_physics(physx, 'physx')
    assert physx.scene.scene.spawn.usd_path == str(source)
    assert physx.sim.physics == 'physx'
    assert not hasattr(physx, 'newton_bagel_assets')
    physics.configure_physics(newton, 'newton_mjwarp')
    assert Path(newton.scene.scene.spawn.usd_path).name == bagels.SCENE_OVERLAYS[scene]
    assert newton.sim.physics.solver_cfg.cone == 'elliptic'
    assert newton.newton_contact_settings['cone'] == 'elliptic'
    assert newton.newton_bagel_assets['authored_sliding_friction'] == 5
    assert newton.sim.dt == 1 / 120 and newton.sim.physics.num_substeps == 4
    assert newton.newton_contact_settings['timeconst'] == .0045
    assert newton.newton_contact_settings['impedance'] == (.99, .999)
    assert newton.scene.robot.actuators['gripper'].stiffness == 10


def test_unrelated_and_custom_scenes_keep_their_existing_defaults(backend, tmp_path):
    for scene in [ROOT / 'assets/scenes/banana_on_plate.usda', tmp_path / 'bagel_plate_banana_bowl.usda']:
        cfg = config(scene)
        physics.configure_physics(cfg, 'newton_mjwarp')
        assert cfg.scene.scene.spawn.usd_path == str(scene)
        assert cfg.sim.physics.solver_cfg.cone == 'pyramidal'
        assert not hasattr(cfg, 'newton_bagel_assets')


def test_explicit_clean_scene_and_user_overrides_are_preserved(backend, monkeypatch):
    cfg = config(ROOT / 'assets/scenes/breakfast_table_clean.usda')
    cfg.newton_contact_settings = {'cone': 'pyramidal'}
    requested = {'mode': 'uniform', 'spec': '.7', 'objects': {'bagel_01': {'static': .7, 'dynamic': .7}}}
    cfg.friction = requested
    seen = []
    module = ModuleType('robolab.core.physics.newton_friction')
    module.configure = lambda c: seen.append(c.friction)
    monkeypatch.setitem(sys.modules, module.__name__, module)
    physics.configure_physics(cfg, 'newton_mjwarp')
    assert cfg.sim.physics.solver_cfg.cone == 'pyramidal'
    assert seen == [requested] and cfg.friction == requested
    assert cfg.newton_bagel_assets['scene'] == 'breakfast_table_clean.usda'


def test_missing_overlay_fails_before_spawning(monkeypatch, tmp_path):
    monkeypatch.setattr(bagels, 'SCENE_DIR', tmp_path)
    with pytest.raises(FileNotFoundError, match='bagel scene overlay missing'):
        bagels.configure_scene(config(tmp_path / 'breakfast_table.usda'))


@pytest.mark.parametrize('name', ['bagel_00', 'bagel_06'])
def test_clean_asset_keeps_visuals_inertia_and_imports_full_hulls(name):
    folder = ROOT / 'assets/objects/objaverse'
    original = Usd.Stage.Open(str(folder / f'{name}.usd'))
    clean = Usd.Stage.Open(str(folder / f'{name}_clean.usda'))
    reference = json.loads((folder / 'bagel_clean_reference_inertia.json').read_text())
    assert hashlib.sha256((folder / f'{name}.usd').read_bytes()).hexdigest() == reference['source_asset_sha256'][name]
    assert sum(p.HasAPI(UsdPhysics.CollisionAPI) for p in Usd.PrimRange(original.GetDefaultPrim())) == 32
    colliders = [p for p in Usd.PrimRange(clean.GetDefaultPrim()) if p.HasAPI(UsdPhysics.CollisionAPI)]
    assert len(colliders) == 64
    for prim in colliders:
        assert '/CleanCollisions/sector_' in str(prim.GetPath())
        count = len(UsdGeom.Mesh(prim).GetPointsAttr().Get())
        assert 64 < count <= prim.GetAttribute('newton:maxHullVertices').Get() == 128
        assert 'NewtonMeshCollisionAPI' in prim.GetMetadata('apiSchemas').GetAppliedItems()
        targets = prim.GetRelationship('material:binding:physics').GetTargets()
        assert len(targets) == 1
        material = UsdPhysics.MaterialAPI(clean.GetPrimAtPath(targets[0]))
        old_material = UsdPhysics.MaterialAPI(original.GetPrimAtPath(targets[0]))
        assert material.GetDynamicFrictionAttr().Get() == material.GetStaticFrictionAttr().Get() == 5
        assert old_material.GetDynamicFrictionAttr().Get() == old_material.GetStaticFrictionAttr().Get() == 10
        for attr in old_material.GetPrim().GetAttributes():
            if attr.GetName() not in ('physics:staticFriction', 'physics:dynamicFriction'):
                assert material.GetPrim().GetAttribute(attr.GetName()).Get() == attr.Get()
    visual_count = 0
    for prim in Usd.PrimRange(original.GetDefaultPrim()):
        if prim.IsA(UsdGeom.Mesh) and not prim.HasAPI(UsdPhysics.CollisionAPI):
            visual_count += 1
            other = clean.GetPrimAtPath(prim.GetPath())
            for attr in prim.GetAttributes():
                assert other.GetAttribute(attr.GetName()).Get() == attr.Get()
            for rel in prim.GetRelationships():
                assert other.GetRelationship(rel.GetName()).GetTargets() == rel.GetTargets()
    assert visual_count > 0
    api = UsdPhysics.MassAPI(clean.GetDefaultPrim())
    inertial = reference['assets'][name]
    assert api.GetMassAttr().Get() == pytest.approx(inertial['mass'])
    np.testing.assert_allclose(api.GetCenterOfMassAttr().Get(), inertial['ipos'], atol=1e-9)
    np.testing.assert_allclose(api.GetDiagonalInertiaAttr().Get(), inertial['inertia'], rtol=1e-6)
    q = api.GetPrincipalAxesAttr().Get()
    np.testing.assert_allclose([q.GetReal(), *q.GetImaginary()], inertial['iquat'], atol=1e-7)


@pytest.mark.parametrize('scene,instances', SCENES.items())
def test_scene_overlays_preserve_placements_and_replace_aliased_bagels(scene, instances):
    original = Usd.Stage.Open(str(ROOT / 'assets/scenes' / scene))
    clean = Usd.Stage.Open(str(ROOT / 'assets/scenes' / bagels.SCENE_OVERLAYS[scene]))
    assert clean.GetDefaultPrim().GetName() == original.GetDefaultPrim().GetName()
    for name, asset in instances.items():
        prim = clean.GetPrimAtPath(f'/world/{name}')
        old = original.GetPrimAtPath(prim.GetPath())
        assert [p.assetPath for p in prim.GetMetadata('payload').GetAppliedItems()] == [f'../objects/objaverse/{asset}_clean.usda']
        assert sum(p.HasAPI(UsdPhysics.CollisionAPI) for p in Usd.PrimRange(prim)) == 64
        for attr in old.GetAttributes():
            if attr.GetName().startswith('xformOp') or attr.GetName() in ('physics:velocity', 'physics:angularVelocity'):
                assert prim.GetAttribute(attr.GetName()).Get() == attr.Get()
    # Other objects must retain their complete authored subtree, including
    # collision geometry and material values, rather than just the root pose.
    for prim in original.Traverse():
        if any(prim.GetPath().HasPrefix(f'/world/{name}') for name in instances):
            continue
        other = clean.GetPrimAtPath(prim.GetPath())
        assert other.IsValid()
        for attr in prim.GetAttributes():
            assert other.GetAttribute(attr.GetName()).Get() == attr.Get()
        for rel in prim.GetRelationships():
            assert other.GetRelationship(rel.GetName()).GetTargets() == rel.GetTargets()


def test_every_bundled_scene_referencing_these_assets_has_an_overlay():
    users = {p.name for p in (ROOT / 'assets/scenes').rglob('*.usda')
             if any(f'@../objects/objaverse/{name}.usd@' in p.read_text()
                    for name in ['bagel_00', 'bagel_06'])}
    assert users == set(SCENES) == set(bagels.SCENE_OVERLAYS)
