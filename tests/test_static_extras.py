# SPDX-License-Identifier: Apache-2.0
"""Static configuration extras expose batched poses without authoring USD."""
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from pxr import Gf, Usd, UsdGeom

from isaaclab.assets import AssetBaseCfg
import robolab.core.world.world_state as world_module


def test_static_extra_pose_origin_quaternion_and_cache(monkeypatch):
    views = pytest.importorskip('isaaclab.sim.views')
    if not hasattr(views, 'UsdFrameView'):
        pytest.skip('Config-only extras are specific to Isaac Lab 3')
    view_type = views.UsdFrameView
    stage = Usd.Stage.CreateInMemory()
    c = 2 ** -.5
    for i, origin in enumerate([-2., 2.]):
        parent = UsdGeom.Xform.Define(stage, f'/World/envs/env_{i}')
        parent.AddTranslateOp().Set(Gf.Vec3d(origin, 0., 0.))
        rack = UsdGeom.Xform.Define(stage, f'/World/envs/env_{i}/rack')
        rack.AddTranslateOp().Set(Gf.Vec3d(.4, .5, .6))
        rack.AddOrientOp().Set(Gf.Quatf(c, 0., 0., c))
        rack.AddScaleOp(precision=UsdGeom.XformOp.PrecisionFloat).Set(Gf.Vec3f(2., 1., .5))
    before = stage.GetRootLayer().ExportToString()
    made = []
    def factory(*args, **kwargs):
        assert kwargs['validate_xform_ops'] is False
        view = view_type(*args, stage=stage, **kwargs)
        made.append(view)
        return view
    monkeypatch.setattr(views, 'UsdFrameView', factory)
    cfg = AssetBaseCfg(prim_path='/World/envs/env_.*/rack', spawn=None,
                       init_state=AssetBaseCfg.InitialStateCfg(rot=(1., 0., 0., 0.)))
    class Environment:
        device = 'cpu'
        num_envs = 2
        env_origins = torch.tensor([[-2., 0., 0.], [2., 0., 0.]])
        scene = SimpleNamespace(env_origins=env_origins, extras={'rack': cfg}, rigid_objects={}, articulations={}, deformable_objects={})
    world = world_module.WorldState(Environment())
    pos, quat = world.get_pose('rack')
    np.testing.assert_allclose(pos, [[.4, .5, .6]]*2, atol=1e-6)
    np.testing.assert_allclose(quat, [[c, 0., 0., c]]*2, atol=1e-6)
    pos_world, _ = world.get_pose('rack', is_relative=False)
    np.testing.assert_allclose(pos_world, [[-1.6, .5, .6], [2.4, .5, .6]], atol=1e-6)
    np.testing.assert_allclose(world.get_pose('rack', env_id=1)[0], [.4, .5, .6], atol=1e-6)
    np.testing.assert_array_equal(world.get_velocity('rack'), np.zeros((2, 6)))
    assert world._get_prim('rack', env_id=1).GetPath().pathString == '/World/envs/env_1/rack'
    assert len(made) == 1 and world.get_body('rack') is made[0]
    assert world.env.scene.extras['rack'] is cfg
    assert stage.GetRootLayer().ExportToString() == before
    world.set_world(Environment())
    assert world.get_body('rack') is made[-1] and len(made) == 2


def test_existing_extra_views_are_returned_unchanged():
    view = object()
    env = SimpleNamespace(scene=SimpleNamespace(extras={'rack': view}, rigid_objects={},
                                               articulations={}, deformable_objects={}))
    world = world_module.WorldState(env)
    assert world.get_body('rack') is view
