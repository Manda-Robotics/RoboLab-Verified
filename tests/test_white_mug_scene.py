# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""USD-only regression for the missing WhiteMug distractors (issue #43).

Run without Isaac: python -m pytest --noconftest tests/test_white_mug_scene.py
Requires usd-core and the scene's Git LFS object/fixture payloads.
Geometry checks do not replace settling the scene in RobolabEnv.
"""

import json
from pathlib import Path

import pytest
from pxr import Usd, UsdGeom, UsdPhysics

ROOT = Path(__file__).resolve().parents[1]
SCENE = "objects_around_table.usda"


@pytest.fixture
def stage():
    stage = Usd.Stage.Open(str(ROOT / "assets/scenes" / SCENE), load=Usd.Stage.LoadNone)
    # Load just the table and task objects, not the unrelated robot stand.
    names = ("table", "bowl", "banana", "mug", "alphabet_soup_can",
             "orange_juice_carton", "smartphone", "milk_carton", "rubiks_cube")
    for name in names:
        stage.GetDefaultPrim().GetChild(name).Load()
    return stage


def bounds(stage, name):
    prim = stage.GetDefaultPrim().GetChild(name)
    result = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_]).ComputeWorldBound(prim).ComputeAlignedRange()
    assert not result.IsEmpty(), f"Missing geometry for {name}; fetch Git LFS payloads"
    return result


@pytest.mark.parametrize("name", ["bowl", "banana"])
def test_distractor_rests_on_table_with_active_collision(stage, name):
    prim = stage.GetDefaultPrim().GetChild(name)
    body = UsdPhysics.RigidBodyAPI(prim)
    assert body and body.GetRigidBodyEnabledAttr().Get()
    assert tuple(body.GetVelocityAttr().Get()) == (0, 0, 0)
    assert tuple(body.GetAngularVelocityAttr().Get()) == (0, 0, 0)
    colliders = [p for p in Usd.PrimRange(prim)
                 if p.IsA(UsdGeom.Mesh) and p.HasAPI(UsdPhysics.CollisionAPI)
                 and UsdPhysics.CollisionAPI(p).GetCollisionEnabledAttr().Get()]
    assert colliders, f"{name} has no active collision mesh"
    table, obj = bounds(stage, "table"), bounds(stage, name)
    gap = obj.GetMin()[2] - table.GetMax()[2]
    assert 0 <= gap <= .001, f"{name} bottom is {gap} m from tabletop"
    for axis in (0, 1):
        assert table.GetMin()[axis] < obj.GetMin()[axis] < obj.GetMax()[axis] < table.GetMax()[axis]


@pytest.mark.parametrize("name", ["bowl", "banana"])
def test_restored_distractor_does_not_overlap_other_objects(stage, name):
    obj = bounds(stage, name)
    for other in ("bowl", "banana", "mug", "alphabet_soup_can",
                  "orange_juice_carton", "smartphone", "milk_carton", "rubiks_cube"):
        if other == name:
            continue
        target = bounds(stage, other)
        assert any(obj.GetMax()[axis] < target.GetMin()[axis]
                   or target.GetMax()[axis] < obj.GetMin()[axis]
                   for axis in range(3)), f"{name} overlaps {other}"


@pytest.mark.parametrize("name", ["bowl", "banana"])
def test_metadata_matches_authored_pose(stage, name):
    entries = json.loads((ROOT / "assets/scenes/_metadata/scene_metadata.json").read_text())[SCENE]
    entry = next(item for item in entries if item["name"] == name)
    prim = stage.GetDefaultPrim().GetChild(name)
    assert entry["position"] == pytest.approx(tuple(prim.GetAttribute("xformOp:translate").Get()))
    q = prim.GetAttribute("xformOp:orient").Get()
    assert entry["rotation"] == pytest.approx((q.GetReal(), *q.GetImaginary()))
