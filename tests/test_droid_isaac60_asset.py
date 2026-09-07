# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
ASSET = REPO_ROOT / "assets" / "robots" / "franka_robotiq_2f_85_isaac60.usd"
DROID_CONFIG = REPO_ROOT / "robolab" / "robots" / "droid.py"
GRIPPER_ROOT = "/panda/Gripper/Robotiq_2F_85"
GRIPPER_LINKS = (
    "base_link",
    "left_outer_knuckle",
    "left_outer_finger",
    "left_inner_finger",
    "left_inner_knuckle",
    "right_outer_knuckle",
    "right_outer_finger",
    "right_inner_finger",
    "right_inner_knuckle",
)


def test_droid_config_uses_isaac60_rendering_asset():
    assert ASSET.is_file()
    source = DROID_CONFIG.read_text()
    assert '"franka_robotiq_2f_85_isaac60.usd"' in source


def test_isaac60_asset_preserves_robot_structure_without_instanceable_visuals():
    pytest.importorskip("pxr", reason="USD Python bindings are provided by Isaac Sim")
    from pxr import Usd, UsdGeom, UsdPhysics

    stage = Usd.Stage.Open(str(ASSET))
    assert stage
    assert str(stage.GetDefaultPrim().GetPath()) == "/panda"

    prims = list(stage.Traverse(Usd.TraverseInstanceProxies()))
    assert sum(prim.IsA(UsdPhysics.Joint) for prim in prims) == 18
    assert sum(prim.HasAPI(UsdPhysics.RigidBodyAPI) for prim in prims) == 20
    assert sum(prim.HasAPI(UsdPhysics.CollisionAPI) for prim in prims) == 28
    assert sum(prim.IsA(UsdGeom.Mesh) for prim in prims) == 22
    assert all(stage.GetPrimAtPath(f"{GRIPPER_ROOT}/{link}") for link in GRIPPER_LINKS)
    assert not [prim.GetPath() for prim in prims if prim.IsInstanceable()]
