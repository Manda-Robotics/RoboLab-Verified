# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Structural checks on the bimanual YAM USD (pure pxr, no Isaac).

The asset is authored by ``assets/robots/_utils/build_bimanual_yam.py`` from I2RT's URDF;
these tests pin what the robot cfg relies on: names, DOF count, joint limits, finger travel,
the merged wrist-camera frame, and that the USD hierarchy composes to the URDF's forward
kinematics.
"""
import math
import os
import pathlib
import re
import sys

import numpy as np
import pytest

pxr = pytest.importorskip("pxr")
from pxr import Sdf, Usd, UsdGeom, UsdPhysics  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
USD = os.path.join(ROOT, "assets", "robots", "bimanual_yam", "bimanual_yam.usd")
sys.path.insert(0, os.path.join(ROOT, "assets", "robots", "_utils"))
import build_bimanual_yam as bb  # noqa: E402

def _is_real_usd(path: str) -> bool:
    """False when the file is absent or a Git LFS pointer stub (a checkout without `git lfs pull`)."""
    if not os.path.exists(path) or os.path.getsize(path) < 4096:
        return False
    with open(path, "rb") as f:
        return not f.read(24).startswith(b"version https://git-lfs")


pytestmark = pytest.mark.skipif(not _is_real_usd(USD), reason="bimanual_yam.usd not built or an LFS pointer")


@pytest.fixture(scope="module")
def stage():
    return Usd.Stage.Open(USD)


def _bodies(stage):
    return {p.GetName(): p for p in stage.Traverse() if p.HasAPI(UsdPhysics.RigidBodyAPI)}


def _joints(stage):
    return {p.GetName(): p for p in stage.Traverse() if p.IsA(UsdPhysics.Joint)}


def test_single_articulation_root(stage):
    roots = [p.GetPath() for p in stage.Traverse() if p.HasAPI(UsdPhysics.ArticulationRootAPI)]
    assert roots == [Sdf.Path("/robot")]


def test_bodies_and_dof(stage):
    bodies, joints = _bodies(stage), _joints(stage)
    for side in ("left", "right"):
        for n in ("base", "link1", "link2", "link3", "link4", "link5", "gripper", "tip_left", "tip_right"):
            assert f"{side}_{n}" in bodies
    assert "torso" in bodies
    assert len(bodies) == 19
    dof = [j for j in joints.values() if j.IsA(UsdPhysics.RevoluteJoint) or j.IsA(UsdPhysics.PrismaticJoint)]
    assert len(dof) == 16, "6 arm + 2 finger joints per arm"


def test_every_body_has_positive_mass(stage):
    for name, prim in _bodies(stage).items():
        assert UsdPhysics.MassAPI(prim).GetMassAttr().Get() > 0, name


def test_arm_joint_limits_match_urdf(stage):
    links, urdf_joints, _ = bb.parse_urdf(bb.SOURCE_URDF)
    joints = _joints(stage)
    for j in urdf_joints:
        if j.type != "revolute":
            continue
        for side in ("left", "right"):
            prim = joints[f"{side}_{j.name}"]
            rj = UsdPhysics.RevoluteJoint(prim)
            assert math.isclose(rj.GetLowerLimitAttr().Get(), math.degrees(j.lower), abs_tol=1e-3)
            assert math.isclose(rj.GetUpperLimitAttr().Get(), math.degrees(j.upper), abs_tol=1e-3)


def test_finger_joints_are_prismatic_closed_at_zero(stage):
    joints = _joints(stage)
    for side in ("left", "right"):
        for n in ("joint7", "joint8"):
            pj = UsdPhysics.PrismaticJoint(joints[f"{side}_{n}"])
            assert pj, f"{side}_{n} is not prismatic"
            assert math.isclose(pj.GetLowerLimitAttr().Get(), -0.04695, abs_tol=1e-5)
            assert math.isclose(pj.GetUpperLimitAttr().Get(), 0.0, abs_tol=1e-6)


def test_client_finger_travel_matches_asset(stage):
    """policies/molmoact2/yam_client.py repeats FINGER_TRAVEL_M so it imports without Isaac."""
    src = pathlib.Path(ROOT, "policies", "molmoact2", "yam_client.py").read_text()
    client_travel = float(re.search(r"^FINGER_TRAVEL_M = ([0-9.]+)", src, re.M).group(1))
    lower = UsdPhysics.PrismaticJoint(_joints(stage)["left_joint7"]).GetLowerLimitAttr().Get()
    assert math.isclose(-lower, client_travel, abs_tol=1e-6)


def test_joints_reference_rigid_bodies(stage):
    for prim in _joints(stage).values():
        j = UsdPhysics.Joint(prim)
        for rel in (j.GetBody0Rel(), j.GetBody1Rel()):
            for t in rel.GetTargets():
                assert stage.GetPrimAtPath(t).HasAPI(UsdPhysics.RigidBodyAPI), f"{prim.GetPath()} -> {t}"


def test_arms_are_mirrored_about_x(stage):
    bodies = _bodies(stage)
    tc = Usd.TimeCode.Default()
    for n in ("base", "gripper"):
        left = UsdGeom.Xformable(bodies[f"left_{n}"]).ComputeLocalToWorldTransform(tc).ExtractTranslation()
        right = UsdGeom.Xformable(bodies[f"right_{n}"]).ComputeLocalToWorldTransform(tc).ExtractTranslation()
        assert math.isclose(left[1] - right[1], bb.ARM_SPACING_M, abs_tol=1e-6)
        assert math.isclose(left[0], right[0], abs_tol=1e-9) and math.isclose(left[2], right[2], abs_tol=1e-9)


def test_hierarchy_composes_to_urdf_fk(stage):
    """Every link's USD world transform at the zero pose equals the URDF FK (+ the arm offset)."""
    links, _, root_name = bb.parse_urdf(bb.SOURCE_URDF)
    t = bb.fk(links, root_name, {})
    bodies = _bodies(stage)
    tc = Usd.TimeCode.Default()
    for side, y in bb.SIDES:
        for name in ("base", "link1", "link2", "link3", "link4", "link5", "gripper", "tip_left", "tip_right"):
            want = t[name][:3, 3] + np.array([0.0, y, 0.0])
            got = np.array(UsdGeom.Xformable(bodies[f"{side}_{name}"]).ComputeLocalToWorldTransform(tc).ExtractTranslation())
            assert np.allclose(got, want, atol=1e-5), f"{side}_{name}: {got} vs {want}"


def test_wrist_camera_frame_is_merged_into_gripper(stage):
    """The camera frame is an Xform under the gripper *body* (not a fixed-jointed link), and its
    pose matches I2RT's published extrinsics: flange -> camera (-0.07, 0, -0.077)."""
    for side in ("left", "right"):
        gripper = _bodies(stage)[f"{side}_gripper"]
        cam = gripper.GetChild(f"{side}_camera")
        assert cam, f"{side}_camera must be a direct child of {side}_gripper"
        assert not cam.HasAPI(UsdPhysics.RigidBodyAPI)
        tc = Usd.TimeCode.Default()
        g = UsdGeom.Xformable(gripper).ComputeLocalToWorldTransform(tc)
        c = UsdGeom.Xformable(cam).ComputeLocalToWorldTransform(tc)
        rel = c * g.GetInverse()
        assert np.allclose(np.array(rel.ExtractTranslation()), [-0.0704, 0.0, -0.077], atol=1e-3)


def test_collision_shapes_where_grasping_happens(stage):
    for side in ("left", "right"):
        for n in ("gripper", "tip_left", "tip_right"):
            body = _bodies(stage)[f"{side}_{n}"]
            coll = [c for c in body.GetChildren() if c.HasAPI(UsdPhysics.CollisionAPI)]
            assert coll, f"{side}_{n} has no collision mesh"
            approx = UsdPhysics.MeshCollisionAPI(coll[0]).GetApproximationAttr().Get()
            assert approx == "convexDecomposition", f"{side}_{n}: {approx}"
