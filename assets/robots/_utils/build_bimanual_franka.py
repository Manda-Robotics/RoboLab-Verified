# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Build the bimanual Franka + Robotiq 2F-85 asset from the single-arm DROID asset.

Two copies of ``/panda`` from ``franka_robotiq_2f_85_flattened.usd`` are mounted on a
shared torso link inside ONE articulation. RoboLab's recorders, ground-truth export and
metrics all assume a single articulation named ``robot``, so two separate arm
articulations are not an option; and IsaacLab resolves joints and bodies by *name*
(the last path component), so the two copies cannot keep identical names either.
Every rigid body and joint prim is therefore renamed with a ``left_arm_`` /
``right_arm_`` prefix, which is the naming ``docs/robots.md`` already sketches for
bimanual robots.

Only pure-Python ``pxr`` is needed (``pip install usd-core``); Isaac Sim is not.

    python assets/robots/_utils/build_bimanual_franka.py

Layout of the output::

    /robot                        Xform, PhysicsArticulationRootAPI (the only one)
    /robot/torso                  rigid body the arms mount on, fixed to the world
    /robot/left_arm               Xform at (0, +ARM_Y, 0); bodies/joints prefixed left_arm_
    /robot/right_arm              Xform at (0, -ARM_Y, 0); bodies/joints prefixed right_arm_
    /robot/root_joint             fixed joint: torso -> world
    /robot/left_mount_joint       fixed joint: torso -> left_arm_panda_link0
    /robot/right_mount_joint      fixed joint: torso -> right_arm_panda_link0
"""

from __future__ import annotations

import argparse
import os
import sys

from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics

HERE = os.path.dirname(os.path.abspath(__file__))
ROBOTS_DIR = os.path.dirname(HERE)
SOURCE_USD = os.path.join(ROBOTS_DIR, "franka_robotiq_2f_85_flattened.usd")
OUTPUT_DIR = os.path.join(ROBOTS_DIR, "bimanual_franka_robotiq_2f85")
OUTPUT_USD = os.path.join(OUTPUT_DIR, "bimanual_franka_robotiq_2f85.usd")

# Half the base-to-base distance. 0.60 m apart keeps both arms inside the 1.0 m wide
# task table and both reach the shared workspace centre (0.55, 0): a Franka reaches
# ~0.85 m, and the far edge of the workspace is ~0.78 m from either base.
ARM_Y = 0.30
ARMS = {"left_arm": +ARM_Y, "right_arm": -ARM_Y}

# Visual-only mounting plate under both bases (no collider: it would sit inside the
# table fixture's collision mesh and only add solver noise to a fixed-base robot).
TORSO_SIZE = (0.22, 2 * ARM_Y + 0.30, 0.02)


def _is_body_or_joint(spec: Sdf.PrimSpec) -> bool:
    if "Joint" in spec.typeName:
        return True
    return any("RigidBodyAPI" in api for api in spec.GetInfo("apiSchemas").GetAddedOrExplicitItems())


def _collect_renames(layer: Sdf.Layer, root: Sdf.Path, prefix: str) -> dict[Sdf.Path, str]:
    renames: dict[Sdf.Path, str] = {}

    def visit(spec: Sdf.PrimSpec):
        for child in spec.nameChildren:
            if _is_body_or_joint(child):
                name = child.name
                # The Robotiq links carry two joints both just called "FixedJoint";
                # qualify them by parent so every joint name in the file is unique.
                if name == "FixedJoint":
                    name = f"{spec.name}_fixed_joint"
                renames[child.path] = prefix + name
            visit(child)

    visit(layer.GetPrimAtPath(root))
    return renames


def _remap_path(path: Sdf.Path, renames: dict[Sdf.Path, str]) -> Sdf.Path:
    """Rebuild ``path`` component by component, substituting renamed prims."""
    prim_path = path.GetPrimPath()
    new = Sdf.Path.absoluteRootPath
    cur = Sdf.Path.absoluteRootPath
    for name in prim_path.pathString.strip("/").split("/"):
        cur = cur.AppendChild(name)
        new = new.AppendChild(renames.get(cur, name))
    if path.IsPropertyPath():
        new = new.AppendProperty(path.name)
    return new


def _remap_targets(layer: Sdf.Layer, root: Sdf.Path, renames: dict[Sdf.Path, str]) -> int:
    """Rewrite relationship targets and attribute connections under ``root``."""
    count = 0

    def rewrite(list_proxy):
        nonlocal count
        for key in ("explicitItems", "addedItems", "prependedItems", "appendedItems",
                    "deletedItems", "orderedItems"):
            old = list(getattr(list_proxy, key))
            if old:
                setattr(list_proxy, key, [_remap_path(p, renames) for p in old])
                count += len(old)

    def visit(spec: Sdf.PrimSpec):
        for prop in spec.properties:
            if isinstance(prop, Sdf.RelationshipSpec):
                rewrite(prop.targetPathList)
            elif isinstance(prop, Sdf.AttributeSpec):
                rewrite(prop.connectionPathList)
        for child in spec.nameChildren:
            visit(child)

    visit(layer.GetPrimAtPath(root))
    return count


def _apply_renames(layer: Sdf.Layer, renames: dict[Sdf.Path, str]) -> None:
    # Deepest first so a parent's rename never invalidates a pending child edit.
    edit = Sdf.BatchNamespaceEdit()
    for old in sorted(renames, key=lambda p: -p.pathElementCount):
        edit.Add(old, old.GetParentPath().AppendChild(renames[old]))
    if not layer.Apply(edit):
        raise RuntimeError("namespace edit failed")


def _add_fixed_joint(stage: Usd.Stage, path: str, body0: str, body1: str | None,
                     local_pos0=(0.0, 0.0, 0.0)) -> None:
    joint = UsdPhysics.FixedJoint.Define(stage, path)
    joint.CreateBody0Rel().SetTargets([Sdf.Path(body0)])
    if body1 is not None:
        joint.CreateBody1Rel().SetTargets([Sdf.Path(body1)])
    joint.CreateLocalPos0Attr().Set(Gf.Vec3f(*local_pos0))
    joint.CreateLocalRot0Attr().Set(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
    joint.CreateLocalPos1Attr().Set(Gf.Vec3f(0.0, 0.0, 0.0))
    joint.CreateLocalRot1Attr().Set(Gf.Quatf(1.0, 0.0, 0.0, 0.0))


def build(source: str = SOURCE_USD, output: str = OUTPUT_USD) -> str:
    # The source is a single flattened layer: /panda plus the instancing prototypes
    # its meshes reference internally, next to stray scene prims (/banana, /bowl,
    # /table, ...) that are never spawned. Copy only what the arm needs.
    src_layer = Sdf.Layer.FindOrOpen(source)
    if src_layer is None or src_layer.subLayerPaths:
        raise RuntimeError(f"expected a single flattened layer at {source}")
    prototypes = [p.name for p in src_layer.rootPrims if p.name.startswith("Flattened_Prototype")]

    os.makedirs(os.path.dirname(output), exist_ok=True)
    stage = Usd.Stage.CreateNew(output)
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    layer = stage.GetRootLayer()

    robot = UsdGeom.Xform.Define(stage, "/robot")
    stage.SetDefaultPrim(robot.GetPrim())
    UsdPhysics.ArticulationRootAPI.Apply(robot.GetPrim())

    torso = UsdGeom.Xform.Define(stage, "/robot/torso")
    UsdPhysics.RigidBodyAPI.Apply(torso.GetPrim())
    UsdPhysics.MassAPI.Apply(torso.GetPrim()).CreateMassAttr().Set(5.0)
    plate = UsdGeom.Cube.Define(stage, "/robot/torso/plate")
    plate.CreateSizeAttr().Set(1.0)
    plate_xf = UsdGeom.Xformable(plate.GetPrim())
    plate_xf.AddTranslateOp().Set(Gf.Vec3d(0.0, 0.0, -TORSO_SIZE[2] / 2))
    plate_xf.AddScaleOp().Set(Gf.Vec3f(*TORSO_SIZE))
    plate.CreateDisplayColorAttr().Set([Gf.Vec3f(0.25, 0.25, 0.27)])

    # Mesh geometry is instanced: both arms reference these once-copied prototypes,
    # so the bimanual file is not twice the size of the single-arm one.
    for name in prototypes:
        if not Sdf.CopySpec(src_layer, Sdf.Path(f"/{name}"), layer, Sdf.Path(f"/{name}")):
            raise RuntimeError(f"failed to copy prototype {name}")

    for arm, y in ARMS.items():
        dst = Sdf.Path(f"/robot/{arm}")
        if not Sdf.CopySpec(src_layer, Sdf.Path("/panda"), layer, dst):
            raise RuntimeError(f"failed to copy /panda to {dst}")
        arm_spec = layer.GetPrimAtPath(dst)

        # One articulation root only: strip the copy's own, and drop its world joint.
        apis = arm_spec.GetInfo("apiSchemas")
        apis.explicitItems = [a for a in apis.GetAddedOrExplicitItems()
                              if a != "PhysicsArticulationRootAPI"]
        arm_spec.SetInfo("apiSchemas", apis)
        root_joint = layer.GetPrimAtPath(dst.AppendChild("rootJoint"))
        if root_joint is not None:
            del arm_spec.nameChildren["rootJoint"]
        # Stale Isaac import metadata from before the Panda hand was swapped for the
        # Robotiq: it names links and joints that no longer exist. Dangling in the
        # source too; nothing reads it, and it would fail the dangling-target check.
        for stale in ("isaac:physics:robotJoints", "isaac:physics:robotjoints",
                      "isaac:physics:robotLinks"):
            if stale in arm_spec.properties:
                del arm_spec.properties[stale]

        prefix = f"{arm}_"
        renames = _collect_renames(layer, dst, prefix)
        n_targets = _remap_targets(layer, dst, renames)
        _apply_renames(layer, renames)

        arm_prim = stage.GetPrimAtPath(dst)
        xf = UsdGeom.Xformable(arm_prim)
        translate = next((op for op in xf.GetOrderedXformOps()
                          if op.GetOpType() == UsdGeom.XformOp.TypeTranslate), None)
        if translate is None:
            translate = xf.AddTranslateOp()
        translate.Set(Gf.Vec3d(0.0, y, 0.0))

        _add_fixed_joint(stage, f"/robot/{arm[:-4]}_mount_joint", "/robot/torso",
                         f"/robot/{arm}/{prefix}panda_link0", local_pos0=(0.0, y, 0.0))
        print(f"{arm}: renamed {len(renames)} prims, remapped {n_targets} targets")

    _add_fixed_joint(stage, "/robot/root_joint", "/robot/torso", None)

    layer.comment = ("Generated by assets/robots/_utils/build_bimanual_franka.py from "
                     "franka_robotiq_2f_85_flattened.usd. Do not edit by hand.")
    stage.GetRootLayer().Save()
    return output


def verify(path: str) -> None:
    """Structural checks the robot cfg relies on; raise on any violation."""
    stage = Usd.Stage.Open(path)
    roots, bodies, joints = [], [], []
    for prim in stage.Traverse():
        if prim.HasAPI(UsdPhysics.ArticulationRootAPI):
            roots.append(prim.GetPath())
        if prim.HasAPI(UsdPhysics.RigidBodyAPI):
            bodies.append(prim.GetName())
        if prim.IsA(UsdPhysics.Joint):
            joints.append(prim)
    assert roots == [Sdf.Path("/robot")], roots
    dup_bodies = {b for b in bodies if bodies.count(b) > 1}
    joint_names = [j.GetName() for j in joints]
    dup_joints = {j for j in joint_names if joint_names.count(j) > 1}
    assert not dup_bodies, f"duplicate body names: {dup_bodies}"
    assert not dup_joints, f"duplicate joint names: {dup_joints}"
    # every joint body target must resolve to a rigid body in this stage
    for j in joints:
        for rel in (UsdPhysics.Joint(j).GetBody0Rel(), UsdPhysics.Joint(j).GetBody1Rel()):
            for t in rel.GetTargets():
                tp = stage.GetPrimAtPath(t)
                assert tp and tp.HasAPI(UsdPhysics.RigidBodyAPI), f"{j.GetPath()} -> {t}"
    # dangling relationship targets / connections anywhere
    for prim in stage.Traverse():
        for rel in prim.GetRelationships():
            for t in rel.GetTargets():
                assert stage.GetObjectAtPath(t), f"dangling rel {rel.GetPath()} -> {t}"
        for attr in prim.GetAttributes():
            for c in attr.GetConnections():
                assert stage.GetObjectAtPath(c), f"dangling connection {attr.GetPath()} -> {c}"
    for arm in ARMS:
        for i in range(1, 8):
            assert f"{arm}_panda_joint{i}" in joint_names
        assert f"{arm}_finger_joint" in joint_names
        assert f"{arm}_base_link" in bodies
        assert f"{arm}_left_inner_finger" in bodies
    print(f"verify OK: {len(bodies)} bodies, {len(joints)} joints, root {roots[0]}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--source", default=SOURCE_USD)
    ap.add_argument("--output", default=OUTPUT_USD)
    ap.add_argument("--verify-only", action="store_true")
    args = ap.parse_args()
    if not args.verify_only:
        print("wrote", build(args.source, args.output))
    verify(args.output)
    sys.exit(0)
