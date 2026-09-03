#!/usr/bin/env python
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Build the bimanual YAM articulation USD from I2RT's URDF, with plain USD (no Isaac).

Why not the Isaac URDF importer: the ALOHA import produced degenerate joint frames and a
wrist camera under a fixed-jointed link that PhysX merged away (the camera never moved).
Authoring the stage directly keeps every frame explicit and testable offline with pxr.

Source: ``assets/robots/yam_i2rt_v1/yam_linear_4310_d405.urdf`` (I2RT, YAM v1 + linear_4310
gripper + D405 wrist bracket). Layout copies Ai2's MolmoAct 2 rig as encoded in their
ManiSkill sim (``sim_eval/robots/bimanual_yam.py``): bases 0.48 m apart on the mount plane.

Stage layout::

    /robot                                Xform, PhysicsArticulationRootAPI (the only one)
    /robot/torso                          rigid body, the 2060 profile, fixed to the world
    /robot/left_arm                       Xform at (0, +0.24, 0)
    /robot/left_arm/left_base             rigid body (URDF ``base``), fixed to torso
    /robot/left_arm/left_link1 ...        every link a *sibling* under the arm Xform, posed at
                                          its zero-pose FK transform. PhysX drops rigid bodies
                                          nested inside other rigid bodies (no xformstack
                                          reset), so the tree is flat like the Isaac importer's;
                                          joint prims live under the parent link
    /robot/left_arm/left_gripper          URDF ``gripper``; the fixed camera chain
                                          (bracket, D405 body, cover, cable holder, ``camera``
                                          frame) is merged into it as plain Xforms/meshes
         /left_gripper/left_tip_left, left_tip_right   prismatic fingers (0 = closed,
                                          -0.04695 = open, as in the URDF and Ai2's sim)
    /robot/right_arm ...                  mirror, prefix ``right_``

Body and joint names carry the ``left_``/``right_`` prefix because Isaac Lab resolves both by
their last path component and rejects duplicates.

Run::

    python assets/robots/_utils/build_bimanual_yam.py            # writes the USD, verifies it
    python assets/robots/_utils/build_bimanual_yam.py --fk       # also print rest-pose FK
"""
from __future__ import annotations

import argparse
import math
import os
import struct
import xml.etree.ElementTree as ET

import numpy as np
from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics, UsdShade

HERE = os.path.dirname(os.path.abspath(__file__))
ROBOTS_DIR = os.path.dirname(HERE)
SOURCE_DIR = os.path.join(ROBOTS_DIR, "yam_i2rt_v1")
SOURCE_URDF = os.path.join(SOURCE_DIR, "yam_linear_4310_d405.urdf")
OUTPUT_USD = os.path.join(ROBOTS_DIR, "bimanual_yam", "bimanual_yam.usd")

ARM_SPACING_M = 0.48          # Ai2 sim: left_arm at y=+0.24, right_arm at y=-0.24
# Colours from Ai2's rig photo (arXiv 2605.02881, Fig. "setup"): the upper-arm and forearm shells
# (link2, link3) are white; base, joints, wrist links and gripper are black.
ARM_RGB = (0.06, 0.06, 0.065)
SHELL_RGB = (0.88, 0.88, 0.87)
WHITE_LINKS = {"link2", "link3"}
SIDES = (("left", +ARM_SPACING_M / 2), ("right", -ARM_SPACING_M / 2))
# The 2060 profile the arms clamp to (Ai2 kit: 80 cm). Visual + a little mass; fixed to world.
PROFILE_SIZE = (0.06, 0.80, 0.02)
PROFILE_MASS_KG = 3.0

# Actuator defaults, from Ai2's ManiSkill config (which mirrors I2RT's MJCF actuator classes):
# dm4340 on joints 1-3, joint 4 softer, dm4310 on joints 5-6. Isaac Lab's ImplicitActuatorCfg
# overrides these at spawn; they are here so the bare USD also behaves.
ARM_DRIVE = {  # joint -> (stiffness, damping, max_force)
    "joint1": (40.0, 2.5, 28.0), "joint2": (40.0, 2.5, 28.0), "joint3": (40.0, 2.5, 28.0),
    "joint4": (20.0, 0.5, 10.0), "joint5": (10.0, 1.0, 10.0), "joint6": (10.0, 1.0, 10.0),
}
FINGER_DRIVE = (2000.0, 40.0, 40.0)
# Finger-pad physics material. Ai2's ManiSkill config overrides the fingers to static 3.0 /
# dynamic 2.5 "so fingers don't slip"; without a material PhysX uses 0.5 and the first benchmark
# probe logged 42-103 dropped-object events per FoodPacking episode.
FINGER_FRICTION = (3.0, 2.5)   # static, dynamic

# Links merged into their parent rigid body (fixed joints): the D405 chain hangs off ``gripper``.
MERGE_FIXED = True
# Collision approximation per link. Fingers and gripper body need faithful shapes for grasping;
# arm links only need to hit the table.
CONVEX_DECOMP = {"gripper", "tip_left", "tip_right"}
NO_COLLISION = {"camera", "camera_body", "camera_cover", "camera_cable_holder", "camera_bracket"}

AXIS_FIX = {
    # URDF axis -> (USD axis token, extra rotation applied to BOTH joint frames so that the
    # signed motion stays the URDF's). Rotating both frames by the same R leaves the joint's
    # geometry untouched and maps USD's +axis onto the URDF's signed axis.
    (1, 0, 0): ("X", None), (0, 1, 0): ("Y", None), (0, 0, 1): ("Z", None),
    (-1, 0, 0): ("X", Gf.Rotation(Gf.Vec3d(0, 0, 1), 180.0)),
    (0, -1, 0): ("Y", Gf.Rotation(Gf.Vec3d(0, 0, 1), 180.0)),
    (0, 0, -1): ("Z", Gf.Rotation(Gf.Vec3d(1, 0, 0), 180.0)),
}


# ----------------------------------------------------------------------------- URDF parsing
def _vec(s: str | None, n: int = 3, default=0.0) -> np.ndarray:
    if not s:
        return np.full(n, default, dtype=float)
    return np.array([float(v) for v in s.split()], dtype=float)


def _rpy_to_mat(rpy: np.ndarray) -> np.ndarray:
    r, p, y = rpy
    cr, sr, cp, sp, cy, sy = math.cos(r), math.sin(r), math.cos(p), math.sin(p), math.cos(y), math.sin(y)
    rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    return rz @ ry @ rx  # URDF: R = Rz(yaw) Ry(pitch) Rx(roll)


def _origin(el: ET.Element | None) -> np.ndarray:
    """4x4 homogeneous transform of an <origin xyz rpy/> element (identity if absent)."""
    t = np.eye(4)
    if el is None:
        return t
    t[:3, :3] = _rpy_to_mat(_vec(el.get("rpy")))
    t[:3, 3] = _vec(el.get("xyz"))
    return t


class Link:
    def __init__(self, el: ET.Element):
        self.name = el.get("name")
        inertial = el.find("inertial")
        self.mass = float(inertial.find("mass").get("value")) if inertial is not None else 0.0
        self.inertial_origin = _origin(inertial.find("origin")) if inertial is not None else np.eye(4)
        self.inertia = np.zeros((3, 3))
        if inertial is not None and inertial.find("inertia") is not None:
            i = inertial.find("inertia")
            ixx, iyy, izz = (float(i.get(k)) for k in ("ixx", "iyy", "izz"))
            ixy, ixz, iyz = (float(i.get(k, 0.0)) for k in ("ixy", "ixz", "iyz"))
            self.inertia = np.array([[ixx, ixy, ixz], [ixy, iyy, iyz], [ixz, iyz, izz]])
        self.visuals = []  # (mesh path, 4x4 transform, rgba)
        for v in el.findall("visual"):
            mesh = v.find("geometry/mesh")
            if mesh is None:
                continue
            rgba = None
            mat = v.find("material/color")
            if mat is not None:
                rgba = _vec(mat.get("rgba"), 4, 1.0)
            self.visuals.append((mesh.get("filename"), _origin(v.find("origin")), rgba))
        self.children: list[Joint] = []
        self.parent_joint: Joint | None = None


class Joint:
    def __init__(self, el: ET.Element):
        self.name = el.get("name")
        self.type = el.get("type")
        self.origin = _origin(el.find("origin"))
        self.axis = _vec(el.find("axis").get("xyz")) if el.find("axis") is not None else np.array([1.0, 0, 0])
        self.parent = el.find("parent").get("link")
        self.child = el.find("child").get("link")
        lim = el.find("limit")
        self.lower = float(lim.get("lower")) if lim is not None else None
        self.upper = float(lim.get("upper")) if lim is not None else None
        self.effort = float(lim.get("effort")) if lim is not None else None
        self.velocity = float(lim.get("velocity")) if lim is not None else None


def parse_urdf(path: str) -> tuple[dict[str, Link], list[Joint], str]:
    root = ET.parse(path).getroot()
    links = {el.get("name"): Link(el) for el in root.findall("link")}
    joints = [Joint(el) for el in root.findall("joint")]
    for j in joints:
        links[j.parent].children.append(j)
        links[j.child].parent_joint = j
    roots = [n for n, l in links.items() if l.parent_joint is None]
    assert len(roots) == 1, roots
    return links, joints, roots[0]


# ----------------------------------------------------------------------------- mesh loading
def load_stl(path: str) -> tuple[np.ndarray, np.ndarray]:
    """Binary STL -> (points (N,3) float32, triangle indices (M,3) int32), vertices welded."""
    data = open(path, "rb").read()
    n = struct.unpack("<I", data[80:84])[0]
    assert 84 + n * 50 == len(data), f"{path}: not a binary STL"
    rec = np.frombuffer(data[84:84 + n * 50], dtype=np.dtype([("n", "<3f4"), ("v", "<9f4"), ("a", "<u2")]))
    tri = rec["v"].reshape(-1, 3, 3).astype(np.float32)
    flat = tri.reshape(-1, 3)
    pts, inv = np.unique(flat, axis=0, return_inverse=True)
    return pts, inv.reshape(-1, 3).astype(np.int32)


# ----------------------------------------------------------------------------- USD helpers
def _gf_rotation(mat3: np.ndarray) -> Gf.Rotation:
    """NumPy rotation (column-vector convention, R @ v) -> Gf.Rotation.

    Gf matrices are row-vector (v * M), so the NumPy matrix must be transposed on the way in;
    feeding the rows straight in yields the inverse rotation. That bug survived every
    translation-only check and only showed on the one asymmetric joint in the chain (the
    wrist-camera bracket), so ``tests/test_bimanual_yam_asset.py`` now compares orientations too."""
    return Gf.Matrix3d(*np.asarray(mat3, dtype=float).T.flatten().tolist()).ExtractRotation()


def _quat(mat3: np.ndarray) -> Gf.Quatf:
    q = _gf_rotation(mat3).GetQuat()
    q.Normalize()
    return Gf.Quatf(q.GetReal(), *q.GetImaginary())


def _set_xform(prim: Usd.Prim, t: np.ndarray) -> None:
    x = UsdGeom.Xformable(prim)
    x.ClearXformOpOrder()
    x.AddTranslateOp().Set(Gf.Vec3d(*t[:3, 3].tolist()))
    x.AddOrientOp().Set(_quat(t[:3, :3]))


def _author_mesh(stage: Usd.Stage, path: str, pts: np.ndarray, tris: np.ndarray, t: np.ndarray,
                 rgba, collision: str | None, material: UsdShade.Material | None,
                 physics_material: UsdShade.Material | None = None, display_rgb=None) -> None:
    mesh = UsdGeom.Mesh.Define(stage, path)
    mesh.CreatePointsAttr().Set([Gf.Vec3f(*p.tolist()) for p in pts])
    mesh.CreateFaceVertexCountsAttr().Set([3] * len(tris))
    mesh.CreateFaceVertexIndicesAttr().Set(tris.flatten().tolist())
    mesh.CreateSubdivisionSchemeAttr().Set(UsdGeom.Tokens.none)
    ext = UsdGeom.PointBased(mesh).ComputeExtent(mesh.GetPointsAttr().Get())
    mesh.CreateExtentAttr().Set(ext)
    _set_xform(mesh.GetPrim(), t)
    if material is not None:
        UsdShade.MaterialBindingAPI.Apply(mesh.GetPrim()).Bind(material)
        # Fallback colour for renderers that do not resolve the preview surface.
        mesh.CreateDisplayColorAttr().Set([Gf.Vec3f(*(display_rgb or ARM_RGB))])
    if collision:
        UsdPhysics.CollisionAPI.Apply(mesh.GetPrim())
        UsdPhysics.MeshCollisionAPI.Apply(mesh.GetPrim()).CreateApproximationAttr().Set(collision)
        if physics_material is not None:
            UsdShade.MaterialBindingAPI.Apply(mesh.GetPrim()).Bind(
                physics_material, materialPurpose="physics")


def _author_mass(prim: Usd.Prim, link: Link) -> None:
    m = UsdPhysics.MassAPI.Apply(prim)
    m.CreateMassAttr().Set(float(max(link.mass, 1e-4)))
    com = link.inertial_origin[:3, 3]
    m.CreateCenterOfMassAttr().Set(Gf.Vec3f(*com.tolist()))
    # Inertia is given about the COM in the inertial frame; USD wants principal moments + axes.
    r_in = link.inertial_origin[:3, :3]
    inertia = r_in @ link.inertia @ r_in.T
    w, v = np.linalg.eigh(inertia)
    if np.linalg.det(v) < 0:
        v[:, 0] *= -1
    m.CreateDiagonalInertiaAttr().Set(Gf.Vec3f(*np.maximum(w, 1e-9).tolist()))
    m.CreatePrincipalAxesAttr().Set(_quat(v))


def _make_material(stage: Usd.Stage, path: str, rgb: tuple[float, float, float]) -> UsdShade.Material:
    mat = UsdShade.Material.Define(stage, path)
    shader = UsdShade.Shader.Define(stage, path + "/Shader")
    shader.CreateIdAttr("UsdPreviewSurface")
    shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*rgb))
    # Matte: any metallic term mirrors the bright room dome and turns the black parts light
    # grey in the policy cameras (seen in the first top-camera renders).
    shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.75)
    shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(0.0)
    # Declare the shader outputs explicitly: without ``outputs:surface`` on the shader the RTX
    # tiled cameras rendered the meshes light grey while the viewport honoured the material.
    shader.CreateOutput("surface", Sdf.ValueTypeNames.Token)
    shader.CreateOutput("displacement", Sdf.ValueTypeNames.Token)
    mat.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
    mat.CreateDisplacementOutput().ConnectToSource(shader.ConnectableAPI(), "displacement")
    return mat


# ----------------------------------------------------------------------------- build
class Builder:
    def __init__(self, stage: Usd.Stage, links: dict[str, Link], side: str, arm_root: Sdf.Path,
                 material: UsdShade.Material, meshes: dict[str, tuple[np.ndarray, np.ndarray]],
                 pad_material: UsdShade.Material | None = None, shell_material: UsdShade.Material | None = None):
        self.stage, self.links, self.side, self.material, self.meshes = stage, links, side, material, meshes
        self.pad_material = pad_material
        self.shell_material = shell_material
        self.arm_root = arm_root
        self.body_paths: dict[str, Sdf.Path] = {}
        self.joint_paths: dict[str, Sdf.Path] = {}
        self.merged_frames: dict[str, np.ndarray] = {}  # merged link -> transform in its body

    def p(self, name: str) -> str:
        return f"{self.side}_{name}"

    def body(self, link: Link, parent_path: Sdf.Path, local_t: np.ndarray, world_t: np.ndarray | None = None):
        """Author ``link`` as a rigid body under the arm root, posed at its zero-pose FK
        transform ``world_t`` (arm frame). ``local_t`` is the URDF joint origin, kept for the
        joint frames; ``parent_path`` is the parent *body* (joint prims go under it)."""
        world_t = np.eye(4) if world_t is None else world_t
        path = self.arm_root.AppendChild(self.p(link.name))
        prim = UsdGeom.Xform.Define(self.stage, path).GetPrim()
        _set_xform(prim, world_t)
        UsdPhysics.RigidBodyAPI.Apply(prim)
        _author_mass(prim, link)
        self._geometry(prim, link, np.eye(4))
        self.body_paths[link.name] = path
        for j in link.children:
            if j.type == "fixed" and MERGE_FIXED:
                self._merge(prim, self.links[j.child], j.origin, link)
            else:
                child = self.body(self.links[j.child], path, j.origin, world_t @ j.origin)
                self._joint(j, path, child)
        return path

    def _merge(self, body_prim: Usd.Prim, link: Link, t_in_body: np.ndarray, body_link: Link) -> None:
        """A fixed-jointed link becomes an Xform under the parent body: geometry keeps its pose,
        mass folds into the body (it is tiny here: camera bracket, D405), no joint prim."""
        frame = UsdGeom.Xform.Define(self.stage, body_prim.GetPath().AppendChild(self.p(link.name)))
        _set_xform(frame.GetPrim(), t_in_body)
        self.merged_frames[link.name] = t_in_body
        self._geometry(frame.GetPrim(), link, np.eye(4), collision=None if link.name in NO_COLLISION else "convexHull")
        for j in link.children:
            assert j.type == "fixed", f"{link.name}: non-fixed joint under a merged link"
            self._merge(body_prim, self.links[j.child], t_in_body @ j.origin, body_link)

    def _geometry(self, prim: Usd.Prim, link: Link, t: np.ndarray, collision: str | None = "auto") -> None:
        for i, (mesh_file, vt, rgba) in enumerate(link.visuals):
            pts, tris = self.meshes[mesh_file]
            approx = collision
            if collision == "auto":
                approx = None if link.name in NO_COLLISION else (
                    "convexDecomposition" if link.name in CONVEX_DECOMP else "convexHull")
            shell = link.name in WHITE_LINKS and self.shell_material is not None
            _author_mesh(self.stage, str(prim.GetPath().AppendChild(f"visual_{i}")), pts, tris, t @ vt,
                         rgba, None, self.shell_material if shell else self.material,
                         display_rgb=SHELL_RGB if shell else ARM_RGB)
            if approx:
                _author_mesh(self.stage, str(prim.GetPath().AppendChild(f"collision_{i}")), pts, tris, t @ vt,
                             None, approx, None,
                             physics_material=self.pad_material if link.name in ("tip_left", "tip_right") else None)
                coll = self.stage.GetPrimAtPath(prim.GetPath().AppendChild(f"collision_{i}"))
                # Guide purpose hides the hull in the viewport; the RTX tiled cameras that feed
                # the policy still drew it as a light-grey shell around every link, so it is
                # made invisible outright (PhysX uses the collision geometry regardless).
                UsdGeom.Imageable(coll).CreatePurposeAttr().Set(UsdGeom.Tokens.guide)
                UsdGeom.Imageable(coll).MakeInvisible()

    def _joint(self, j: Joint, parent_path: Sdf.Path, child_path: Sdf.Path) -> None:
        jpath = parent_path.AppendChild(self.p(j.name))
        key = tuple(int(round(a)) for a in j.axis)
        assert key in AXIS_FIX, f"{j.name}: axis {j.axis} is not a signed principal axis"
        axis_token, fix = AXIS_FIX[key]
        if j.type == "revolute":
            joint = UsdPhysics.RevoluteJoint.Define(self.stage, jpath)
            joint.CreateLowerLimitAttr().Set(math.degrees(j.lower))
            joint.CreateUpperLimitAttr().Set(math.degrees(j.upper))
            drive = UsdPhysics.DriveAPI.Apply(joint.GetPrim(), "angular")
            k, d, f = ARM_DRIVE.get(j.name, (10.0, 1.0, 10.0))
        elif j.type == "prismatic":
            joint = UsdPhysics.PrismaticJoint.Define(self.stage, jpath)
            joint.CreateLowerLimitAttr().Set(j.lower)
            joint.CreateUpperLimitAttr().Set(j.upper)
            drive = UsdPhysics.DriveAPI.Apply(joint.GetPrim(), "linear")
            k, d, f = FINGER_DRIVE
        else:
            raise ValueError(j.type)
        joint.CreateAxisAttr().Set(axis_token)
        drive.CreateTypeAttr().Set("force")
        drive.CreateStiffnessAttr().Set(k)
        drive.CreateDampingAttr().Set(d)
        drive.CreateMaxForceAttr().Set(f)
        joint.CreateBody0Rel().SetTargets([parent_path])
        joint.CreateBody1Rel().SetTargets([child_path])
        r0 = _gf_rotation(j.origin[:3, :3])
        r1 = Gf.Rotation(Gf.Quatd(1, 0, 0, 0))
        if fix is not None:
            r0 = fix * r0   # apply fix in the joint frame, then the origin rotation
            r1 = fix * r1
        q0, q1 = r0.GetQuat(), r1.GetQuat()
        joint.CreateLocalPos0Attr().Set(Gf.Vec3f(*j.origin[:3, 3].tolist()))
        joint.CreateLocalRot0Attr().Set(Gf.Quatf(q0.GetReal(), *q0.GetImaginary()))
        joint.CreateLocalPos1Attr().Set(Gf.Vec3f(0, 0, 0))
        joint.CreateLocalRot1Attr().Set(Gf.Quatf(q1.GetReal(), *q1.GetImaginary()))
        self.joint_paths[j.name] = jpath


def build(source: str = SOURCE_URDF, output: str = OUTPUT_USD) -> str:
    links, joints, root_name = parse_urdf(source)
    src_dir = os.path.dirname(source)
    meshes = {}
    for l in links.values():
        for mesh_file, _, _ in l.visuals:
            if mesh_file not in meshes:
                meshes[mesh_file] = load_stl(os.path.join(src_dir, mesh_file))

    os.makedirs(os.path.dirname(output), exist_ok=True)
    stage = Usd.Stage.CreateNew(output)
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    robot = UsdGeom.Xform.Define(stage, "/robot")
    stage.SetDefaultPrim(robot.GetPrim())
    UsdPhysics.ArticulationRootAPI.Apply(robot.GetPrim())
    looks = UsdGeom.Scope.Define(stage, "/robot/Looks")
    arm_mat = _make_material(stage, "/robot/Looks/yam_black", ARM_RGB)
    shell_mat = _make_material(stage, "/robot/Looks/yam_white", SHELL_RGB)
    profile_mat = _make_material(stage, "/robot/Looks/profile_grey", (0.55, 0.56, 0.58))
    pad_mat = UsdShade.Material.Define(stage, "/robot/Looks/finger_pad_physics")
    pad_api = UsdPhysics.MaterialAPI.Apply(pad_mat.GetPrim())
    pad_api.CreateStaticFrictionAttr().Set(FINGER_FRICTION[0])
    pad_api.CreateDynamicFrictionAttr().Set(FINGER_FRICTION[1])
    pad_api.CreateRestitutionAttr().Set(0.0)

    # Torso = the 2060 profile, fixed to the world.
    torso = UsdGeom.Xform.Define(stage, "/robot/torso")
    UsdPhysics.RigidBodyAPI.Apply(torso.GetPrim())
    UsdPhysics.MassAPI.Apply(torso.GetPrim()).CreateMassAttr().Set(PROFILE_MASS_KG)
    cube = UsdGeom.Cube.Define(stage, "/robot/torso/profile")
    cube.CreateSizeAttr().Set(1.0)
    UsdGeom.Xformable(cube).AddTranslateOp().Set(Gf.Vec3d(0, 0, -PROFILE_SIZE[2] / 2))
    UsdGeom.Xformable(cube).AddScaleOp().Set(Gf.Vec3f(*PROFILE_SIZE))
    UsdShade.MaterialBindingAPI.Apply(cube.GetPrim()).Bind(profile_mat)
    UsdPhysics.CollisionAPI.Apply(cube.GetPrim())
    root_joint = UsdPhysics.FixedJoint.Define(stage, "/robot/torso/root_joint")
    root_joint.CreateBody1Rel().SetTargets(["/robot/torso"])   # body0 empty = the world

    report = {}
    for side, y in SIDES:
        arm_root = Sdf.Path(f"/robot/{side}_arm")
        arm = UsdGeom.Xform.Define(stage, arm_root)
        placement = np.eye(4)
        placement[:3, 3] = [0.0, y, 0.0]
        _set_xform(arm.GetPrim(), placement)
        b = Builder(stage, links, side, arm_root, arm_mat, meshes, pad_material=pad_mat, shell_material=shell_mat)
        base_path = b.body(links[root_name], arm_root, np.eye(4))
        mount = UsdPhysics.FixedJoint.Define(stage, arm_root.AppendChild(f"{side}_mount_joint"))
        mount.CreateBody0Rel().SetTargets(["/robot/torso"])
        mount.CreateBody1Rel().SetTargets([base_path])
        mount.CreateLocalPos0Attr().Set(Gf.Vec3f(0.0, y, 0.0))
        mount.CreateLocalRot0Attr().Set(Gf.Quatf(1, 0, 0, 0))
        mount.CreateLocalPos1Attr().Set(Gf.Vec3f(0, 0, 0))
        mount.CreateLocalRot1Attr().Set(Gf.Quatf(1, 0, 0, 0))
        report[side] = b

    stage.GetRootLayer().Save()
    return output


# ----------------------------------------------------------------------------- verification
def fk(links: dict[str, Link], root_name: str, q: dict[str, float]) -> dict[str, np.ndarray]:
    """Forward kinematics on the URDF (world = base frame). Returns link -> 4x4."""
    out = {root_name: np.eye(4)}
    stack = [root_name]
    while stack:
        name = stack.pop()
        for j in links[name].children:
            t = out[name] @ j.origin
            if j.type == "revolute":
                a = j.axis / np.linalg.norm(j.axis)
                ang = q.get(j.name, 0.0)
                k = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
                r = np.eye(3) + math.sin(ang) * k + (1 - math.cos(ang)) * (k @ k)
                m = np.eye(4); m[:3, :3] = r
                t = t @ m
            elif j.type == "prismatic":
                m = np.eye(4); m[:3, 3] = j.axis * q.get(j.name, 0.0)
                t = t @ m
            out[j.child] = t
            stack.append(j.child)
    return out


def verify(path: str) -> None:
    stage = Usd.Stage.Open(path)
    roots = [p.GetPath() for p in stage.Traverse() if p.HasAPI(UsdPhysics.ArticulationRootAPI)]
    assert roots == [Sdf.Path("/robot")], roots
    bodies = [p for p in stage.Traverse() if p.HasAPI(UsdPhysics.RigidBodyAPI)]
    names = [p.GetName() for p in bodies]
    assert len(names) == len(set(names)), "duplicate body names"
    joints = [p for p in stage.Traverse() if p.IsA(UsdPhysics.Joint)]
    jnames = [p.GetName() for p in joints]
    assert len(jnames) == len(set(jnames)), "duplicate joint names"
    for j in joints:
        for rel in (UsdPhysics.Joint(j).GetBody0Rel(), UsdPhysics.Joint(j).GetBody1Rel()):
            for t in rel.GetTargets():
                tp = stage.GetPrimAtPath(t)
                assert tp and tp.HasAPI(UsdPhysics.RigidBodyAPI), f"{j.GetPath()} -> {t}"
    for b in bodies:
        anc = b.GetParent()
        while anc and anc.GetPath() != Sdf.Path("/"):
            assert not anc.HasAPI(UsdPhysics.RigidBodyAPI), f"{b.GetPath()} is nested under rigid body {anc.GetPath()}"
            anc = anc.GetParent()
    dof = [p for p in joints if p.IsA(UsdPhysics.RevoluteJoint) or p.IsA(UsdPhysics.PrismaticJoint)]
    for side in ("left", "right"):
        for n in ("base", "link1", "link2", "link3", "link4", "link5", "gripper", "tip_left", "tip_right"):
            assert f"{side}_{n}" in names, f"missing body {side}_{n}"
        for n in ("joint1", "joint2", "joint3", "joint4", "joint5", "joint6", "joint7", "joint8"):
            assert f"{side}_{n}" in jnames, f"missing joint {side}_{n}"
        gripper = f"/robot/{side}_arm/{side}_gripper"
        cam = stage.GetPrimAtPath(f"{gripper}/{side}_camera")
        assert cam, f"{side}: camera frame missing under the gripper body"
        # The merged camera frame must compose to the URDF's FK at the zero pose.
        links, _, root_name = parse_urdf(SOURCE_URDF)
        want = fk(links, root_name, {})["camera"][:3, 3] + np.array([0.0, dict(SIDES)[side], 0.0])
        got = np.array(UsdGeom.Xformable(cam).ComputeLocalToWorldTransform(Usd.TimeCode.Default()).ExtractTranslation())
        assert np.allclose(got, want, atol=1e-4), f"{side} camera: USD {got} vs FK {want}"
    print(f"OK {path}: {len(bodies)} bodies, {len(joints)} joints ({len(dof)} DOF), "
          f"{os.path.getsize(path) / 1e6:.1f} MB")


def print_fk(source: str = SOURCE_URDF) -> None:
    links, _, root_name = parse_urdf(source)
    for label, q in (("home (all zero)", {}),
                     ("Ai2 rest (j2=pi/4, j3=pi/2)", {"joint2": math.pi / 4, "joint3": math.pi / 2})):
        t = fk(links, root_name, q)
        g, c = t["gripper"], t["camera"]
        print(f"{label}: gripper at {np.round(g[:3, 3], 3)}, approach (-Z of gripper) = {np.round(-g[:3, 2], 3)}")
        print(f"    camera at {np.round(c[:3, 3], 3)}, optical axis (+Z) = {np.round(c[:3, 2], 3)}")
    # gripper -> camera, constant (fixed chain), for the robot cfg
    t = fk(links, root_name, {})
    g2c = np.linalg.inv(t["gripper"]) @ t["camera"]
    q = Gf.Matrix3d(*g2c[:3, :3].flatten().tolist()).ExtractRotation().GetQuat()
    print(f"gripper->camera: xyz {np.round(g2c[:3, 3], 4)} quat(w,x,y,z) "
          f"({q.GetReal():.4f}, {q.GetImaginary()[0]:.4f}, {q.GetImaginary()[1]:.4f}, {q.GetImaginary()[2]:.4f})")
    tip = np.linalg.inv(t["gripper"]) @ t["tip_left"]
    print(f"gripper->tip_left (closed): xyz {np.round(tip[:3, 3], 4)}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default=SOURCE_URDF)
    ap.add_argument("--output", default=OUTPUT_USD)
    ap.add_argument("--fk", action="store_true", help="print rest-pose forward kinematics")
    a = ap.parse_args()
    if a.fk:
        print_fk(a.source)
    out = build(a.source, a.output)
    verify(out)
