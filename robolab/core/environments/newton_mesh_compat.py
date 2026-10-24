# SPDX-License-Identifier: Apache-2.0
"""Narrow mesh-compiler compatibility for the pinned EA and GA Newton / MuJoCo stacks.

Keep native decomposition settings and solid collision geometry. The adapters
are process-wide and must enclose one environment's initialization/lifetime.
"""
from contextlib import contextmanager
import importlib.metadata
import json
import re

import numpy as np


def _explicit_mesh_users(spec, mesh, mujoco):
    users = [g for g in spec.geoms if g.meshname == mesh.name]
    return (users and all(g.parent.explicitinertial and g.parent.mass > 0 for g in users)
            and spec.compiler.inertiafromgeom != mujoco.mjtInertiaFromGeom.mjINERTIAFROMGEOM_TRUE)


def _rejected_small_meshes(spec, compile_original, mujoco):
    """Confirm tiny-volume failures in isolation instead of recompiling a scene.

    Probe only inline meshes <= 0.1 mm across with explicit body inertia. The
    original compiler must reject the exact copied mesh with its tiny-volume
    error; size alone never selects the SHELL fallback. Other meshes retain the
    ordinary full-model compiler path.
    """
    if spec.compiler.inertiafromgeom == mujoco.mjtInertiaFromGeom.mjINERTIAFROMGEOM_TRUE:
        return
    safe_users = {}
    for geom in spec.geoms:
        if geom.meshname:
            safe_users[geom.meshname] = (safe_users.get(geom.meshname, True)
                                        and geom.parent.explicitinertial and geom.parent.mass > 0)
    for mesh in spec.meshes:
        if (mesh.file or mesh.needsdf
                or mesh.inertia == mujoco.mjtMeshInertia.mjMESH_INERTIA_SHELL):
            continue
        vertices = np.asarray(mesh.uservert).reshape(-1, 3) * mesh.scale
        if (len(vertices) < 4 or not np.isfinite(vertices).all()
                or np.linalg.norm(np.ptp(vertices, axis=0)) > 1e-4
                or not safe_users.get(mesh.name, False)):
            continue
        probe = mujoco.MjSpec()
        probe.compiler.inertiafromgeom = spec.compiler.inertiafromgeom
        probe.compiler.alignfree = spec.compiler.alignfree
        probe.add_mesh(name=mesh.name, uservert=mesh.uservert, userface=mesh.userface,
                       scale=mesh.scale, refpos=mesh.refpos, refquat=mesh.refquat,
                       maxhullvert=mesh.maxhullvert, inertia=mesh.inertia)
        body = probe.worldbody.add_body(mass=1, inertia=[1, 1, 1], explicitinertial=True)
        body.add_freejoint()
        body.add_geom(type=mujoco.mjtGeom.mjGEOM_MESH, meshname=mesh.name)
        try:
            compile_original(probe)
        except ValueError as error:
            match = re.search(r"Error: mesh volume is too small: (.*?) \. Try setting inertia to shell", str(error))
            if match and match[1] == mesh.name:
                yield mesh


def compile_with_explicit_inertia(spec, compile_original, mujoco, reports):
    """Retry only rejected meshes whose every user has explicit body inertia.

    SHELL changes mesh inertia preprocessing, not the collision surface. Never
    allow it to determine a body's mass/inertia. Restore the input spec even
    when another compiler error follows a successful retry.
    """
    changed = []
    batch_checked = False

    def select(mesh):
        changed.append((mesh, mesh.inertia))
        mesh.inertia = mujoco.mjtMeshInertia.mjMESH_INERTIA_SHELL
        vertices = np.asarray(mesh.uservert).reshape(-1, 3) * mesh.scale
        report = {"kind": "tiny_mesh_explicit_inertia", "mesh": mesh.name,
                  "extent_m": np.ptp(vertices, axis=0).tolist()}
        reports.append(report)
        print("NEWTON_MESH_COMPAT=" + json.dumps(report), flush=True)

    try:
        while True:
            try:
                return compile_original(spec)
            except ValueError as error:
                match = re.search(r"Error: mesh volume is too small: (.*?) \. Try setting inertia to shell", str(error))
                if not match:
                    raise
                mesh = next((m for m in spec.meshes if m.name == match[1]), None)
                if (mesh is None or mesh.inertia == mujoco.mjtMeshInertia.mjMESH_INERTIA_SHELL
                        or not _explicit_mesh_users(spec, mesh, mujoco)):
                    raise
                vertices = np.asarray(mesh.uservert).reshape(-1, 3) * mesh.scale
                if len(vertices) < 4 or not np.isfinite(vertices).all():
                    raise
                select(mesh)
                if not batch_checked:
                    batch_checked = True
                    for rejected in _rejected_small_meshes(spec, compile_original, mujoco):
                        select(rejected)
    finally:
        for mesh, inertia in changed:
            mesh.inertia = inertia


def thicken_generated_fragment(vertices, scale):
    """Give a flat CoACD fragment <= 0.1 mm across a 1 micrometre thickness.

    Work in scaled physical coordinates. Every generated vertex moves at most
    0.5 micrometres from an original vertex; authored meshes, larger fragments,
    lines/points and non-flat solid hulls are left for native validation.
    """
    from scipy.spatial import ConvexHull

    scale = np.asarray(scale, dtype=np.float64)
    v = np.asarray(vertices, dtype=np.float64) * scale
    if len(v) < 3 or not np.isfinite(v).all() or np.any(scale == 0):
        return None
    _, singular, basis = np.linalg.svd(v - v.mean(0), full_matrices=False)
    diameter = float(np.linalg.norm(np.ptp(v, axis=0)))
    if diameter > 1e-4 or singular[1] <= 1e-12 or singular[-1] > 1e-6 * diameter:
        return None
    normal = basis[-1]
    points = np.concatenate([v + normal * 5e-7, v - normal * 5e-7])
    hull = ConvexHull(points)
    faces = hull.simplices.copy()
    for i, face in enumerate(faces):
        direction = np.cross(points[face[1]] - points[face[0]], points[face[2]] - points[face[0]])
        if np.dot(direction, hull.equations[i, :3]) < 0:
            faces[i] = face[[0, 2, 1]]
    return points / scale, faces, diameter


@contextmanager
def generated_fragment_context(builder_class, convex_type, reports):
    """Repair only newly generated CoACD hulls before model replication."""
    original = builder_class.approximate_meshes

    def approximate(builder, *args, **kwargs):
        method = args[0] if args else kwargs.get("method", "convex_hull")
        before = list(builder.shape_source) if method == "coacd" else []
        result = original(builder, *args, **kwargs)
        if method != "coacd":
            return result
        for i, mesh in enumerate(builder.shape_source):
            if (builder.shape_type[i] != convex_type
                    or (i < len(before) and mesh is before[i])):
                continue
            repair = thicken_generated_fragment(mesh.vertices, builder.shape_scale[i])
            if repair is None:
                continue
            vertices, faces, diameter = repair
            builder.shape_source[i] = mesh.copy(vertices=vertices, indices=faces)
            report = {"kind": "thin_generated_fragment", "shape": builder.shape_label[i],
                      "diameter_m": diameter, "added_thickness_m": 1e-6}
            reports.append(report)
            print("NEWTON_MESH_COMPAT=" + json.dumps(report), flush=True)
        return result

    builder_class.approximate_meshes = approximate
    try:
        yield
    finally:
        builder_class.approximate_meshes = original


_active = False


@contextmanager
def mesh_compatibility():
    """Scope and restore both adapters; reject unsupported versions/nesting."""
    global _active
    if _active:
        raise RuntimeError("Concurrent/nested Newton mesh compatibility contexts are unsupported")
    versions = tuple(importlib.metadata.version(package) for package in ("newton", "mujoco"))
    if versions not in (("1.5.2", "3.11.0"), ("1.6.1", "3.12.0")):
        raise RuntimeError(f"Re-audit mesh compatibility for Newton/MuJoCo {versions}")
    import mujoco
    from newton import GeoType, ModelBuilder

    original_compile = mujoco.MjSpec.compile
    reports = []

    def compile(spec):
        return compile_with_explicit_inertia(spec, original_compile, mujoco, reports)

    _active = True
    try:
        mujoco.MjSpec.compile = compile
        with generated_fragment_context(ModelBuilder, GeoType.CONVEX_MESH, reports):
            yield reports
    finally:
        mujoco.MjSpec.compile = original_compile
        _active = False
