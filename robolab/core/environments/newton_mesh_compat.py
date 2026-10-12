# SPDX-License-Identifier: Apache-2.0
"""Narrow mesh-compiler compatibility for Newton 1.5.2 / MuJoCo 3.11.

Keep native decomposition settings and solid collision geometry. The adapters
are process-wide and must enclose one environment's initialization/lifetime.
"""
from contextlib import contextmanager
import importlib.metadata
import json
import re

import numpy as np


def compile_with_explicit_inertia(spec, compile_original, mujoco, reports):
    """Retry only a rejected mesh whose every user has explicit body inertia.

    SHELL changes mesh inertia preprocessing, not the collision surface. Never
    allow it to determine a body's mass/inertia. Restore the input spec even
    when another compiler error follows a successful retry.
    """
    changed = []
    try:
        while True:
            try:
                return compile_original(spec)
            except ValueError as error:
                match = re.search(r"Error: mesh volume is too small: (.*?) \. Try setting inertia to shell", str(error))
                if not match:
                    raise
                mesh = next((m for m in spec.meshes if m.name == match[1]), None)
                if mesh is None or mesh.inertia == mujoco.mjtMeshInertia.mjMESH_INERTIA_SHELL:
                    raise
                users = [g for g in spec.geoms if g.meshname == mesh.name]
                if (not users or any(not g.parent.explicitinertial or not g.parent.mass > 0 for g in users)
                        or spec.compiler.inertiafromgeom == mujoco.mjtInertiaFromGeom.mjINERTIAFROMGEOM_TRUE):
                    raise
                vertices = np.asarray(mesh.uservert).reshape(-1, 3) * mesh.scale
                if len(vertices) < 4 or not np.isfinite(vertices).all():
                    raise
                changed.append((mesh, mesh.inertia))
                mesh.inertia = mujoco.mjtMeshInertia.mjMESH_INERTIA_SHELL
                report = {"kind": "tiny_mesh_explicit_inertia", "mesh": mesh.name,
                          "extent_m": np.ptp(vertices, axis=0).tolist()}
                reports.append(report)
                print("NEWTON_MESH_COMPAT=" + json.dumps(report), flush=True)
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
    for package, expected in (("newton", "1.5.2"), ("mujoco", "3.11.0")):
        if importlib.metadata.version(package) != expected:
            raise RuntimeError(f"Re-audit mesh compatibility before changing {package} {expected}")
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
