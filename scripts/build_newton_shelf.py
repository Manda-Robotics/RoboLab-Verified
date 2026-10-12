# SPDX-License-Identifier: Apache-2.0
"""Compile rack_l04 into 44 convex collision pieces, without running CoACD.

Keep the four post envelopes, four foot plates, eight braces, four continuous
shelf decks and 24 rails/brackets. Fasteners are omitted and the 1 mm slat gaps
(at scene scale 0.2) are closed. Original rendering and body mass properties are
retained by the Newton importer. Requires NumPy, SciPy, USD and Newton 1.5.2.
"""
import argparse
import hashlib
import importlib.util
import itertools
import json
from pathlib import Path

import numpy as np
from scipy.spatial import ConvexHull

ROOT = Path(__file__).resolve().parents[1]
SOURCE_SHA256 = "94f332ada51a44466386d2d23e40b77d5ead54aa2c918aad32f9d76da71dff0b"


def convex_hull(vertices):
    vertices = np.asarray(vertices, dtype=float)
    h = ConvexHull(vertices)
    ids = np.unique(h.simplices)
    inverse = np.full(len(vertices), -1)
    inverse[ids] = np.arange(len(ids))
    faces = inverse[h.simplices]
    vertices = vertices[ids]
    normals = np.cross(vertices[faces[:, 1]] - vertices[faces[:, 0]],
                       vertices[faces[:, 2]] - vertices[faces[:, 0]])
    inward = np.einsum("ij,ij->i", normals, vertices[faces].mean(1) - vertices.mean(0)) < 0
    faces[inward] = faces[inward][:, [0, 2, 1]]
    return vertices, faces


def box(lo, hi):
    return convex_hull(np.array(list(itertools.product(*zip(lo, hi)))))


def brace_box(vertices):
    """Minimum-area rectangle in the brace's XZ plane, extruded along Y."""
    points = np.asarray(vertices, dtype=float)[:, [0, 2]]
    hull = points[ConvexHull(points).vertices]
    edges = np.roll(hull, -1, axis=0) - hull
    best = None
    for edge in edges:
        u = edge / np.linalg.norm(edge)
        axes = np.array([u, [-u[1], u[0]]]).T
        projected = points @ axes
        lo, hi = projected.min(0), projected.max(0)
        area = np.prod(hi - lo)
        if best is None or area < best[0]:
            best = area, axes, lo, hi
    _, axes, lo, hi = best
    xz = np.array(list(itertools.product(*zip(lo, hi)))) @ axes.T
    return convex_hull(np.array([[x, y, z] for x, z in xz
                                for y in (vertices[:, 1].min(), vertices[:, 1].max())]))


def source_components(source):
    from pxr import Usd, UsdGeom
    from newton import Mesh
    from newton._src.usd.utils import fan_triangulate_faces
    from newton._src.utils.mesh import split_mesh_components

    if hashlib.sha256(source.read_bytes()).hexdigest() != SOURCE_SHA256:
        raise ValueError("rack_l04 source changed; review component indexing before rebaking")
    stage = Usd.Stage.Open(str(source))
    result = {}
    for prim in stage.Traverse():
        if not prim.IsA(UsdGeom.Mesh):
            continue
        mesh = UsdGeom.Mesh(prim)
        vertices = np.array(mesh.GetPointsAttr().Get())
        faces = fan_triangulate_faces(np.array(mesh.GetFaceVertexCountsAttr().Get()),
                                      np.array(mesh.GetFaceVertexIndicesAttr().Get())).reshape(-1, 3)
        result[prim.GetName()] = (vertices, faces, split_mesh_components(Mesh(vertices, faces, compute_inertia=False)))
    if {k: len(v[2]) for k, v in result.items()} != {
        "M_Rack_L01_Frame": 4, "M_Rack_L01_FrameJoints": 732,
        "M_Rack_L01_Shelf": 56, "M_Rack_L01_ShelfElements": 24,
    }:
        raise ValueError("Source component topology changed")
    return result


def simplify(name, components):
    if name.endswith("Frame"):
        return [(f"post_{i}", *convex_hull(v)) for i, (v, _) in enumerate(components)]
    if name.endswith("FrameJoints"):
        parts = []
        for i, (v, _) in enumerate(components[:4]):
            # Tabs above the 0.4 mm base plate are contained in the post hull.
            base = v[v[:, 2] <= .002001]
            lo, hi = base.min(0).astype(float), base.max(0).astype(float)
            hi[2] = .002
            parts.append((f"foot_{i}", *box(lo, hi)))
        parts.extend((f"brace_{i}", *brace_box(v)) for i, (v, _) in enumerate(components[4:12]))
        return parts
    if name.endswith("Shelf"):
        parts = []
        for i in range(4):
            v = np.vstack([v for v, _ in components[14*i:14*(i+1)]])
            parts.append((f"deck_{i}", *box(v.min(0), v.max(0))))
        return parts
    if name.endswith("ShelfElements"):
        return [(f"rail_bracket_{i}", *box(v.min(0), v.max(0))) for i, (v, _) in enumerate(components)]
    raise ValueError(name)


def main():
    from pxr import Usd, UsdGeom, UsdPhysics, Vt
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--repo", type=Path, default=ROOT)
    p.add_argument("--output", type=Path)
    args = p.parse_args()
    source = args.repo / "assets/objects/vomp/rack_l04/rack_l04.usd"
    output = args.output or source.with_name("rack_l04_colliders.usda")
    spec = importlib.util.spec_from_file_location("shelf", args.repo / "robolab/core/environments/newton_shelf.py")
    shelf = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(shelf)
    stage = Usd.Stage.CreateInMemory()
    root = UsdGeom.Xform.Define(stage, "/RackColliders")
    stage.SetDefaultPrim(root.GetPrim())
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.)
    stage.GetRootLayer().documentation = (
        "SPDX-License-Identifier: CC-BY-4.0. Derived from VOMP rack_l04, copyright 2026 NVIDIA Corporation. Generated by scripts/build_newton_shelf.py. "
        "Collision geometry library; original scene supplies visuals, materials and body mass properties. "
        "Units and coordinates match source; benchmark instances use scale 0.2.")
    manifest = dict(format_version=2, source_asset_sha256=SOURCE_SHA256, max_validated_scale=.2,
                    newton="1.5.2", part_count=0, meshes={},
                    approximations=["Omit bolt, nut, washer and thread colliders", "Close 1 mm shelf slat gaps at scale 0.2",
                                    "Use post convex envelopes and solid rectangular rails, brackets and braces"])
    colors = [(0.3, .5, .7), (.9, .5, .2), (.6, .7, .8), (.4, .65, .4)]
    for color, (name, (vertices, faces, components)) in zip(colors, source_components(source).items()):
        parts = simplify(name, components)
        entry = dict(source_geometry_sha256=shelf.geometry_key(vertices, faces),
                     source_component_count=len(components), part_count=len(parts), paths=[])
        for label, v, f in parts:
            path = f"/RackColliders/{name}/{label}"
            mesh = UsdGeom.Mesh.Define(stage, path)
            mesh.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(v.astype(np.float32)))
            mesh.CreateFaceVertexCountsAttr([3] * len(f))
            mesh.CreateFaceVertexIndicesAttr(f.ravel().tolist())
            mesh.CreateSubdivisionSchemeAttr(UsdGeom.Tokens.none)
            mesh.CreateDisplayColorAttr([color])
            UsdPhysics.CollisionAPI.Apply(mesh.GetPrim()).CreateCollisionEnabledAttr(True)
            UsdPhysics.MeshCollisionAPI.Apply(mesh.GetPrim()).CreateApproximationAttr("convexHull")
            entry["paths"].append(path)
        manifest["meshes"][name] = entry
        manifest["part_count"] += len(parts)
        print(name, len(parts), flush=True)
    if manifest["part_count"] != 44:
        raise ValueError("Unexpected shelf collider count")
    output.parent.mkdir(parents=True, exist_ok=True)
    stage.GetRootLayer().Export(str(output))
    manifest["collision_asset_sha256"] = hashlib.sha256(output.read_bytes()).hexdigest()
    output.with_suffix(".json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    shelf.load_colliders(output)
    print(output, output.stat().st_size, "bytes", flush=True)


if __name__ == "__main__":
    main()
