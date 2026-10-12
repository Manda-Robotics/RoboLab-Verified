# SPDX-License-Identifier: Apache-2.0
"""Load the compiled rack collision USD through Newton's native mesh importer.

Only fingerprinted rack meshes are substituted. Native code still owns visuals,
transforms, materials, body mass/inertia, filters and shape registration.
"""
from contextlib import contextmanager
from contextvars import ContextVar
import hashlib
import importlib.metadata
import json
from pathlib import Path

import numpy as np

MESH_NAMES = frozenset(f"M_Rack_L01_{part}" for part in ("Frame", "FrameJoints", "Shelf", "ShelfElements"))
_active = False


def geometry_key(vertices, faces):
    digest = hashlib.sha256()
    for value, dtype in ((vertices, "<f8"), (np.asarray(faces).reshape(-1, 3), "<i8")):
        array = np.ascontiguousarray(value, dtype=dtype)
        digest.update(str(array.shape).encode())
        digest.update(array.tobytes())
    return digest.hexdigest()


def load_colliders(path):
    from pxr import Usd, UsdGeom
    path = Path(path)
    manifest = json.loads(path.with_suffix(".json").read_text())
    if (manifest["format_version"] != 2 or not 0 < manifest["part_count"] <= 64
            or set(manifest["meshes"]) != MESH_NAMES
            or hashlib.sha256(path.read_bytes()).hexdigest() != manifest["collision_asset_sha256"]):
        raise ValueError("Unsupported or changed shelf collision asset; rebuild and revalidate")
    stage = Usd.Stage.Open(str(path))
    parts = {}
    for entry in manifest["meshes"].values():
        if entry["part_count"] != len(entry["paths"]) or entry["source_component_count"] <= 0:
            raise ValueError("Invalid shelf part mapping")
        for prim_path in entry["paths"]:
            mesh = UsdGeom.Mesh(stage.GetPrimAtPath(prim_path))
            if not mesh or prim_path in parts:
                raise ValueError("Missing or duplicate shelf collision mesh")
            vertices = np.asarray(mesh.GetPointsAttr().Get(), dtype=np.float32)
            counts = np.asarray(mesh.GetFaceVertexCountsAttr().Get())
            faces = np.asarray(mesh.GetFaceVertexIndicesAttr().Get(), dtype=np.int32)
            if (vertices.ndim != 2 or vertices.shape[1] != 3 or not 4 <= len(vertices) <= 64
                    or not np.isfinite(vertices).all() or len(counts) < 4 or not np.all(counts == 3)
                    or len(faces) != 3 * len(counts) or faces.min() < 0 or faces.max() >= len(vertices)):
                raise ValueError("Invalid shelf collision geometry; rebuild the asset")
            parts[prim_path] = (vertices, faces.reshape(-1, 3))
    if len(parts) != manifest["part_count"]:
        raise ValueError("Invalid shelf collider count")
    return manifest, parts


@contextmanager
def shelf_colliders(builder_class=None, coacd_module=None, path=None):
    """Substitute compiled geometry before replication; restore on all exits.

    The native importer visits every connected component of the original mesh.
    After validating the complete source, return its compiled geometry on the
    first component, then empty results on subsequent components. No CoACD work
    is performed for the shelf. A ContextVar isolates concurrent importer threads.
    """
    global _active
    if _active:
        raise RuntimeError("Concurrent/nested shelf collider contexts are unsupported")
    if builder_class is None:
        if importlib.metadata.version("newton") != "1.5.2":
            raise RuntimeError("Re-audit shelf colliders before changing Newton 1.5.2")
        from newton import ModelBuilder
        builder_class = ModelBuilder
    if coacd_module is None:
        import coacd as coacd_module
    if path is None:
        path = Path(__file__).resolve().parents[3] / "assets/objects/vomp/rack_l04/rack_l04_colliders.usda"
    current = ContextVar("robolab_shelf_mesh", default=None)
    original_approximate = builder_class.approximate_meshes
    original_decompose = coacd_module.run_coacd
    loaded = None
    reports = []

    def decompose(mesh, *args, **kwargs):
        selection = current.get()
        if selection is None:
            return original_decompose(mesh, *args, **kwargs)
        entry, parts, seen = selection
        seen.append(None)
        return [(v.copy(), f.copy()) for v, f in (parts[p] for p in entry["paths"])] if len(seen) == 1 else []

    def approximate(builder, method="convex_hull", shape_indices=None, **kwargs):
        nonlocal loaded
        if method != "coacd" or shape_indices is None:
            return original_approximate(builder, method=method, shape_indices=shape_indices, **kwargs)
        indices = list(shape_indices)
        selected = [i for i in indices if builder.shape_label[i].split("/")[-1] in MESH_NAMES
                    and "/SM_Rack_L04_01/" in builder.shape_label[i]]
        if not selected:
            return original_approximate(builder, method=method, shape_indices=indices, **kwargs)
        if loaded is None:
            loaded = load_colliders(path)
        manifest, parts = loaded
        result = set()
        for i in selected:
            label = builder.shape_label[i]
            entry = manifest["meshes"][label.split("/")[-1]]
            mesh = builder.shape_source[i]
            if geometry_key(mesh.vertices, mesh.indices) != entry["source_geometry_sha256"]:
                raise RuntimeError(f"Shelf source geometry changed at {label}; rebuild and revalidate collision parts")
            scale = np.asarray(builder.shape_scale[i])
            if not np.isfinite(scale).all() or np.max(np.abs(scale)) > manifest["max_validated_scale"] * (1 + 1e-6):
                raise RuntimeError(f"Shelf scale exceeds the validated 0.2 scale at {label}")
            seen = []
            token = current.set((entry, parts, seen))
            before_count = len(builder.shape_source)
            try:
                changed = original_approximate(builder, method=method, shape_indices=[i],
                                               **dict(kwargs, raise_on_failure=True))
            finally:
                current.reset(token)
            if len(seen) != entry["source_component_count"] or i not in changed:
                raise RuntimeError(f"Shelf import did not consume the expected components at {label}")
            created = [i] + [j for j in range(before_count, len(builder.shape_source))
                             if not builder.shape_label[j].endswith("_visual")]
            if len(created) != entry["part_count"]:
                raise RuntimeError(f"Unexpected shelf collider count at {label}; refusing importer fallback")
            result |= changed
            report = {"shape": label, "parts": entry["part_count"], "asset_sha256": manifest["collision_asset_sha256"]}
            reports.append(report)
            print("ROBOLAB_SHELF_COLLIDERS=" + json.dumps(report), flush=True)
        remaining = [i for i in indices if i not in selected]
        if remaining:
            result |= original_approximate(builder, method=method, shape_indices=remaining, **kwargs)
        return result

    _active = True
    builder_class.approximate_meshes = approximate
    coacd_module.run_coacd = decompose
    try:
        yield reports
    finally:
        coacd_module.run_coacd = original_decompose
        builder_class.approximate_meshes = original_approximate
        _active = False
