# SPDX-License-Identifier: Apache-2.0
"""Load fingerprinted compiled collision USD through Newton's native mesh importer.

Only fingerprinted meshes are substituted. Native code still owns visuals,
transforms, materials, body mass/inertia, filters and shape registration.
"""
from contextlib import contextmanager
from contextvars import ContextVar
import hashlib
import importlib.metadata
import json
from pathlib import Path

import numpy as np


def geometry_key(vertices, faces):
    digest = hashlib.sha256()
    for value, dtype in ((vertices, "<f8"), (np.asarray(faces).reshape(-1, 3), "<i8")):
        array = np.ascontiguousarray(value, dtype=dtype)
        digest.update(str(array.shape).encode())
        digest.update(array.tobytes())
    return digest.hexdigest()


def load_colliders(path, *, mesh_names, max_parts, asset_name):
    from pxr import Usd, UsdGeom
    path = Path(path)
    manifest = json.loads(path.with_suffix(".json").read_text())
    if (manifest["format_version"] != 2 or not 0 < manifest["part_count"] <= max_parts
            or set(manifest["meshes"]) != mesh_names
            or hashlib.sha256(path.read_bytes()).hexdigest() != manifest["collision_asset_sha256"]):
        raise ValueError(f"Unsupported or changed {asset_name} collision asset; rebuild and revalidate")
    stage = Usd.Stage.Open(str(path))
    parts = {}
    for entry in manifest["meshes"].values():
        if entry["part_count"] != len(entry["paths"]) or entry["source_component_count"] <= 0:
            raise ValueError(f"Invalid {asset_name} part mapping")
        for prim_path in entry["paths"]:
            mesh = UsdGeom.Mesh(stage.GetPrimAtPath(prim_path))
            if not mesh or prim_path in parts:
                raise ValueError(f"Missing or duplicate {asset_name} collision mesh")
            vertices = np.asarray(mesh.GetPointsAttr().Get(), dtype=np.float32)
            counts = np.asarray(mesh.GetFaceVertexCountsAttr().Get())
            faces = np.asarray(mesh.GetFaceVertexIndicesAttr().Get(), dtype=np.int32)
            if (vertices.ndim != 2 or vertices.shape[1] != 3 or not 4 <= len(vertices) <= 64
                    or not np.isfinite(vertices).all() or len(counts) < 4 or not np.all(counts == 3)
                    or len(faces) != 3 * len(counts) or faces.min() < 0 or faces.max() >= len(vertices)):
                raise ValueError(f"Invalid {asset_name} collision geometry; rebuild the asset")
            parts[prim_path] = (vertices, faces.reshape(-1, 3))
    if len(parts) != manifest["part_count"]:
        raise ValueError(f"Invalid {asset_name} collider count")
    return manifest, parts


@contextmanager
def compiled_colliders(builder_class=None, coacd_module=None, path=None, *,
                       mesh_names, label_fragment, max_parts, asset_name):
    """Preserve native import while replacing decomposition of verified meshes.

    Supply all compiled pieces on the first connected component, then consume
    the remaining source components without generating duplicate colliders.
    ContextVar scopes the substitution to the current importer thread.
    """
    if builder_class is None:
        if importlib.metadata.version("newton") != "1.5.2":
            raise RuntimeError("Re-audit compiled colliders before changing Newton 1.5.2")
        from newton import ModelBuilder
        builder_class = ModelBuilder
    if coacd_module is None:
        import coacd as coacd_module
    current = ContextVar(f"robolab_{asset_name}_mesh", default=None)
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
        selected = [i for i in indices if builder.shape_label[i].split("/")[-1] in mesh_names
                    and label_fragment in builder.shape_label[i]]
        if not selected:
            return original_approximate(builder, method=method, shape_indices=indices, **kwargs)
        if loaded is None:
            loaded = load_colliders(path, mesh_names=mesh_names, max_parts=max_parts, asset_name=asset_name)
        manifest, parts = loaded
        result = set()
        for i in selected:
            label = builder.shape_label[i]
            entry = manifest["meshes"][label.split("/")[-1]]
            mesh = builder.shape_source[i]
            if geometry_key(mesh.vertices, mesh.indices) != entry["source_geometry_sha256"]:
                raise RuntimeError(f"{asset_name} source geometry changed at {label}; rebuild and revalidate collision parts")
            scale = np.asarray(builder.shape_scale[i])
            if not np.isfinite(scale).all() or np.max(np.abs(scale)) > manifest["max_validated_scale"] * (1 + 1e-6):
                raise RuntimeError(f"{asset_name} scale exceeds the validated scale at {label}")
            seen = []
            token = current.set((entry, parts, seen))
            before_count = len(builder.shape_source)
            try:
                changed = original_approximate(builder, method=method, shape_indices=[i],
                                               **dict(kwargs, raise_on_failure=True))
            finally:
                current.reset(token)
            if len(seen) != entry["source_component_count"] or i not in changed:
                raise RuntimeError(f"{asset_name} import did not consume the expected components at {label}")
            created = [i] + [j for j in range(before_count, len(builder.shape_source))
                             if not builder.shape_label[j].endswith("_visual")]
            if len(created) != entry["part_count"]:
                raise RuntimeError(f"Unexpected {asset_name} collider count at {label}; refusing importer fallback")
            result |= changed
            report = {"shape": label, "parts": entry["part_count"], "asset_sha256": manifest["collision_asset_sha256"]}
            reports.append(report)
            print(f"ROBOLAB_{asset_name.upper()}_COLLIDERS=" + json.dumps(report), flush=True)
        remaining = [i for i in indices if i not in selected]
        if remaining:
            result |= original_approximate(builder, method=method, shape_indices=remaining, **kwargs)
        return result

    builder_class.approximate_meshes = approximate
    coacd_module.run_coacd = decompose
    try:
        yield reports
    finally:
        coacd_module.run_coacd = original_decompose
        builder_class.approximate_meshes = original_approximate
