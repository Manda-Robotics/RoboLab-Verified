"""Scoped Newton 1.2 mesh-import compatibility, independent of object names.

Isaac Lab beta2's cloner skips authored approximations and convexifies every
mesh. Restore Newton's authored-approximation path; also replace Newton 1.2's
coarse CoACD threshold (0.5) and shallow search with explicit quality settings.
Keep upstream mesh/body/material/filter conversion. Neither source USD nor
installed packages change.
This process-global compatibility shim is only for single-simulation processes;
it restores upstream functions even if environment initialization fails.
"""

from contextlib import contextmanager
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import tempfile
import weakref

import numpy as np

_active = False


def decomposition_key(vertices, indices, settings, version):
    digest = hashlib.sha256()
    for array, dtype in ((vertices, "<f8"), (indices, "<i8")):
        value = np.ascontiguousarray(array, dtype=dtype)
        digest.update(str(value.shape).encode())
        digest.update(value.tobytes())
    digest.update(json.dumps({"settings": settings, "coacd": version}, sort_keys=True).encode())
    return digest.hexdigest()


def validate_parts(parts):
    if not parts:
        raise ValueError("Convex decomposition returned no parts")
    for vertices, faces in parts:
        if (vertices.ndim != 2 or vertices.shape[1] != 3 or len(vertices) < 4
                or not np.isfinite(vertices).all() or faces.ndim != 2
                or faces.shape[1] != 3 or len(faces) < 4
                or not np.issubdtype(faces.dtype, np.integer)
                or faces.min() < 0 or faces.max() >= len(vertices)):
            raise ValueError("Invalid convex decomposition geometry")


@contextmanager
def mesh_decomposition_context(threshold=0.05, cache_dir=None, contact_timeconst=None):
    """Improve every requested convex decomposition, preserving other methods.

    ``threshold`` is CoACD's normalized concavity tolerance, NOT PhysX's
    errorPercentage and NOT a penetration tolerance in metres.
    """
    global _active
    if not math.isfinite(threshold) or not 0 < threshold < 1:
        raise ValueError("Decomposition threshold must be finite and between zero and one")
    if contact_timeconst is not None and (not math.isfinite(contact_timeconst) or contact_timeconst <= 0):
        raise ValueError("Contact time constant must be finite and positive")
    if _active:
        raise RuntimeError("Concurrent/nested Newton mesh-import contexts are unsupported")
    if importlib.metadata.version("newton") != "1.2.1":
        raise RuntimeError("Re-audit mesh import before using this compatibility shim with a different Newton version")
    import coacd
    from newton import ModelBuilder

    version = importlib.metadata.version("coacd")
    cache = Path(cache_dir) if cache_dir else Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache")) / "robolab/newton-convex-v1"
    cache.mkdir(parents=True, exist_ok=True)
    original_approximate = ModelBuilder.approximate_meshes
    original_add_usd = ModelBuilder.add_usd
    original_decompose = coacd.run_coacd
    reports = []
    authored_builders = weakref.WeakSet()

    def add_usd(builder, *args, **kwargs):
        restore_authored = kwargs.get("skip_mesh_approximation") is True
        if restore_authored:
            kwargs["skip_mesh_approximation"] = False
        defaults = builder.default_shape_cfg if contact_timeconst is not None else None
        if defaults is not None:
            previous = defaults.ke, defaults.kd
            defaults.ke, defaults.kd = 1 / contact_timeconst ** 2, 2 / contact_timeconst
        try:
            result = original_add_usd(builder, *args, **kwargs)
        finally:
            if defaults is not None:
                defaults.ke, defaults.kd = previous
        if restore_authored:
            authored_builders.add(builder)
            print("ROBOLAB_AUTHORED_COLLIDERS=" + str(kwargs.get("root_path", "/")), flush=True)
        return result

    def cached_decompose(mesh, **settings):
        key = decomposition_key(mesh.vertices, mesh.indices, settings, version)
        path = cache / (key + ".npz")
        hit = path.is_file()
        if hit:
            with np.load(path, allow_pickle=False) as data:
                parts = [(data[f"vertices_{i}"].copy(), data[f"faces_{i}"].copy())
                         for i in range(int(data["count"]))]
        else:
            parts = [(np.asarray(v), np.asarray(f)) for v, f in original_decompose(mesh, **settings)]
        validate_parts(parts)
        if not hit:
            arrays = {"count": np.asarray(len(parts))}
            for i, (vertices, faces) in enumerate(parts):
                arrays[f"vertices_{i}"] = vertices
                arrays[f"faces_{i}"] = faces
            with tempfile.NamedTemporaryFile(dir=cache, suffix=".npz", delete=False) as handle:
                temporary = Path(handle.name)
            try:
                np.savez_compressed(temporary, **arrays)
                os.replace(temporary, path)
            finally:
                temporary.unlink(missing_ok=True)
        report = {"mesh_sha256": key, "hulls": len(parts), "cache_hit": hit, "settings": settings}
        reports.append(report)
        print("ROBOLAB_CONVEX_DECOMPOSITION=" + json.dumps(report), flush=True)
        return parts

    def approximate(builder, method="convex_hull", shape_indices=None, **kwargs):
        # This is the cloner's blanket second pass, after add_usd completed all
        # authored approximations. Do not convexify its decomposed/visual meshes
        # again. Other explicit convex-hull calls continue through upstream.
        if (builder in authored_builders and method == "convex_hull"
                and shape_indices is None and kwargs.get("keep_visual_shapes") is True):
            authored_builders.discard(builder)
            return set()
        if method == "coacd":
            # Override only absent call-site options; retain upstream handling of
            # materials, body mass, shape transforms and collision-filter pairs.
            for key, value in {"threshold": threshold, "mcts_nodes": 20,
                               "mcts_iterations": 150, "mcts_max_depth": 3,
                               "seed": 0}.items():
                kwargs.setdefault(key, value)
            kwargs.setdefault("raise_on_failure", True)
        return original_approximate(builder, method=method, shape_indices=shape_indices, **kwargs)

    _active = True
    ModelBuilder.approximate_meshes = approximate
    ModelBuilder.add_usd = add_usd
    coacd.run_coacd = cached_decompose
    try:
        yield reports
    finally:
        ModelBuilder.approximate_meshes = original_approximate
        ModelBuilder.add_usd = original_add_usd
        coacd.run_coacd = original_decompose
        _active = False
