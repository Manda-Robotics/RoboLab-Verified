# SPDX-License-Identifier: Apache-2.0
"""Bounded candidate-only storage growth for pinned MuJoCo-Warp, in eager mode.

Broadphase candidates are not contacts. MJWarp initially gives both the same
capacity; a dense group of convex pieces can exhaust candidates long before it
exhausts contacts. Retry broadphase before any narrowphase/solver side effect.
"""
from contextlib import contextmanager
from functools import wraps


class _BroadphaseData:
    def __init__(self, data, capacity):
        self._data = data
        self.naconmax = capacity

    def __getattr__(self, name):
        return getattr(self._data, name)


def candidate_launch_dimensions(kernel, kwargs):
    """Cover enlarged candidates without raising the contact-output capacity."""
    labels = [arg.label for arg in kernel.adj.args]
    values = dict(zip(labels, kwargs.get("inputs", [])))
    if not {"ncollision_in", "collision_pair_in", "naconmax_in"}.issubset(values):
        return kwargs
    capacity = values["collision_pair_in"].shape[0]
    if capacity <= values["naconmax_in"] or "grid_stride_in" in values:
        return kwargs  # CCD grid-stride kernels already iterate all candidates.
    key = kernel.key
    if "primitive_narrowphase" in key or "ccd_hfield" in key:
        if not isinstance(kwargs.get("dim"), int):
            raise RuntimeError("Unsupported candidate narrowphase launch dimensions")
        return {**kwargs, "dim": capacity}
    if "sdf_narrowphase" in key:
        dim = kwargs.get("dim")
        if not isinstance(dim, tuple) or len(dim) != 2:
            raise RuntimeError("Unsupported SDF candidate launch dimensions")
        return {**kwargs, "dim": (dim[0], capacity)}
    raise RuntimeError(f"Unsupported candidate narrowphase kernel: {key}")


@contextmanager
def candidate_storage(wp, driver=None, max_multiplier=8):
    """Grow three temporary arrays, retaining contact/constraint/CCD budgets.

    Both SAP and NXN are supported, including incremental sleeping passes.
    Broadphase only writes its candidate arrays and counter. On overflow,
    allocate the next power-of-two capacity and rerun from a zero counter;
    never clamp the count or let narrowphase consume a partial list.
    Growth is bounded to eight times the original per-batch contact capacity.
    This adapter requires eager execution, as does the existing CCD guard.
    """
    if driver is None:
        from mujoco_warp._src import collision_driver as driver
    originals = {name: getattr(driver, name) for name in ("nxn_broadphase", "sap_broadphase")}
    original_launch = wp.launch
    cache = {}  # Keep data objects alive so identities cannot be recycled.

    def wrap(original):
        @wraps(original)
        def broadphase(model, data, ctx, awake_prev=None):
            entry = cache.get(id(data))
            if entry is None:
                entry = {"data": data, "capacity": data.naconmax, "arrays": None,
                         "stats": {"contact_capacity": data.naconmax,
                                   "candidate_capacity": data.naconmax,
                                   "peak_candidates": 0, "growth_events": 0}}
                cache[id(data)] = entry
                data._robolab_candidate_storage = entry["stats"]
            if entry["arrays"] is not None:
                for name, array in entry["arrays"].items():
                    setattr(ctx, name, array)
            while True:
                view = _BroadphaseData(data, entry["capacity"])
                original(model, view, ctx, awake_prev)
                count = int(data.ncollision.numpy()[0])
                entry["stats"]["peak_candidates"] = max(entry["stats"]["peak_candidates"], count)
                if count < 0:
                    raise RuntimeError("Negative collision candidate count")
                if count <= entry["capacity"]:
                    return
                capacity = 1 << (count - 1).bit_length()
                limit = data.naconmax * max_multiplier
                if capacity > limit:
                    raise RuntimeError(f"Collision candidates require {capacity} slots, above bounded limit {limit}")
                arrays = {name: wp.empty(capacity, dtype=getattr(ctx, name).dtype,
                                         device=getattr(ctx, name).device)
                          for name in ("collision_pair", "collision_pairid", "collision_worldid")}
                entry.update(capacity=capacity, arrays=arrays)
                entry["stats"].update(candidate_capacity=capacity,
                                       growth_events=entry["stats"]["growth_events"] + 1)
                for name, array in arrays.items():
                    setattr(ctx, name, array)
                data.ncollision.zero_()
                print("COLLISION_CANDIDATE_GROWTH", entry["stats"], flush=True)
        return broadphase

    @wraps(original_launch)
    def launch(kernel, *args, **kwargs):
        if getattr(kernel, "adj", None) is not None:
            kwargs = candidate_launch_dimensions(kernel, kwargs)
            labels = [arg.label for arg in kernel.adj.args] if "ccd_kernel" in kernel.key else []
            inputs = kwargs.get("inputs", [])
            values = dict(zip(labels, inputs))
            if ("grid_stride_in" in values
                    and values["collision_pair_in"].shape[0] > values["naconmax_in"]):
                # MJWarp 3.12 bounds the CCD loop by min(ncollision, naconmax).
                # Enlarging only candidate storage would silently skip later
                # pairs. Process bounded views into that storage instead. Keep
                # the original contact output budget, counter, CCD scratch and
                # overflow checks across launches; never enlarge naconmax.
                limit = values["naconmax_in"]
                count = int(values["ncollision_in"].numpy()[0])
                capacity = values["collision_pair_in"].shape[0]
                if count < 0 or count > capacity:
                    raise RuntimeError(f"CCD candidate overflow: {count} > {capacity}")
                if count > limit:
                    if limit <= 0:
                        raise RuntimeError("CCD requires positive contact capacity")
                    names = ("collision_pair_in", "collision_pairid_in", "collision_worldid_in")
                    if not all(name in values and values[name].shape[0] >= count for name in names):
                        raise RuntimeError("Unsupported CCD candidate array layout")
                    counter = values["ncollision_in"]
                    batch_counter = wp.empty(1, dtype=counter.dtype, device=counter.device)
                    result = None
                    for start in range(0, count, limit):
                        end = min(start + limit, count)
                        batch_counter.fill_(end - start)
                        batch = list(inputs)
                        batch[labels.index("ncollision_in")] = batch_counter
                        for name in names:
                            batch[labels.index(name)] = values[name][start:end]
                        result = original_launch(kernel, *args, **{**kwargs, "inputs": batch})
                    return result
        return original_launch(kernel, *args, **kwargs)

    try:
        for name, original in originals.items():
            setattr(driver, name, wrap(original))
        wp.launch = launch
        yield
    finally:
        for name, original in originals.items():
            setattr(driver, name, original)
        wp.launch = original_launch
        for entry in cache.values():
            print("COLLISION_CANDIDATE_SUMMARY", entry["stats"], flush=True)
