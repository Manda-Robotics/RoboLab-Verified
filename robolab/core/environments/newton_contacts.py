"""Scoped contact adapters for the pinned EA and GA Newton / MJWarp stacks.

These adapters deliberately use private solver APIs. Keep the installer pinned;
revalidate the kernel argument layout and both model arrays before upgrading.
"""
import numpy as np
from contextlib import contextmanager


def is_fingertip_pad(label):
    return ("/Gripper/Robotiq_2F_85/" in label
            and any(f"/{side}_inner_finger/" in label for side in ("left", "right"))
            and "/Defeatured_2F_85_PAD_OPEN_fingertipsstep_01/" in label
            and not label.endswith("_visual"))


@contextmanager
def single_hull_fingertips(builder_class, reports):
    """Change only explicitly requested CoACD pad meshes, before replication.

    Native conversion retains transforms, materials and visual-copy handling.
    No USD, installed package or global cache is edited. Intended for one
    simulation per process; the original method is restored on exit.
    """
    original = builder_class.approximate_meshes

    def approximate(builder, method="convex_hull", shape_indices=None, **kwargs):
        if method != "coacd" or shape_indices is None:
            return original(builder, method=method, shape_indices=shape_indices, **kwargs)
        indices = list(shape_indices)
        pads = [i for i in indices if is_fingertip_pad(builder.shape_label[i])]
        if not pads:
            return original(builder, method=method, shape_indices=indices, **kwargs)
        labels = [builder.shape_label[i] for i in pads]
        remaining = [i for i in indices if i not in pads]
        # Convex-hull calls must not inherit CoACD-specific remeshing options.
        result = original(builder, method="convex_hull", shape_indices=pads,
                          keep_visual_shapes=kwargs.get("keep_visual_shapes", False),
                          raise_on_failure=True)
        if not set(pads).issubset(result):
            raise RuntimeError("Fingertip convex approximation did not cover every pad")
        reports.extend({"label": label, "method": "convex_hull"} for label in labels)
        if remaining:
            result |= original(builder, method=method, shape_indices=remaining, **kwargs)
        return result

    builder_class.approximate_meshes = approximate
    try:
        yield
    finally:
        builder_class.approximate_meshes = original


@contextmanager
def guard_ccd_capacity(wp, stats):
    """Fail before out-of-bounds CCD reads; never truncate candidate pairs."""
    original = wp.launch

    def launch(kernel, *args, **kwargs):
        if "ccd_kernel" in getattr(kernel, "key", ""):
            values = dict(zip((a.label for a in kernel.adj.args), kwargs.get("inputs", [])))
            if not {"ncollision_in", "collision_pair_in"}.issubset(values):
                raise RuntimeError("Unsupported CCD kernel arguments; cannot safely check candidate capacity")
            if "ncollision_in" in values and "collision_pair_in" in values:
                count = int(values["ncollision_in"].numpy()[0])
                capacity = values["collision_pair_in"].shape[0]
                stats["peak_candidates"] = max(stats.get("peak_candidates", 0), count)
                stats["candidate_capacity"] = capacity
                if count < 0 or count > capacity:
                    raise RuntimeError(f"CCD candidate overflow: {count} > {capacity}; refusing unsafe kernel launch")
        return original(kernel, *args, **kwargs)

    wp.launch = launch
    try:
        yield
    finally:
        wp.launch = original


def override_contact_impedance(solver, endpoints):
    """Set global geom resistance; preserve equality constraints and solref."""
    if endpoints is None:
        return
    values = np.asarray(endpoints)
    if values.shape != (2,) or not np.isfinite(values).all() or not 0 < values[0] <= values[1] < 1:
        raise ValueError("Impedance requires finite 0 < DMIN <= DMAX < 1")
    solver.mj_model.geom_solimp[:, :2] = values
    gpu = solver.mjw_model.geom_solimp
    updated = gpu.numpy().copy()
    updated[..., :2] = values
    gpu.assign(updated)
    if not np.allclose(gpu.numpy(), solver.mj_model.geom_solimp, rtol=1e-6, atol=1e-8):
        raise RuntimeError("Contact impedance CPU/GPU mismatch")


def override_contact_response(solver, timeconst, wp):
    if timeconst is None:
        return
    dt = float(solver.mjw_model.opt.timestep.numpy().reshape(-1)[0])
    # Configuration uses Python doubles; Warp stores the timestep as float32.
    # Permit only that conversion's roundoff at the exact stability boundary.
    at_boundary = np.isclose(timeconst, 2 * dt, rtol=2 * np.finfo(np.float32).eps, atol=0)
    if not np.isfinite(timeconst) or (timeconst < 2 * dt and not at_boundary):
        raise ValueError("Contact time constant must be finite and >= twice the solver timestep")
    solver.mj_model.geom_solref[:] = [timeconst, 1.0]
    solver.mjw_model.geom_solref.fill_(wp.vec2(timeconst, 1.0))
    print(f"CONTACT_RESPONSE_OVERRIDE timeconst={timeconst} damping_ratio=1", flush=True)


_active = False


@contextmanager
def contact_runtime(settings):
    """Scope process-wide adapters to one simulation; restore on every exit.

    Eager execution is required: Python guards cannot inspect replayed graphs.
    Contact response is applied to both native models before their first step.
    USD geometry, visuals, equality constraints and PhysX are not modified.
    """
    global _active
    if _active:
        raise RuntimeError("Concurrent/nested Newton contact runtimes are unsupported")
    import warp as wp
    from newton import ModelBuilder
    from newton.solvers import SolverMuJoCo

    original = SolverMuJoCo._mujoco_warp_step
    owners = []
    reports = []
    stats = {}

    def step(solver):
        if not any(owner is solver for owner in owners):
            if settings["single_hull_fingertips"] and len(reports) != 2:
                raise RuntimeError("Expected exactly two DROID fingertip pads before replication")
            if settings.get("filter_immovable_contacts", False):
                from newton import BodyFlags
                from robolab.core.environments.newton_collision_filter import filter_immovable_contacts
                filter_immovable_contacts(solver, wp, int(BodyFlags.KINEMATIC))
            if settings.get("support_rolling_friction") is not None:
                from newton import BodyFlags
                from robolab.core.environments.newton_collision_filter import override_support_rolling
                override_support_rolling(solver, settings["support_rolling_friction"], wp,
                                         int(BodyFlags.KINEMATIC))
            override_contact_response(solver, settings["timeconst"], wp)
            override_contact_impedance(solver, settings["impedance"])
            owners.append(solver)
        return original(solver)

    from contextlib import nullcontext
    pads = single_hull_fingertips(ModelBuilder, reports) if settings["single_hull_fingertips"] else nullcontext()
    from robolab.core.environments.newton_candidates import candidate_storage
    _active = True
    try:
        with pads, candidate_storage(wp), guard_ccd_capacity(wp, stats):
            SolverMuJoCo._mujoco_warp_step = step
            try:
                yield
            finally:
                SolverMuJoCo._mujoco_warp_step = original
    finally:
        _active = False
