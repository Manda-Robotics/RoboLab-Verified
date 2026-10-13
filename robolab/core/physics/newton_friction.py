# SPDX-License-Identifier: Apache-2.0
"""Material controls for the pinned Newton/MuJoCo-Warp native-contact backend.

Use the requested dynamic coefficient as MuJoCo's single sliding coefficient.
Do not emulate PhysX's static/dynamic split or its material-combination rule.
Only sliding friction changes; rolling, torsion and contact response survive.
"""
from contextlib import contextmanager
from functools import wraps
import re

import numpy as np


MAPPING = "dynamic_to_sliding"


def configure(cfg):
    """Replace generated PhysX events with a scoped Newton solver adapter."""
    from robolab.core.physics.friction import EVENT_PREFIX
    requested = getattr(cfg, "friction", None) or {"mode": "upstream"}
    targets = {"objects": {}, "gripper": {}}
    names = list(requested.get("objects") or {})
    if requested.get("mode", "upstream") == "upstream":
        from isaaclab.assets import RigidObjectCfg
        names = [k for k, v in vars(cfg.scene).items() if isinstance(v, RigidObjectCfg)]
    for name in names:
        targets["objects"][name] = getattr(cfg.scene, name).prim_path
    robot_path = getattr(cfg.scene.robot, "prim_path", "")
    for body in requested.get("gripper_bodies") or []:
        targets["gripper"][body] = robot_path + r"/(?:.*/)?" + re.escape(body)
    if requested.get("mode", "upstream") != "upstream":
        for name in [*names, "robot_pads"]:
            # Configclass fields can be inherited; None disables the term reliably.
            setattr(cfg.events, EVENT_PREFIX + name, None)
    cfg.newton_friction_targets = targets
    cfg.friction = {**requested, "physics_backend": "newton_mjwarp", "mapping": MAPPING}


def target_geoms(solver, targets):
    """Resolve all worlds through the solver's shape mapping, not body ordering."""
    mapping = np.asarray(solver.mjc_geom_to_newton_shape.numpy())
    if mapping.ndim != 2 or mapping.shape[1] != solver.mj_model.ngeom:
        raise RuntimeError("Unsupported Newton geom mapping")
    labels = solver.model.shape_label
    resolved = {"objects": {}, "gripper": {}}
    for group, entries in targets.items():
        for name, path in entries.items():
            pattern = re.compile(path.replace("{ENV_REGEX_NS}", r"/World/envs/env_[^/]+") + r"(?:/.*)?$")
            rows = []
            for world, shapes in enumerate(mapping):
                geoms = [i for i, shape in enumerate(shapes) if shape >= 0 and pattern.fullmatch(labels[shape])]
                if not geoms:
                    raise RuntimeError(f"Newton friction target {group}/{name} has no compiled geoms in world {world}")
                rows.extend((world, geom, int(shapes[geom])) for geom in geoms)
            resolved[group][name] = rows
    return resolved


def apply(solver, requested, targets):
    if not solver._use_mujoco_contacts:
        raise RuntimeError("Newton friction overrides require native MuJoCo contacts")
    resolved = target_geoms(solver, targets)
    model, gpu = solver.mj_model, solver.mjw_model
    geom = model.geom_friction.copy()
    device = gpu.geom_friction.numpy().copy()
    mu = solver.model.shape_material_mu.numpy().copy()
    assigned = {}
    for group, entries in resolved.items():
        for name, rows in entries.items():
            material = requested["objects"][name] if group == "objects" else requested["gripper"]
            value = float(material["dynamic"])
            if not np.isfinite(value) or not 0 <= value <= 20:
                raise ValueError(f"Invalid Newton sliding friction for {name}: {value}")
            for world, index, shape in rows:
                if index in assigned and assigned[index] != value:
                    raise RuntimeError("Shared Newton geom has conflicting friction requests")
                assigned[index] = value
                geom[index, 0] = value
                device[..., index, 0] = value
                mu[shape] = value
    # Explicit pairs bypass geom friction. Apply the native priority/max rule to
    # sliding directions only, keeping the pair's torsion/rolling coefficients.
    pairs = model.pair_friction.copy()
    device_pairs = gpu.pair_friction.numpy().copy()
    pair_updates = []
    for index, (a, b) in enumerate(zip(model.pair_geom1, model.pair_geom2)):
        if int(a) not in assigned and int(b) not in assigned:
            continue
        pa, pb = model.geom_priority[a], model.geom_priority[b]
        value = geom[a, 0] if pa > pb else geom[b, 0] if pb > pa else max(geom[a, 0], geom[b, 0])
        pairs[index, :2] = value
        device_pairs[..., index, :2] = value
        pair_updates.append({"pair": index, "sliding": float(value)})
    # Validate all targets before mutating any model array.
    solver.model.shape_material_mu.assign(mu)
    model.geom_friction[:] = geom
    gpu.geom_friction.assign(device)
    if model.npair:
        model.pair_friction[:] = pairs
        gpu.pair_friction.assign(device_pairs)
    solver._robolab_friction_pairs = pair_updates
    report = read_solver(solver, requested, targets)
    errors = check_applied(report)
    if errors:
        raise RuntimeError("Newton friction readback mismatch: " + "; ".join(errors))


def read_solver(solver, requested, targets):
    resolved = target_geoms(solver, targets)
    values = solver.mjw_model.geom_friction.numpy()
    mu = solver.model.shape_material_mu.numpy()
    out = {"physics_backend": "newton_mjwarp", "status": "measured", "mapping": MAPPING,
           "requested": requested, "objects": {}, "gripper": {}, "summary": {"objects": {}, "gripper": {}},
           "explicit_pairs": []}
    for group, entries in resolved.items():
        for name, indices in entries.items():
            rows = []
            for world, geom, shape in indices:
                row = values[geom] if values.ndim == 2 else values[0 if values.shape[0] == 1 else world, geom]
                rows.append({"world": world, "geom": geom, "shape": shape,
                             "sliding": float(row[0]), "torsional": float(row[1]), "rolling": float(row[2]),
                             "cpu_sliding": float(solver.mj_model.geom_friction[geom, 0]),
                             "newton_sliding": float(mu[shape])})
            out[group][name] = rows
            out["summary"][group][name] = {"shapes": len(rows), "worlds": len({r["world"] for r in rows}),
                                                   "sliding": sorted({r["sliding"] for r in rows})}
    pair_values = solver.mjw_model.pair_friction.numpy()
    for updated in getattr(solver, "_robolab_friction_pairs", []):
        index = updated["pair"]
        out["explicit_pairs"].append({**updated,
            "cpu": solver.mj_model.pair_friction[index, :2].tolist(),
            "gpu": pair_values[..., index, :2].reshape(-1, 2).tolist()})
    return out


def check_applied(applied, tol=1e-3):
    """Pure validation, including missing targets and non-finite measurements."""
    if applied.get("status") != "measured" or applied.get("mapping") != MAPPING:
        return ["Newton friction was not measured with the supported mapping"]
    requested = applied.get("requested") or {}
    problems = []
    for group, names in (("objects", requested.get("objects") or {}),
                         ("gripper", requested.get("gripper_bodies") or [])):
        for name in names:
            mat = requested["objects"][name] if group == "objects" else requested.get("gripper")
            rows = applied.get(group, {}).get(name)
            if not isinstance(rows, list) or not rows:
                problems.append(f"{group}/{name}: no Newton readback")
                continue
            if mat is None:
                continue
            expected = mat["dynamic"]
            for row in rows:
                if any(not np.isfinite(row.get(key, np.nan)) or abs(row[key] - expected) > tol
                       for key in ("sliding", "cpu_sliding", "newton_sliding")):
                    problems.append(f"{group}/{name}: expected sliding {expected}, measured {row}")
                    break
    for pair in applied.get("explicit_pairs", []):
        values = np.asarray([pair["cpu"], *pair["gpu"]])
        if not np.isfinite(values).all() or not np.allclose(values, pair["sliding"], atol=tol, rtol=0):
            problems.append(f"Explicit pair {pair['pair']}: CPU/GPU friction mismatch")
    return problems


@contextmanager
def friction_runtime(requested, targets):
    """Apply once after solver construction, before startup/step/reset; restore hook."""
    from newton.solvers import SolverMuJoCo
    original = SolverMuJoCo.__init__

    @wraps(original)
    def initialize(self, *args, **kwargs):
        original(self, *args, **kwargs)
        if requested.get("mode", "upstream") != "upstream":
            apply(self, requested, targets)

    SolverMuJoCo.__init__ = initialize
    try:
        yield
    finally:
        SolverMuJoCo.__init__ = original


def read_applied(env, requested):
    from isaaclab_newton.physics import NewtonManager
    return read_solver(NewtonManager._solver, requested, env.cfg.newton_friction_targets)
