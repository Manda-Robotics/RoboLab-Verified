# SPDX-FileCopyrightText: Copyright (c) 2026 Manda Robotics
# SPDX-License-Identifier: Apache-2.0
"""Exclude response contacts between immovable bodies in pinned MuJoCo-Warp.

Newton represents some kinematic bodies with free joints and large armature.
Native MuJoCo collision filtering consequently treats them as movable. Contacts
between overlapping kinematic scenery can then perturb their internal solver
state. Those bodies have prescribed motion and cannot respond to each other.
"""
import numpy as np


def immovable_body_mask(model, body_mapping, body_flags, kinematic_flag):
    """Classify using Newton's actual mapping, including replicated worlds.

    A jointless child follows its parent; a jointless child of a dynamic body
    is therefore NOT static. Unknown bodies with joints remain collision-active.
    Different mobility across worlds cannot share MJWarp's single pair table.
    """
    mapping = np.asarray(body_mapping)
    flags = np.asarray(body_flags)
    if mapping.ndim != 2 or mapping.shape[1] != model.nbody or mapping.shape[0] == 0:
        raise RuntimeError("Unsupported MuJoCo/Newton body mapping")
    if np.any(mapping < -1) or np.any(mapping >= len(flags)):
        raise RuntimeError("Invalid Newton body index in MuJoCo mapping")
    fixed = np.zeros(mapping.shape, dtype=bool)
    fixed[:, 0] = True
    for bid in range(1, model.nbody):
        parent = int(model.body_parentid[bid])
        if not 0 <= parent < bid:
            raise RuntimeError("Unsupported MuJoCo body ordering")
        ids = mapping[:, bid]
        known = ids >= 0
        fixed[known, bid] = (flags[ids[known]] & kinematic_flag) != 0
        fixed[:, bid] |= model.body_mocapid[bid] >= 0
        if model.body_jntnum[bid] == 0:
            fixed[:, bid] |= fixed[:, parent]
    if not np.all(fixed == fixed[0]):
        raise RuntimeError("Immovable-contact filtering requires identical body mobility in every world")
    return fixed[0]


def filter_immovable_contacts(solver, wp, kinematic_flag):
    """Update both native pair caches before stepping; retain collision sensors.

    This adapter targets MJWarp 3.11's native contacts only. It does not change
    geom masks, materials, dynamic pairs, or explicit dynamic contact pairs.
    Both columns of pairid matter: -2 disables contact response, while the
    second column can still request geometric collision sensing. CPU geometric
    recomputation from mj_model does not include this GPU response filter.
    Body mobility must remain fixed for the lifetime of this RoboLab runtime.
    """
    if not solver._use_mujoco_contacts:
        return {"applied": False, "reason": "Newton collision pipeline"}
    model, gpu = solver.mj_model, solver.mjw_model
    fixed = immovable_body_mask(model, solver.mjc_body_to_newton.numpy(),
                                solver.model.body_flags.numpy(), kinematic_flag)
    geom_fixed = fixed[model.geom_bodyid]
    pairs = gpu.nxn_geom_pair.numpy()
    ids = gpu.nxn_pairid.numpy().copy()
    if pairs.ndim != 2 or pairs.shape[1] != 2 or ids.shape != pairs.shape:
        raise RuntimeError("Unsupported MuJoCo-Warp collision pair cache")
    drop = geom_fixed[pairs[:, 0]] & geom_fixed[pairs[:, 1]]
    removed = int(np.count_nonzero(drop & (ids[:, 0] > -2)))
    ids[drop, 0] = -2
    keep = (ids[:, 0] > -2) | (ids[:, 1] >= 0)
    # Allocate before mutation. The full table is used by SAP; the compact
    # table is used by NXN. Updating just masks or one table is insufficient.
    replacements = {}
    for field, values in (("nxn_geom_pair_filtered", pairs[keep]),
                          ("nxn_pairid_filtered", ids[keep])):
        previous = getattr(gpu, field)
        replacements[field] = wp.array(values, dtype=previous.dtype, device=previous.device)
    gpu.nxn_pairid.assign(ids)
    for field, value in replacements.items():
        setattr(gpu, field, value)
    report = {"applied": True, "immovable_bodies": int(fixed.sum()),
              "immovable_geoms": int(geom_fixed.sum()), "removed_contact_pairs": removed,
              "worlds": int(solver.mjc_body_to_newton.shape[0])}
    solver._robolab_immovable_contact_filter = report
    print("IMMOVABLE_CONTACT_FILTER", report, flush=True)
    return report


def override_support_rolling(solver, coefficient, wp, kinematic_flag):
    """Opt-in rolling resistance only for contacts with immovable supports.

    Coefficient is a length in metres, not sliding friction. Raising only the
    support contact dimension leaves object/finger and other dynamic/dynamic
    contact dimensions unchanged. Existing sliding/torsional values survive.
    Both compiled models are updated, including explicit support contact pairs.
    """
    if not np.isfinite(coefficient) or coefficient <= 0:
        raise ValueError("Support rolling friction must be finite and positive")
    if not solver._use_mujoco_contacts:
        raise ValueError("Support rolling friction requires native MuJoCo contacts")
    model, gpu = solver.mj_model, solver.mjw_model
    fixed = immovable_body_mask(model, solver.mjc_body_to_newton.numpy(),
                                solver.model.body_flags.numpy(), kinematic_flag)
    selected = fixed[model.geom_bodyid]
    updates = [("geom_condim", selected, None, 6),
               ("geom_friction", selected, slice(2, 3), coefficient)]
    explicit = selected[model.pair_geom1] | selected[model.pair_geom2]
    if model.npair:
        updates.extend([("pair_dim", explicit, None, 6),
                        ("pair_friction", explicit, slice(3, 5), coefficient)])
    staged = []
    for field, mask, columns, value in updates:
        cpu = getattr(model, field).copy()
        device = getattr(gpu, field).numpy().copy()
        if columns is None:
            cpu[mask] = value
            device[..., mask] = value
        else:
            cpu[mask, columns] = value
            device[..., mask, columns] = value
        staged.append((field, cpu, device))
    for field, cpu, device in staged:
        getattr(model, field)[:] = cpu
        getattr(gpu, field).assign(device)
    report = {"coefficient_m": float(coefficient), "support_geoms": int(selected.sum()),
              "explicit_support_pairs": int(explicit.sum()),
              "dynamic_dynamic_contact_parameters_unchanged": True}
    solver._robolab_support_rolling = report
    print("SUPPORT_ROLLING_OVERRIDE", report, flush=True)
    return report
