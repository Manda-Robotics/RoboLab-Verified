# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""P62: per-pad gripper contact is recorded, so a tow can be told from a wedge.

The reviewer labelled six open-hand carries: the three real tows are held by one pad, and
the two he rejected (`redonion`, `tuna_can`) are wedged between both. Jaw-axis
geometry did not separate those classes at all -- tows measured 1.1/1.2/10.6 cm
and non-tows 3.9/0.1/2.8 cm. Per-pad contact does, and it was being computed
every step and thrown away.
"""
import sys
import types

import torch

import robolab.core.task.grasp as G


class _Env:
    num_envs = 2
    device = "cpu"


class _World:
    """Stands in for WorldState: force between two bodies, per env."""
    def __init__(self, force): self._f = force
    def get_contact_force(self, a, b, env_id=None): return self._f(a, b)


def _run(contact_fn, objs=("banana", "bowl"), force=None):
    """Stub predicate_logic just for this call, then put sys.modules back.

    The helper imports it lazily, so a stub is enough. Leaving the stub installed
    breaks every later test module that imports real names from it (`_and` was the
    casualty) -- pytest shares one interpreter across files.
    """
    name = "robolab.core.task.predicate_logic"
    saved = sys.modules.get(name)
    stub = types.ModuleType(name)
    stub.in_contact = contact_fn
    sys.modules[name] = stub
    world = _World(force or (lambda a, b: torch.zeros(2, 3)))
    try:
        return G.pad_contact_columns(_Env(), world, objs)
    finally:
        if saved is not None:
            sys.modules[name] = saved
        else:
            sys.modules.pop(name, None)


def test_pad_forces_have_one_column_per_pad():
    out = _run(lambda w, obj, pad, env_id=None: torch.tensor([True, False]))
    assert out["banana"].shape == (2, 2), "(num_envs, [left, right])"
    assert out["banana"].dtype == torch.float32, "force, not a boolean"


def test_a_firm_grip_and_a_magnetic_hold_are_distinguishable():
    """The whole point of P77: a boolean cannot tell these apart, a force can."""
    firm = _run(None, force=lambda a, b: torch.full((2, 3), 3.0))
    ghost = _run(None, force=lambda a, b: torch.full((2, 3), 1e-4))
    assert firm["banana"][0].sum() > 1.0
    assert ghost["banana"][0].sum() < 0.01


def test_destination_contact_is_recorded_for_containers():
    out = _run(lambda w, obj, dest, env_id=None: torch.tensor([True, False]))
    # 'bowl' looks like a destination, so banana-vs-bowl contact is recorded
    assert "banana__bowl" in out, "placement predicates need object-to-container contact"
    assert out["banana__bowl"].dtype == torch.uint8


def test_failing_lookup_degrades_to_zero():
    def boom(*a, **k):
        raise RuntimeError("sensor missing")
    out = _run(boom, force=boom)
    assert out["banana"].sum().item() == 0, "recording must never take a run down"


def test_no_destination_columns_between_two_destinations():
    out = _run(lambda w, obj, dest, env_id=None: torch.tensor([True, True]))
    assert "bowl__bowl" not in out


def test_no_objects_records_nothing():
    assert _run(lambda *a, **k: torch.tensor([True]), objs=()) == {}
