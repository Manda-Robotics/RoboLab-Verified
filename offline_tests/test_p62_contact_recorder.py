# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""P62: per-pad gripper contact is recorded, so a tow can be told from a wedge.

Finn labelled six open-hand carries: the three real tows are held by one pad, and
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


def _run(contact_fn, objs=("banana", "bowl")):
    # predicate_logic pulls in isaaclab, which is not available offline; the helper
    # imports it lazily, so a stub module is enough to exercise the logic.
    stub = types.ModuleType("robolab.core.task.predicate_logic")
    stub.in_contact = contact_fn
    sys.modules["robolab.core.task.predicate_logic"] = stub
    return G.pad_contact_columns(_Env(), object(), objs)


def test_shape_and_dtype():
    out = _run(lambda w, obj, pad, env_id=None: torch.tensor([True, False]))
    assert set(out) == {"banana", "bowl"}
    assert out["banana"].shape == (2, 2) and out["banana"].dtype == torch.uint8


def test_single_pad_hold_is_visible_as_a_tow():
    out = _run(lambda w, obj, pad, env_id=None: torch.tensor([pad == "gripper_left", False]))
    assert out["banana"][0].tolist() == [1, 0]


def test_wedged_object_reads_both_pads():
    out = _run(lambda w, obj, pad, env_id=None: torch.tensor([True, True]))
    assert out["banana"][0].tolist() == [1, 1], "wedged between both pads is not a tow"


def test_failing_lookup_degrades_to_no_contact():
    def boom(w, obj, pad, env_id=None):
        raise RuntimeError("sensor missing")
    out = _run(boom)
    assert out["banana"].sum().item() == 0, "recording must never take a run down"


def test_scalar_result_is_broadcast_to_every_env():
    out = _run(lambda w, obj, pad, env_id=None: True)
    assert out["banana"].shape == (2, 2)


def test_no_objects_records_nothing():
    assert _run(lambda *a, **k: torch.tensor([True]), objs=()) == {}
