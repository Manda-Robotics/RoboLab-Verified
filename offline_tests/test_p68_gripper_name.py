# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""P68: `object_picked_up` takes `gripper_name`, like every other predicate.

`gripper_name` is an upstream parameter on `object_on_top`, `object_in_container`
and friends; `object_picked_up` was the only one without it, so there was no way
to write "lift this with two arms" into a success condition. A list means every
listed gripper must hold the object. Adopted so the bimanual rigs (P37) ship with
the hook that makes user-authored two-arm tasks possible — Verified ships the
rigs, not the tasks.
"""
import inspect

import robolab.core.task.conditionals as C


def test_signature_takes_gripper_name_with_the_usual_default():
    sig = inspect.signature(C.object_picked_up)
    assert "gripper_name" in sig.parameters, "the whole point of P68"
    assert sig.parameters["gripper_name"].default == "gripper", "default must be unchanged"


def test_gripper_name_is_forwarded_to_object_grabbed(monkeypatch):
    seen = {}

    def fake_grabbed(env, object, gripper_name="gripper", env_id=None):
        seen["gripper_name"] = gripper_name
        return True

    def fake_above(env, object, reference_object, env_id=None, z_margin=0.0):
        return True

    monkeypatch.setattr(C, "object_grabbed", fake_grabbed)
    monkeypatch.setattr(C, "object_above", fake_above)
    C.object_picked_up(None, object="cube", surface="table", gripper_name=["left", "right"])
    assert seen["gripper_name"] == ["left", "right"]


def test_the_default_still_reaches_object_grabbed(monkeypatch):
    seen = {}
    monkeypatch.setattr(C, "object_grabbed",
                        lambda env, object, gripper_name="gripper", env_id=None: seen.setdefault("g", gripper_name) or True)
    monkeypatch.setattr(C, "object_above",
                        lambda env, object, reference_object, env_id=None, z_margin=0.0: True)
    C.object_picked_up(None, object="cube", surface="table")
    assert seen["g"] == "gripper"


def test_positional_callers_are_unbroken():
    """`object_picked_up(env, obj, surface, 0.05)` must still mean distance=0.05."""
    sig = inspect.signature(C.object_picked_up)
    names = list(sig.parameters)
    assert names[:4] == ["env", "object", "surface", "distance"], \
        "gripper_name must come AFTER distance or positional callers silently change meaning"
