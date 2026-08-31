# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""P65 (H-R9-T4): the last rung of a ladder must be the task's success predicate.

`GrabABagel` / `GrabAFruit` succeed on `object_picked_up(..., distance=0.05)` — a
50 mm lift — while their ladders were `object_grabbed` (contact) alone, so
"Completed subtask" fired on episodes the task scored as failures. Anchor:
`isaac60_robolab120_pi05/GrabAFruitTask` env1 logs
"Completed subtask 'grab_a_fruit' 1/1" at 6.20 s with `success: false`.

Static and Isaac-free: read the two task files and check that every object named
by a `success*` termination has a ladder group ending in the same predicate.
"""
import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TASKS = ("grab_a_bagel.py", "grab_a_fruit.py")


def _literal_str_kwarg(call, name):
    for k in call.keywords:
        if k.arg == name and isinstance(k.value, ast.Constant) and isinstance(k.value.value, str):
            return k.value.value
    return None


def _success_terms(tree):
    """{object_name: predicate_func_name} from every `success*` DoneTerm."""
    out = {}
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign):
            continue
        if not any(isinstance(t, ast.Name) and t.id.startswith("success") for t in node.targets):
            continue
        for call in ast.walk(node.value):
            if not isinstance(call, ast.Call):
                continue
            func = None
            obj = None
            for k in call.keywords:
                if k.arg == "func" and isinstance(k.value, ast.Name):
                    func = k.value.id
                if k.arg == "params" and isinstance(k.value, ast.Dict):
                    for key, val in zip(k.value.keys, k.value.values):
                        if isinstance(key, ast.Constant) and key.value == "object" \
                                and isinstance(val, ast.Constant):
                            obj = val.value
            if func and obj:
                out[obj] = func
    return out


def _ladder_last_rungs(tree):
    """{group_name: last condition's predicate func name} for dict-form Subtasks."""
    out = {}
    for call in ast.walk(tree):
        if not (isinstance(call, ast.Call) and getattr(call.func, "id", None) == "Subtask"):
            continue
        for k in call.keywords:
            if k.arg != "conditions" or not isinstance(k.value, ast.Dict):
                continue
            for key, val in zip(k.value.keys, k.value.values):
                if not (isinstance(key, ast.Constant) and isinstance(val, ast.List) and val.elts):
                    continue
                last = val.elts[-1]
                if isinstance(last, ast.Call) and last.args and isinstance(last.args[0], ast.Name):
                    out[key.value] = last.args[0].id
    return out


def test_every_success_object_has_a_ladder_ending_in_the_success_predicate():
    for name in TASKS:
        tree = ast.parse((ROOT / "robolab/tasks/benchmark" / name).read_text())
        success = _success_terms(tree)
        last = _ladder_last_rungs(tree)
        assert success, f"{name}: no success* terms parsed"
        for obj, func in success.items():
            assert obj in last, f"{name}: '{obj}' can succeed but has no ladder group"
            assert last[obj] == func, (
                f"{name}: ladder for '{obj}' ends in {last[obj]}, success needs {func} "
                "— completion would not imply success (H-R9-T4)"
            )


def test_the_lift_distance_matches_the_success_term():
    """A ladder ending in object_picked_up with a different distance would still lie."""
    for name in TASKS:
        tree = ast.parse((ROOT / "robolab/tasks/benchmark" / name).read_text())
        distances = set()
        for call in ast.walk(tree):
            if isinstance(call, ast.Call) and call.args and getattr(call.args[0], "id", None) == "object_picked_up":
                for k in call.keywords:
                    if k.arg == "distance" and isinstance(k.value, ast.Constant):
                        distances.add(k.value.value)
        assert distances == {0.05}, f"{name}: ladder lift distances {distances} != success's 0.05"
