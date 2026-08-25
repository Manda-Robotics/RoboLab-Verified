"""Target/container extraction from subtask conditions (VERIFIED_PATCHES P14).

Pure-Python: builds partials over dummy predicates in the three authoring forms
RoboLab tasks use, plus an AST sweep over every benchmark task asserting that
no subtask's target set is empty or a synthetic group name.
"""
import ast
from functools import partial
from pathlib import Path
from types import SimpleNamespace

from robolab.core.task.target_objects import (
    condition_objects, subtask_containers, subtask_targets, task_containers, task_targets,
)

ROOT = Path(__file__).resolve().parents[1]


def grabbed(env, object, env_id=None): ...
def in_container(env, object, container, env_id=None): ...
def left_of(env, object, reference_object, env_id=None): ...
def stacked(env, objects, order=None, env_id=None): ...
def groups_in_containers(env, groups, env_id=None): ...


def _subtask(conditions, group_names=None):
    return SimpleNamespace(conditions=conditions, group_names=group_names or list(conditions))


def test_list_form_reads_object_kwargs_not_group_names():
    st = _subtask({
        "group1": [(partial(grabbed, object="rubiks_cube"), 0.33)],
        "group2": [(partial(left_of, object="rubiks_cube", reference_object="bowl"), 0.33)],
        "group3": [(partial(grabbed, object="rubiks_cube"), 0.34)],
    })
    assert subtask_targets(st) == {"rubiks_cube"}          # not group1..3, not the bowl
    assert subtask_containers(st) == set()


def test_pick_and_place_form():
    st = _subtask({
        "mustard": [(partial(grabbed, object="mustard"), 0.0),
                    (partial(in_container, object="mustard", container="bin_b03"), 1.0)],
        "sugar_box": [(partial(grabbed, object="sugar_box"), 0.0),
                      (partial(in_container, object="sugar_box", container="bin_b03"), 1.0)],
    })
    assert subtask_targets(st) == {"mustard", "sugar_box"}
    assert subtask_containers(st) == {"bin_b03"}


def test_objects_list_and_groups_dict():
    st = _subtask({"conditions": [(partial(stacked, objects=["red", "blue"]), 1.0)]})
    assert subtask_targets(st) == {"red", "blue"}
    g = partial(groups_in_containers, groups=[{"object": ["a"], "container": "x"}, {"object": ["b", "c"], "container": "y"}])
    assert condition_objects(g) == {"a", "b", "c"}


def test_fallback_to_group_names_filtered_by_scene():
    st = _subtask({"banana": [(grabbed, 1.0)]})            # bare callable, nothing to read
    assert subtask_targets(st, objects_in_scene=["banana", "table"]) == {"banana"}
    st2 = _subtask({"group1": [(grabbed, 1.0)]})
    assert subtask_targets(st2, objects_in_scene=["banana", "table"]) == set()


def test_task_level_union():
    a = _subtask({"mustard": [(partial(in_container, object="mustard", container="bin_b03"), 1.0)]})
    b = _subtask({"coffee_can": [(partial(in_container, object="coffee_can", container="bin_a06"), 1.0)]})
    assert task_targets([a, b]) == {"mustard", "coffee_can"}
    assert task_containers([a, b]) == {"bin_a06", "bin_b03"}


def _static_targets(call: ast.Call) -> set[str]:
    out = set()
    for kw in call.keywords:
        if kw.arg in ("object", "objects"):
            try:
                v = ast.literal_eval(kw.value)
            except Exception:
                continue
            out |= {v} if isinstance(v, str) else set(v)
    return out


def test_every_benchmark_subtask_has_readable_targets():
    """Static mirror of the runtime rule: each Subtask(...) / pick_and_place(...) in
    robolab/tasks/benchmark binds at least one object= / objects= literal."""
    problems = []
    for f in sorted((ROOT / "robolab/tasks/benchmark").glob("*.py")):
        tree = ast.parse(f.read_text())
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            fname = getattr(node.func, "id", None) or getattr(node.func, "attr", None)
            if fname not in ("Subtask", "pick_and_place", "pick_and_place_grouped", "pick_and_place_on_surface"):
                continue
            targets = set()
            for inner in ast.walk(node):
                if isinstance(inner, ast.Call):
                    targets |= _static_targets(inner)
            if not targets:
                problems.append(f"{f.name}:{node.lineno}")
    assert not problems, f"subtasks with no object=/objects= literal: {problems}"
