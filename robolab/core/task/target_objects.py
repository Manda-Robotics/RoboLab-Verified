# SPDX-License-Identifier: Apache-2.0
"""Which scene objects a subtask is *about*.

The event tracker needs to know, per env, which objects the policy is supposed
to handle (so a grasp of anything else is ``WRONG_OBJECT_GRABBED``) and which
objects are destinations (so brushing the bin while releasing is not a wrong
grab). Upstream derived the first set from ``Subtask.group_names``; that works
for the dict / ``pick_and_place`` forms, where groups are keyed by object name,
but the list-of-callables form names its groups ``group1..N`` and the keyword
form names its single group ``conditions`` — so for those tasks the target
itself was flagged as a wrong object on every grasp (VERIFIED_PLAN B4, H-B2,
H-R6-12, H-R8-22).

Here the names come from the conditions themselves: every condition is a
``functools.partial`` over a predicate whose keyword arguments name the objects
it tests. No task file needs to change.
"""
from __future__ import annotations

from functools import partial
from typing import Callable, Iterable

# Keyword arguments that name the object(s) a predicate manipulates.
TARGET_KWARGS = ("object", "objects")
# Keyword arguments that name where those objects go / what they rest on.
CONTAINER_KWARGS = ("container", "containers", "surface")
# Keyword arguments naming objects that are only referenced, never handled
# (``object_left_of(object=cube, reference_object=bowl)``).
REFERENCE_KWARGS = ("reference_object", "reference_objects")


def _names(value) -> set[str]:
    if value is None:
        return set()
    if isinstance(value, str):
        return {value}
    if isinstance(value, dict):
        # ``object_groups_in_containers`` style: {"object": [...], "container": ...}
        out = set()
        for k in TARGET_KWARGS:
            out |= _names(value.get(k))
        return out
    if isinstance(value, Iterable):
        out = set()
        for v in value:
            out |= _names(v)
        return out
    return set()


def _kwargs_of(condition: Callable) -> dict:
    """Bound keyword arguments of a condition, following nested partials."""
    kw: dict = {}
    while isinstance(condition, partial):
        kw = {**condition.keywords, **kw}
        condition = condition.func
    return kw


def condition_objects(condition: Callable, kwargs: tuple[str, ...] = TARGET_KWARGS) -> set[str]:
    """Object names bound to ``kwargs`` on ``condition`` (empty for non-partials)."""
    kw = _kwargs_of(condition)
    out: set[str] = set()
    for k in kwargs:
        if k in kw:
            out |= _names(kw[k])
    if "groups" in kw and kwargs is TARGET_KWARGS:
        out |= _names(kw["groups"])
    return out


def _iter_conditions(subtask) -> Iterable[Callable]:
    conditions = getattr(subtask, "conditions", None) or {}
    if isinstance(conditions, dict):
        for group in conditions.values():
            for item in group:
                yield item[0] if isinstance(item, tuple) else item
    else:
        for item in conditions:
            yield item[0] if isinstance(item, tuple) else item


def subtask_targets(subtask, objects_in_scene: Iterable[str] | None = None) -> set[str]:
    """Objects the subtask handles.

    Union of the ``object``/``objects`` kwargs of every condition. If none of
    the conditions is a partial (nothing to read), fall back to the group
    names — filtered to real scene objects when ``objects_in_scene`` is given,
    so ``group1`` can never leak through.
    """
    names: set[str] = set()
    for cond in _iter_conditions(subtask):
        names |= condition_objects(cond, TARGET_KWARGS)
    if not names:
        names = set(getattr(subtask, "group_names", []) or [])
        if objects_in_scene is not None:
            names &= set(objects_in_scene)
    return names


def subtask_containers(subtask) -> set[str]:
    """Destinations / support surfaces named by the subtask's conditions."""
    names: set[str] = set()
    for cond in _iter_conditions(subtask):
        names |= condition_objects(cond, CONTAINER_KWARGS)
    return names


def task_targets(subtasks: Iterable, objects_in_scene: Iterable[str] | None = None) -> set[str]:
    """Every object any stage of the task handles (H-R8-6: a later stage's
    target is not a wrong object; H-R8-24: neither is a just-completed one)."""
    out: set[str] = set()
    for st in subtasks or []:
        out |= subtask_targets(st, objects_in_scene)
    return out


def task_containers(subtasks: Iterable) -> set[str]:
    out: set[str] = set()
    for st in subtasks or []:
        out |= subtask_containers(st)
    return out
