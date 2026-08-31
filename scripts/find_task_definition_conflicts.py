#!/usr/bin/env python3
"""Compare each benchmark task's SUCCESS termination against its SUBTASK ladder.

The success DoneTerm and the `subtasks` list are written independently, so they
can disagree about which container an object belongs in. When they do, the
subtask score measures progress toward a goal that is not the scored goal: a
policy can hold subtask score 1.0 and be structurally unable to succeed.

Emits one line per task; exit code 1 if any conflict is found.

    python scripts/find_task_definition_conflicts.py [--tasks-dir DIR]
"""
from __future__ import annotations

import argparse
import ast
import sys
from pathlib import Path

PLACEMENT_FUNCS = {"object_in_container", "object_groups_in_containers", "pick_and_place"}


def _literal(node):
    try:
        return ast.literal_eval(node)
    except (ValueError, SyntaxError):
        return None


def _pairs_from_call(call: ast.Call) -> list[tuple[str, str]]:
    """(object, container) pairs from one placement call, including `groups=[...]`."""
    kw = {k.arg: k.value for k in call.keywords if k.arg}
    out: list[tuple[str, str]] = []

    if "groups" in kw:
        for group in _literal(kw["groups"]) or []:
            if not isinstance(group, dict):
                continue
            container = group.get("container")
            objs = group.get("object") or group.get("objects")
            if container is None or objs is None:
                continue
            for obj in [objs] if isinstance(objs, str) else objs:
                out.append((obj, container))
        return out

    container = _literal(kw.get("container")) if "container" in kw else None
    objs = _literal(kw.get("object")) if "object" in kw else None
    if objs is None and "objects" in kw:
        objs = _literal(kw["objects"])
    if container is None or objs is None:
        return out
    for obj in [objs] if isinstance(objs, str) else objs:
        if isinstance(obj, str):
            out.append((obj, container))
    return out


def _pairs_from_doneterm(call: ast.Call) -> list[tuple[str, str]]:
    """DoneTerm(func=object_in_container, params={...}) -- the predicate is passed
    by reference, not called, so its arguments live in the `params` dict."""
    kw = {k.arg: k.value for k in call.keywords if k.arg}
    fn = kw.get("func")
    name = fn.id if isinstance(fn, ast.Name) else getattr(fn, "attr", None)
    if name not in PLACEMENT_FUNCS or "params" not in kw:
        return []
    params = _literal(kw["params"])
    if not isinstance(params, dict):
        return []

    out: list[tuple[str, str]] = []
    for group in params.get("groups") or []:
        if not isinstance(group, dict):
            continue
        container, objs = group.get("container"), group.get("object") or group.get("objects")
        if container is None or objs is None:
            continue
        for obj in [objs] if isinstance(objs, str) else objs:
            out.append((obj, container))
    if out:
        return out

    container = params.get("container")
    objs = params.get("object") or params.get("objects")
    if container is None or objs is None:
        return []
    for obj in [objs] if isinstance(objs, str) else objs:
        if isinstance(obj, str):
            out.append((obj, container))
    return out


def _collect(node) -> list[tuple[str, str]]:
    pairs: list[tuple[str, str]] = []
    for sub in ast.walk(node):
        if not isinstance(sub, ast.Call):
            continue
        fn = sub.func
        name = fn.id if isinstance(fn, ast.Name) else getattr(fn, "attr", None)
        if name in PLACEMENT_FUNCS:
            pairs += _pairs_from_call(sub)
        elif name == "DoneTerm":
            pairs += _pairs_from_doneterm(sub)
    return pairs


def analyse(path: Path) -> dict | None:
    tree = ast.parse(path.read_text())
    success_pairs: list[tuple[str, str]] = []
    subtask_pairs: list[tuple[str, str]] = []

    for cls in [n for n in ast.walk(tree) if isinstance(n, ast.ClassDef)]:
        if cls.name.endswith("Terminations"):
            for stmt in cls.body:
                targets = getattr(stmt, "targets", []) or [getattr(stmt, "target", None)]
                names = {t.id for t in targets if isinstance(t, ast.Name)}
                if any(n.startswith("success") for n in names):
                    success_pairs += _collect(stmt)
        else:
            for stmt in cls.body:
                targets = getattr(stmt, "targets", []) or [getattr(stmt, "target", None)]
                if any(isinstance(t, ast.Name) and t.id == "subtasks" for t in targets):
                    subtask_pairs += _collect(stmt)

    if not success_pairs or not subtask_pairs:
        return None

    success_map: dict[str, set[str]] = {}
    for obj, container in success_pairs:
        success_map.setdefault(obj, set()).add(container)
    subtask_map: dict[str, set[str]] = {}
    for obj, container in subtask_pairs:
        subtask_map.setdefault(obj, set()).add(container)

    conflicts = [
        (obj, sorted(subtask_map[obj]), sorted(success_map[obj]))
        for obj in sorted(set(subtask_map) & set(success_map))
        if subtask_map[obj] != success_map[obj]
    ]
    unscored = sorted(set(success_map) - set(subtask_map))
    return {"conflicts": conflicts, "unscored": unscored,
            "success": success_map, "subtasks": subtask_map}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks-dir", default="robolab/tasks/benchmark")
    args = ap.parse_args()

    files = sorted(Path(args.tasks_dir).glob("*_task.py"))
    if not files:
        print(f"no task files under {args.tasks_dir}", file=sys.stderr)
        return 2

    checked = bad = 0
    for path in files:
        try:
            res = analyse(path)
        except SyntaxError as exc:
            print(f"SKIP  {path.name}: {exc}")
            continue
        if res is None:
            continue
        checked += 1
        if res["conflicts"]:
            bad += 1
            print(f"\nCONFLICT  {path.name}")
            for obj, sub, suc in res["conflicts"]:
                print(f"    {obj:20s} subtask -> {', '.join(sub):12s} | success -> {', '.join(suc)}")
            if res["unscored"]:
                print(f"    (in success but no subtask: {', '.join(res['unscored'])})")

    print(f"\n{checked} tasks with both a success placement and a subtask ladder; "
          f"{bad} disagree about a container.")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
