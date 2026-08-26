#!/usr/bin/env python3
"""Static audit of every benchmark task definition against its scene.

`find_task_definition_conflicts.py` only catches one failure mode: the success
termination and the subtask ladder naming different containers.  Several other
mistakes are just as silent and just as damaging, and each has bitten us at
least once:

  A  object named in success/ladder does not exist in the scene USD
     -> the condition can never fire; the task is unwinnable or the ladder is dead.
  B  object named in success/ladder is missing from `contact_object_list`
     -> no contact sensor is built for it, so grasp/contact predicates on it are
        silently always-false (this is the class of bug P48 fixed for the gripper).
  C  `contact_object_list` names something that is not in the scene
     -> a sensor is requested for a prim that does not exist (dead entry, and it
        hides typos like `redonion` vs `red_onion`).
  D  success requires an object the ladder never mentions
     -> the ladder can read 1.0 while the task is structurally unfinished
        (this is what H-R8-2 was: `sugar_box` missing from FoodPackingByColor).
  E  ladder and success disagree about `logical`/`K`
     -> partial credit is measured against a different rule than the score.
  F  duplicate task class names across task subfolders
     -> os.walk order decides which definition wins (the FruitsOrangesOnPlate
        episode_length_s 800-vs-90 bug documented in task_utils.resolve_task_path).

Read-only. Emits one section per finding class; exit 1 if anything is found.

    python scripts/audit_task_definitions.py [--tasks-dir DIR] [--scenes-dir DIR]
"""
from __future__ import annotations

import argparse
import ast
import sys
from collections import defaultdict
from pathlib import Path

PLACEMENT_FUNCS = {
    "object_in_container", "object_groups_in_containers", "pick_and_place",
    "object_on_top", "object_on_center", "object_behind", "object_in_front",
    "object_left_of", "object_right_of", "object_grabbed", "object_moved",
    "object_reoriented", "object_lifted", "object_dropped", "object_upright",
}
# kwargs whose value names a scene object
OBJECT_KWARGS = {
    "object", "objects", "container", "reference_object", "target",
    "surface", "on_top_of", "reference", "support", "destination",
}


def _literal(node):
    try:
        return ast.literal_eval(node)
    except (ValueError, SyntaxError):
        return None


TARGET_KWARGS = {"object", "objects"}          # the thing the robot must move
REFERENCE_KWARGS = OBJECT_KWARGS - TARGET_KWARGS  # containers, surfaces, landmarks


def _names_from_mapping(d: dict) -> tuple[set[str], dict]:
    """Scene-object names inside a literal params/group dict, plus its logical/K."""
    names: set[str] = set()
    meta: dict = {}
    if not isinstance(d, dict):
        return names, meta
    for group in d.get("groups") or []:
        n, m = _names_from_mapping(group)
        names |= n
        for k, v in m.items():
            meta.setdefault(k, v)
    for key in OBJECT_KWARGS:
        v = d.get(key)
        vals = [v] if isinstance(v, str) else [x for x in (v or []) if isinstance(x, str)] if isinstance(v, (list, tuple)) else []
        names.update(vals)
        if key in TARGET_KWARGS:
            meta.setdefault("_targets", set()).update(vals)
    for key in ("logical", "K"):
        if d.get(key) is not None:
            meta.setdefault(key, d[key])
    return names, meta


def _names_from_call(call: ast.Call) -> tuple[set[str], dict]:
    """Scene-object names referenced by one conditional call, plus its logical/K."""
    names: set[str] = set()
    meta: dict = {}
    kw = {k.arg: k.value for k in call.keywords if k.arg}

    # DoneTerm(func=<conditional>, params={...}) keeps every name inside `params`.
    if "params" in kw:
        n, m = _names_from_mapping(_literal(kw["params"]) or {})
        names |= n
        for k, v in m.items():
            meta.setdefault(k, v)

    if "groups" in kw:
        for group in _literal(kw["groups"]) or []:
            if not isinstance(group, dict):
                continue
            for key in ("object", "objects", "container"):
                v = group.get(key)
                if isinstance(v, str):
                    names.add(v)
                elif isinstance(v, (list, tuple)):
                    names.update(x for x in v if isinstance(x, str))
            if "logical" in group:
                meta.setdefault("logical", group["logical"])
            if "K" in group:
                meta.setdefault("K", group["K"])

    for key in OBJECT_KWARGS:
        if key not in kw:
            continue
        v = _literal(kw[key])
        vals = [v] if isinstance(v, str) else [x for x in (v or []) if isinstance(x, str)] if isinstance(v, (list, tuple)) else []
        names.update(vals)
        if key in TARGET_KWARGS:
            meta.setdefault("_targets", set()).update(vals)

    for key in ("logical", "K"):
        if key in kw:
            v = _literal(kw[key])
            if v is not None:
                meta.setdefault(key, v)
    return names, meta


def _walk_calls(node) -> list[ast.Call]:
    return [n for n in ast.walk(node) if isinstance(n, ast.Call)]


def _call_name(call: ast.Call) -> str | None:
    f = call.func
    if isinstance(f, ast.Name):
        # partial(object_grabbed, object=...) -> look at the first arg
        if f.id == "partial" and call.args and isinstance(call.args[0], ast.Name):
            return call.args[0].id
        return f.id
    if isinstance(f, ast.Attribute):
        return f.attr
    return None


def _refs(node) -> tuple[set[str], dict]:
    """All scene-object names + logical/K under an AST node."""
    names: set[str] = set()
    meta: dict = {}
    # Structural, not a whitelist: inside a `success` term or a `subtasks` list every
    # call is a condition, so any call carrying an object-naming kwarg counts.  A
    # whitelist silently under-reports — `pick_and_place_on_surface` was missing from
    # it and made 20+ tasks look like their ladder mentioned nothing.
    for call in _walk_calls(node):
        n, m = _names_from_call(call)
        names |= n
        for k, v in m.items():
            if k == "_targets":
                meta.setdefault("_targets", set()).update(v)
            else:
                meta.setdefault(k, v)
        if _call_name(call) == "Subtask":
            for k in call.keywords:
                if k.arg in ("logical", "K"):
                    v = _literal(k.value)
                    if v is not None:
                        meta.setdefault(k.arg, v)
    return names, meta


def parse_task_file(path: Path) -> list[dict]:
    """One dict per Task class in the file."""
    try:
        tree = ast.parse(path.read_text())
    except SyntaxError as e:
        return [{"file": path, "class": "<unparseable>", "error": str(e)}]

    # terminations classes: name -> success node
    term_success: dict[str, ast.AST] = {}
    for node in tree.body:
        if not isinstance(node, ast.ClassDef):
            continue
        for stmt in node.body:
            if isinstance(stmt, ast.Assign) and any(
                isinstance(t, ast.Name) and t.id == "success" for t in stmt.targets
            ):
                term_success[node.name] = stmt.value

    out = []
    for node in tree.body:
        if not isinstance(node, ast.ClassDef):
            continue
        assigns = {
            t.id: s.value
            for s in node.body if isinstance(s, ast.Assign)
            for t in s.targets if isinstance(t, ast.Name)
        }
        if "scene" not in assigns and "subtasks" not in assigns:
            continue

        scene_file = None
        if "scene" in assigns:
            for call in _walk_calls(assigns["scene"]):
                if _call_name(call) == "import_scene" and call.args:
                    scene_file = _literal(call.args[0])

        contacts = _literal(assigns.get("contact_object_list")) if "contact_object_list" in assigns else None
        contacts = [c for c in (contacts or []) if isinstance(c, str)]

        succ_names, succ_meta = (set(), {})
        term_name = None
        if "terminations" in assigns and isinstance(assigns["terminations"], ast.Name):
            term_name = assigns["terminations"].id
            if term_name in term_success:
                succ_names, succ_meta = _refs(term_success[term_name])

        sub_names, sub_meta = (set(), {})
        if "subtasks" in assigns:
            sub_names, sub_meta = _refs(assigns["subtasks"])

        out.append({
            "file": path, "class": node.name, "scene": scene_file,
            "contacts": contacts, "success": succ_names, "success_meta": succ_meta,
            "ladder": sub_names, "ladder_meta": sub_meta,
            "has_ladder": "subtasks" in assigns,
        })
    return out


def scene_prim_names(scene_path: Path) -> set[str] | None:
    try:
        from pxr import Usd
    except ImportError:
        return None
    try:
        stage = Usd.Stage.Open(str(scene_path))
    except Exception:
        return None
    return {p.GetName() for p in stage.Traverse()}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks-dir", default="robolab/tasks/benchmark")
    ap.add_argument("--scenes-dir", default="assets/scenes")
    ap.add_argument("--all-task-dirs", default="robolab/tasks",
                    help="root scanned for duplicate class names (check F)")
    args = ap.parse_args()

    tasks_dir, scenes_dir = Path(args.tasks_dir), Path(args.scenes_dir)
    tasks = [t for f in sorted(tasks_dir.glob("*.py")) if f.name != "__init__.py"
             for t in parse_task_file(f)]

    scene_cache: dict[str, set[str] | None] = {}

    def prims(scene: str) -> set[str] | None:
        if scene not in scene_cache:
            scene_cache[scene] = scene_prim_names(scenes_dir / scene)
        return scene_cache[scene]

    A: list[str] = []   # referenced object not in scene
    B: list[str] = []   # referenced object not in contact_object_list
    C: list[str] = []   # contact_object_list entry not in scene
    D1: list[str] = []  # success TARGET object missing from ladder
    D2: list[str] = []  # success reference/exclusion object missing from ladder
    E: list[str] = []   # logical/K mismatch
    unparsed: list[str] = []
    no_scene: set[str] = set()

    for t in tasks:
        if t.get("error"):
            unparsed.append(f"{t['file'].name}: {t['error']}")
            continue
        loc = f"{t['class']} ({t['file'].name})"
        referenced = t["success"] | t["ladder"]
        names = prims(t["scene"]) if t["scene"] else None
        if t["scene"] and names is None:
            no_scene.add(t["scene"])

        if names:
            for obj in sorted(referenced):
                if obj not in names:
                    A.append(f"{loc}: '{obj}' referenced by a condition but not a prim in {t['scene']}")
            for obj in sorted(t["contacts"]):
                if obj not in names:
                    C.append(f"{loc}: contact_object_list has '{obj}', not a prim in {t['scene']}")

        if t["contacts"]:
            for obj in sorted(referenced):
                if obj not in t["contacts"]:
                    B.append(f"{loc}: '{obj}' is scored but absent from contact_object_list")

        if t["has_ladder"] and t["success"]:
            missing = t["success"] - t["ladder"]
            targets = t["success_meta"].get("_targets", set())
            if t["success_meta"].get("logical", "all") == "all":
                mt = sorted(missing & targets)
                mr = sorted(missing - targets)
                if mt:
                    D1.append(f"{loc}: success must move {mt} but the ladder never mentions {'it' if len(mt)==1 else 'them'}")
                if mr:
                    D2.append(f"{loc}: success refers to {mr} (container/surface/landmark) but the ladder never mentions {'it' if len(mr)==1 else 'them'}")

        sm, lm = t["success_meta"], t["ladder_meta"]
        if t["has_ladder"] and sm.get("logical") and lm.get("logical") and sm["logical"] != lm["logical"]:
            E.append(f"{loc}: success logical='{sm['logical']}' but ladder logical='{lm['logical']}'")
        if t["has_ladder"] and sm.get("K") and lm.get("K") and sm["K"] != lm["K"]:
            E.append(f"{loc}: success K={sm['K']} but ladder K={lm['K']}")

    # F: duplicate class names across task subfolders
    seen: dict[str, list[Path]] = defaultdict(list)
    for f in Path(args.all_task_dirs).rglob("*.py"):
        if f.name == "__init__.py" or "__pycache__" in f.parts:
            continue
        try:
            tree = ast.parse(f.read_text())
        except SyntaxError:
            continue
        for node in tree.body:
            if isinstance(node, ast.ClassDef) and node.name.endswith("Task"):
                seen[node.name].append(f)
    F = [f"{name}: defined in {[str(p) for p in paths]}"
         for name, paths in sorted(seen.items()) if len(paths) > 1]

    sections = [
        ("A  condition references an object that is not in the scene", A),
        ("B  scored object missing from contact_object_list (contact predicates silently false)", B),
        ("C  contact_object_list entry that is not in the scene (dead sensor / typo)", C),
        ("D1 success TARGET object the ladder never mentions (ladder can read 1.0 unfinished)", D1),
        ("D2 success reference/exclusion object the ladder never mentions (weaker: check the intent)", D2),
        ("E  success and ladder disagree about logical/K", E),
        ("F  duplicate task class names across subfolders (resolution order decides)", F),
    ]

    print(f"audited {len(tasks)} task classes in {tasks_dir}")
    if no_scene:
        print(f"  (note: {len(no_scene)} scene file(s) could not be opened; "
              f"checks A and C skipped for them)")
    if unparsed:
        print(f"  (note: {len(unparsed)} file(s) failed to parse)")
        for u in unparsed:
            print(f"    {u}")
    total = 0
    for title, items in sections:
        print(f"\n## {title} — {len(items)}")
        for line in items:
            print(f"  {line}")
        total += len(items)
    print(f"\ntotal findings: {total}")
    return 1 if total else 0


if __name__ == "__main__":
    sys.exit(main())
