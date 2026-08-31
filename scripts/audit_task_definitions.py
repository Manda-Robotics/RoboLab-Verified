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
    #
    # P65 (H-R9-T5): match the `success*` PREFIX, not the exact name `success`.
    # GrabABagel names its terms success_bagel_01/_02/_07 and GrabAFruit names
    # its success_banana/_banana_01/_apple/_orange, so under exact matching both
    # tasks had no success term at all as far as this sweep was concerned and
    # every check below silently skipped them. `find_task_definition_conflicts.py`
    # already used startswith("success"); the two lints now agree. A class may
    # carry several success terms, so they are collected and walked together.
    term_success: dict[str, ast.AST] = {}
    for node in tree.body:
        if not isinstance(node, ast.ClassDef):
            continue
        nodes = [
            stmt.value for stmt in node.body
            if isinstance(stmt, ast.Assign) and any(
                isinstance(t, ast.Name) and t.id.startswith("success") for t in stmt.targets
            )
        ]
        if nodes:
            term_success[node.name] = nodes[0] if len(nodes) == 1 else ast.Tuple(elts=nodes, ctx=ast.Load())

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
        succ_func = None
        term_name = None
        if "terminations" in assigns and isinstance(assigns["terminations"], ast.Name):
            term_name = assigns["terminations"].id
            if term_name in term_success:
                succ_names, succ_meta = _refs(term_success[term_name])
                for call in _walk_calls(term_success[term_name]):
                    for k in call.keywords:
                        if k.arg == "func" and isinstance(k.value, ast.Name):
                            succ_func = k.value.id

        sub_names, sub_meta = (set(), {})
        if "subtasks" in assigns:
            sub_names, sub_meta = _refs(assigns["subtasks"])

        out.append({
            "file": path, "class": node.name, "scene": scene_file,
            "contacts": contacts, "success": succ_names, "success_meta": succ_meta,
            "ladder": sub_names, "ladder_meta": sub_meta, "success_func": succ_func,
            "has_ladder": "subtasks" in assigns,
        })
    return out


# Predicates whose goal is "near the container" vs "away from it". A target that
# already satisfies its goal relation at spawn needs no action, so a ladder that
# leaves it out is correct, not a conflict. Ignoring this produced two confident
# false positives (UnstackRubiksCube, BananasInCrate) that the reviewer caught by simply
# looking at the scene -- the bottom cube already sits on the table 25 cm from the
# bin, and banana_01 already sits 2 cm from the crate centre.
NEAR_GOAL_FUNCS = {"object_in_container", "pick_and_place", "object_on_top", "object_on_center"}
AWAY_GOAL_FUNCS = {"object_outside_of_and_on_surface", "pick_and_place_on_surface"}
SPAWN_NEAR_M = 0.06


def spawn_satisfied(task: dict, obj: str, states_dir: Path) -> tuple[bool, str]:
    """Does `obj` already satisfy the success relation at step 0? (verdict, evidence)."""
    if states_dir is None:
        return False, "no --states-dir given"
    func = task.get("success_func")
    if func not in NEAR_GOAL_FUNCS | AWAY_GOAL_FUNCS:
        return False, "unknown predicate"
    containers = sorted(task["success"] - (task["success_meta"].get("_targets") or set()))
    if not containers:
        return False, "no container named"
    try:
        import h5py  # noqa: PLC0415
        import numpy as np  # noqa: PLC0415
    except ImportError:
        return False, "h5py unavailable"
    files = sorted((states_dir / task["class"]).glob("*.hdf5"))
    if not files:
        return False, "no run to read spawn poses from"
    try:
        with h5py.File(files[0]) as f:
            g = f[f"data/{list(f['data'])[0]}/states/rigid_object"]
            if obj not in g:
                return False, f"{obj} not in the recorded state"
            o = g[obj]["root_pose"][0, :2]
            for c in containers:
                if c not in g:
                    continue
                d = float(np.linalg.norm(o - g[c]["root_pose"][0, :2]))
                near = d < SPAWN_NEAR_M
                if (func in NEAR_GOAL_FUNCS and near) or (func in AWAY_GOAL_FUNCS and not near):
                    return True, f"{obj} starts {d*100:.0f} cm from {c} — already satisfied at spawn"
    except (OSError, KeyError, IndexError) as e:
        return False, f"could not read spawn poses ({type(e).__name__})"
    return False, "not satisfied at spawn"


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
    ap.add_argument("--states-dir", default=None,
                    help="a recorded run directory whose step-0 poses give each task's spawn state "
                         "(optional; without it, candidates that need the spawn state are listed as "
                         "NOT asserted rather than reported)")
    ap.add_argument("--all-task-dirs", default="robolab/tasks",
                    help="root scanned for duplicate class names (check F)")
    args = ap.parse_args()

    tasks_dir, scenes_dir = Path(args.tasks_dir), Path(args.scenes_dir)
    states_dir = Path(args.states_dir) if args.states_dir else None
    tasks = [t for f in sorted(tasks_dir.glob("*.py")) if f.name != "__init__.py"
             for t in parse_task_file(f)]

    scene_cache: dict[str, set[str] | None] = {}

    def prims(scene: str) -> set[str] | None:
        if scene not in scene_cache:
            scene_cache[scene] = scene_prim_names(scenes_dir / scene)
        return scene_cache[scene]

    dismissed: list[str] = []   # cleared by the spawn-state check
    unverified: list[str] = []  # spawn state could not be read -> NOT asserted as findings
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
                mt = []
                for o in sorted(missing & targets):
                    done, why = spawn_satisfied(t, o, states_dir)
                    if done:
                        dismissed.append(f"{loc}: {why}")
                    elif why != "not satisfied at spawn":
                        # could not check -> never assert a conflict we did not verify
                        unverified.append(f"{loc}: '{o}' unchecked ({why})")
                    else:
                        mt.append(o)
                mr = sorted(missing - targets)
                if mt:
                    D1.append(f"{loc}: success must move {mt} but the ladder never mentions {'it' if len(mt)==1 else 'them'}")
                if mr:
                    D2.append(f"{loc}: success refers to {mr} (container/surface/landmark) but the ladder never mentions {'it' if len(mr)==1 else 'them'}")

        sm, lm = t["success_meta"], t["ladder_meta"]
        if t["has_ladder"] and sm.get("logical") and lm.get("logical") and sm["logical"] != lm["logical"]:
            E.append(f"{loc}: success logical='{sm['logical']}' but ladder logical='{lm['logical']}'")
        if t["has_ladder"] and sm.get("K") and lm.get("K") and sm["K"] != lm["K"]:
            # a target already in its goal relation at spawn counts toward success's K
            prestaged = 0
            blocked = False
            for o in sorted(sm.get("_targets") or set()):
                done, why = spawn_satisfied(t, o, states_dir)
                if done:
                    prestaged += 1
                    dismissed.append(f"{loc}: {why}")
                elif why != "not satisfied at spawn":
                    blocked = True
            if blocked:
                unverified.append(f"{loc}: K={sm['K']} vs ladder K={lm['K']} unchecked (spawn state unreadable)")
            elif lm["K"] + prestaged < sm["K"]:
                E.append(f"{loc}: success K={sm['K']} but ladder K={lm['K']} (+{prestaged} pre-staged at spawn)")

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
    if unverified:
        print(f"\n## NOT asserted — spawn state could not be read — {len(unverified)}")
        print("   (a target already in its goal relation at spawn needs no ladder step; without the")
        print("    spawn poses this cannot be told apart from a real gap, so it is not a finding)")
        for line in unverified:
            print(f"  {line}")
    if dismissed:
        print(f"\n## cleared by the spawn-state check (already satisfied before the robot moves) — {len(dismissed)}")
        for line in dismissed:
            print(f"  {line}")
    # A-C, D1 and E are conflicts: a definition that cannot score what it says. D2 (a success
    # term's landmark the ladder never names -- normal for "take X off Y") and F (a duplicate
    # class name under test_tasks/) are notes. Only conflicts fail the run, so CI can gate on it.
    conflicts = sum(len(items) for title, items in sections if not title.startswith(("D2", "F ")))
    notes = total - conflicts
    print(f"\nconflicts: {conflicts}   notes: {notes}   not asserted: {len(unverified)}")
    return 1 if conflicts else 0


if __name__ == "__main__":
    sys.exit(main())
