"""A task's success termination and its subtask ladder must agree on where each
object goes (P11 / findings.md H-R8-3). Uses scripts/find_task_definition_conflicts.py."""
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
KNOWN_CONFLICTS: set[str] = set()  # food_packing_by_color (H-R8-2) fixed in P19


def test_task_definitions_consistent():
    out = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "find_task_definition_conflicts.py")],
        capture_output=True, text=True, cwd=ROOT,
    ).stdout
    conflicts = {line.split()[1] for line in out.splitlines() if line.startswith("CONFLICT")}
    new = conflicts - KNOWN_CONFLICTS
    fixed = KNOWN_CONFLICTS - conflicts
    assert not new, f"new task-definition conflicts: {sorted(new)}\n{out}"
    assert not fixed, f"remove from KNOWN_CONFLICTS, now clean: {sorted(fixed)}"


def test_list_form_all_sequences_are_about_one_object():
    """A list-form Subtask with logical='all' is a sequence over ONE object (P28).
    A list of conditions about different objects under 'all' almost certainly
    meant parallel groups — write it as a dict."""
    import ast
    bad = []
    for f in sorted((ROOT / "robolab/tasks/benchmark").glob("*.py")):
        tree = ast.parse(f.read_text())
        for node in ast.walk(tree):
            if not (isinstance(node, ast.Call) and getattr(node.func, "id", None) == "Subtask"):
                continue
            kw = {k.arg: k.value for k in node.keywords}
            conds = kw.get("conditions")
            logical = ast.literal_eval(kw["logical"]) if "logical" in kw else "all"
            if not isinstance(conds, ast.List) or logical != "all":
                continue
            # Each step must share at least one object with the step before it:
            # grabbed(yellow) → stacked([yellow, red]) is a sequence; grabbed(a) →
            # grabbed(b) is not.
            per_step = []
            for elt in conds.elts:
                objs = set()
                for c in ast.walk(elt):
                    if isinstance(c, ast.Call):
                        for k in c.keywords:
                            if k.arg in ("object", "objects"):
                                try:
                                    v = ast.literal_eval(k.value)
                                except Exception:
                                    continue
                                objs |= {v} if isinstance(v, str) else set(v)
                per_step.append(objs)
            for a, b in zip(per_step, per_step[1:]):
                if a and b and not (a & b):
                    bad.append(f"{f.name}:{node.lineno} {sorted(a)} → {sorted(b)}")
                    break
    assert not bad, "list-form 'all' subtasks whose consecutive steps are about different objects: " + "; ".join(bad)
