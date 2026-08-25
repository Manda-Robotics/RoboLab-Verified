"""A task's success termination and its subtask ladder must agree on where each
object goes (P11 / VERIFIED_PLAN H-R8-3). Uses scripts/find_task_definition_conflicts.py."""
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
KNOWN_CONFLICTS = {
    "food_packing_by_color_task.py",  # H-R8-2: ladder inverts the success bins
}


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
