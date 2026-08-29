# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The audit must NOT report a ladder as incomplete when the missing object
already satisfies the success relation before the robot moves.

Both cases below were reported as conflicts by the first version of this tool and
were wrong. The reviewer checked the scenes and said so: the Rubik's tower's bottom cube
already sits on the table, and one banana already sits in the crate. Measured
from the corpus run's step-0 poses: 25 cm from the bin, and 2 cm from the crate
centre. These are the regression anchors for the spawn-state check.
"""
import ast
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _run():
    return subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "audit_task_definitions.py")],
        capture_output=True, text=True, cwd=ROOT,
    ).stdout


def test_script_parses():
    ast.parse((ROOT / "scripts" / "audit_task_definitions.py").read_text())


def test_unstack_rubiks_cube_is_not_reported_as_a_missing_target():
    out = _run()
    d1 = out.split("## D1")[1].split("## D2")[0] if "## D1" in out else ""
    assert "UnstackRubiksCubeTask" not in d1, "bottom cube starts on the table; not a conflict"


def test_bananas_in_crate_is_not_reported_as_a_k_mismatch():
    out = _run()
    e = out.split("## E ")[1].split("## F")[0] if "## E " in out else ""
    assert "BananasInCrateTask" not in e, "one banana starts in the crate; K=1 + 1 pre-staged = 2"


def test_spawn_check_explains_itself_or_says_it_could_not_run():
    """Either the check ran and shows its measurement, or it says it could not run.

    What must never happen is a silent pass-through that turns an unverified
    candidate into an asserted conflict -- that is the bug this file guards.
    """
    out = _run()
    assert ("already satisfied at spawn" in out) or ("spawn state could not be read" in out)


# --- P65 (H-R9-T5): success terms are matched by prefix, not by exact name ----

def _parse_task_definitions():
    sys.path.insert(0, str(ROOT / "scripts"))
    import importlib
    mod = importlib.import_module("audit_task_definitions")
    return mod


def test_prefixed_success_terms_are_seen(tmp_path):
    """`success_bagel_02 = DoneTerm(...)` must register as a success term.

    Under exact `success` matching, GrabABagel and GrabAFruit had no success term
    as far as this sweep was concerned, so every check silently skipped them
    (H-R9-T5). `find_task_definition_conflicts.py` already matched the prefix.
    """
    mod = _parse_task_definitions()
    f = tmp_path / "prefixed.py"
    f.write_text(
        "class Terminations:\n"
        "    time_out = DoneTerm(func=mdp.time_out, time_out=True)\n"
        "    success_bagel_02 = DoneTerm(func=object_picked_up, params={'object': 'bagel_02', 'surface': 'table'})\n"
        "    success_bagel_01 = DoneTerm(func=object_picked_up, params={'object': 'bagel_01', 'surface': 'table'})\n"
        "\n"
        "class ThingTask(Task):\n"
        "    scene = import_scene('breakfast_table.usda', [])\n"
        "    terminations = Terminations\n"
        "    subtasks = [Subtask(name='x', conditions=[partial(object_grabbed, object='bagel_02')])]\n"
    )
    tasks = [t for t in mod.parse_task_file(f) if t.get("class") == "ThingTask"]
    assert tasks, "task class not parsed"
    assert {"bagel_01", "bagel_02"} <= tasks[0]["success"], "prefixed success terms were skipped"


def test_time_out_is_not_a_success_term(tmp_path):
    mod = _parse_task_definitions()
    f = tmp_path / "timeout_only.py"
    f.write_text(
        "class Terminations:\n"
        "    time_out = DoneTerm(func=mdp.time_out, time_out=True)\n"
        "\n"
        "class ThingTask(Task):\n"
        "    scene = import_scene('breakfast_table.usda', [])\n"
        "    terminations = Terminations\n"
        "    subtasks = [Subtask(name='x', conditions=[partial(object_grabbed, object='bagel_02')])]\n"
    )
    tasks = [t for t in mod.parse_task_file(f) if t.get("class") == "ThingTask"]
    assert tasks and not tasks[0]["success"]
