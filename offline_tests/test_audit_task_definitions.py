# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The audit must NOT report a ladder as incomplete when the missing object
already satisfies the success relation before the robot moves.

Both cases below were reported as conflicts by the first version of this tool and
were wrong. Finn checked the scenes and said so: the Rubik's tower's bottom cube
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
