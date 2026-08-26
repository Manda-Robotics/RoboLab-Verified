# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The task-definition audit must keep finding the two conflicts we verified by hand.

Both were found by scripts/audit_task_definitions.py and confirmed by reading the
task files; they are the regression anchors for the extractor, which has already
under-reported once (a whitelist of condition names missed
`pick_and_place_on_surface` and made 20+ ladders look empty).
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


def test_finds_unstack_rubiks_cube_missing_target():
    # success needs middle+top+bottom out of the bin (logical='all');
    # the ladder only covers middle+top, so it reads 1.0 with the bottom cube stuck.
    out = _run()
    assert "UnstackRubiksCubeTask" in out
    assert "rubiks_cube_bottom" in out


def test_finds_bananas_in_crate_k_mismatch():
    # success is choose K=2 of 5 bananas; the ladder is choose K=1 of 4.
    out = _run()
    assert "BananasInCrateTask" in out
    assert "K=2" in out and "K=1" in out
