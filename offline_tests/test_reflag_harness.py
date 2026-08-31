# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The re-flagger must apply the post-processable rules and admit what it cannot check."""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from reflag import reflag  # noqa: E402


def _e(step, name, info):
    return {"step": step, "code": 0, "name": name, "info": info, "score": 0.0}


def test_p71_detector_line_is_renamed_and_ladder_line_is_kept():
    ev = [_e(100, "OBJECT_GRABBED_SUCCESS", "'can' grasped (carry established)"),
          _e(120, "OBJECT_GRABBED_SUCCESS", "success: object_grabbed(object=can). advanced")]
    out, _ = reflag(ev, 1 / 15)
    assert [e["name"] for e in out] == ["OBJECT_CARRIED", "OBJECT_GRABBED_SUCCESS"]


def test_p72_container_attempts_are_dropped():
    ev = [_e(100, "GRASP_ATTEMPT_FAILED", "Grasp attempt on 'bin_a06' failed")]
    out, removed = reflag(ev, 1 / 15)
    assert out == [] and removed[0]["why"] == "P72 container"


def test_p73_attempt_right_after_a_release_is_dropped():
    ev = [_e(100, "OBJECT_RELEASED", "'can' released (hand opened)"),
          _e(110, "GRASP_ATTEMPT_FAILED", "Grasp attempt on 'can' failed")]
    out, removed = reflag(ev, 1 / 15)
    assert [e["name"] for e in out] == ["OBJECT_RELEASED"]
    assert removed[0]["why"] == "P73 after release"


def test_a_genuine_later_attempt_survives():
    ev = [_e(100, "OBJECT_RELEASED", "'can' released (hand opened)"),
          _e(400, "GRASP_ATTEMPT_FAILED", "Grasp attempt on 'can' failed")]
    out, _ = reflag(ev, 1 / 15)
    assert len(out) == 2


def test_p75_placed_without_lift_is_dropped():
    ev = [_e(4, "PLACED_WITHOUT_LIFT", "'keyboard' placed without ever being carried")]
    out, removed = reflag(ev, 1 / 15)
    assert out == [] and removed[0]["why"] == "P75 retired"


def test_labels_file_is_wellformed():
    n = 0
    for line in (ROOT / "analysis" / "flag_labels.jsonl").read_text().splitlines():
        if not line.strip():
            continue
        lb = json.loads(line)
        assert {"run", "task", "env", "flag", "verdict", "note", "source"} <= set(lb)
        assert lb["verdict"] in {"wrong", "missing", "correct", "correct-absent", "ambiguous"}
        n += 1
    assert n >= 20, "the labelled set is the point; it should not shrink"
