# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Both bimanual rigs are adopted; only one has a working policy.

The ALOHA/ViperX rig is included because Finn wants the embodiment available, with
its limitation flagged. The flag is the deliverable here: an ALOHA score is a
statement about the released checkpoint (0/6, out of distribution), not about the rig
or the benchmark. These tests keep that caveat attached to the code rather than
living only in a conversation.
"""
import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _doc(rel):
    return ast.get_docstring(ast.parse((ROOT / rel).read_text())) or ""


def test_both_rigs_are_present():
    assert (ROOT / "robolab/robots/bimanual_franka.py").exists()
    assert (ROOT / "robolab/robots/aloha.py").exists()
    assert (ROOT / "robolab/robots/bimanual_station.py").exists()


def test_aloha_carries_its_limitation_where_someone_will_read_it():
    doc = _doc("robolab/robots/aloha.py")
    assert "LIMITED USE" in doc
    assert "0/6" in doc, "the actual number, not a vague warning"
    assert "fine-tun" in doc.lower(), "say what the path forward is"


def test_aloha_points_at_the_rig_that_does_work():
    assert "bimanual_franka" in _doc("robolab/robots/aloha.py")


def test_the_bimanual_readme_states_the_reporting_rule():
    txt = (ROOT / "robolab/robots/README_bimanual.md").read_text()
    assert "0/6" in txt and "6/6" in txt
    assert "P67" in txt, "the silent-no-metrics dependency must be recorded"


def test_per_arm_metrics_dependency_is_actually_in_this_tree():
    # both rigs declare left_ee_pose/right_ee_pose, so a bimanual demo has no ee_pose
    # group; without P67 compute_metrics returns None and the runs look empty.
    assert "ee_channels" in (ROOT / "robolab/core/metrics/compute_metrics.py").read_text()


def test_aloha_wrist_camera_support_came_with_the_rig():
    assert (ROOT / "robolab/core/sensors/body_attached_camera.py").exists()
