# SPDX-License-Identifier: Apache-2.0
"""Check explicit authored rotations at RoboLab's WXYZ config boundary.

Isaac Lab 3 defaults are XYZW. Leaving a nested pose at its native class default
and subsequently treating it as authored WXYZ can introduce a half turn.
Inspect keyword arguments on each call, not text in nested calls or comments.
"""
from __future__ import annotations

import ast
import textwrap


def _name(node: ast.expr) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return _name(node.value) + "." + node.attr
    return ""


def missing_rotations(src: str) -> list[str]:
    """Return config calls with an implicit pose rotation, without importing Kit."""
    problems = []
    for node in ast.walk(ast.parse(textwrap.dedent(src))):
        if not isinstance(node, ast.Call):
            continue
        name = _name(node.func)
        kind = name.rsplit(".", 1)[-1]
        keywords = {keyword.arg for keyword in node.keywords}
        required = None
        if kind in {"OffsetCfg", "InitialStateCfg"}:
            required = "rot"
        elif name.endswith("FrameTransformerCfg.FrameCfg"):
            required = "offset"
            kind = "FrameCfg"
        elif kind == "FrameTransformerCfg":
            required = "source_frame_offset"
        elif kind == "AssetBaseCfg":
            # Crucial for the dome: absent init_state still instantiates Lab's
            # native default, so checking explicit InitialStateCfg calls alone
            # cannot detect the original defect.
            required = "init_state"
        if required is not None and required not in keywords:
            problems.append(f"{kind} without {required}= at line {node.lineno}")
    return problems
