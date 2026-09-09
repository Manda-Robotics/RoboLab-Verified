# SPDX-License-Identifier: Apache-2.0
"""Static check that every RoboLab-authored config rotation is explicit.

RoboLab authors WXYZ quaternions and converts them at the Isaac Lab boundary
(``isaaclab_compat.prepare_env_cfg``). A ``rot`` left at the class default therefore
inherits Isaac Lab's own convention, XYZW identity ``(0, 0, 0, 1)`` on Isaac Lab 3, and the
boundary conversion then turns it into a 180-degree rotation about Z at runtime. The dome
light, the sphere lights and the FrameTransformer offsets shipped like that in every
Isaac Sim 6.0 recording (docs/isaac_sim_6.md). The rule: spell identity out.
"""
from __future__ import annotations

import re

CALLS = ("OffsetCfg(", "InitialStateCfg(")     # every call must carry rot=
FRAME_CFG = "FrameTransformerCfg.FrameCfg("       # every frame needs an offset
FRAME_TRANSFORMER = "FrameTransformerCfg("        # every transformer needs a source_frame_offset


def call_spans(src: str, marker: str) -> list[tuple[int, int]]:
    """(start, end) of every ``marker ... )`` call with balanced parentheses."""
    out, i = [], 0
    while True:
        i = src.find(marker, i)
        if i < 0:
            return out
        depth, j = 0, i + len(marker) - 1
        while j < len(src):
            if src[j] == "(":
                depth += 1
            elif src[j] == ")":
                depth -= 1
                if depth == 0:
                    break
            j += 1
        out.append((i, j + 1))
        i = j + 1


def missing_rotations(src: str) -> list[str]:
    """Human-readable list of config calls in ``src`` whose rotation is implicit."""
    problems = []
    for marker in CALLS:
        for a, b in call_spans(src, marker):
            body = src[a:b]
            if not re.search(r"\brot\s*=", body):
                problems.append(f"{marker[:-1]} without rot= at offset {a}: {body[:80]!r}")
    for a, b in call_spans(src, FRAME_CFG):
        if "offset=" not in src[a:b]:
            problems.append(f"FrameCfg without offset= at offset {a}: {src[a:b][:80]!r}")
    for a, b in call_spans(src, FRAME_TRANSFORMER):
        if src[a - 1:a] == ".":        # FrameTransformerCfg.FrameCfg, handled above
            continue
        if "source_frame_offset=" not in src[a:b]:
            problems.append(f"FrameTransformerCfg without source_frame_offset= at offset {a}")
    return problems
