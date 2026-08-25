# SPDX-License-Identifier: Apache-2.0
"""One line per transition (VERIFIED_PATCHES P45).

The recorder writes two kinds of lines per env per tick: tracker events
(grasp attempt / release / drop / tow / wrong object / bump …) and the subtask
ladder's own transition line ("success: <pred>(…) advanced …", "failed: <pred>(…)
regressing …", "Completed subtask …"). One physical change used to produce up to
three lines — 24 % of all corpus events shared a tick with a twin, e.g.

    26.00s  TARGET_OBJECT_DROPPED    Target object dropped during transport
    26.00s  OBJECT_GRABBED_FAILURE   failed: object_grabbed(lime01_01). regressing to step 0

Rules applied here, on the lines of one env for one tick (pure functions, no
simulator):

a. the ladder's ``failed: object_grabbed(X) … regressing`` line is dropped when a
   tracker release/drop/tow line for X was written on the same tick;
b. a ladder line is named after the predicate that flipped
   (``object_on_top`` → ``OBJECT_ON_TOP_SUCCESS``), not after whatever code the
   previous transition had;
c. ``WRONG_OBJECT_DETACHED`` is dropped when a tracker release/drop for the same
   object was written on the same tick (the release line carries the fact);
d. a ``Completed subtask`` line absorbs the predicate line of the same tick.
Scores, ladders and detectors are untouched; only what is written and its name.
"""
from __future__ import annotations

import re

from robolab.core.task.status import StatusCode, get_status_name

_PRED = re.compile(r"^(success|failed): (\w+)\(object=(\w+)")
_OBJ_IN_INFO = re.compile(r"'(\w+)'")
_RELEASE_CODES = {int(StatusCode.OBJECT_RELEASED), int(StatusCode.OBJECT_DROPPED), int(StatusCode.TOWED_WITHOUT_GRASP)}


def predicate_of(info: str):
    m = _PRED.match(info or "")
    return (m.group(1), m.group(2), m.group(3)) if m else None


def name_for_ladder_line(code: int, info: str) -> str:
    """Rule b: the event name follows the predicate in the line."""
    p = predicate_of(info)
    if not p:
        return get_status_name(code)
    kind, pred, _ = p
    want = f"{pred.upper()}_{'SUCCESS' if kind == 'success' else 'FAILURE'}"
    return want if hasattr(StatusCode, want) else get_status_name(code)


def released_objects(tracker_lines: list[dict]) -> set[str]:
    out = set()
    for ln in tracker_lines:
        if int(ln.get("code", -1)) in _RELEASE_CODES:
            m = _OBJ_IN_INFO.search(ln.get("info", "") or "")
            if m:
                out.add(m.group(1))
    return out


def dedupe_tick(tracker_lines: list[dict], ladder_line: dict | None) -> tuple[list[dict], dict | None]:
    """Apply rules a–d to one env's lines of one tick. Returns (tracker_lines, ladder_line_or_None)."""
    released = released_objects(tracker_lines)
    # rule c
    kept = []
    for ln in tracker_lines:
        if int(ln.get("code", -1)) == int(StatusCode.WRONG_OBJECT_DETACHED):
            m = _OBJ_IN_INFO.search(ln.get("info", "") or "")
            if m and m.group(1) in released:
                continue
        kept.append(ln)
    if ladder_line is None:
        return kept, None
    info = ladder_line.get("info", "") or ""
    p = predicate_of(info)
    # rule a
    if p and p[0] == "failed" and p[1] == "object_grabbed" and p[2] in released:
        return kept, None
    # rule b (+ d: a completion line keeps its own name and text)
    if "Completed subtask" not in info:
        ladder_line = dict(ladder_line, name=name_for_ladder_line(int(ladder_line.get("code", 0)), info))
    return kept, ladder_line
