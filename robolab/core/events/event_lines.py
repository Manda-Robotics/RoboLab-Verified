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
d. a ``Completed subtask`` line absorbs the predicate line of the same tick;
e. (P57) ``WRONG_OBJECT_DETACHED`` is folded into a release/drop for the same
   object that lands within a short window, not only on the same tick — rule c
   only caught the same-tick case, and in the rc2 corpus **43 of 43** detach
   lines were followed by a release/drop for the same object within 0.5 s
   (typically 0.07 s later), so the pair is 100 % redundant in practice. A
   detach with no release after it is kept: that is the informative case where
   the wrong object is knocked out of the hand without the gripper opening.
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
    # rule b: predicate lines are named after their predicate; a completion line is
    # SUBTASK_COMPLETED (it used to borrow the stage's first condition's code — a
    # pick_and_place completion read "OBJECT_GRABBED_SUCCESS · Completed subtask").
    if "Completed subtask" in info:
        ladder_line = dict(ladder_line, name="SUBTASK_COMPLETED", code=int(StatusCode.SUBTASK_COMPLETED))
    else:
        ladder_line = dict(ladder_line, name=name_for_ladder_line(int(ladder_line.get("code", 0)), info))
    return kept, ladder_line


def fold_detach_into_release(events: list[dict], window_steps: int) -> list[dict]:
    """Rule e: drop a WRONG_OBJECT_DETACHED that a release/drop for the same object follows.

    Operates on one env's finished event list (pure; ``events`` is not mutated).
    ``window_steps`` is how far ahead a release may land and still absorb the
    detach line. Order is preserved.
    """
    if window_steps <= 0:
        return list(events)

    releases: dict[str, list[int]] = {}
    for ev in events:
        if int(ev.get("code", -1)) in _RELEASE_CODES:
            m = _OBJ_IN_INFO.search(ev.get("info", "") or "")
            if m:
                releases.setdefault(m.group(1), []).append(int(ev.get("step", 0)))

    out = []
    for ev in events:
        if int(ev.get("code", -1)) == int(StatusCode.WRONG_OBJECT_DETACHED):
            m = _OBJ_IN_INFO.search(ev.get("info", "") or "")
            step = int(ev.get("step", 0))
            if m and any(step <= r <= step + window_steps for r in releases.get(m.group(1), ())):
                continue
        out.append(ev)
    return out
