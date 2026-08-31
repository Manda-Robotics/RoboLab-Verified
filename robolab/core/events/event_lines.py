# SPDX-License-Identifier: Apache-2.0
"""One line per transition (changes.md P45).

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


_TRACKER_GRAB = re.compile(r"^'(\w+)' grasped")


def fold_duplicate_grab_lines(events: list[dict], window_steps: int = 0) -> list[dict]:
    """P60: keep one grab line per grasp.

    The tracker now emits the grasp itself, so a task whose ladder also has an
    ``object_grabbed`` rung would print the same grasp twice. The ladder line wins:
    it carries the progress text ("advanced 1 step") the tracker line does not.
    Tasks with no such rung -- the five stacking tasks -- keep the tracker line,
    which is the whole point of P60.

    Pairing is structural, not a time window. The first rc3 run showed the ladder
    line lagging the tracker by 0.60-2.20 s (a fixed 0.53 s window caught none of
    the five duplicates), because the ladder re-evaluates its own predicate on its
    own schedule. Instead: a tracker grab is a duplicate when a ladder grab for the
    same object follows it with no release/drop for that object in between -- two
    genuinely separate grasps always have a release between them.

    ``window_steps`` is accepted and ignored; kept so callers need not change.
    """
    ladder: dict[str, list[int]] = {}
    for ev in events:
        p = predicate_of(ev.get("info", "") or "")
        if p and p[0] == "success" and p[1] == "object_grabbed":
            ladder.setdefault(p[2], []).append(int(ev.get("step", 0)))
    if not ladder:
        return list(events)

    # steps at which each object left the hand
    partings: dict[str, list[int]] = {}
    for ev in events:
        if int(ev.get("code", -1)) in _RELEASE_CODES:
            m = _OBJ_IN_INFO.search(ev.get("info", "") or "")
            if m:
                partings.setdefault(m.group(1), []).append(int(ev.get("step", 0)))

    out = []
    for ev in events:
        m = _TRACKER_GRAB.match(ev.get("info", "") or "")
        if m:
            obj = m.group(1)
            step = int(ev.get("step", 0))
            later = [l for l in ladder.get(obj, ()) if l >= step]
            if later:
                nxt = min(later)
                if not any(step < r < nxt for r in partings.get(obj, ())):
                    continue
        out.append(ev)
    return out
