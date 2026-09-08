# SPDX-License-Identifier: Apache-2.0
"""score_gold.py scoring conventions (docs/verified/dense_annotations.md §9.12).

Three rules the H8 number depends on, none of which the segment maths itself enforces:
a review pass supersedes the free-form marks on the same episode, machine segments outside
the annotated span are unknown rather than false positives, and a 1-based step ``k`` spans
``[(k-1)*dt, k*dt)`` in both the scorer and the dashboard.
"""
import importlib.util
import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_spec = importlib.util.spec_from_file_location("score_gold", os.path.join(_ROOT, "scripts", "score_gold.py"))
score_gold = importlib.util.module_from_spec(_spec)
sys.modules["score_gold"] = score_gold
_spec.loader.exec_module(score_gold)


def _review(seg_index, t_start, t_end, label="pick", **kw):
    row = {"kind": "review", "seg_index": seg_index, "t_start": t_start, "t_end": t_end,
           "label": label, "object": "can", "result": "pass",
           "verdict": {"label": True, "result": True, "bounds": True}, "corrected": None}
    row.update(kw)
    return row


def test_review_pass_supersedes_free_form_marks():
    """A narrated episode re-reviewed later keeps the review pass, not both readings."""
    rows = [
        {"kind": "segment", "t_start": 0.0, "t_end": 5.0, "label": "pick", "object": "can", "result": "pass"},
        {"kind": "segment", "t_start": 5.0, "t_end": 9.0, "label": "place", "object": "can", "result": "pass"},
        _review(0, 0.0, 4.6), _review(1, 4.6, 7.9, label="place"),
    ]
    score_gold.REVIEW_STATS.clear()
    segs, _ = score_gold.segments_of(rows)
    assert [(s[0], s[1]) for s in segs] == [(0.0, 4.6), (4.6, 7.9)]


def test_a_reviewer_added_segment_survives_the_review_pass():
    """A mark filling a gap the machine missed overlaps no reviewed segment, so it stands."""
    rows = [_review(0, 0.0, 4.6), _review(1, 4.6, 7.9, label="place"),
            {"kind": "segment", "t_start": 9.0, "t_end": 11.0, "label": "pick", "object": "can", "result": "fail"}]
    score_gold.REVIEW_STATS.clear()
    segs, _ = score_gold.segments_of(rows)
    assert [(s[0], s[1]) for s in segs] == [(0.0, 4.6), (4.6, 7.9), (9.0, 11.0)]


def test_free_form_marks_stand_when_nothing_was_reviewed():
    rows = [{"kind": "segment", "t_start": 0.0, "t_end": 5.0, "label": "pick", "object": "can", "result": "pass"}]
    score_gold.REVIEW_STATS.clear()
    segs, _ = score_gold.segments_of(rows)
    assert [(s[0], s[1]) for s in segs] == [(0.0, 5.0)]


def test_boundary_marks_survive_a_review_pass():
    rows = [{"kind": "boundary", "t_start": 3.0}, _review(0, 0.0, 6.0)]
    score_gold.REVIEW_STATS.clear()
    _, bounds = score_gold.segments_of(rows)
    assert 3.0 in bounds


EP = ("run", "task", 0, 0)


def _scored(pred, gold, **kw):
    gb = {round(b, 3) for a, b_, *_ in gold for b in (a, b_)}
    pb = {round(b, 3) for a, b_, *_ in pred for b in (a, b_)}
    return score_gold.score({EP: (pred, pb)}, {EP: (gold, gb)}, "t", **kw)


def test_machine_segments_past_the_annotated_span_are_not_false_positives():
    """The annotator stopped at 10 s; what the machine did afterwards is unknown."""
    gold = [(0.0, 5.0, "pick", "can", "pass"), (5.0, 10.0, "place", "can", "pass")]
    pred = gold + [(10.0, 14.0, "pick", "can", "pass"), (14.0, 20.0, "place", "can", "fail")]
    assert _scored(pred, gold)["P"] == 1.0
    assert _scored(pred, gold, span_clip=False)["P"] == 0.5


def test_a_segment_straddling_the_last_gold_edge_still_counts():
    """Half or more inside the span is in: only the wholly unannotated tail drops out."""
    gold = [(0.0, 10.0, "pick", "can", "pass")]
    pred = [(0.0, 10.0, "pick", "can", "pass"), (9.0, 11.0, "place", "can", "pass")]
    assert _scored(pred, gold)["n_pred"] == 2


def test_step_to_seconds_matches_the_dashboard():
    """dashboard.app and machine_segments must place a 1-based step k at [(k-1)*dt, k*dt)."""
    src = open(os.path.join(_ROOT, "dashboard", "app.py")).read()
    assert 'start_s=(a["start"] - 1) * dt, end_s=a["end"] * dt' in src
    assert 'start_s=(p["start"] - 1) * dt, end_s=p["end"] * dt' in src
