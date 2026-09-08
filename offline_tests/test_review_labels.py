"""Review-mode labels (plan §9.4, 2026-09-02): a verdict per machine segment becomes a gold
segment in score_gold.py, and the run path writes phases_<run>_env<env>.json (P107)."""
import importlib.util
import json
import os
import shutil
import sys

ROOT = os.path.join(os.path.dirname(__file__), "..")


def _score_gold():
    spec = importlib.util.spec_from_file_location("score_gold", os.path.join(ROOT, "scripts", "score_gold.py"))
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    return m


def _review(i, label="pick", obj="banana", result="pass", t0=1.0, t1=4.0, ok=None, corrected=None):
    return {"id": f"r{i}", "kind": "review", "seg_index": i, "t_start": t0, "t_end": t1, "label": label,
            "object": obj, "result": result, "verdict": {"label": True, "result": True, "bounds": True, **(ok or {})},
            "corrected": corrected}


def test_review_verdicts_become_gold_segments():
    S = _score_gold()
    rows = [
        _review(0),                                                                       # machine right
        _review(1, t0=5.0, t1=8.0, result="fail", ok={"result": False}, corrected={"result": "pass"}),
        _review(2, t0=9.0, t1=12.0, ok={"bounds": False}, corrected={"t_start": 9.5, "t_end": 11.0}),
        _review(3, t0=13.0, t1=14.0, ok={"label": False}, corrected={"label": "none"}),   # should not exist
        _review(2, t0=9.0, t1=12.0),                                                      # later review of seg 2 wins
        {"id": "m1", "kind": "segment", "t_start": 15.0, "t_end": 17.0, "label": "place", "object": "banana", "result": "pass"},
    ]
    segs, bounds = S.segments_of(rows)
    assert segs == [(1.0, 4.0, "pick", "banana", "pass"), (5.0, 8.0, "pick", "banana", "pass"),
                    (9.0, 12.0, "pick", "banana", "pass"), (15.0, 17.0, "place", "banana", "pass")]
    assert S.REVIEW_STATS["n"] == 4 and S.REVIEW_STATS["removed"] == 1 and S.REVIEW_STATS["label_ok"] == 3
    assert {1.0, 4.0, 5.0, 8.0, 9.0, 12.0, 15.0, 17.0} <= bounds


def test_write_phases_puts_the_file_next_to_the_log(tmp_path):
    src = os.path.join(ROOT, "offline_tests", "fixtures", "dense", "t1_BananaInBowl", "BananaInBowlTask")
    td = tmp_path / "BananaInBowlTask"
    shutil.copytree(src, td)
    sys.path.insert(0, ROOT)
    from robolab.eval.phases import phases_path, write_phases
    a = write_phases(str(td), 0, 0, use_replay=False)
    pf = phases_path(str(td), 0, 0)
    assert os.path.exists(pf) and not os.path.exists(pf + ".tmp")
    doc = json.load(open(pf))
    assert doc["attempts"] == a.doc["attempts"] and "n_picks" in doc["summary"]
