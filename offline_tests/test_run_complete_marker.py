"""`run_complete.json` is written at the end of a run (P10)."""
import json

from robolab.eval.runner import write_run_complete_marker


def test_marker_contents(tmp_path):
    rows = [{"env_name": "A"}, {"env_name": "A"}, {"env_name": "B"}]
    p = write_run_complete_marker(str(tmp_path), rows)
    d = json.load(open(p))
    assert d["tasks"] == 2 and d["episodes"] == 3 and d["finished_at"].endswith("+00:00")
