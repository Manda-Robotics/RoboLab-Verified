"""scripts/friction_sweep_report.py: the per-condition table for the P79 / C1 sweep."""
import importlib.util
import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
_SPEC = importlib.util.spec_from_file_location("friction_sweep_report", ROOT / "scripts" / "friction_sweep_report.py")
R = importlib.util.module_from_spec(_SPEC)
sys.modules["friction_sweep_report"] = R
_SPEC.loader.exec_module(R)


def _run(tmp_path, name, successes, events_per_ep, complete=True, applied=None):
    d = tmp_path / name
    task = d / "XTask"
    task.mkdir(parents=True)
    with open(d / "episode_results.jsonl", "w") as f:
        for i, s in enumerate(successes):
            f.write(json.dumps({"env_id": i, "success": s, "score": 1.0 if s else 0.0, "physics_artifact": False}) + "\n")
    for i, evs in enumerate(events_per_ep):
        json.dump({"task": "XTask", "env_id": i, "run": 0, "dt": 0.1, "success": False, "final_step": 100,
                   "events": [{"step": k, "code": c, "name": "x", "info": "'o'"} for k, c in enumerate(evs)]},
                  open(task / f"log_0_env{i}.json", "w"))
    if complete:
        (d / "run_complete.json").write_text("{}")
    if applied is not None:
        json.dump(applied, open(task / "friction_applied.json", "w"))
    return str(d)


GOOD = {"objects": {"o": [[0.5, 0.5, 0.0]]}, "gripper": {"l": [[0.5, 0.5, 0.0]]},
        "requested": {"mode": "uniform", "spec": "0.5", "gripper_bodies": ["l"],
                      "gripper": {"static": 0.5, "dynamic": 0.5}, "objects": {"o": {"static": 0.5, "dynamic": 0.5, "source": "uniform"}}}}
UP = {"objects": {"o": [[2.0, 2.0, 0.0]]}, "gripper": {}, "requested": {"mode": "upstream", "objects": {}, "gripper_bodies": []}}


def test_groups_by_mode_and_task_and_counts_the_events(tmp_path):
    a = _run(tmp_path, "rc7_upstream_X", [True, True, False, False], [[283, 267], [283, 268], [266, 266], []], applied=UP)
    b = _run(tmp_path, "rc7_0.5_X", [True, False, False, False], [[283, 267], [266], [266, 266, 266], [275]],
             complete=False, applied=GOOD)
    groups = R.group_runs([a, b], R.DEFAULT_PATTERN)
    assert set(groups) == {"upstream", "0.5"} and groups["0.5"] == {"X": b}
    s = R.load_run(a)
    assert (s["episodes"], s["successes"], s["carried"], s["failed"], s["dropped"], s["released"]) == (4, 2, 2, 2, 1, 1)
    text = R.render(groups, ["upstream", "1.0", "0.5", "realistic"])
    head = text.splitlines()[0]
    assert head == "| task | upstream | 0.5 |"                    # only modes that exist, in the requested order
    assert "| X | 2/4 (50%) | 1/4 (25%) ⏳ |" in text             # incomplete run is flagged, never hidden
    assert "| success rate | 2/4 (50%) | 1/4 (25%) |" in text
    assert "| carries / episode | 0.5 | 0.2 |" in text
    assert "| failed attempts / episode | 0.5 | 1.0 |" in text
    assert "| tows (`TOWED_WITHOUT_GRASP`) / physics-artifact episodes | 0 / 0 | 1 / 0 |" in text
    assert "| P79 verdict per run | N/A×1 | PASS×1 |" in text


def test_a_run_whose_materials_did_not_land_is_marked_in_the_table(tmp_path):
    bad = dict(GOOD, objects={"o": [[2.0, 2.0, 0.0]]})
    b = _run(tmp_path, "rc7_0.5_X", [True], [[283]], applied=bad)
    text = R.render(R.group_runs([b], R.DEFAULT_PATTERN))
    assert "| X | 1/1 (100%) **FAIL** |" in text


def test_no_match_is_an_error_not_an_empty_table(tmp_path, capsys):
    d = _run(tmp_path, "something_else", [True], [[]])
    sys.argv = ["x", d]
    assert R.main() == 2


def test_an_in_flight_or_unreadable_hdf5_degrades_to_na_not_a_traceback(tmp_path):
    """The report ran while the sync loop was copying rc7_upstream_FoodPacking2Cans and died
    inside h5py.File on the half-written HDF5."""
    d = _run(tmp_path, "rc7_0.5_X", [True], [[283]], complete=False)
    (pathlib.Path(d) / "XTask" / "run_0.hdf5").write_bytes(b"not an hdf5 file")
    assert R.one_pad_share(d) is None
    (pathlib.Path(d) / "run_complete.json").write_text("{}")
    assert R.one_pad_share(d) is None          # complete but unreadable: still no traceback
    assert "n/a (h5py)" in R.render(R.group_runs([d], R.DEFAULT_PATTERN))
