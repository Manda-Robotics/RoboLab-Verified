"""P45: one line per transition — twins collapse, ladder lines carry their predicate's name."""
import sys, types
for n in ("isaaclab", "isaaclab.utils"):
    sys.modules.setdefault(n, types.ModuleType(n))
from robolab.core.events.event_lines import dedupe_tick, name_for_ladder_line
from robolab.core.task.status import StatusCode

def L(code, info): return {"step": 10, "code": int(code), "name": "x", "info": info, "score": 0.0}

def test_drop_and_ladder_regression_become_one_line():
    tr = [L(StatusCode.OBJECT_DROPPED, "'lime01_01' dropped (left the closed hand)")]
    ladder = L(StatusCode.OBJECT_GRABBED_FAILURE, "failed: object_grabbed(object=lime01_01). regressing to step 0 for lime01_01.")
    kept, ld = dedupe_tick(tr, ladder)
    assert len(kept) == 1 and ld is None

def test_unrelated_regression_is_kept():
    tr = [L(StatusCode.OBJECT_DROPPED, "'banana' dropped (left the closed hand)")]
    ladder = L(StatusCode.OBJECT_GRABBED_FAILURE, "failed: object_grabbed(object=lime01_01). regressing to step 0 for lime01_01.")
    kept, ld = dedupe_tick(tr, ladder)
    assert ld is not None and ld["name"] == "OBJECT_GRABBED_FAILURE"

def test_ladder_line_named_after_its_predicate():
    ladder = L(StatusCode.OBJECT_GRABBED_SUCCESS, "success: object_on_top(object=lime01_01, reference_object=clay_plates). advanced 1 step(s)")
    _, ld = dedupe_tick([], ladder)
    assert ld["name"] == "OBJECT_ON_TOP_SUCCESS"
    assert name_for_ladder_line(int(StatusCode.OBJECT_GRABBED_SUCCESS), "success: object_in_container(object=banana, container=bowl)") == "OBJECT_IN_CONTAINER_SUCCESS"

def test_completion_line_keeps_its_text_and_wrong_detach_folds():
    tr = [L(StatusCode.OBJECT_RELEASED, "'rubiks_cube' released (hand opened)"), L(StatusCode.WRONG_OBJECT_DETACHED, "Wrong object that was grabbed is now detached: 'rubiks_cube'")]
    ladder = L(StatusCode.OBJECT_IN_CONTAINER_SUCCESS, "Completed subtask 'pick_and_place' 1/1")
    kept, ld = dedupe_tick(tr, ladder)
    assert [k["code"] for k in kept] == [int(StatusCode.OBJECT_RELEASED)]
    assert ld is not None and "Completed subtask" in ld["info"] and ld["name"] == "SUBTASK_COMPLETED"
