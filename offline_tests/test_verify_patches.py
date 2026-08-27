"""The verifier decides what may be called "verified on hardware", so a wrong PASS
is the expensive failure. These tests give each predicate a hand-built recording of
the pre-patch behaviour it exists to catch, and one of the post-patch behaviour."""
import importlib.util
import json
import pathlib
import sys

import pytest

_SPEC = importlib.util.spec_from_file_location(
    "verify_patches", pathlib.Path(__file__).resolve().parents[1] / "scripts" / "verify_patches.py")
V = importlib.util.module_from_spec(_SPEC)
sys.modules["verify_patches"] = V          # @dataclass resolves annotations via sys.modules
_SPEC.loader.exec_module(V)

DT = 1 / 15.0


def ev(step, code, name, info, detected=None):
    e = {"step": step, "code": code, "name": name, "info": info, "score": 0.0}
    if detected is not None:
        e["detected_step"] = detected
    return e


def episode(events, *, final_step=900, env_id=0, task="T", success=False):
    return V.Episode("<mem>", task, env_id, 0, DT, success, final_step, events)


def write_run(tmp_path, episodes, task="T"):
    """A run directory shaped like a real one, so `load` is exercised too."""
    d = tmp_path / f"rc_{task}"
    (d / task).mkdir(parents=True)
    for e in episodes:
        (d / task / f"log_0_env{e.env_id}.json").write_text(json.dumps(
            {"schema_version": 2, "dt": e.dt, "task": task, "env_id": e.env_id, "run": 0,
             "success": e.success, "final_step": e.final_step, "events": e.events}))
    return str(d)


# --- P61 ------------------------------------------------------------------- #

def test_p61_flags_an_onset_stamped_after_its_detection():
    bad = episode([ev(100, 283, "OBJECT_CARRIED", "'mug' carried", detected=90)])
    assert V.p61_onset_stamping([bad]).verdict == "FAIL"


def test_p61_passes_when_onset_precedes_detection():
    good = episode([ev(90, 283, "OBJECT_CARRIED", "'mug' carried", detected=100)])
    r = V.p61_onset_stamping([good])
    assert r.verdict == "PASS" and r.opportunities == 1


# --- P71 ------------------------------------------------------------------- #

def test_p71_catches_the_trackers_green_grab_line():
    """rc3's tracker line: code 139, and no `success:` prefix — it is not a ladder line."""
    bad = episode([ev(50, 139, "OBJECT_GRABBED_SUCCESS", "'mug' grasped (carry established)")])
    assert V.p71_no_green_on_tracker_grabs([bad]).verdict == "FAIL"


def test_p71_passes_with_a_neutral_tracker_line_beside_a_ladder_line():
    good = episode([ev(50, 283, "OBJECT_CARRIED", "'mug' carried (grasp established)"),
                    ev(60, 139, "OBJECT_GRABBED_SUCCESS", "success: object_grabbed(object=mug).")])
    assert V.p71_no_green_on_tracker_grabs([good]).verdict == "PASS"


# --- P72 ------------------------------------------------------------------- #

@pytest.mark.parametrize("obj", ["rack_l04", "grey_bin", "bin_a06", "clay_plates"])
def test_p72_catches_an_attempt_on_a_container_or_fixture(obj):
    bad = episode([ev(30, 266, "GRASP_ATTEMPT_FAILED", f"Grasp attempt on '{obj}' failed")])
    assert V.p72_no_attempts_on_containers([bad]).verdict == "FAIL"


def test_p72_passes_on_attempts_against_real_objects():
    good = episode([ev(30, 266, "GRASP_ATTEMPT_FAILED", "Grasp attempt on 'banana_02' failed")])
    assert V.p72_no_attempts_on_containers([good]).verdict == "PASS"


def test_p72_is_not_a_pass_when_nothing_was_attempted():
    assert V.p72_no_attempts_on_containers([episode([])]).verdict == "N/A"


# --- P73 ------------------------------------------------------------------- #

def test_p73_catches_an_attempt_inside_the_burst_window_after_a_release():
    inside = int(V.GRASP_ATTEMPT_BURST_S / DT) - 1
    bad = episode([ev(100, 267, "OBJECT_RELEASED", "'mug' released (hand opened)"),
                   ev(100 + inside, 266, "GRASP_ATTEMPT_FAILED", "Grasp attempt on 'mug' failed")])
    assert V.p73_no_attempt_after_release([bad]).verdict == "FAIL"


def test_p73_allows_a_genuine_retry_after_the_window():
    outside = int(V.GRASP_ATTEMPT_BURST_S / DT) + 2
    good = episode([ev(100, 267, "OBJECT_RELEASED", "'mug' released (hand opened)"),
                    ev(100 + outside, 266, "GRASP_ATTEMPT_FAILED", "Grasp attempt on 'mug' failed")])
    assert V.p73_no_attempt_after_release([good]).verdict == "PASS"


def test_p73_does_not_confuse_two_different_objects():
    inside = int(V.GRASP_ATTEMPT_BURST_S / DT) - 1
    good = episode([ev(100, 267, "OBJECT_RELEASED", "'mug' released (hand opened)"),
                    ev(100 + inside, 266, "GRASP_ATTEMPT_FAILED", "Grasp attempt on 'banana' failed")])
    assert V.p73_no_attempt_after_release([good]).verdict == "PASS"


# --- P74 ------------------------------------------------------------------- #

def test_p74_catches_the_black_items_free_credit_at_027s():
    bad = episode([ev(4, 125, "OBJECT_IN_CONTAINER_SUCCESS",
                      "success: object_in_container(object=keyboard, container=grey_bin)")])
    r = V.p74_nothing_credited_before_settle([bad])
    assert r.verdict == "FAIL" and "settle warm-up" in r.detail


def test_p74_passes_when_the_first_credit_is_after_the_warmup():
    good = episode([ev(200, 139, "OBJECT_GRABBED_SUCCESS", "success: object_grabbed(object=mug).")])
    assert V.p74_nothing_credited_before_settle([good]).verdict == "PASS"


def test_p74_ignores_early_non_success_events():
    """A gripper bump at 0.27s is fine; only *credit* is forbidden before settle."""
    good = episode([ev(4, 255, "GRIPPER_HIT_TABLE", "Gripper hit table"),
                    ev(200, 139, "OBJECT_GRABBED_SUCCESS", "success: object_grabbed(object=mug).")])
    assert V.p74_nothing_credited_before_settle([good]).verdict == "PASS"


# --- P75 / P76 -------------------------------------------------------------- #

def test_p75_catches_a_surviving_placed_without_lift():
    bad = episode([ev(4, 274, "PLACED_WITHOUT_LIFT", "'keyboard' placed without ever being carried")])
    assert V.p75_placed_without_lift_retired([bad]).verdict == "FAIL"


def test_p76_catches_an_episode_that_ran_on_after_losing_its_destination():
    bad = episode([ev(139, 273, "TARGET_LOST", "Target lost: rack_l04 off the table")], final_step=2700)
    assert V.p76_lost_destination_ends_episode([bad]).verdict == "FAIL"


def test_p76_passes_when_the_episode_stops_on_that_step():
    good = episode([ev(139, 273, "TARGET_LOST", "Target lost: rack_l04 off the table")], final_step=139)
    assert V.p76_lost_destination_ends_episode([good]).verdict == "PASS"


def test_p76_is_not_a_pass_when_no_destination_was_lost():
    assert V.p76_lost_destination_ends_episode([episode([])]).verdict == "N/A"


# --- end to end ------------------------------------------------------------- #

def test_a_run_directory_round_trips_and_a_clean_run_exits_zero(tmp_path, capsys):
    run = write_run(tmp_path, [episode([
        ev(90, 283, "OBJECT_CARRIED", "'mug' carried (grasp established)", detected=100),
        ev(100, 139, "OBJECT_GRABBED_SUCCESS", "success: object_grabbed(object=mug)."),
        ev(139, 273, "TARGET_LOST", "Target lost: rack_l04 off the table"),
    ], final_step=139)])
    argv = sys.argv
    try:
        sys.argv = ["verify_patches", run]
        assert V.main() == 0
    finally:
        sys.argv = argv
    out = capsys.readouterr().out
    assert "[PASS] P71" in out and "[PASS] P76" in out


def test_a_regressed_run_exits_nonzero(tmp_path):
    run = write_run(tmp_path, [episode([
        ev(4, 274, "PLACED_WITHOUT_LIFT", "'keyboard' placed without ever being carried")])])
    argv = sys.argv
    try:
        sys.argv = ["verify_patches", run]
        assert V.main() == 1
    finally:
        sys.argv = argv


def test_an_unfinished_run_is_not_reported_as_a_regression(tmp_path):
    """The fetcher copies a run directory while it is still filling up."""
    d = tmp_path / "rc_inflight"; (d / "T").mkdir(parents=True)
    results = V.report(str(d))
    assert [r.verdict for r in results] == ["N/A"]
