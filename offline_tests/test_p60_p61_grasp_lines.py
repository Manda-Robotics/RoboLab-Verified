# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""P60 (the tracker emits the grasp) and P61 (lines are stamped at their onset).

Both come from Finn's r3 review. P60: BowlStackingRightOnLeft printed a release
and a drop but never a grab, because the grab line was only ever a side effect of
a subtask ladder containing an `object_grabbed` step -- and that task's ladder is
a single placement condition. P61: "the grasp attempt is already over and then it
is counted as a fail" -- the burst is flushed GRASP_ATTEMPT_BURST_S after the last
blip, so the line landed ~2 s late while the onset step was known all along.
"""
import torch

import robolab.core.task.grasp as G
from robolab.core.events.event_lines import fold_duplicate_grab_lines
from robolab.core.task.status import StatusCode
from offline_tests.test_grasp_tracker import _Env, _Script, _tick


def _carry(env, t, s):
    for x in (0.00, 0.01, 0.02):
        grasped, ev = _tick(env, t, s, True, hand=[x, 0, 0], obj=[x + 0.05, 0, 0], closure=0.6)
    return grasped, ev


def test_p60_tracker_emits_the_grasp_itself():
    env = _Env(dt=0.1); s = _Script(); s.install(); t = G.GraspTracker(env)
    env.episode_length_buf[:] = 1
    grasped, ev = _carry(env, t, s)
    assert grasped is True
    grabs = [e for e in ev if e[3] == "grabbed"]
    assert len(grabs) == 1, "a carry must produce exactly one grasp event"
    assert grabs[0][1] == "banana"


def test_p60_grasp_is_emitted_once_not_every_tick():
    env = _Env(dt=0.1); s = _Script(); s.install(); t = G.GraspTracker(env)
    env.episode_length_buf[:] = 1
    evs = []
    for x in (0.00, 0.01, 0.02, 0.03, 0.04, 0.05):
        _, ev = _tick(env, t, s, True, hand=[x, 0, 0], obj=[x + 0.05, 0, 0], closure=0.6)
        evs += ev
    assert len([e for e in evs if e[3] == "grabbed"]) == 1


def test_p61_grasp_carries_the_contact_onset_step():
    env = _Env(dt=0.1); s = _Script(); s.install(); t = G.GraspTracker(env)
    env.episode_length_buf[:] = 1
    _, ev = _carry(env, t, s)
    grab = [e for e in ev if e[3] == "grabbed"][0]
    onset = grab[4]["onset_step"]
    # contact began on the first of the three carry ticks, well before the carry
    # criteria were satisfied on the third
    assert onset < int(env.episode_length_buf[0]), "the grasp must be stamped before it was detected"


def _line(step, info, code=int(StatusCode.OBJECT_GRABBED_SUCCESS)):
    return {"step": step, "code": code, "name": "OBJECT_GRABBED_SUCCESS", "info": info, "score": 0.0}


def test_p60_ladder_grab_line_wins_when_both_exist():
    # a task whose ladder has an object_grabbed step keeps printing exactly one grab
    events = [
        _line(100, "'bowl_1' grasped (carry established)"),
        _line(102, "success: object_grabbed(object=bowl_1). advanced 1 step(s) to step 1"),
    ]
    out = fold_duplicate_grab_lines(events, window_steps=8)
    assert len(out) == 1 and "advanced" in out[0]["info"]


def test_p60_tracker_grab_line_survives_when_the_ladder_has_none():
    # the five stacking tasks: this line is the only record that a grasp happened
    events = [_line(100, "'bowl_1' grasped (carry established)")]
    out = fold_duplicate_grab_lines(events, window_steps=8)
    assert len(out) == 1 and "grasped" in out[0]["info"]


def test_p60_two_separate_grasps_are_not_collapsed():
    # a release between them means these are two different grasps
    events = [
        _line(100, "'bowl_1' grasped (carry established)"),
        {"step": 200, "code": int(StatusCode.OBJECT_RELEASED), "name": "OBJECT_RELEASED",
         "info": "'bowl_1' released (hand opened)", "score": 0.0},
        _line(400, "success: object_grabbed(object=bowl_1). advanced 1 step(s) to step 1"),
    ]
    out = fold_duplicate_grab_lines(events, window_steps=8)
    assert len([e for e in out if "grasped (carry" in e["info"]]) == 1


def test_p60_ladder_line_lagging_far_behind_still_folds():
    """rc3 measured the ladder lagging the tracker by 0.60-2.20 s (9-33 steps).

    The original fixed 0.53 s window caught none of the five duplicates in that
    run, which is why the rule is now structural.
    """
    for lag in (9, 10, 18, 23, 33):
        events = [
            _line(100, "'banana' grasped (carry established)"),
            _line(100 + lag, "success: object_grabbed(object=banana). advanced 1 step(s) to step 1"),
        ]
        out = fold_duplicate_grab_lines(events, window_steps=8)
        assert len(out) == 1 and "advanced" in out[0]["info"], f"lag {lag} not folded"


def test_p60_a_grab_of_a_different_object_is_not_folded():
    events = [
        _line(100, "'bowl_2' grasped (carry established)"),
        _line(101, "success: object_grabbed(object=bowl_1). advanced 1 step(s) to step 1"),
    ]
    assert len(fold_duplicate_grab_lines(events, window_steps=8)) == 2
