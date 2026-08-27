# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""P71/P72/P73 — the three flag-quality complaints from Finn's rc3 review.

P71  The detector's grab line was OBJECT_GRABBED_SUCCESS, identical in name and
     colour to the ladder's progress line. Finn: "there are two object grab
     success flags... let me know if this was on purpose". Worse, it put a green
     success on 28 grasps of objects the task then flagged as WRONG. It is now
     OBJECT_CARRIED, a neutral physical observation.
P72  14 of 81 attempt lines were on a container or fixture. Finn: "I don't think
     this was a grasp attempt on bin".
P73  10 of 81 followed a release of that same object: contact flickers as the
     object leaves the hand. Finn: "there's often a grasp attempt failed after a
     release, which I don't really fully see".
"""
import torch

import robolab.core.task.grasp as G
from robolab.core.task.status import NEUTRAL_STATUS_CODES, StatusCode
from offline_tests.test_grasp_tracker import _Env, _Script, _tick


def test_p71_object_carried_exists_and_is_neutral():
    assert int(StatusCode.OBJECT_CARRIED) in NEUTRAL_STATUS_CODES
    assert int(StatusCode.OBJECT_CARRIED) != int(StatusCode.OBJECT_GRABBED_SUCCESS)


def test_p71_carried_is_not_a_success_code():
    # anything >= 200 is failure/neutral class; the green SUCCESS range is < 200.
    # OBJECT_CARRIED must not sit in the success range, or it reads as progress.
    assert int(StatusCode.OBJECT_CARRIED) >= 200


def test_p73_no_attempt_is_opened_right_after_a_release():
    env = _Env(dt=0.1); s = _Script(); s.install(); t = G.GraspTracker(env)
    env.episode_length_buf[:] = 1
    for x in (0.00, 0.01, 0.02):                     # establish a carry
        _tick(env, t, s, True, hand=[x, 0, 0], obj=[x + 0.05, 0, 0], closure=0.6)
    _tick(env, t, s, False, hand=[0.03, 0, 0], obj=[0.08, 0, 0], closure=0.0, cmd_open=True)  # release
    evs = []
    for x in (0.04, 0.05):                           # contact flickers as it leaves
        evs += _tick(env, t, s, True, hand=[x, 0, 0], obj=[0.08, 0, 0], closure=0.6)[1]
        evs += _tick(env, t, s, False, hand=[x, 0, 0], obj=[0.08, 0, 0], closure=0.6)[1]
    for _ in range(30):
        evs += _tick(env, t, s, False, closure=0.0)[1]
    assert [e[3] for e in evs if e[3] == "attempt_failed"] == [], \
        "a flicker right after a release must not count as a fresh failed attempt"


def test_p73_a_genuine_attempt_much_later_still_counts():
    env = _Env(dt=0.1); s = _Script(); s.install(); t = G.GraspTracker(env)
    env.episode_length_buf[:] = 1
    for x in (0.00, 0.01, 0.02):
        _tick(env, t, s, True, hand=[x, 0, 0], obj=[x + 0.05, 0, 0], closure=0.6)
    _tick(env, t, s, False, hand=[0.03, 0, 0], obj=[0.08, 0, 0], closure=0.0, cmd_open=True)
    for _ in range(40):                              # long quiet gap
        _tick(env, t, s, False, closure=0.0)
    evs = []
    evs += _tick(env, t, s, True, hand=[0.1, 0, 0], obj=[0.15, 0, 0], closure=0.6)[1]
    evs += _tick(env, t, s, False, closure=0.6)[1]
    for _ in range(30):
        evs += _tick(env, t, s, False, closure=0.0)[1]
    assert any(e[3] == "attempt_failed" for e in evs), "a real later attempt must still be reported"
