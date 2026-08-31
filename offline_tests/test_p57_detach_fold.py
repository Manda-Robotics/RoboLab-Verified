# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""P57: a WRONG_OBJECT_DETACHED absorbed by the release/drop that follows it.

Measured on the rc2 runs (24 episodes): 43 of 43 detach lines were followed by a
release/drop for the same object within 0.5 s, median 0.07 s. Rule c already
handled the same-tick case; this covers the near-tick case, and must still keep a
detach that no release follows.
"""
from robolab.core.events.event_lines import fold_detach_into_release
from robolab.core.task.status import StatusCode

DETACH = int(StatusCode.WRONG_OBJECT_DETACHED)
RELEASED = int(StatusCode.OBJECT_RELEASED)
DROPPED = int(StatusCode.OBJECT_DROPPED)
BUMPED = int(StatusCode.OBJECT_BUMPED)


def _ev(step, code, obj):
    return {"step": step, "code": code, "name": "x", "info": f"'{obj}' something", "score": 0.0}


def test_detach_folded_when_release_follows_within_window():
    events = [_ev(100, DETACH, "orange_01"), _ev(101, RELEASED, "orange_01")]
    out = fold_detach_into_release(events, window_steps=8)
    assert [e["code"] for e in out] == [RELEASED]


def test_detach_folded_when_drop_follows():
    events = [_ev(100, DETACH, "lime01_01"), _ev(107, DROPPED, "lime01_01")]
    assert len(fold_detach_into_release(events, window_steps=8)) == 1


def test_detach_kept_when_no_release_follows():
    # the informative case: knocked out of the hand without the gripper opening
    events = [_ev(100, DETACH, "orange_01"), _ev(101, BUMPED, "orange_01")]
    out = fold_detach_into_release(events, window_steps=8)
    assert [e["code"] for e in out] == [DETACH, BUMPED]


def test_detach_kept_when_release_is_outside_the_window():
    events = [_ev(100, DETACH, "orange_01"), _ev(200, RELEASED, "orange_01")]
    assert len(fold_detach_into_release(events, window_steps=8)) == 2


def test_detach_kept_when_release_is_for_a_different_object():
    events = [_ev(100, DETACH, "orange_01"), _ev(101, RELEASED, "lemon_02")]
    assert len(fold_detach_into_release(events, window_steps=8)) == 2


def test_release_before_detach_does_not_absorb_a_later_detach():
    # a release that already happened cannot explain a detach that comes after it
    events = [_ev(90, RELEASED, "orange_01"), _ev(100, DETACH, "orange_01")]
    out = fold_detach_into_release(events, window_steps=8)
    assert [e["code"] for e in out] == [RELEASED, DETACH]


def test_window_zero_disables_folding_and_input_is_not_mutated():
    events = [_ev(100, DETACH, "orange_01"), _ev(101, RELEASED, "orange_01")]
    assert len(fold_detach_into_release(events, window_steps=0)) == 2
    assert len(events) == 2
