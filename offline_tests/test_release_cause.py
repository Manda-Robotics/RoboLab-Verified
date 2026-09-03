# SPDX-License-Identifier: Apache-2.0
"""The release cause (plan §9.13): why an object left the hand, as a third axis of a place
next to label and result; and the prepare/derive split of annotate() that the stability study
relies on producing the same attempts as annotate() itself."""
import os
import sys
import types

import numpy as np

ROOT = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, ROOT)
from robolab.eval import phases as P  # noqa: E402

DT = 1 / 15


def _rec(pairs=None, forces=None, T=60):
    r = types.SimpleNamespace()
    r.T = T
    r.pair_contact = {k: np.array(v, dtype=np.uint8) for k, v in (pairs or {}).items()}
    r.body_forces = {k: np.array(v, dtype=np.float32) for k, v in (forces or {}).items()}
    return r


def _col(T, on):
    c = np.zeros(T, dtype=np.uint8)
    for a, b in on:
        c[a:b] = 1
    return c


def test_commanded_open_is_released_whatever_was_touched():
    rec = _rec({"can__bin": _col(60, [(28, 40)])})
    assert P._release_cause(rec, "can", 30, True, DT) == ("released", None)


def test_new_contact_just_before_leaving_a_closed_hand_is_knocked():
    rec = _rec({"can__bin": _col(60, [(28, 40)]), "can__table": _col(60, [])})
    cause, culprits = P._release_cause(rec, "can", 30, False, DT)
    assert cause == "knocked" and culprits == ["bin"]


def test_contact_that_was_already_continuous_is_not_a_knock():
    """A dragged object touches the table throughout: the table did not knock it out."""
    rec = _rec({"can__table": _col(60, [(0, 60)])})
    assert P._release_cause(rec, "can", 30, False, DT) == ("slipped", None)


def test_no_contact_at_all_is_slipped():
    rec = _rec({"can__bin": _col(60, []), "other__bin": _col(60, [(20, 40)])})
    assert P._release_cause(rec, "can", 30, False, DT) == ("slipped", None)


def test_hand_body_hitting_something_else_counts_with_a_hand_prefix():
    rec = _rec(pairs={"can__bin": _col(60, [])}, forces={"left_outer_knuckle__bin": _col(60, [(29, 31)])})
    cause, culprits = P._release_cause(rec, "can", 30, False, DT)
    assert cause == "knocked" and culprits == ["hand:bin"]


def test_without_contact_pairs_the_cause_is_unclear():
    assert P._release_cause(_rec(), "can", 30, False, DT) == ("unclear", None)


def test_every_place_carries_a_cause_and_derive_matches_annotate():
    td = os.path.join(ROOT, "offline_tests", "fixtures", "dense", "t1_BananaInBowl", "BananaInBowlTask")
    a = P.annotate(td, 0, 0, use_replay=False)
    for at in a.doc["attempts"]:
        if at["label"] == "place":
            assert at["cause"] in ("released", "knocked", "slipped", "unclear", None)  # None: still held at the cap
        else:
            assert "cause" not in at
    assert "n_places_by_cause" in a.doc["summary"]
    prep = P.prepare_recording(td, 0, 0, use_replay=False)
    _, segs, _, _ = P.derive(prep, [])
    strip = lambda ss: [{k: v for k, v in s.items() if k != "flags"} for s in ss]  # noqa: E731
    assert strip(segs) == strip(a.doc["attempts"])
