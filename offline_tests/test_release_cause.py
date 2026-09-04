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


def _rec(pairs=None, forces=None, T=60, fall_from=None):
    r = types.SimpleNamespace()
    r.T = T
    v = np.zeros((T, 3))
    if fall_from is not None:                       # picks up speed after leaving the hand
        v[fall_from:, 2] = -0.9
    r.obj_vel = {"can": v}
    r.pair_contact = {k: np.array(v, dtype=np.uint8) for k, v in (pairs or {}).items()}
    r.body_forces = {k: np.array(v, dtype=np.float32) for k, v in (forces or {}).items()}
    return r


def _col(T, on):
    c = np.zeros(T, dtype=np.uint8)
    for a, b in on:
        c[a:b] = 1
    return c


def test_commanded_open_onto_a_support_is_released():
    rec = _rec({"can__table": _col(60, [(31, 60)])})          # lands the row after leaving
    assert P._release_cause(rec, "can", 30, True, DT) == ("released", None)


def test_commanded_open_after_hitting_the_wall_and_falling_is_knocked():
    """rc5 FoodPacking2Cans env 2: can on the bin wall while gripped, open commanded, can falls."""
    rec = _rec({"can__bin": _col(60, [(26, 32)]), "can__table": _col(60, [(36, 60)])})
    cause, culprits = P._release_cause(rec, "can", 30, True, DT, rest_at=45)
    assert cause == "knocked" and culprits == ["bin"]


def test_a_short_drop_onto_the_bin_floor_is_still_released():
    """rc4 BlackItemsInBin: smartphone let go over the bin, lands on the keyboard inside it."""
    rec = _rec({"can__keyboard": _col(60, [(33, 60)])})
    assert P._release_cause(rec, "can", 30, True, DT) == ("released", None)


def test_a_rim_bounce_that_settles_inside_the_bin_is_released():
    """rc5 FoodPacking2Cans env 3, 50.7 s: let go 18 cm up, bounces on the rim, rests inside."""
    rec = _rec({"can__bin": _col(60, [(28, 33), (34, 60)])})
    assert P._release_cause(rec, "can", 30, True, DT, rest_at=50) == ("released", None)


def test_set_down_on_the_bin_floor_after_touching_it_is_released():
    rec = _rec({"can__bin": _col(60, [(28, 60)])})       # touched the bin just before, stays on it
    assert P._release_cause(rec, "can", 30, True, DT) == ("released", None)


def test_new_contact_just_before_leaving_a_closed_hand_is_knocked():
    rec = _rec({"can__bin": _col(60, [(28, 30)]), "can__table": _col(60, [(40, 60)])})
    assert P._release_cause(rec, "can", 30, False, DT, rest_at=50)[0] == "knocked"


def test_sliding_out_of_a_closed_hand_onto_the_table_it_was_lowered_to_is_slipped():
    """rc7_upstream FoodPacking2Cans env 1, 45.9 s: box touches the table as it is lowered, leaves the
    closed hand, rests on the table. The reviewer: slipped, no other object involved."""
    rec = _rec({"box__table": _col(60, [(29, 30), (31, 60)])})
    assert P._release_cause(rec, "box", 30, False, DT, rest_at=50) == ("slipped", None)


def test_contact_that_was_already_continuous_is_not_a_knock():
    """A dragged object touches the table throughout: the table did not knock it out."""
    rec = _rec({"can__table": _col(60, [(0, 60)])})
    assert P._release_cause(rec, "can", 30, False, DT) == ("slipped", None)
    assert P._release_cause(rec, "can", 30, True, DT) == ("released", None)   # and let go on it: released


def test_no_contact_at_all_is_slipped():
    rec = _rec({"can__bin": _col(60, []), "can__table": _col(60, [(40, 60)]), "other__bin": _col(60, [(20, 40)])})
    assert P._release_cause(rec, "can", 30, False, DT) == ("slipped", None)


def test_hand_body_hitting_something_else_counts_with_a_hand_prefix():
    rec = _rec(pairs={"can__bin": _col(60, []), "can__table": _col(60, [(40, 60)])}, forces={"left_outer_knuckle__bin": _col(60, [(29, 31)])})
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


def test_in_box_uses_the_oriented_footprint_not_the_axis_aligned_one():
    """A 30 cm square bin rotated 45 deg has a 42 cm axis-aligned extent; a point in the AABB's
    corner is outside the bin."""
    import math
    c, s_ = math.cos(math.pi / 4), math.sin(math.pi / 4)
    R = np.array([[c, -s_, 0], [s_, c, 0], [0, 0, 1]])
    local = np.array([[x, y, z] for x in (-0.15, 0.15) for y in (-0.15, 0.15) for z in (0.0, 0.1)])
    corners = local @ R.T
    assert P._in_box(corners, np.array([0.0, 0.0, 0.05]))              # centre
    assert P._in_box(corners, np.array([0.2, 0.0, 0.05]))              # on a diagonal, inside the rotated square (half-diagonal 0.21)
    assert not P._in_box(corners, np.array([0.19, 0.19, 0.05]))        # AABB corner region, outside the bin
    assert P._in_box(corners, np.array([0.0, 0.0, 0.14]))              # 4 cm above the top: on it
    assert not P._in_box(corners, np.array([0.0, 0.0, 0.2]))           # 10 cm above: carried over it


def test_in_box_on_a_flat_wide_box_whose_face_diagonal_is_shorter_than_its_long_edge():
    """d6 BananasInCrate: crate 0.30 x 0.21 x 0.15, banana resting inside read outside."""
    local = np.array([[x, y, z] for x in (-0.15, 0.15) for y in (-0.105, 0.105) for z in (0.0, 0.147)])
    assert P._in_box(local, np.array([0.12, 0.08, 0.09]))              # inside, near a corner
    assert not P._in_box(local, np.array([0.18, 0.0, 0.05]))           # past the long side
