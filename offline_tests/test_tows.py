"""Towing artifact detection (docs/verified/towing.md, P124): the Robotiq forward kinematics, the
hook classifier, and the end-to-end pass over the clean dense fixture. No simulator."""
import os

import numpy as np
import pytest

import robolab.constants as C
from robolab.eval import tows as T

FIX = os.path.join(os.path.dirname(__file__), "fixtures", "dense", "t1_BananaInBowl", "BananaInBowlTask")


def _pad_face_y(theta):
    """|y| of the pad's inner face for both fingers at finger_joint = theta, from the FK and the pad box."""
    jp = np.zeros(13)
    # the mimic pattern of the recorded joints: outer right +theta, inner knuckles -theta, inner fingers -/+theta
    jp[7], jp[8], jp[9], jp[10], jp[11], jp[12] = theta, theta, -theta, theta, -theta, -theta
    poses = T.gripper_link_poses(np.zeros(3), np.array([1.0, 0, 0, 0]), jp)
    out = []
    for side in ("left", "right"):
        p, q = poses[T.PAD_BODY[side]]
        lo, hi = T.PAD_BOX[side]
        corners = np.array([[x, y, z] for x in (lo[0], hi[0]) for y in (lo[1], hi[1]) for z in (lo[2], hi[2])])
        world = (T.quat_to_R(q) @ corners.T).T + p
        out.append(np.abs(world[:, 1]).min())
    return out


def test_fk_pad_planes():
    # open: the two pad faces are 42.5 mm off the jaw axis (85 mm aperture); closed: they meet
    yl, yr = _pad_face_y(0.0)
    assert abs(yl - 0.0425) < 0.001 and abs(yr - 0.0425) < 0.001
    yl, yr = _pad_face_y(T.FINGER_CLOSED)
    assert yl < 0.003 and yr < 0.003
    # a fixed identity joint keeps the outer finger on its knuckle
    poses = T.gripper_link_poses(np.zeros(3), np.array([1.0, 0, 0, 0]), np.zeros(13))
    assert np.allclose(poses["left_outer_finger"][0], poses["left_outer_knuckle"][0], atol=1e-6)


def test_pad_signed_distance_sign():
    poses = T.gripper_link_poses(np.zeros(3), np.array([1.0, 0, 0, 0]), np.zeros(13))
    inside = np.array([[0.13, 0.045, 0.0]])          # 2.5 mm behind the left pad face
    outside = np.array([[0.13, 0.030, 0.0]])         # 12.5 mm in front of it, in the aperture
    sd_in, n_in = T.pad_signed_distance("left", inside, poses)
    sd_out, n_out = T.pad_signed_distance("left", outside, poses)
    assert -2.6 < sd_in < -2.4 and n_in == 1
    assert 12.4 < sd_out < 12.6 and n_out == 0


def _rows(depth, n, far=40.0, fj=0.0, pad="right", step=0.2):
    sL, sR = (-depth, far) if pad == "left" else (far, -depth)
    return [[i * step, sL, sR, fj] for i in range(n)]


def test_classify_tow_drag_press_squeeze():
    step = 0.2
    # a 3 s hook at the open limit, object lifted 5 cm: tier-A tow
    segs = T.classify(_rows(6.0, 15), step, lambda t0, t1: (0.05, 0.10))
    assert len(segs) == 1 and segs[0].cls == "tow" and segs[0].tier == "A" and segs[0].pad == "right"
    # same hook, object pushed along the table: drag
    segs = T.classify(_rows(6.0, 15), step, lambda t0, t1: (0.005, 0.10))
    assert segs[0].cls == "drag" and segs[0].tier is None
    # same hook, object stationary: press
    segs = T.classify(_rows(6.0, 15), step, lambda t0, t1: (0.0, 0.0))
    assert segs[0].cls == "press"
    # mid-range finger with the far pad touching: a squeeze, not a hook
    segs = T.classify(_rows(4.0, 15, far=0.3, fj=0.4), step, lambda t0, t1: (0.1, 0.3))
    assert segs[0].cls == "squeeze"
    # mid-range finger but the far pad clear: still a tow, tier C (not at a limit)
    segs = T.classify(_rows(4.0, 15, far=30.0, fj=0.4), step, lambda t0, t1: (0.1, 0.3))
    assert segs[0].cls == "tow" and segs[0].tier == "C"
    # too short for a hook
    assert T.classify(_rows(6.0, 2), step, lambda t0, t1: (0.05, 0.1)) == []
    # shallow contact is not a hook
    assert T.classify(_rows(C.TOW_HOOK_DEPTH_MM - 0.5, 15), step, lambda t0, t1: (0.05, 0.1)) == []
    # short hook at a limit, lifted: tier B
    segs = T.classify(_rows(4.0, 6), step, lambda t0, t1: (0.025, 0.05))
    assert segs[0].tier == "B"


def test_summary_tiers():
    segs = [dict(cls="tow", tier="B", object="mug", duration_s=1.2, t_start=3.0, depth_max_mm=4.0),
            dict(cls="tow", tier="A", object="bowl", duration_s=5.0, t_start=10.0, depth_max_mm=8.0),
            dict(cls="press", tier=None, object="bowl", duration_s=1.0, t_start=20.0, depth_max_mm=3.0)]
    s = T._summary(segs)
    assert s["tow_tier"] == "A" and s["towed_objects"] == ["bowl", "mug"] and s["n_press"] == 1
    assert abs(s["tow_time_s"] - 6.2) < 1e-6 and s["first_tow_s"] == 3.0
    assert T._summary([])["tow_tier"] is None


def test_clean_fixture_has_no_tow():
    pytest.importorskip("h5py")
    pytest.importorskip("pxr")
    if not os.path.isfile(os.path.join(C.ASSET_DIR, "scenes", "banana_bowl.usda")):
        pytest.skip("scene assets not pulled")
    for env in range(4):
        doc = T.analyse_episode(FIX, env, 0)
        assert doc.get("status") is None and "banana" in doc["objects_checked"]
        assert doc["summary"]["tow_tier"] is None, doc["segments"]
