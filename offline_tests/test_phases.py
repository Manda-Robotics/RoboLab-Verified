"""Dense phase annotation (docs/verified/dense_annotations.md): invariants, the tracker replay,
and the two fixture episodes that anchor the L2 semantics. No simulator; needs h5py and torch
(the replay); the geometric fallback is exercised separately."""
import json
import os

import numpy as np
import pytest

pytest.importorskip("h5py")

from robolab.eval import phases as P  # noqa: E402
from robolab.eval.tracker_replay import compare_events, load_recording, replay  # noqa: E402

h5py = P._real_h5py()      # test_p67 leaves a bare stub named h5py in sys.modules

FIX = os.path.join(os.path.dirname(__file__), "fixtures", "dense")
T1 = os.path.join(FIX, "t1_BananaInBowl", "BananaInBowlTask")            # 4/4 clean successes
T3 = os.path.join(FIX, "t3_BowlStackLeftOnRight", "BowlStackingLeftOnRightTask")  # 0/4, indecision then a late grasp


def _log(task_dir, env):
    return json.load(open(os.path.join(task_dir, f"log_0_env{env}.json")))["events"]


def _episodes():
    for td in (T1, T3):
        for env in range(4):
            yield td, env


# --------------------------------------------------------------------------- roles
def test_roles_from_env_cfg():
    assert P.task_roles(T1) == ({"banana"}, {"bowl"})
    assert P.task_roles(T3) == ({"bowl_2"}, {"bowl_1"})      # the pick target is a bowl (P81)
    assert P.task_kind(T1) == "place" and P.task_kind(T3) == "place"


# --------------------------------------------------------------------------- H1 invariants
@pytest.mark.parametrize("td,env", list(_episodes()))
def test_h1_coverage_and_order(td, env):
    d = P.annotate(td, env, log_events=_log(td, env)).doc
    ph = d["phases"]
    assert ph[0]["start"] == 1 and ph[-1]["end"] == d["num_steps"]
    assert all(ph[i]["end"] + 1 == ph[i + 1]["start"] for i in range(len(ph) - 1))
    assert all(p["label"] in P.FAMILY_OF for p in ph)
    at = d["attempts"]
    assert all(at[i]["end"] < at[i + 1]["start"] for i in range(len(at) - 1))
    assert all(a["result"] in ("pass", "fail", "unknown") for a in at)
    assert d["step_convention"].startswith("1-based")
    assert d["annotator"]["tracker_replay"] is True
    # every threshold the file was written with is named in it
    assert set(P.THRESHOLDS) <= set(d["annotator"]["thresholds"])


# --------------------------------------------------------------------------- anchors
def test_t1_env0_is_the_textbook_chain():
    d = P.annotate(T1, 0, log_events=_log(T1, 0)).doc
    labels = [p["label"] for p in d["phases"]]
    for a, b in (("reach", "approach"), ("approach", "close_on"), ("close_on", "lift"), ("lift", "transport"),
                 ("lower", "release")):
        assert labels.index(a) < labels.index(b), (a, b, labels)
    at = d["attempts"]
    assert [(a["label"], a["result"], a["object"]) for a in at] == [("pick", "pass", "banana"), ("place", "pass", "banana")]
    assert at[0]["start"] == 1                       # the initial reach belongs to the first pick
    assert at[1]["destination"] == "bowl"
    assert at[1]["end"] == d["num_steps"]
    # a disagreement between the tracker replay and the geometric in-hand channel is allowed
    # (the pick window is short); a disagreement with the event log is not
    assert not any(f for a in at for f in a["flags"] if not f.startswith("held_disagree"))
    s = d["summary"]
    assert s["n_picks_pass"] == 1 and s["n_places_pass"] == 1 and s["n_drops"] == 0


def test_t3_env1_indecision_then_a_pick_that_never_places():
    d = P.annotate(T3, 1, log_events=_log(T3, 1)).doc
    at = d["attempts"]
    kinds = [(a["label"], a["result"]) for a in at]
    assert kinds[0] == ("no_completed_subtask", "fail")
    assert at[0]["end"] * d["dt"] > 8.0                # the 10 s of approach / retreat alternation
    assert ("pick", "pass") in kinds
    # still in hand at the 20 s cap, held above the other bowl: a place in progress
    assert kinds[-1] == ("place", "unknown")
    assert any("still held at the end" in x for x in at[-1]["attributes"])


def test_t3_env3_picks_the_wrong_bowl():
    d = P.annotate(T3, 3, log_events=_log(T3, 3)).doc
    picks = [a for a in d["attempts"] if a["label"] == "pick"]
    assert picks and picks[0]["object"] == "bowl_1"
    assert "destination_as_target" in picks[0]["attributes"]


# --------------------------------------------------------------------------- H2 / H5
def test_h2_events_land_in_the_phase_that_names_them():
    checked = consistent = 0
    for td, env in _episodes():
        log = _log(td, env)
        a = P.annotate(td, env, log_events=log)
        c = P.check_events(a.doc, log, a.channels)
        checked += c["checked"]; consistent += c["consistent"]
    assert checked > 20
    assert consistent / checked >= 0.85, (consistent, checked)


def test_h5_replay_reproduces_the_logged_tracker_lines():
    logged = matched = 0
    for td, env in _episodes():
        cfg = json.load(open(os.path.join(td, "env_cfg.json")))
        dt = cfg["sim"]["dt"] * cfg["decimation"]
        with h5py.File(os.path.join(td, "run_0.hdf5"), "r") as f:
            rec = load_recording(f["data"][f"demo_{env}"])
        rr = replay(rec, dt, containers=P.task_roles(td)[1])
        c = compare_events(rr.events, _log(td, env), tol_steps=2, by="detected")
        logged += c["logged"]; matched += c["matched"]
    # the fixture logs predate P84 (carry onset never before the grip), so onsets differ on
    # every t1 carry; detection is compared instead (plan H5)
    assert matched / logged >= 0.9, (matched, logged)


def test_replay_restores_the_accessors():
    import robolab.core.task.grasp as G
    before = (G.hand_contact, G.hand_position, G.object_position, G.jaw_offset, G.hand_closed)
    with h5py.File(os.path.join(T1, "run_0.hdf5"), "r") as f:
        rec = load_recording(f["data"]["demo_0"])
    replay(rec, 1 / 15)
    assert (G.hand_contact, G.hand_position, G.object_position, G.jaw_offset, G.hand_closed) == before


# --------------------------------------------------------------------------- degraded modes
def test_geometry_only_mode_is_flagged():
    d = P.annotate(T1, 0, log_events=_log(T1, 0), use_replay=False).doc
    assert d["annotator"]["tracker_replay"] is False
    assert all("geometry only" in " ".join(a["flags"]) for a in d["attempts"])
    labels = [p["label"] for p in d["phases"]]
    assert "lift" in labels and "release" in labels


def test_contact_proxy_when_no_contact_group(tmp_path):
    """An upstream-era recording (no contact/ group) still annotates, with a warning."""
    src = os.path.join(T1, "run_0.hdf5")
    dst = tmp_path / "BananaInBowlTask"
    dst.mkdir()
    with h5py.File(src, "r") as fi, h5py.File(dst / "run_0.hdf5", "w") as fo:
        g = fo.create_group("data").create_group("demo_0")
        for key in ("actions", "ee_pose/position", "ee_pose/orientation", "states/articulation/robot/joint_position"):
            g.create_dataset(key, data=fi["data/demo_0"][key][:])
        for o in fi["data/demo_0/states/rigid_object"]:
            for k in ("root_pose", "root_velocity"):
                g.create_dataset(f"states/rigid_object/{o}/{k}", data=fi[f"data/demo_0/states/rigid_object/{o}/{k}"][:])
        for o in fi["data/demo_0/bbox/bbox_mm"]:
            g.create_dataset(f"bbox/bbox_mm/{o}", data=fi[f"data/demo_0/bbox/bbox_mm/{o}"][:])
    for name in ("env_cfg.json", "log_0_env0.json"):
        (dst / name).write_text(open(os.path.join(T1, name)).read())
    d = P.annotate(str(dst), 0).doc
    assert d["annotator"]["contact_source"] == "proxy"
    assert any("no contact/ group" in w for w in d["warnings"])
    assert any(a["label"] == "pick" for a in d["attempts"])


# --------------------------------------------------------------------------- segmentation helpers
def test_segment_absorbs_short_runs_but_never_release():
    labels = ["idle"] * 10 + ["hover"] * 2 + ["idle"] * 10 + ["release"] * 2 + ["idle"] * 3
    objs = [None] * 27
    segs, warnings = P.segment(labels, objs, min_run=4)
    # the 2-step hover is absorbed; the 2-step release survives; the 3-step tail folds into it
    assert [s["label"] for s in segs] == ["idle", "release"]
    assert segs[0]["start"] == 1 and segs[0]["end"] == 22 and segs[-1]["end"] == 27
    assert warnings and "absorbed" in warnings[0]


def test_write_roundtrip(tmp_path):
    d = P.annotate(T1, 0, log_events=_log(T1, 0)).doc
    path = P.write(d, str(tmp_path))
    back = json.load(open(path))
    assert back["schema_version"] == P.SCHEMA_VERSION
    assert back["phases"] == d["phases"]
    assert os.path.basename(path) == "phases_0_env0.json"
