# SPDX-License-Identifier: Apache-2.0
"""Dense phase annotation of a recorded episode, from simulator state alone.

Design and evidence: ``docs/verified/dense_annotations.md``. Two layers are produced from
the per-step arrays in ``run_<i>.hdf5`` plus the roles in ``env_cfg.json``:

* **L1 phases**: one label per step, contiguous and exhaustive. Vocabulary in
  :data:`PHASES`. Rules are evaluated top to bottom in :func:`label_steps`; the first match
  wins. The in-hand state comes from the runtime :class:`GraspTracker`, replayed over the
  recording (:mod:`robolab.eval.tracker_replay`), so "carried" means what the event log
  means by it. A purely geometric second opinion (both pads and lifted) is kept for the
  disagreement flag.
* **L2 attempts**: ``pick`` / ``place`` / ``drop`` / ``no_completed_subtask`` segments built
  from L1 by :func:`attempts`, each with ``result``, derived ``attributes``, a templated
  ``description`` and ``flags`` that name every disagreement with the log or with the
  second derivation.

Everything reads the run directory only. No simulator, no ``isaaclab``; ``torch`` is
needed for the tracker replay and the module degrades to the geometric in-hand channel
without it. Thresholds that the event tracker owns are imported from
:mod:`robolab.constants`; the ones introduced here are module constants, named in every
output file under ``annotator.thresholds`` so a reader can tell which build wrote it.
"""
from __future__ import annotations

import collections
import json
import math
import os
import re
from dataclasses import dataclass, field

import numpy as np



def _real_h5py():
    """The real h5py, even when an earlier import left a stub in ``sys.modules`` (an offline
    test does that to import a module that only needs the name)."""
    import importlib
    import sys
    mod = sys.modules.get("h5py")
    if mod is not None and not hasattr(mod, "File"):
        del sys.modules["h5py"]
        mod = None
    return mod or importlib.import_module("h5py")

from robolab.eval.tracker_replay import Recording, load_recording, quat_rotmat

try:  # the event tracker's own constants; never re-declared here
    import robolab.constants as _C
    SETTLE_WARMUP_S = float(getattr(_C, "SETTLE_WARMUP_S", 1.0))
    GRASP_HOLD_S = float(getattr(_C, "GRASP_HOLD_S", 0.2))
    GRASP_ATTEMPT_BURST_S = float(getattr(_C, "GRASP_ATTEMPT_BURST_S", 2.0))
    SUCCESS_REST_S = float(getattr(_C, "SUCCESS_REST_S", 0.2))
    SUCCESS_MAX_SPEED = float(getattr(_C, "SUCCESS_MAX_SPEED", 0.02))
except Exception:  # pragma: no cover
    SETTLE_WARMUP_S, GRASP_HOLD_S, GRASP_ATTEMPT_BURST_S, SUCCESS_REST_S, SUCCESS_MAX_SPEED = 1.0, 0.2, 2.0, 0.2, 0.02

SCHEMA_VERSION = 1
ANNOTATOR = "robolab.eval.phases"
ANNOTATOR_VERSION = "0.1"

# --- thresholds introduced by this module (plan §2.3; sensitivity: plan H4) -------------
TCP_OFFSET = (0.15, 0.03, 0.0)   # base_link -> fingertip midpoint, base_link frame (measured, plan §1.1)
V_MOVE = 0.02                    # m/s   TCP counts as moving
V_LIFT = 0.03                    # m/s   object rising / descending while held
LIFT_M = 0.01                    # m     above its own rest height = lifted
NEAR_M = 0.08                    # m     hover radius around an object
APPROACH_M = 0.25                # m     approach (inside) vs reach (outside)
CLOSING_RATE = 0.01              # m/s   |d dist/dt| below this is neither approach nor retreat
DISP_WIN_S = 0.53                # s     window for object displacement (8 steps at 15 Hz)
DISP_M = 0.01                    # m     displacement over the window = the object moved
REST_M = 0.002                   # m     an untouched object that moves less than this over DISP_WIN_S is still (settle)
SLIP_V = 0.05                    # m/s   object speed relative to the TCP while held
SMOOTH_W = 5                     # steps mode filter
MIN_RUN = 4                      # steps runs shorter than this are absorbed into the predecessor
RELEASE_S = 0.5                  # s     the release phase after the hand opens on a held object
GAP_NCS_S = 5.0                  # s     a stretch this long with no attempt is no_completed_subtask ...
NCS_BREAKER_SHARE = 0.4          #       ... if at least this share of it is retreat / idle / hover / table / repositioning
PICK_MIN_HOLD_S = 0.5            # s     held this long (and lifted) = pick complete
BREAK_RUN_S = 0.5                # s     a retreat/idle run this long, or an approach to another object, ends the approach that belongs to a pick
CONTACT_PROXY_M = 0.02           # m     TCP to object box, when no contact/ group is recorded
THRESHOLDS = {k: globals()[k] for k in (
    "TCP_OFFSET", "V_MOVE", "V_LIFT", "LIFT_M", "NEAR_M", "APPROACH_M", "CLOSING_RATE", "DISP_WIN_S", "DISP_M",
    "REST_M", "SLIP_V", "SMOOTH_W", "MIN_RUN", "RELEASE_S", "GAP_NCS_S", "NCS_BREAKER_SHARE", "PICK_MIN_HOLD_S",
    "CONTACT_PROXY_M", "BREAK_RUN_S", "SETTLE_WARMUP_S", "GRASP_HOLD_S", "GRASP_ATTEMPT_BURST_S",
    "SUCCESS_REST_S", "SUCCESS_MAX_SPEED")}

PHASES = {
    # family: phases. Colour and lane grouping in the dashboard follow the family.
    "settle": ["settle"],
    "in_hand": ["lift", "lower", "transport", "hold", "slip"],
    "grasp": ["close_on", "pinch_no_lift", "open_contact", "close_empty", "release"],
    "table": ["press_table"],
    "motion": ["approach", "reach", "retreat", "reposition"],
    "still": ["hover", "idle"],
    "scene": ["disturb"],
}
FAMILY_OF = {p: f for f, ps in PHASES.items() for p in ps}
OBJECT_PHASES = {"lift", "lower", "transport", "hold", "slip", "close_on", "pinch_no_lift", "open_contact",
                 "release", "approach", "reach", "hover", "disturb"}

# What an event log line should fall into (plan H2). Row = step - 1; a tolerance of one
# phase boundary either side is allowed by the checker.
EVENT_EXPECTS = {
    "OBJECT_GRIPPED": {"close_on", "pinch_no_lift", "lift", "hold", "transport"},
    "OBJECT_CARRIED": {"close_on", "pinch_no_lift", "lift", "hold", "transport", "lower", "slip"},
    "OBJECT_GRABBED_SUCCESS": {"pinch_no_lift", "lift", "hold", "transport", "lower", "close_on"},
    "OBJECT_RELEASED": {"release", "lower", "hold", "transport"},
    "OBJECT_DROPPED": {"disturb", "release", "lower", "transport", "hold", "slip", "lift", "close_empty", "idle", "retreat"},
    "GRIPPER_HIT_TABLE": {"press_table"},
    "GRASP_ATTEMPT_FAILED": {"close_on", "pinch_no_lift", "open_contact", "close_empty"},
    "TOWED_WITHOUT_GRASP": {"open_contact", "lift", "transport", "hold"},
    "GRIPPER_FULLY_CLOSED": {"close_empty"},
}


# --------------------------------------------------------------------------- roles
def task_roles(task_dir: str) -> tuple[set[str], set[str]]:
    """(targets, destinations) from the run's ``env_cfg.json``: the ladder and the success
    terminations serialise ``object=`` / ``container=`` / ``surface=`` / ``reference_object=``
    arguments (the kwarg families of ``robolab/core/task/target_objects.py``). Same reading as
    ``scripts/verify_patches.py::task_roles`` (P81), which deliberately does not import this
    package, plus the reference objects of the on-top / next-to predicates, which for the
    attempt layer are where the object goes."""
    targets, dests = set(), set()
    path = os.path.join(task_dir, "env_cfg.json")
    if os.path.exists(path):
        cfg = json.load(open(path))
        text = json.dumps(cfg.get("subtasks")) + json.dumps(cfg.get("terminations", ""))
        for key, bucket in (("object", targets), ("objects", targets), ("container", dests),
                            ("containers", dests), ("surface", dests), ("surfaces", dests),
                            ("reference_object", dests), ("reference_objects", dests)):
            for m in re.finditer(r"\b%s=(\[[^\]]*\]|'[^']*')" % key, text):
                bucket.update(re.findall(r"'([^']+)'", m.group(1)))
    dests.discard("table")
    return targets, dests


def task_kind(task_dir: str) -> str:
    """``place`` when any success termination needs the object somewhere (in / on / stacked),
    ``pick`` when every success termination is a pick-up or grab, ``unknown`` otherwise."""
    path = os.path.join(task_dir, "env_cfg.json")
    if not os.path.exists(path):
        return "unknown"
    funcs = []
    for name, term in (json.load(open(path)).get("terminations") or {}).items():
        if isinstance(term, dict) and "func" in term and name != "time_out" and "lost" not in name:
            funcs.append(str(term["func"]).rsplit(":", 1)[-1])
    if not funcs:
        return "unknown"
    if all(f in ("object_picked_up", "object_grabbed", "object_in_contact") for f in funcs):
        return "pick"
    return "place"


def resolve_dt(task_dir: str, default: float = 1 / 15) -> float:
    path = os.path.join(task_dir, "env_cfg.json")
    try:
        cfg = json.load(open(path))
        return float(cfg["sim"]["dt"]) * float(cfg["decimation"])
    except Exception:
        return default


def find_hdf5(task_dir: str, run_index: int = 0) -> str | None:
    for cand in (f"run_{run_index}.hdf5", "data.hdf5"):
        p = os.path.join(task_dir, cand)
        if os.path.exists(p):
            return p
    hits = sorted(f for f in os.listdir(task_dir) if f.startswith("run_") and f.endswith(".hdf5"))
    return os.path.join(task_dir, hits[0]) if hits else None


# --------------------------------------------------------------------------- channels
def _win_disp(pos: np.ndarray, w: int) -> np.ndarray:
    out = np.zeros(pos.shape[0])
    if w < pos.shape[0]:
        out[w:] = np.linalg.norm(pos[w:] - pos[:-w], axis=1)
    return out


def _speed(pos: np.ndarray, dt: float) -> np.ndarray:
    if pos.shape[0] < 3:
        return np.zeros(pos.shape[0])
    return np.linalg.norm(np.gradient(pos, dt, axis=0), axis=1)


def _bbox_dist(point: np.ndarray, corners: np.ndarray) -> float:
    """Distance from a point to the axis-aligned box spanned by 8 corners (0 inside)."""
    lo, hi = corners.min(axis=0), corners.max(axis=0)
    d = np.maximum(np.maximum(lo - point, 0), point - hi)
    return float(np.linalg.norm(d))


def proxy_pads(rec: Recording, dt: float) -> dict[str, np.ndarray]:
    """Synthetic ``contact/<obj>`` columns for a recording without the contact group: pad 1 =
    the TCP within CONTACT_PROXY_M of the object's box AND evidence of interaction (jaws
    part-closed, a close command, or the object moving); pad 2 = the same with the jaws at
    least 30 % closed (so "both pads" means a pinch). Distance alone read "touch" for 6.2 s of
    hovering next to the mustard in the first narrated episode; this composite reads 0.6 s
    outside the narrated contact windows and misses 0.3 of 7.9 s inside. Feeding these to the
    tracker replay gives coupled-motion carries on such recordings instead of a distance blink."""
    T = rec.T
    R = quat_rotmat(rec.ee_quat)
    tcp = rec.ee_pos + np.einsum("tij,j->ti", R, np.asarray(TCP_OFFSET))
    closure = rec.closure()
    cmd = rec.actions[:, -1] > 0.5
    w = max(1, int(round(DISP_WIN_S / dt)))
    pads = {}
    for o in rec.objects:
        p = rec.obj_pos[o]
        corners = rec.bbox_corners.get(o)
        if corners is not None:
            dd = np.array([_bbox_dist(tcp[t], corners[t]) for t in range(T)])
        else:
            dd = np.linalg.norm(tcp - p, axis=1) - 0.04
        engaged = (closure > 0.15) | cmd | (_win_disp(p, w) > 0.002)
        touch = (dd < CONTACT_PROXY_M) & engaged
        pads[o] = np.stack([touch, touch & (closure > 0.3)], axis=1).astype(np.uint8)
    return pads


@dataclass
class Channels:
    T: int
    dt: float
    objects: list[str]
    roles: dict[str, str]
    tcp: np.ndarray                        # (T, 3)
    tcp_speed: np.ndarray                  # (T,)
    grip_cmd: np.ndarray                   # (T,) bool closed commanded
    closure: np.ndarray                    # (T,) measured closure fraction
    touch: dict[str, np.ndarray]           # (T,) bool, either pad
    pinch: dict[str, np.ndarray]           # (T,) bool, both pads
    table_touch: np.ndarray                # (T,) bool
    dist: dict[str, np.ndarray]            # (T,) TCP to object centroid
    closing_rate: dict[str, np.ndarray]    # (T,) d dist / dt
    lift: dict[str, np.ndarray]            # (T,) z above rest
    obj_speed: dict[str, np.ndarray]       # (T,)
    disp: dict[str, np.ndarray]            # (T,) windowed displacement
    at_rest: dict[str, np.ndarray]         # (T,) bool
    held_tracker: list[str | None]         # per step
    held_geom: list[str | None]
    closed_on: dict[str, np.ndarray]       # (T,) bool: jaws closed on it during this contact (tracker attempt_closed)
    towed: dict[str, np.ndarray]           # (T,) bool from the replay
    replay_events: list[dict]
    contact_source: str                    # "pads" | "proxy"
    replay_ok: bool


def compute_channels(rec: Recording, dt: float, targets: set[str], dests: set[str],
                     replay_result=None, contact_source: str = "pads") -> Channels:
    T = rec.T
    objects = list(rec.objects)
    roles = {o: ("target" if o in targets else "destination" if o in dests else "distractor") for o in objects}
    R = quat_rotmat(rec.ee_quat)
    tcp = rec.ee_pos + np.einsum("tij,j->ti", R, np.asarray(TCP_OFFSET))
    tcp_speed = _speed(tcp, dt)
    grip_cmd = rec.actions[:, -1] > 0.5
    closure = rec.closure()
    w_disp = max(1, int(round(DISP_WIN_S / dt)))
    # "at rest" exactly as P30 confirms a success: the recorded speed below SUCCESS_MAX_SPEED for
    # SUCCESS_REST_S in a row. The recorded velocity jitters by up to 5 cm/s on some resting
    # objects (plan §4 finding 3), which affects P30 the same way; kept identical on purpose.
    w_rest = max(1, int(round(SUCCESS_REST_S / dt)))
    settle_rows = max(1, int(round(SETTLE_WARMUP_S / dt)))

    touch, pinch, dist, rate, lift, speed, disp, rest = {}, {}, {}, {}, {}, {}, {}, {}
    for o in objects:
        p = rec.obj_pos[o]
        touch[o], pinch[o] = rec.contact(o), rec.pinch(o)     # recorded pads, or proxy_pads()
        dist[o] = np.linalg.norm(tcp - p, axis=1)
        rate[o] = np.gradient(dist[o], dt) if T > 2 else np.zeros(T)
        rest_z = float(np.median(p[min(settle_rows, T - 1):min(settle_rows + 15, T), 2])) if T > settle_rows + 1 else float(p[0, 2])
        lift[o] = p[:, 2] - rest_z
        speed[o] = _speed(p, dt)
        disp[o] = _win_disp(p, w_disp)
        slow = np.linalg.norm(rec.obj_vel[o], axis=1) < SUCCESS_MAX_SPEED
        rest[o] = np.array([slow[max(0, t - w_rest + 1): t + 1].all() and t + 1 >= w_rest for t in range(T)])
    tp = rec.pads.get("table")
    if tp is not None:
        thr = 0.1 if np.issubdtype(tp.dtype, np.floating) else 0
        table_touch = (tp > thr).any(axis=1)
    else:
        table_touch = np.zeros(T, bool)

    held_geom: list[str | None] = []
    for t in range(T):
        cands = [o for o in objects if pinch[o][t] and lift[o][t] > LIFT_M]
        held_geom.append(min(cands, key=lambda o: dist[o][t]) if cands else None)

    towed = {o: np.zeros(T, bool) for o in objects}
    replay_events: list[dict] = []
    held_tracker: list[str | None]
    if replay_result is not None:
        held_tracker = []
        for t in range(T):
            cands = [o for o in objects if replay_result.grasped.get(o, np.zeros(T, bool))[t]]
            held_tracker.append(min(cands, key=lambda o: dist[o][t]) if cands else None)
        towed = {o: replay_result.towed.get(o, np.zeros(T, bool)) for o in objects}
        closed_on = {o: replay_result.attempt_closed.get(o, np.zeros(T, bool)) for o in objects}
        replay_events = replay_result.events
        replay_ok = True
    else:
        held_tracker = list(held_geom)
        closed_on = {o: touch[o] & (closure > 0.3) for o in objects}
        replay_ok = False
    return Channels(T, dt, objects, roles, tcp, tcp_speed, grip_cmd, closure, touch, pinch, table_touch, dist,
                    rate, lift, speed, disp, rest, held_tracker, held_geom, closed_on, towed, replay_events,
                    contact_source, replay_ok)


# --------------------------------------------------------------------------- L1
def label_steps(ch: Channels) -> tuple[list[str], list[str | None]]:
    """Per-step (phase, object). Rules in plan §2.3 order."""
    T, dt = ch.T, ch.dt
    settle_rows = max(1, int(round(SETTLE_WARMUP_S / dt)))
    release_rows = max(1, int(round(RELEASE_S / dt)))
    labels: list[str] = []
    objs: list[str | None] = []
    release_until: dict[str, int] = {}
    vz = {o: (np.gradient(ch.lift[o], dt) if T > 2 else np.zeros(T)) for o in ch.objects}
    tcp_v = np.gradient(ch.tcp, dt, axis=0) if T > 2 else np.zeros_like(ch.tcp)
    for t in range(T):
        held = ch.held_tracker[t]
        prev_held = ch.held_tracker[t - 1] if t > 0 else None
        if prev_held is not None and held != prev_held and not ch.grip_cmd[t]:
            release_until[prev_held] = t + release_rows
        touching = [o for o in ch.objects if ch.touch[o][t]]
        pinching = [o for o in ch.objects if ch.pinch[o][t]]
        near = min(ch.objects, key=lambda o: ch.dist[o][t]) if ch.objects else None
        moving_free = [o for o in ch.objects if ch.disp[o][t] > DISP_M and o not in touching and o != held]

        if t < settle_rows and held is None and not touching:
            still_settling = any(ch.disp[o][t] > REST_M for o in ch.objects) or t < 3
            if still_settling or t < 3:
                labels.append("settle"); objs.append(None); continue
        rel = [o for o, until in release_until.items() if t < until]
        if rel and held is None:
            labels.append("release"); objs.append(rel[0]); continue
        if held is not None:
            o = held
            if ch.lift[o][t] > LIFT_M or ch.pinch[o][t]:
                if ch.obj_speed[o][t] > SLIP_V and np.linalg.norm(tcp_v[t]) < ch.obj_speed[o][t] - SLIP_V:
                    labels.append("slip")
                elif vz[o][t] > V_LIFT:
                    labels.append("lift")
                elif vz[o][t] < -V_LIFT:
                    labels.append("lower")
                elif ch.tcp_speed[t] > V_MOVE:
                    labels.append("transport")
                else:
                    labels.append("hold")
                objs.append(o); continue
            # tracker says carried but not lifted and not pinched: a drag / tow
            labels.append("pinch_no_lift" if ch.grip_cmd[t] else "open_contact"); objs.append(o); continue
        if pinching and ch.grip_cmd[t]:
            labels.append("pinch_no_lift"); objs.append(pinching[0]); continue
        if touching and ch.grip_cmd[t]:
            labels.append("close_on"); objs.append(touching[0]); continue
        if touching:
            labels.append("open_contact"); objs.append(touching[0]); continue
        if ch.table_touch[t]:
            labels.append("press_table"); objs.append(None); continue
        if ch.grip_cmd[t] and ch.closure[t] > 0.6:
            labels.append("close_empty"); objs.append(None); continue
        if moving_free:
            o = max(moving_free, key=lambda o: ch.disp[o][t])
            labels.append("disturb"); objs.append(o); continue
        if ch.tcp_speed[t] > V_MOVE and near is not None:
            r = ch.closing_rate[near][t]
            if r < -CLOSING_RATE:
                labels.append("approach" if ch.dist[near][t] < APPROACH_M else "reach"); objs.append(near)
            elif r > CLOSING_RATE:
                labels.append("retreat"); objs.append(None)
            else:
                labels.append("reposition"); objs.append(None)
            continue
        if near is not None and ch.dist[near][t] < NEAR_M:
            labels.append("hover"); objs.append(near); continue
        labels.append("idle"); objs.append(None)
    return labels, objs


def smooth_labels(labels: list[str], objs: list[str | None], w: int = SMOOTH_W) -> tuple[list[str], list[str | None]]:
    """Mode filter over ``w`` steps on (label, object) pairs; ``release`` is exempt."""
    keys = list(zip(labels, objs))
    out = list(keys)
    h = w // 2
    for t in range(len(keys)):
        if keys[t][0] == "release":
            continue
        win = [k for k in keys[max(0, t - h): t + h + 1] if k[0] != "release"]
        out[t] = collections.Counter(win).most_common(1)[0][0] if win else keys[t]
    return [k[0] for k in out], [k[1] for k in out]


def segment(labels: list[str], objs: list[str | None], min_run: int = MIN_RUN) -> tuple[list[dict], list[str]]:
    """Runs of equal (label, object) -> segments with 1-based inclusive ``start``/``end`` steps.
    Runs shorter than ``min_run`` are absorbed into their predecessor (``release`` exempt);
    every absorption is reported as a warning."""
    segs: list[list] = []
    for t, key in enumerate(zip(labels, objs)):
        if segs and (segs[-1][2], segs[-1][3]) == key:
            segs[-1][1] = t
        else:
            segs.append([t, t, key[0], key[1]])
    out: list[list] = []
    absorbed = 0
    for s in segs:
        n = s[1] - s[0] + 1
        if out and n < min_run and s[2] != "release":
            out[-1][1] = s[1]; absorbed += 1
        else:
            out.append(s)
    merged: list[list] = []
    for s in out:
        if merged and (merged[-1][2], merged[-1][3]) == (s[2], s[3]):
            merged[-1][1] = s[1]
        else:
            merged.append(s)
    warnings = [f"absorbed {absorbed} runs shorter than {min_run} steps"] if absorbed else []
    return [{"start": s[0] + 1, "end": s[1] + 1, "label": s[2], "object": s[3]} for s in merged], warnings


# --------------------------------------------------------------------------- L2
def _in_destination(rec: Recording, ch: Channels, o: str, d: str, t: int) -> bool | None:
    """Object ``o`` inside / on destination ``d`` at row ``t``: the recorded pair-contact column
    if present, else the centroid inside the destination's box footprint and not above its
    top by more than 5 cm. ``None`` when nothing can be said."""
    col = rec.pair_contact.get(f"{o}__{d}")
    if col is not None and col[t]:
        return True
    corners = rec.bbox_corners.get(d)
    if corners is None:
        return None
    lo, hi = corners[t].min(axis=0), corners[t].max(axis=0)
    p = rec.obj_pos[o][t]
    inside_xy = lo[0] <= p[0] <= hi[0] and lo[1] <= p[1] <= hi[1]
    return bool(inside_xy and p[2] <= hi[2] + 0.05 and p[2] >= lo[2] - 0.02)


def attempts(rec: Recording, ch: Channels, phases: list[dict], targets: set[str], dests: set[str],
             kind: str, log_events: list[dict] | None) -> list[dict]:
    """L2 segments. A pick opens when the jaws close on an object (the tracker's
    ``attempt_closed``, i.e. >= 30 % closure while touching, P47) or when the tracker says it is
    carried; an open-hand brush is not an attempt. The approach that precedes the pick belongs
    to it, back to the last retreat / idle / table-pressing run of at least ``BREAK_RUN_S``;
    anything before that of at least ``GAP_NCS_S`` is ``no_completed_subtask``. Consecutive
    failed picks on one object closer than ``GRASP_ATTEMPT_BURST_S`` are one pick with a
    ``regrasp`` count, the way P47 folds attempt lines."""
    T, dt = ch.T, ch.dt
    lab = [None] * T
    obj = [None] * T
    for s in phases:
        for t in range(s["start"] - 1, s["end"]):
            lab[t], obj[t] = s["label"], s["object"]
    held = ch.held_tracker
    min_hold = max(1, int(round(PICK_MIN_HOLD_S / dt)))
    break_run = max(1, int(round(BREAK_RUN_S / dt)))
    burst = max(1, int(round(GRASP_ATTEMPT_BURST_S / dt)))
    log_events = log_events or []
    BREAKERS = {"retreat", "idle", "press_table", "close_empty", "disturb", "reposition"}

    def engaged(o, u):
        return ch.touch[o][u] or held[u] == o or ch.closed_on[o][u]

    raw: list[dict] = []
    last_parting: dict[str, int] = {}
    t = 0
    while t < T:
        cand = held[t] or next((o for o in ch.objects if ch.closed_on[o][t]), None)
        if cand is None:
            t += 1
            continue
        o = cand
        # walk forward while engaged with this object (allow 2-step contact flicker)
        u, gap, held_at, lifted = t, 0, None, False
        while u < T:
            if engaged(o, u):
                gap = 0
                if held[u] == o:
                    held_at = u if held_at is None else held_at
                    lifted = lifted or ch.lift[o][u] > LIFT_M
            else:
                gap += 1
                if gap > 2:
                    break
            u += 1
        end_contact = u - 1 - min(gap, u - 1 - t)
        held_len = 0
        if held_at is not None:
            v = held_at
            while v <= end_contact and held[v] == o:
                v += 1
            held_len = v - held_at
        # a pick completes when the tracker's carry has lasted PICK_MIN_HOLD_S (the fork's own
        # definition of a grasp: coupled motion, P31); whether it was lifted is an attribute
        passed = held_at is not None and held_len >= min_hold
        # P73: contact flicker right after a parting is not a new attempt; nor is a single
        # step of contact (the tracker needs a contact streak too)
        if not passed and ((o in last_parting and t - last_parting[o] <= burst) or end_contact - t + 1 < 2):
            t = end_contact + 1
            continue
        raw.append({"o": o, "first": t, "end_contact": end_contact, "held_at": held_at, "lifted": lifted,
                    "held_len": held_len, "passed": passed})
        if held_at is not None:
            last_parting[o] = held_at + held_len
        t = end_contact + 1

    # fold failed picks on the same object inside one burst (P47)
    folded: list[dict] = []
    for r in raw:
        prev = folded[-1] if folded else None
        if prev and not prev["passed"] and not r["passed"] and prev["o"] == r["o"] and r["first"] - prev["end_contact"] <= burst:
            prev["end_contact"] = r["end_contact"]
            prev["n_attempts"] = prev.get("n_attempts", 1) + 1
        else:
            r.setdefault("n_attempts", 1)
            folded.append(r)

    segs: list[dict] = []
    for idx, r in enumerate(folded):
        o, first = r["o"], r["first"]
        nxt_first = folded[idx + 1]["first"] if idx + 1 < len(folded) else T
        prev_end_row = segs[-1]["end"] if segs else 0          # 1-based end == first free row
        # back-extend over the approach until a breaker run of >= break_run steps
        back = first
        run_lab, run_len = None, 0
        while back - 1 >= prev_end_row:
            l = lab[back - 1]
            if l in ("close_on", "pinch_no_lift", "open_contact") and obj[back - 1] != o:
                break
            if l in ("approach", "reach", "hover") and obj[back - 1] not in (None, o):
                break                                # heading for a different object: not this pick's approach
            if l in BREAKERS:
                run_len = run_len + 1 if l == run_lab else 1
                run_lab = l
                if run_len >= break_run:
                    back += run_len - 1          # leave the breaker run outside the pick
                    break
            else:
                run_lab, run_len = None, 0
            back -= 1
        start = min(max(back, prev_end_row), first)
        gap = start - prev_end_row
        # the stretch before the approach is its own segment only when it is long and is
        # mostly not approaching (retreats, idling, pressing the table): a slow reach belongs
        # to the pick (the references' convention), indecision does not
        breaker_share = (sum(1 for k in range(prev_end_row, start) if lab[k] in BREAKERS or lab[k] == "hover") / gap) if gap else 0.0
        if gap * dt >= GAP_NCS_S and breaker_share >= NCS_BREAKER_SHARE:
            segs.append(_make_segment("no_completed_subtask", None, prev_end_row, start - 1, "fail", ch, lab, obj, targets, dests, dt))
        else:
            start = prev_end_row
        pick_end = min((r["held_at"] + min_hold - 1) if r["passed"] else r["end_contact"], T - 1)
        seg = _make_segment("pick", o, start, pick_end, "pass" if r["passed"] else "fail", ch, lab, obj, targets, dests, dt)
        if r["n_attempts"] > 1:
            seg["attributes"].append(f"regrasp x{r['n_attempts']}")
        if r["passed"] and not r["lifted"]:
            seg["attributes"].append("drag (carried, never lifted)")
        if not r["passed"] and r["held_at"] is not None:
            seg["attributes"].append(f"carried {r['held_len'] * dt:.1f}s only")
        if not r["passed"] and any(lab[k] == "pinch_no_lift" and obj[k] == o for k in range(first, r["end_contact"] + 1)):
            seg["attributes"].append("drag")
        segs.append(seg)
        if not r["passed"] or pick_end >= T - 1:
            continue                              # nothing after the pick to segment
        # carry until the object leaves the hand
        v = pick_end + 1
        while v < T and held[v] == o:
            v += 1
        if v >= T:
            segs.append(_make_segment("carry", o, pick_end + 1, T - 1, "unknown", ch, lab, obj, targets, dests, dt,
                                      note="still held at the end of the episode"))
            continue
        leave = v
        if leave >= nxt_first:
            continue                              # straight into the next attempt: no place to segment
        deliberate = (not ch.grip_cmd[leave]) or (leave + 1 < T and not ch.grip_cmd[leave + 1])
        # the object comes to rest, or the next attempt begins first (a re-grasp before rest)
        w = leave
        while w < T and w < nxt_first and not ch.at_rest[o][w]:
            w += 1
        settled = w < T and w < nxt_first
        interrupted = (not settled) and nxt_first < T and w >= nxt_first
        rest_at = max(leave, min(w if settled else w - 1 if interrupted else T - 1, T - 1))
        dest = None
        if dests:
            dest = min((d for d in dests if d in rec.obj_pos),
                       key=lambda d: np.linalg.norm(rec.obj_pos[o][rest_at, :2] - rec.obj_pos[d][rest_at, :2]), default=None)
        label = "place" if deliberate else "drop"
        in_dest = _in_destination(rec, ch, o, dest, rest_at) if dest else None
        result = "fail"
        if label == "place":
            result = "unknown" if kind == "pick" or in_dest is None or not settled else ("pass" if in_dest else "fail")
        seg = _make_segment(label, o, pick_end + 1, rest_at, result, ch, lab, obj, targets, dests, dt, dest=dest)
        if interrupted:
            seg["attributes"].append("re-engaged before coming to rest")
        elif not settled:
            seg["attributes"].append("not at rest at episode end")
        if leave >= T - 3:
            seg["attributes"].append("released on the final steps")
        if label == "drop" and in_dest:
            seg["attributes"].append("landed_in_destination")
        segs.append(seg)

    if not segs:
        last_move = max([t for t in range(T) if ch.tcp_speed[t] > V_MOVE] or [0])
        if last_move > 0:
            segs.append(_make_segment("no_completed_subtask", None, 0, last_move, "fail", ch, lab, obj, targets, dests, dt))
    _flag_against_log(segs, ch, log_events, dt, kind)
    for seg in segs:                       # attributes added after the segment was built
        base = seg["description"].split(" [")[0]
        seg["description"] = base + (" [" + "; ".join(seg["attributes"]) + "]" if seg["attributes"] else "")
    return segs


def _make_segment(label, o, start_row, end_row, result, ch: Channels, lab, obj, targets, dests, dt, dest=None, note=None) -> dict:
    start_row = min(max(0, start_row), ch.T - 1); end_row = max(start_row, min(end_row, ch.T - 1))
    rows = range(start_row, end_row + 1)
    inside = collections.Counter(lab[k] for k in rows)
    attrs: list[str] = []
    if o is not None:
        role = ch.roles.get(o, "distractor")
        if role != "target":
            attrs.append("wrong_object" if role == "distractor" else "destination_as_target")
        if inside.get("slip"):
            attrs.append("slip")
        if any(lab[k] == "open_contact" and obj[k] == o and ch.lift[o][k] > LIFT_M for k in rows):
            attrs.append("open_hand_carry")
        if any(ch.towed[o][k] for k in rows):
            attrs.append("towed")
        others = {obj[k] for k in rows if lab[k] == "disturb" and obj[k] not in (None, o)}
        for x in sorted(others):
            attrs.append(f"disturbed {x}")
        multi = any(sum(1 for q in ch.objects if ch.pinch[q][k]) >= 2 for k in rows)
        if multi:
            attrs.append("multi_object")
        dis = sum(1 for k in rows if (ch.held_tracker[k] == o) != (ch.held_geom[k] == o))
        n_hand = sum(1 for k in rows if ch.held_tracker[k] == o or ch.held_geom[k] == o)
    else:
        dis, n_hand = 0, 0
    if inside.get("press_table"):
        attrs.append(f"press_table {inside['press_table'] * dt:.1f}s")
    if inside.get("close_empty"):
        attrs.append(f"close_empty x{_runs(lab, obj, rows, 'close_empty')}")
    flags: list[str] = []
    if n_hand and dis / n_hand > 0.2:
        flags.append(f"held_disagree {dis / n_hand:.0%}")
    desc = _describe(label, o, result, attrs, inside, dt, dest, note, ch, start_row, end_row)
    return {"start": start_row + 1, "end": end_row + 1, "label": label, "object": o, "destination": dest,
            "result": result, "attributes": attrs, "description": desc, "flags": flags,
            "phases": {k: round(v * dt, 2) for k, v in inside.most_common()}}


def _runs(lab, obj, rows, name) -> int:
    n = 0
    prev = None
    for k in rows:
        cur = lab[k] == name
        if cur and not prev:
            n += 1
        prev = cur
    return n


def _describe(label, o, result, attrs, inside, dt, dest, note, ch, a, b) -> str:
    dur = (b - a + 1) * dt
    if label == "pick":
        head = f"pick {o}: " + ("lifted" if result == "pass" else "no lift")
        bits = []
        if inside.get("close_on"):
            bits.append(f"{inside['close_on'] * dt:.1f}s closing on it")
        if inside.get("approach") or inside.get("reach"):
            bits.append(f"{(inside.get('approach', 0) + inside.get('reach', 0)) * dt:.1f}s approaching")
        if inside.get("hover"):
            bits.append(f"{inside['hover'] * dt:.1f}s hovering")
        text = head + (", " + ", ".join(bits) if bits else "") + f", {dur:.1f}s"
    elif label in ("place", "drop"):
        where = f" near {dest}" if dest else ""
        text = f"{label} {o}{where}: {result}, {dur:.1f}s until at rest"
    elif label == "carry":
        text = f"carry {o}: {note or ''}".strip()
    else:
        top = ", ".join(f"{k} {v * dt:.1f}s" for k, v in inside.most_common(3))
        text = f"no completed subtask for {dur:.1f}s ({top})"
    if attrs:
        text += " [" + "; ".join(attrs) + "]"
    return text


def _flag_against_log(segs: list[dict], ch: Channels, log_events: list[dict], dt: float, kind: str) -> None:
    """Every L2 segment that disagrees with the run's own event log gets a flag naming it."""
    def obj_of(e):
        m = re.search(r"'([^']+)'", e.get("info", ""))
        return m.group(1) if m else None
    carries = [(int(e["step"]), obj_of(e)) for e in log_events if e.get("name") == "OBJECT_CARRIED"]
    fails = [(int(e["step"]), obj_of(e)) for e in log_events if e.get("name") == "GRASP_ATTEMPT_FAILED"]
    def credits(e, o):
        info = e.get("info", "")
        if e.get("name") == "SUBTASK_COMPLETED":
            return True
        if not e.get("name", "").endswith("_SUCCESS") or "grabbed" in info.lower():
            return False
        return (f"object={o}" in info) or (f"'{o}'" in info) or (f"for {o}" in info)
    completed_by = {s["object"]: [int(e["step"]) for e in log_events if credits(e, s["object"])] for s in segs if s["object"]}
    tol = int(round(1.0 / dt))
    for s in segs:
        a, b, o = s["start"], s["end"], s["object"]
        if s["label"] == "pick":
            has_carry = any(o == co and a - tol <= st <= b + tol for st, co in carries)
            has_fail = any(o == fo and a - tol <= st <= b + tol for st, fo in fails)
            if s["result"] == "pass" and not has_carry:
                s["flags"].append("no OBJECT_CARRIED in log")
            if s["result"] == "fail" and not has_fail and not has_carry:
                s["flags"].append("no GRASP_ATTEMPT_FAILED in log")
            if s["result"] == "fail" and has_carry:
                s["flags"].append("log has OBJECT_CARRIED; carry shorter than PICK_MIN_HOLD_S")
        if s["label"] == "place" and s["result"] in ("pass", "fail") and kind == "place":
            done = any(a - tol <= st <= b + 3 * tol for st in completed_by.get(o, []))
            if s["result"] == "pass" and not done:
                s["flags"].append("place pass, no SUBTASK_COMPLETED in log")
            if s["result"] == "fail" and done:
                s["flags"].append("place fail, log credits SUBTASK_COMPLETED")
    if log_events and not ch.replay_ok:
        for s in segs:
            s["flags"].append("in-hand from geometry only (no torch)")


# --------------------------------------------------------------------------- driver
@dataclass
class Annotation:
    doc: dict
    channels: Channels = field(repr=False)


def annotate(task_dir: str, env_id: int, run_index: int = 0, log_events: list[dict] | None = None,
             use_replay: bool = True) -> Annotation:
    h5py = _real_h5py()

    h5 = find_hdf5(task_dir, run_index)
    if h5 is None:
        raise FileNotFoundError(f"no run_*.hdf5 in {task_dir}")
    dt = resolve_dt(task_dir)
    targets, dests = task_roles(task_dir)
    kind = task_kind(task_dir)
    with h5py.File(h5, "r") as f:  # noqa: F821
        g = f["data"]
        key = next((k for k in (f"demo_{env_id}", f"demo_{run_index}_{env_id}", f"episode_{env_id}") if k in g), None)
        if key is None:
            raise KeyError(f"no demo for env {env_id} in {h5}")
        rec = load_recording(g[key])
    rec.dt = dt
    if log_events is None:
        lp = os.path.join(task_dir, f"log_{run_index}_env{env_id}.json")
        if os.path.exists(lp):
            lg = json.load(open(lp))
            log_events = lg.get("events", lg if isinstance(lg, list) else [])
        else:
            log_events = []
    rr = None
    warnings: list[str] = []
    contact_source = "pads" if rec.pads else "proxy"
    if contact_source == "proxy":
        rec.pads = proxy_pads(rec, dt)
        warnings.append(f"no contact/ group: touch is TCP within {CONTACT_PROXY_M * 100:.0f} cm of the object box with "
                        "evidence of interaction; the tracker replay runs on that proxy")
    if use_replay:
        try:
            from robolab.eval.tracker_replay import replay
            rr = replay(rec, dt, containers=dests)
        except Exception as exc:  # torch missing, or an unexpected recording layout
            warnings.append(f"tracker replay unavailable ({type(exc).__name__}: {exc}); in-hand from geometry")
    ch = compute_channels(rec, dt, targets, dests, rr, contact_source)
    labels, objs = label_steps(ch)
    labels, objs = smooth_labels(labels, objs)
    phases, w2 = segment(labels, objs)
    warnings += w2
    segs = attempts(rec, ch, phases, targets, dests, kind, log_events)
    for p in phases:
        p["role"] = ch.roles.get(p["object"]) if p["object"] else None
        p["family"] = FAMILY_OF[p["label"]]
    task_name = os.path.basename(os.path.normpath(task_dir))
    doc = {
        "schema_version": SCHEMA_VERSION,
        "dt": dt,
        "task": task_name, "env_id": env_id, "run": run_index,
        "num_steps": rec.T,
        "step_convention": "1-based, same as log_*.json (episode_length_buf = HDF5 row + 1); time_s = step * dt",
        "annotator": {"name": ANNOTATOR, "version": ANNOTATOR_VERSION, "thresholds": THRESHOLDS,
                      "contact_source": ch.contact_source, "tracker_replay": ch.replay_ok},
        "task_kind": kind,
        "roles": ch.roles,
        "phases": phases,
        "attempts": segs,
        "summary": summarize(phases, segs, ch),
        "warnings": warnings,
    }
    return Annotation(doc, ch)


def summarize(phases: list[dict], segs: list[dict], ch: Channels) -> dict:
    dt, T = ch.dt, ch.T
    fam = collections.Counter()
    for p in phases:
        fam[p["family"]] += (p["end"] - p["start"] + 1)
    picks = [s for s in segs if s["label"] == "pick"]
    places = [s for s in segs if s["label"] in ("place", "drop")]
    first_touch = next((p["start"] for p in phases if p["label"] in ("close_on", "pinch_no_lift", "open_contact")), None)
    return {
        "time_by_family_s": {k: round(v * dt, 2) for k, v in fam.most_common()},
        "n_phases": len(phases),
        "n_picks": len(picks),
        "n_picks_pass": sum(1 for s in picks if s["result"] == "pass"),
        "n_picks_wrong_object": sum(1 for s in picks if any(a.startswith("wrong_object") for a in s["attributes"])),
        "n_drops": sum(1 for s in places if s["label"] == "drop"),
        "n_places_pass": sum(1 for s in places if s["label"] == "place" and s["result"] == "pass"),
        "first_contact_s": round(first_touch * dt, 2) if first_touch else None,
        "time_in_hand_s": round(sum(1 for h in ch.held_tracker if h) * dt, 2),
        "time_press_table_s": round(int(ch.table_touch.sum()) * dt, 2),
        "n_flags": sum(len(s["flags"]) for s in segs),
    }


def check_events(doc: dict, log_events: list[dict], ch: Channels | None = None) -> dict:
    """Plan H2: does each log event land in a phase that names it? A hit counts if the phase at
    the event's step, or the neighbouring phase on either side, is in ``EVENT_EXPECTS``. Two
    events describe a fact the phase priority can hide (a table hit while touching an object,
    a closed hand at the instant a drop registers); for those the recorded channel at the step
    decides when ``ch`` is given, and the row says ``by_channel``."""
    phases = doc["phases"]
    T = doc["num_steps"]
    def channel_ok(name, step):
        if ch is None:
            return False
        rows = [k for k in (step - 2, step - 1, step) if 0 <= k < T]
        if name == "GRIPPER_HIT_TABLE":
            return any(ch.table_touch[k] for k in rows)
        if name == "GRIPPER_FULLY_CLOSED":
            return any(ch.closure[k] > 0.9 and not any(ch.pinch[o][k] for o in ch.objects) for k in rows)
        return False
    def phase_index(step):
        for i, p in enumerate(phases):
            if p["start"] <= step <= p["end"]:
                return i
        return None
    rows = []
    for e in log_events:
        name = e.get("name", "")
        if name not in EVENT_EXPECTS:
            continue
        i = phase_index(int(e["step"]))
        if i is None:
            rows.append((name, int(e["step"]), None, False)); continue
        cands = {phases[j]["label"] for j in (i - 1, i, i + 1) if 0 <= j < len(phases)}
        ok = bool(cands & EVENT_EXPECTS[name])
        if not ok and channel_ok(name, int(e["step"])):
            ok = True
            rows.append((name, int(e["step"]), phases[i]["label"] + " (by_channel)", ok)); continue
        rows.append((name, int(e["step"]), phases[i]["label"], ok))
    n = len(rows)
    return {"checked": n, "consistent": sum(1 for r in rows if r[3]), "rows": rows}


def write(doc: dict, task_dir: str) -> str:
    path = os.path.join(task_dir, f"phases_{doc['run']}_env{doc['env_id']}.json")
    with open(path, "w") as fh:
        json.dump(doc, fh, indent=1, default=_json_default)
    return path


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if isinstance(o, tuple):
        return list(o)
    raise TypeError(str(type(o)))
