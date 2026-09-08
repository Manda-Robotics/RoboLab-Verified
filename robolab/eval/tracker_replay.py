# SPDX-License-Identifier: Apache-2.0
"""Replay the runtime :class:`GraspTracker` over a recorded episode, with no simulator.

The tracker reads the world through five module-level accessors in
:mod:`robolab.core.task.grasp` (``hand_contact``, ``hand_position``, ``object_position``,
``jaw_offset``, ``hand_closed``) plus ``gripper_open_commanded`` (the last action column).
``offline_tests/test_grasp_tracker.py`` scripts those from a test; this module binds them to
the arrays in ``run_<i>.hdf5`` instead, so grip / carry / release / drop / tow / attempt are
decided per step by the same code and the same constants (``robolab/constants.py``) that
produced the run's ``log_*.json``. Nothing here is a re-implementation of a rule.

What the recording gives the tracker, and where it differs from the live env:

* ``hand_contact``: ``contact/<obj>`` > 0.1 N on either pad (P77 forces; on P62-era
  recordings the columns are booleans and any nonzero counts). Same threshold as
  ``WorldState.in_contact``.
* ``object_position``: ``states/rigid_object/<obj>/root_pose[:, :3]`` (env-local, as
  ``get_pose`` returns it).
* ``jaw_offset``: exact. ``GRASP_JAW_BODY`` is ``base_link`` and ``ee_pose`` records that
  body's pose.
* ``hand_closed``: ``joint_position[:, 7] / (pi/4)`` >= threshold, the closure fraction
  ``gripper_fully_closed`` computes from ``gripper_closure_cfg``.
* ``hand_position``: **the one deviation.** Live, this is the left inner finger body; the
  recording has only ``base_link``. The finger moves a few centimetres relative to
  ``base_link`` while the jaws close, so ``rel_dev`` (object-to-hand drift over
  ``GRASP_HOLD_S``) excludes finger travel here and the replayed carry can begin up to a
  step or two earlier than the live one. Measured by ``scripts/annotate_phases.py
  --check-events`` (plan H5); fix by recording the finger body (plan R1).
* Step index: the recorder writes row ``t`` with the pre-step action and the post-step
  state; the tracker ran at post-step with ``episode_length_buf == t + 1``. The log's
  ``step`` is that value, so replayed onsets are compared to the log as ``row + 1``.

Only the tracker's own events are reproduced (``OBJECT_GRIPPED``, ``OBJECT_CARRIED``,
``GRASP_ATTEMPT_FAILED``, ``OBJECT_RELEASED``, ``OBJECT_DROPPED``, ``TOWED_WITHOUT_GRASP``),
with the same container exclusions the event tracker applies (P72/P78). Ladder lines,
bumps, off-table and collateral events are not part of the replay.
"""
from __future__ import annotations

import contextlib
import math
from dataclasses import dataclass, field

import numpy as np

# Robotiq 2F-85 finger_joint: column 7 of the 13-joint articulation (droid.py:83-96)
FINGER_JOINT_COL = 7
FINGER_JOINT_CLOSED = math.pi / 4
CONTACT_FORCE_N = 0.1          # WorldState.in_contact default
HAND = "gripper"

KIND_TO_CODE = {
    "gripped": (284, "OBJECT_GRIPPED"),
    "grabbed": (283, "OBJECT_CARRIED"),
    "attempt_failed": (266, "GRASP_ATTEMPT_FAILED"),
    "released": (267, "OBJECT_RELEASED"),
    "dropped": (268, "OBJECT_DROPPED"),
    "towed": (275, "TOWED_WITHOUT_GRASP"),
}


@dataclass
class Recording:
    """The arrays one episode's replay and phase derivation read. All (T, ...) numpy."""
    dt: float
    actions: np.ndarray                    # (T, A); last column = gripper command, 1 = close
    ee_pos: np.ndarray                     # (T, 3) base_link, env-local
    ee_quat: np.ndarray                    # (T, 4) wxyz
    joint_pos: np.ndarray                  # (T, J)
    obj_pos: dict[str, np.ndarray]         # (T, 3) per rigid object (table included if recorded)
    obj_quat: dict[str, np.ndarray]        # (T, 4)
    obj_vel: dict[str, np.ndarray]         # (T, 3) linear velocity as PhysX reports it (jitters at rest)
    pads: dict[str, np.ndarray]            # (T, 2) pad force (or boolean) per object, table included
    pair_contact: dict[str, np.ndarray]    # "<a>__<b>" -> (T,) uint8
    bbox_corners: dict[str, np.ndarray] = field(default_factory=dict)   # (T, 8, 3) metres
    objects: list[str] = field(default_factory=list)   # manipulable objects: rigid objects minus table
    # R1/R2/R4 channels (docs/verified/dense_annotations.md §5), present on recordings made
    # with the dense recorder terms; None / empty on older ones.
    finger_left: np.ndarray | None = None      # (T, 3) left inner finger body, env-local (what the live tracker reads)
    finger_right: np.ndarray | None = None     # (T, 3)
    tracker_state: dict[str, np.ndarray] = field(default_factory=dict)   # obj -> (T, 2) uint8 [grasped, attempt_closed], live
    conditions: dict[str, np.ndarray] = field(default_factory=dict)      # legend key -> (T,) uint8, the ladder each step
    body_forces: dict[str, np.ndarray] = field(default_factory=dict)     # "<label>__<obj>" -> (T,) float32 N
    joint_names: list[str] | None = None       # from env_cfg.json (robot_joint_names) when stamped

    @property
    def T(self) -> int:
        return int(self.actions.shape[0])

    def closure(self) -> np.ndarray:
        col = FINGER_JOINT_COL
        if self.joint_names and "finger_joint" in self.joint_names:
            col = self.joint_names.index("finger_joint")
        return self.joint_pos[:, col] / FINGER_JOINT_CLOSED

    @property
    def hand_pos(self) -> np.ndarray:
        """(T, 3) what ``GraspTracker.hand_position`` reads live: the left inner finger when
        recorded (R1), else ``base_link`` (the pre-R1 deviation documented above)."""
        return self.finger_left if self.finger_left is not None else self.ee_pos

    @property
    def finger_mid(self) -> np.ndarray | None:
        """(T, 3) midpoint of the two inner finger *link origins*, or None on pre-R1 recordings.
        Not a TCP: on the DROID asset both link origins coincide with ``base_link`` when the
        jaws are open and part by ~4.6 cm as they close (d1, 2026-09-02), so the pads are a
        constant offset inside each link that the recording does not carry. The channels
        exist for the tracker replay, which reads the left one exactly as the live tracker
        does; the TCP stays ``base_link`` + ``TCP_OFFSET``."""
        if self.finger_left is None or self.finger_right is None:
            return None
        return 0.5 * (self.finger_left + self.finger_right)

    def contact(self, obj: str) -> np.ndarray:
        """(T,) bool: either pad on ``obj`` above the runtime contact threshold."""
        p = self.pads.get(obj)
        if p is None:
            return np.zeros(self.T, dtype=bool)
        thr = CONTACT_FORCE_N if np.issubdtype(p.dtype, np.floating) else 0
        return (p > thr).any(axis=1)

    def pinch(self, obj: str) -> np.ndarray:
        """(T,) bool: both pads on ``obj``."""
        p = self.pads.get(obj)
        if p is None:
            return np.zeros(self.T, dtype=bool)
        thr = CONTACT_FORCE_N if np.issubdtype(p.dtype, np.floating) else 0
        return (p > thr).all(axis=1)


def load_recording(h5_demo) -> Recording:
    """Build a :class:`Recording` from an open ``h5py`` group ``data/demo_<env>``."""
    d = h5_demo
    obj_pos, obj_quat, obj_vel, pads, pair, bbox = {}, {}, {}, {}, {}, {}
    for name in d["states/rigid_object"].keys():
        rp = d[f"states/rigid_object/{name}/root_pose"][:]
        obj_pos[name] = rp[:, :3].astype(np.float64)
        obj_quat[name] = rp[:, 3:7].astype(np.float64)
        vk = f"states/rigid_object/{name}/root_velocity"
        obj_vel[name] = d[vk][:, :3].astype(np.float64) if vk in d else np.zeros_like(obj_pos[name])
    if "contact" in d:
        for key in d["contact"].keys():
            arr = d["contact"][key][:]
            (pair if "__" in key else pads)[key] = arr
    if "bbox" in d and "bbox_mm" in d["bbox"]:
        for name in d["bbox/bbox_mm"].keys():
            bbox[name] = d["bbox/bbox_mm"][name][:].astype(np.float64) / 1000.0
    objects = [o for o in obj_pos if o != "table"]
    fl = d["finger_left/position"][:].astype(np.float64) if "finger_left" in d else None
    fr = d["finger_right/position"][:].astype(np.float64) if "finger_right" in d else None
    tracker_state = {k: d["tracker"][k][:] for k in d["tracker"].keys()} if "tracker" in d else {}
    conditions = {k: d["conditions"][k][:] for k in d["conditions"].keys()} if "conditions" in d else {}
    body_forces = {k: d["contact_body"][k][:].astype(np.float64) for k in d["contact_body"].keys()} if "contact_body" in d else {}
    return Recording(
        dt=0.0,
        actions=d["actions"][:].astype(np.float64),
        ee_pos=d["ee_pose/position"][:].astype(np.float64),
        ee_quat=d["ee_pose/orientation"][:].astype(np.float64),
        joint_pos=d["states/articulation/robot/joint_position"][:].astype(np.float64),
        obj_pos=obj_pos, obj_quat=obj_quat, obj_vel=obj_vel, pads=pads, pair_contact=pair,
        bbox_corners=bbox, objects=objects,
        finger_left=fl, finger_right=fr, tracker_state=tracker_state, conditions=conditions, body_forces=body_forces,
    )


def quat_rotmat(q: np.ndarray) -> np.ndarray:
    """(..., 4) wxyz -> (..., 3, 3)."""
    w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    R = np.empty(q.shape[:-1] + (3, 3))
    R[..., 0, 0] = 1 - 2 * (y * y + z * z); R[..., 0, 1] = 2 * (x * y - z * w); R[..., 0, 2] = 2 * (x * z + y * w)
    R[..., 1, 0] = 2 * (x * y + z * w); R[..., 1, 1] = 1 - 2 * (x * x + z * z); R[..., 1, 2] = 2 * (y * z - x * w)
    R[..., 2, 0] = 2 * (x * z - y * w); R[..., 2, 1] = 2 * (y * z + x * w); R[..., 2, 2] = 1 - 2 * (x * x + y * y)
    return R


@dataclass
class ReplayResult:
    grasped: dict[str, np.ndarray]          # obj -> (T,) bool: tracker says "carried" at this row
    attempt_closed: dict[str, np.ndarray]   # obj -> (T,) bool: jaws closed on it during this contact
    towed: dict[str, np.ndarray]            # obj -> (T,) bool
    contact_streak: dict[str, np.ndarray]   # obj -> (T,) int
    events: list[dict]                      # log-schema dicts: step (onset, 1-based), detected_step, code, name, info, object


class _ActionManager:
    def __init__(self, n_cols: int):
        import torch
        self.action = torch.zeros((1, n_cols))


class _FakeEnv:
    """Only what GraspTracker touches: num_envs, device, step_dt, episode_length_buf,
    common_step_counter, action_manager."""

    def __init__(self, dt: float, n_cols: int):
        import torch
        self.num_envs = 1
        self.device = "cpu"
        self.step_dt = float(dt)
        self.episode_length_buf = torch.zeros(1, dtype=torch.long)
        self.common_step_counter = 0
        self.action_manager = _ActionManager(n_cols)


@contextlib.contextmanager
def _bound_accessors(rec: Recording, cursor: dict):
    """Point grasp.py's accessors at the recording for the duration of a replay."""
    import torch

    import robolab.core.task.grasp as G

    contact = {o: rec.contact(o) for o in rec.pads}
    closure = rec.closure()
    R = quat_rotmat(rec.ee_quat)            # (T, 3, 3)
    ay = R[:, :, 1]                          # jaw axis = +y of base_link, world coords

    def hand_contact(env, obj, hand):
        c = contact.get(obj)
        return torch.tensor([bool(c[cursor["t"]])]) if c is not None else torch.tensor([False])

    hand_pos = rec.hand_pos                  # left inner finger on R1 recordings, base_link before

    def hand_position(env, hand):
        return torch.tensor(hand_pos[cursor["t"]], dtype=torch.float32).unsqueeze(0)

    def object_position(env, obj):
        return torch.tensor(rec.obj_pos[obj][cursor["t"]], dtype=torch.float32).unsqueeze(0)

    def jaw_offset(env, obj):
        t = cursor["t"]
        d = rec.obj_pos[obj][t] - rec.ee_pos[t]
        return torch.tensor([abs(float(np.dot(d, ay[t])))], dtype=torch.float32)

    def hand_closed(env, hand, thr):
        return torch.tensor([bool(closure[cursor["t"]] >= thr)])

    saved = {k: getattr(G, k) for k in ("hand_contact", "hand_position", "object_position", "jaw_offset", "hand_closed")}
    G.hand_contact, G.hand_position, G.object_position, G.jaw_offset, G.hand_closed = (
        hand_contact, hand_position, object_position, jaw_offset, hand_closed)
    try:
        yield G
    finally:
        for k, v in saved.items():
            setattr(G, k, v)


def replay(rec: Recording, dt: float, objects: list[str] | None = None,
           containers: set[str] | None = None) -> ReplayResult:
    """Run the tracker over every row of ``rec`` for ``objects`` (default: every recorded
    object except the table) and return its per-step state plus its events in the log's
    schema. ``containers`` reproduces the event tracker's P72/P78 exclusions (no grip or
    attempt line on a destination)."""
    import torch

    objects = list(objects if objects is not None else rec.objects)
    containers = set(containers or ())
    T = rec.T
    cursor = {"t": 0}
    out = ReplayResult({o: np.zeros(T, bool) for o in objects}, {o: np.zeros(T, bool) for o in objects},
                       {o: np.zeros(T, bool) for o in objects}, {o: np.zeros(T, int) for o in objects}, [])
    with _bound_accessors(rec, cursor) as G:
        env = _FakeEnv(dt, rec.actions.shape[1])
        tracker = G.GraspTracker(env)
        raw: list[tuple[int, str, str, dict]] = []      # (detected_row, obj, kind, extra)
        for t in range(T):
            cursor["t"] = t
            env.episode_length_buf[:] = t + 1
            env.common_step_counter = t + 1
            env.action_manager.action = torch.tensor(rec.actions[t], dtype=torch.float32).unsqueeze(0)
            for o in objects:
                st = tracker.update(o, HAND)
                out.grasped[o][t] = bool(st.grasped[0])
                out.attempt_closed[o][t] = bool(st.attempt_closed[0])
                out.towed[o][t] = bool(st.towed_now[0])
                out.contact_streak[o][t] = int(st.contact_streak[0])
            for eid, o, hand, kind, extra in tracker.pop_events():
                raw.append((t, o, kind, extra))
        tracker.flush_attempts()
        for eid, o, hand, kind, extra in tracker.pop_events():
            raw.append((T - 1, o, kind, extra))

    for t, o, kind, extra in raw:
        code, name = KIND_TO_CODE[kind]
        detected = t + 1
        step = detected
        if kind == "gripped":
            if o in containers:
                continue
            step = int(extra.get("onset_step", detected))
            info = f"'{o}' gripped (jaws closed on it)"
        elif kind == "grabbed":
            step = int(extra.get("onset_step", detected))
            info = f"'{o}' carried (grasp established)"
        elif kind == "attempt_failed":
            if o in containers:
                continue
            n = int(extra.get("count", 1))
            step = int(extra.get("first_step", 0)) or detected
            span = (int(extra.get("last_step", 0)) - int(extra.get("first_step", 0))) * dt
            info = (f"Grasp attempt on '{o}' failed (contact lost before a carry was established)" if n <= 1 else
                    f"Grasp attempts on '{o}' failed ×{n} over {span:.1f}s (contact never became a carry)")
        elif kind == "released":
            info = f"'{o}' released (hand opened)"
        elif kind == "dropped":
            info = f"'{o}' dropped (left the closed hand)"
        else:
            info = f"'{o}' towed without a grasp (moving with an open hand; physics artifact)"
        out.events.append({"step": step, "detected_step": detected, "code": code, "name": name,
                           "info": info, "object": o})
    out.events.sort(key=lambda e: (e["step"], e["detected_step"]))
    return out


def compare_state(rr: ReplayResult, rec: Recording) -> dict | None:
    """R4 check: the replayed per-step tracker state against the state the live tracker
    recorded (``tracker/<obj>``). Returns None on recordings without the channel. Row t of
    the recording is the live tracker's state after its update on step t + 1, which is what
    the replay computes at row t, so the two should agree row for row."""
    if not rec.tracker_state:
        return None
    out = {"objects": {}, "steps": 0, "grasped_agree": 0, "attempt_agree": 0,
           "grasped_onset_diffs": [], "attempt_onset_diffs": []}
    for o, st in rec.tracker_state.items():
        if o not in rr.grasped:
            continue
        g_rec, a_rec = st[:, 0].astype(bool), st[:, 1].astype(bool)
        g_rep, a_rep = rr.grasped[o], rr.attempt_closed[o]
        T = min(len(g_rec), len(g_rep))
        ga = int((g_rec[:T] == g_rep[:T]).sum()); aa = int((a_rec[:T] == a_rep[:T]).sum())
        out["objects"][o] = {"steps": T, "grasped_agree": ga, "attempt_agree": aa,
                             "grasped_steps_rec": int(g_rec[:T].sum()), "grasped_steps_rep": int(g_rep[:T].sum())}
        out["steps"] += T; out["grasped_agree"] += ga; out["attempt_agree"] += aa
        for a, b, key in ((g_rec, g_rep, "grasped_onset_diffs"), (a_rec, a_rep, "attempt_onset_diffs")):
            ia = np.flatnonzero(a[:T]); ib = np.flatnonzero(b[:T])
            if len(ia) and len(ib):
                out[key].append(int(ib[0]) - int(ia[0]))
    return out


def compare_events(replayed: list[dict], logged: list[dict], tol_steps: int = 1, by: str = "onset") -> dict:
    """Match replayed tracker lines to the run's log by (name, object) and a step within
    ``tol_steps``: the onset (``step``) when ``by="onset"``, the detection step when
    ``by="detected"``. Logs written before P84 back-date a carry to the start of the contact
    while this build clamps it to the grip, so onsets differ there and detection is the fairer
    comparison across builds.

    Returns counts and the unmatched lines on each side. Only the six tracker codes are
    compared; ladder and other event-tracker lines in the log are ignored."""
    import re

    codes = {c for c, _ in KIND_TO_CODE.values()}
    def obj_of(e):
        m = re.search(r"'([^']+)'", e.get("info", ""))
        return e.get("object") or (m.group(1) if m else None)
    log = [e for e in logged if e.get("code") in codes]
    used = [False] * len(log)
    matched, miss_replay = 0, []
    for r in replayed:
        hit = None
        for i, l in enumerate(log):
            if used[i] or l["code"] != r["code"] or obj_of(l) != r["object"]:
                continue
            if by == "detected":
                close = abs(int(l.get("detected_step", l["step"])) - int(r["detected_step"])) <= tol_steps
            else:
                close = abs(int(l["step"]) - int(r["step"])) <= tol_steps
            if close:
                hit = i
                break
        if hit is None:
            miss_replay.append(r)
        else:
            used[hit] = True
            matched += 1
    miss_log = [l for i, l in enumerate(log) if not used[i]]
    return {"logged": len(log), "replayed": len(replayed), "matched": matched,
            "only_in_replay": miss_replay, "only_in_log": miss_log}
