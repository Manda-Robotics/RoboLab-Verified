# SPDX-License-Identifier: Apache-2.0
"""Stable-grasp tracking: a grasp is a *carry*, not a touch.

Upstream ``object_grabbed`` was "object in contact with the gripper". On the
Verified corpus 64 % of recorded grabs had an open hand and only ~13 % were
followed by the object moving with the hand — the rest were touches and failed
attempts, each producing a grab-success / drop / grab-failure triplet
(VERIFIED_PLAN B1, B2, B6, F§1, H-R5-4, H-R6-3/4, H-R8-12/17/18).

Definition here (Finn, 2026-08-25): the object counts as grasped once it has been
in contact with the hand for ``GRASP_HOLD_S`` (0.2 s) while its offset to the
hand changed by less than ``GRASP_COUPLING_M`` (0.5 cm) and the hand itself
moved at least ``GRASP_HAND_MOVE_M`` (1 cm) — i.e. the hand is carrying it.
A carry with the hand essentially open (< ``GRASP_TOW_CLOSURE``) **and** the object
off-centre along the jaw axis (>= ``GRASP_TOW_OFFSET_M``) is ``TOWED_WITHOUT_GRASP``
— the object stuck to one finger and dragged along, a physics artifact that cannot
happen in the real world (Finn: an episode with it is bogus). Calibrated on the
corpus against Finn's verdicts: known tows sit 4-11 cm off-centre at closure 0.00;
a can as wide as the aperture reads closure 0.00 but sits centred (a grip); an
orange reads 0.23 (a grip). Any other carry is a grasp. Contact that ends before a
carry, with the hand at least partly closed, is one ``GRASP_ATTEMPT_FAILED``. After a grasp, losing contact is ``OBJECT_RELEASED``
when the hand is opening and ``OBJECT_DROPPED`` when it is still closed.

State lives in one :class:`GraspTracker` per env instance, updated at most once
per sim step per (object, hand); ``object_grabbed`` and the event tracker both
read it, so the ladder, the drop/release events and the wrong-object check all
share one notion of "grasped".
"""
from __future__ import annotations

import weakref
from collections import deque
from typing import Any

import torch

import robolab.constants

_TRACKERS: "weakref.WeakKeyDictionary[Any, GraspTracker]" = weakref.WeakKeyDictionary()


def get_grasp_tracker(env) -> "GraspTracker":
    try:
        t = _TRACKERS.get(env)
    except TypeError:                       # env not weak-referenceable (tests)
        t = getattr(env, "_grasp_tracker", None)
    if t is None:
        t = GraspTracker(env)
        try:
            _TRACKERS[env] = t
        except TypeError:
            env._grasp_tracker = t
    return t


# --- geometry access (module-level so tests can patch them) ---------------
def object_position(env, obj: str) -> torch.Tensor:
    """(N, 3) object position relative to each env origin."""
    from robolab.core.world.world_state import get_world
    pos, _ = get_world(env).get_pose(obj, env_id=None)
    return pos


def hand_position(env, hand_label: str) -> torch.Tensor:
    """(N, 3) position of the hand's contact body (leaf of its contact_gripper prim path)."""
    from robolab.core.world.world_state import get_world
    world = get_world(env)
    labels = world.resolve_contact_bodies(hand_label)
    label = labels[0] if labels else hand_label
    prim = (getattr(env.cfg, "contact_gripper", None) or {}).get(label, label)
    leaf = str(prim).rstrip("/").split("/")[-1]
    robot = env.scene["robot"]
    idx = robot.body_names.index(leaf)
    return robot.data.body_pos_w[:, idx] - env.scene.env_origins


def jaw_offset(env, obj: str) -> torch.Tensor:
    """(N,) |offset| of the object along the jaw axis (+y of GRASP_JAW_BODY) in the
    hand frame. Falls back to zeros (→ never a tow by geometry) when the body is
    unknown, so the closure rule alone decides."""
    try:
        from robolab.core.world.world_state import get_world
        body = getattr(robolab.constants, "GRASP_JAW_BODY", "base_link")
        robot = env.scene["robot"]
        idx = robot.body_names.index(body)
        pos = robot.data.body_pos_w[:, idx] - env.scene.env_origins
        q = robot.data.body_quat_w[:, idx]                       # (N, 4) wxyz
        obj_pos, _ = get_world(env).get_pose(obj, env_id=None)
        d = obj_pos - pos
        w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
        # y-axis of the body frame expressed in world coordinates (2nd column of R)
        ay = torch.stack([2 * (x * y - z * w), 1 - 2 * (x * x + z * z), 2 * (y * z + x * w)], dim=-1)
        return (d * ay).sum(dim=-1).abs()
    except Exception:
        return torch.zeros(env.num_envs, device=env.device)


def hand_contact(env, obj: str, hand_label: str) -> torch.Tensor:
    from robolab.core.task.predicate_logic import in_contact
    from robolab.core.world.world_state import get_world
    r = in_contact(get_world(env), obj, hand_label, env_id=None)
    return r if isinstance(r, torch.Tensor) else torch.as_tensor(r, dtype=torch.bool, device=env.device).reshape(-1)


def gripper_open_commanded(env) -> torch.Tensor:
    """(N,) bool: the policy *commanded* the gripper open on this step.

    The measured finger joint lags and can even be closing at the instant an object
    leaves the hand, so it cannot separate a deliberate release from a slip. The
    action channel can: RoboLab's binary gripper term takes 1 = close, 0 = open
    (``BinaryJointPositionZeroToOneActionCfg``), so the last action's final column is
    the policy's intent (Finn 2026-08-26: "was this a release on purpose or on
    accident? how can we reliably tell?"). Falls back to all-False when the action is
    unavailable, in which case the measured-closure rule decides as before."""
    try:
        a = env.action_manager.action
        return a[:, -1] < 0.5
    except Exception:
        return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)


def hand_closed(env, hand_label: str, threshold: float) -> torch.Tensor:
    from robolab.core.task.conditionals import gripper_fully_closed
    r = gripper_fully_closed(env, closed_threshold=threshold, env_id=None, gripper_name=hand_label)
    return r if isinstance(r, torch.Tensor) else torch.as_tensor(r, dtype=torch.bool, device=env.device).reshape(-1)


def _step_counter(env) -> int:
    c = getattr(env, "common_step_counter", None)
    if c is not None:
        return int(c)
    return int(env.episode_length_buf.max().item())


class _PairState:
    def __init__(self, n: int, k: int, device):
        self.contact_streak = torch.zeros(n, dtype=torch.long, device=device)
        self.grasped = torch.zeros(n, dtype=torch.bool, device=device)
        self.prev_contact = torch.zeros(n, dtype=torch.bool, device=device)
        self.attempt_closed = torch.zeros(n, dtype=torch.bool, device=device)   # hand was closing during the contact
        self.last_grasped_step = torch.full((n,), -10**9, dtype=torch.long, device=device)
        self.towed_flagged = torch.zeros(n, dtype=torch.bool, device=device)  # TOWED_WITHOUT_GRASP raised this episode
        self.contact_start_z = torch.zeros(n, device=device)                  # object height when the current contact began
        self.contact_start_step = torch.zeros(n, dtype=torch.long, device=device)  # episode step when it began (P61 onset stamp)
        self.recent_open_cmd = torch.zeros(n, dtype=torch.long, device=device)  # steps since the policy commanded "open"
        self.attempt_count = torch.zeros(n, dtype=torch.long, device=device)   # failed attempts in the open burst
        self.attempt_first = torch.zeros(n, dtype=torch.long, device=device)
        self.attempt_last = torch.zeros(n, dtype=torch.long, device=device)
        self.towed_now = torch.zeros(n, dtype=torch.bool, device=device)
        self.rel_hist: deque = deque(maxlen=k + 1)     # object - hand offsets, newest last
        self.hand_hist: deque = deque(maxlen=k + 1)
        self.last_step = -1


class GraspTracker:
    def __init__(self, env):
        self.env = env
        self.hold_s = float(getattr(robolab.constants, "GRASP_HOLD_S", 0.2))
        self.coupling_m = float(getattr(robolab.constants, "GRASP_COUPLING_M", 0.005))
        self.hand_move_m = float(getattr(robolab.constants, "GRASP_HAND_MOVE_M", 0.01))
        self.attempt_closure = float(getattr(robolab.constants, "GRASP_ATTEMPT_CLOSURE", 0.3))
        self.release_closure = float(getattr(robolab.constants, "GRASP_RELEASE_CLOSURE", 0.1))
        self.tow_closure = float(getattr(robolab.constants, "GRASP_TOW_CLOSURE", 0.1))
        self.tow_offset = float(getattr(robolab.constants, "GRASP_TOW_OFFSET_M", 0.03))
        self.tow_lift = float(getattr(robolab.constants, "GRASP_TOW_LIFT_M", 0.02))
        self._pairs: dict[tuple[str, str], _PairState] = {}
        self._events: list[tuple[int, str, str, str, dict]] = []   # (env_id, object, hand, kind, extra)
        self.burst_s = float(getattr(robolab.constants, "GRASP_ATTEMPT_BURST_S", 2.0))
        self.open_cmd_memory = max(1, int(round(0.3 / float(getattr(env, "step_dt", 0.0) or 1 / 15))))  # "commanded open" counts for 0.3 s

    # -- parameters --------------------------------------------------------
    def hold_steps(self) -> int:
        dt = float(getattr(self.env, "step_dt", 0.0) or 0.0)
        return max(1, int(round(self.hold_s / dt))) if dt > 0 else 1

    def _state(self, obj: str, hand: str) -> _PairState:
        key = (obj, hand)
        if key not in self._pairs:
            self._pairs[key] = _PairState(self.env.num_envs, self.hold_steps(), self.env.device)
        return self._pairs[key]

    # -- per-step update ---------------------------------------------------
    def update(self, obj: str, hand: str) -> _PairState:
        st = self._state(obj, hand)
        step = _step_counter(self.env)
        if st.last_step == step:
            return st
        st.last_step = step
        env = self.env
        n = env.num_envs
        k = self.hold_steps()

        fresh = env.episode_length_buf <= 1
        if bool(fresh.any()):
            st.contact_streak[fresh] = 0
            st.grasped[fresh] = False
            st.prev_contact[fresh] = False
            st.attempt_closed[fresh] = False
            st.towed_flagged[fresh] = False
            st.contact_start_step[fresh] = 0
            st.attempt_count[fresh] = 0
            st.towed_now[fresh] = False
            if bool(fresh.all()):
                st.rel_hist.clear(); st.hand_hist.clear()

        contact = hand_contact(env, obj, hand)
        obj_pos = object_position(env, obj)
        hand_pos = hand_position(env, hand)
        rel = obj_pos - hand_pos
        st.rel_hist.append(rel); st.hand_hist.append(hand_pos)

        closed_attempt = hand_closed(env, hand, self.attempt_closure)
        still_closed = hand_closed(env, hand, self.release_closure)
        hand_open = ~hand_closed(env, hand, self.tow_closure)      # essentially fully open

        st.contact_streak = torch.where(contact, st.contact_streak + 1, torch.zeros_like(st.contact_streak))
        st.attempt_closed = (st.attempt_closed | closed_attempt) & contact

        # stable grasp: contact for k steps, offset to the hand steady, hand moved
        if len(st.rel_hist) > k:
            rel_dev = torch.linalg.norm(st.rel_hist[-1] - st.rel_hist[0], dim=-1)
            hand_moved = torch.linalg.norm(st.hand_hist[-1] - st.hand_hist[0], dim=-1)
        else:
            rel_dev = torch.full((n,), float("inf"), device=env.device)
            hand_moved = torch.zeros(n, device=env.device)
        carry = contact & (st.contact_streak >= k) & (rel_dev < self.coupling_m) & (hand_moved >= self.hand_move_m)
        # A carry with an OPEN hand is not a grasp: it is the object stuck to one
        # finger and towed (Finn, reviews 06/08 — "looks magnetic", impossible in
        # the real world; a PhysX high-friction artifact). Flag it once per object
        # and never credit it as a grasp; a real grasp always reads closed here.
        off_centre = jaw_offset(env, obj) >= self.tow_offset
        # height of the object when this contact began — a tow lifts it clear of the
        # table, a drag keeps it at the same height
        began = contact & ~st.prev_contact
        st.contact_start_z = torch.where(began, obj_pos[:, 2], st.contact_start_z)
        st.contact_start_step = torch.where(began, env.episode_length_buf, st.contact_start_step)
        lifted = (obj_pos[:, 2] - st.contact_start_z) >= self.tow_lift
        towed = carry & hand_open & off_centre & lifted & ~st.grasped & ~st.towed_flagged
        for eid in towed.nonzero(as_tuple=False).flatten().tolist():
            self._events.append((eid, obj, hand, "towed", {}))
        st.towed_flagged |= towed
        st.towed_now = carry & hand_open & off_centre & lifted
        # a carry that is not a tow is a grasp — wide objects read only 0.2-0.35 closed,
        # and a can as wide as the open jaws reads 0.0 but sits centred
        newly = (~st.grasped) & carry & ~st.towed_now
        # P60: emit the grasp from the tracker. It used to appear only as a subtask-ladder
        # transition line, so the five stacking tasks -- whose ladder is a single placement
        # condition with no object_grabbed step -- showed releases and drops but never a
        # grab (Finn, r3 BowlStackingRightOnLeft: "why is there a drop without a pick?").
        for eid in newly.nonzero(as_tuple=False).flatten().tolist():
            self._events.append((eid, obj, hand, "grabbed",
                                 {"onset_step": int(st.contact_start_step[eid])}))

        lost = st.grasped & ~contact
        ended_attempt = (~st.grasped) & st.prev_contact & ~contact & ~fresh
        # P47: a fumble is one line with a count, not one line per contact blip.
        ep = env.episode_length_buf                      # per-env step index
        burst = max(1, int(round(self.burst_s / float(getattr(env, "step_dt", 0.0) or 1 / 15))))
        for eid in ended_attempt.nonzero(as_tuple=False).flatten().tolist():
            if not bool(self._attempt_flag(st, eid)):
                continue
            if int(st.attempt_count[eid]) == 0:
                st.attempt_first[eid] = int(ep[eid])
            st.attempt_count[eid] += 1
            st.attempt_last[eid] = int(ep[eid])
        # flush a burst that has gone quiet, or that a real grasp ended
        stale = (st.attempt_count > 0) & ((ep - st.attempt_last) > burst)
        for eid in (stale | (newly & (st.attempt_count > 0))).nonzero(as_tuple=False).flatten().tolist():
            self._events.append((eid, obj, hand, "attempt_failed",
                                 {"count": int(st.attempt_count[eid]),
                                  "first_step": int(st.attempt_first[eid]),
                                  "last_step": int(st.attempt_last[eid])}))
            st.attempt_count[eid] = 0
        # release vs drop: the commanded gripper is the policy's intent; the measured
        # joint only decides when no action is available (see gripper_open_commanded)
        commanded_open = gripper_open_commanded(env)
        deliberate = commanded_open | (st.recent_open_cmd > 0)
        st.recent_open_cmd = torch.where(commanded_open, torch.full_like(st.recent_open_cmd, self.open_cmd_memory),
                                         torch.clamp(st.recent_open_cmd - 1, min=0))
        for eid in lost.nonzero(as_tuple=False).flatten().tolist():
            released = bool(deliberate[eid]) if bool(commanded_open.any()) or bool((st.recent_open_cmd > 0).any()) else (not bool(still_closed[eid]))
            self._events.append((eid, obj, hand, "released" if released else "dropped", {}))

        st.grasped = (st.grasped | newly) & contact
        st.last_grasped_step = torch.where(st.grasped, env.episode_length_buf, st.last_grasped_step)
        st.prev_contact = contact.clone()
        # remember whether the hand was closing while the (now ended) contact lasted
        st._last_attempt_closed = st.attempt_closed.clone()
        return st

    def _attempt_flag(self, st: _PairState, eid: int) -> bool:
        # attempt_closed was ANDed with contact this step (now False); use the value from before
        prev = getattr(st, "_last_attempt_closed", None)
        return bool(prev[eid]) if prev is not None else False

    # -- queries -----------------------------------------------------------
    def grasped(self, obj: str, hand: str, env_id: int | None = None):
        st = self.update(obj, hand)
        return bool(st.grasped[env_id]) if env_id is not None else st.grasped.clone()

    def recently_held(self, obj: str, hand: str, within_s: float = 2.0) -> torch.Tensor:
        """(N,) bool: the object was grasped within the last ``within_s`` seconds (or is now)."""
        st = self._pairs.get((obj, hand))
        if st is None:
            return torch.zeros(self.env.num_envs, dtype=torch.bool, device=self.env.device)
        dt = float(getattr(self.env, "step_dt", 0.0) or 1 / 15)
        k = max(1, int(round(within_s / dt)))
        return (self.env.episode_length_buf - st.last_grasped_step) <= k

    def flush_attempts(self) -> None:
        """Emit any open attempt burst — called at episode end so a fumble that runs
        into the buzzer is still reported."""
        for (obj, hand), st in self._pairs.items():
            for eid in (st.attempt_count > 0).nonzero(as_tuple=False).flatten().tolist():
                self._events.append((eid, obj, hand, "attempt_failed",
                                     {"count": int(st.attempt_count[eid]),
                                      "first_step": int(st.attempt_first[eid]),
                                      "last_step": int(st.attempt_last[eid])}))
                st.attempt_count[eid] = 0

    def towed_objects(self) -> dict[int, set[str]]:
        """env_id -> objects that were towed without a grasp this episode."""
        out: dict[int, set[str]] = {}
        for (obj, hand), st in self._pairs.items():
            for eid in st.towed_flagged.nonzero(as_tuple=False).flatten().tolist():
                out.setdefault(eid, set()).add(obj)
        return out

    def pop_events(self) -> list[tuple[int, str, str, str]]:
        ev, self._events = self._events, []
        return ev


PAD_LABELS = ("gripper_left", "gripper_right")


def pad_contact_columns(env, world, object_names) -> dict:
    """P62: per-object (num_envs, 2) uint8 contact, columns [left pad, right pad].

    Lives here rather than in the recorder so it can be tested without isaaclab.
    A lookup that fails degrades to "no contact" for that pad -- recording must
    never take a run down.
    """
    from robolab.core.task.predicate_logic import in_contact  # noqa: PLC0415

    n = env.num_envs
    out = {}
    for name in object_names:
        cols = []
        for pad in PAD_LABELS:
            try:
                r = in_contact(world, name, pad, env_id=None)
                t = r if isinstance(r, torch.Tensor) else torch.as_tensor(
                    r, dtype=torch.bool, device=env.device
                ).reshape(-1)
                if t.numel() != n:
                    t = t.reshape(-1)[:1].expand(n)
            except Exception:
                t = torch.zeros(n, dtype=torch.bool, device=env.device)
            cols.append(t.to(torch.uint8))
        out[name] = torch.stack(cols, dim=-1)
    return out
