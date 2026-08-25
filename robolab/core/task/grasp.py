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
Contact that ends before that, with the hand at least partly closed, is one
``GRASP_ATTEMPT_FAILED``. After a grasp, losing contact is ``OBJECT_RELEASED``
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


def hand_contact(env, obj: str, hand_label: str) -> torch.Tensor:
    from robolab.core.task.predicate_logic import in_contact
    from robolab.core.world.world_state import get_world
    r = in_contact(get_world(env), obj, hand_label, env_id=None)
    return r if isinstance(r, torch.Tensor) else torch.as_tensor(r, dtype=torch.bool, device=env.device).reshape(-1)


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
        self._pairs: dict[tuple[str, str], _PairState] = {}
        self._events: list[tuple[int, str, str, str]] = []      # (env_id, object, hand, kind)

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
            if bool(fresh.all()):
                st.rel_hist.clear(); st.hand_hist.clear()

        contact = hand_contact(env, obj, hand)
        obj_pos = object_position(env, obj)
        hand_pos = hand_position(env, hand)
        rel = obj_pos - hand_pos
        st.rel_hist.append(rel); st.hand_hist.append(hand_pos)

        closed_attempt = hand_closed(env, hand, self.attempt_closure)
        still_closed = hand_closed(env, hand, self.release_closure)

        st.contact_streak = torch.where(contact, st.contact_streak + 1, torch.zeros_like(st.contact_streak))
        st.attempt_closed = (st.attempt_closed | closed_attempt) & contact

        # stable grasp: contact for k steps, offset to the hand steady, hand moved
        if len(st.rel_hist) > k:
            rel_dev = torch.linalg.norm(st.rel_hist[-1] - st.rel_hist[0], dim=-1)
            hand_moved = torch.linalg.norm(st.hand_hist[-1] - st.hand_hist[0], dim=-1)
        else:
            rel_dev = torch.full((n,), float("inf"), device=env.device)
            hand_moved = torch.zeros(n, device=env.device)
        newly = (~st.grasped) & contact & (st.contact_streak >= k) & (rel_dev < self.coupling_m) & (hand_moved >= self.hand_move_m)

        lost = st.grasped & ~contact
        ended_attempt = (~st.grasped) & st.prev_contact & ~contact & ~fresh
        for eid in ended_attempt.nonzero(as_tuple=False).flatten().tolist():
            if bool(self._attempt_flag(st, eid)):
                self._events.append((eid, obj, hand, "attempt_failed"))
        for eid in lost.nonzero(as_tuple=False).flatten().tolist():
            self._events.append((eid, obj, hand, "dropped" if bool(still_closed[eid]) else "released"))

        st.grasped = (st.grasped | newly) & contact
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

    def pop_events(self) -> list[tuple[int, str, str, str]]:
        ev, self._events = self._events, []
        return ev
