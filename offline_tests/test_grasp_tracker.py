"""Stable grasp = carry (P31 v3 / B1, B6, H-R5-4). The tracker's geometry/contact
accessors are module-level so they can be scripted here without a simulator."""
import types

import torch

import robolab.core.task.grasp as G

ENV = None


class _Env:
    def __init__(self, n=1, dt=0.1):
        self.num_envs = n; self.device = "cpu"; self.step_dt = dt
        self.episode_length_buf = torch.zeros(n, dtype=torch.long)
        self.common_step_counter = 0


class _Script:
    """Per-step scripted world: contact flag, hand position, object position, closure fraction."""
    def __init__(self):
        self.contact = False; self.hand = torch.zeros(3); self.obj = torch.tensor([0.05, 0.0, 0.0]); self.closure = 0.0

    def install(self):
        self.offset = 0.06                                      # 6 cm off-centre unless a test says otherwise
        G.jaw_offset = lambda env, obj: torch.tensor([self.offset])
        G.hand_contact = lambda env, obj, hand: torch.tensor([self.contact])
        G.hand_position = lambda env, hand: self.hand.clone().unsqueeze(0)
        G.object_position = lambda env, obj: self.obj.clone().unsqueeze(0)
        G.hand_closed = lambda env, hand, thr: torch.tensor([self.closure >= thr])


def _kinds(evs):
    return [(e[0], e[1], e[3], e[4].get("count")) for e in evs]


def _tick(env, tracker, script, contact, hand=None, obj=None, closure=None):
    env.common_step_counter += 1; env.episode_length_buf += 1
    script.contact = contact
    if hand is not None: script.hand = torch.tensor(hand, dtype=torch.float)
    if obj is not None: script.obj = torch.tensor(obj, dtype=torch.float)
    if closure is not None: script.closure = closure
    return bool(tracker.grasped("banana", "gripper", env_id=0)), tracker.pop_events()


def test_open_hand_brush_is_nothing():
    env = _Env(); s = _Script(); s.install(); t = G.GraspTracker(env)
    env.episode_length_buf[:] = 1                      # skip the reset tick
    assert _tick(env, t, s, True, closure=0.0) == (False, [])
    assert _tick(env, t, s, False) == (False, [])      # ended with an open hand → no attempt flag


def test_short_closed_contact_is_one_failed_attempt():
    """The line appears once the burst goes quiet (P47), with a count of 1."""
    env = _Env(); s = _Script(); s.install(); t = G.GraspTracker(env)
    env.episode_length_buf[:] = 1
    assert _tick(env, t, s, True, closure=0.5)[0] is False
    assert _tick(env, t, s, True, closure=0.6)[0] is False
    grasped, ev = _tick(env, t, s, False, closure=0.6)
    assert grasped is False and ev == []                    # burst still open
    evs = []
    for _ in range(25):                                     # quiet > GRASP_ATTEMPT_BURST_S
        evs += _tick(env, t, s, False, closure=0.0)[1]
    assert _kinds(evs) == [(0, "banana", "attempt_failed", 1)]


def test_carry_becomes_a_grasp_then_release_or_drop():
    env = _Env(dt=0.1); s = _Script(); s.install(); t = G.GraspTracker(env)   # hold_steps = 2
    env.episode_length_buf[:] = 1
    # contact, hand moving 1 cm/tick, object riding along (offset constant)
    for x in (0.00, 0.01, 0.02):
        grasped, ev = _tick(env, t, s, True, hand=[x, 0, 0], obj=[x + 0.05, 0, 0], closure=0.6)
    assert grasped is True and ev == []
    # keep carrying, then open the hand → released
    _tick(env, t, s, True, hand=[0.03, 0, 0], obj=[0.08, 0, 0], closure=0.6)
    grasped, ev = _tick(env, t, s, False, hand=[0.04, 0, 0], obj=[0.09, 0, 0], closure=0.0)
    assert grasped is False and _kinds(ev) == [(0, "banana", "released", None)]
    # new grasp, then lose it with the hand still closed → dropped
    for x in (0.05, 0.06, 0.07):
        grasped, ev = _tick(env, t, s, True, hand=[x, 0, 0], obj=[x + 0.05, 0, 0], closure=0.6)
    assert grasped is True
    grasped, ev = _tick(env, t, s, False, hand=[0.08, 0, 0], obj=[0.30, 0, -0.2], closure=0.6)
    assert grasped is False and _kinds(ev) == [(0, "banana", "dropped", None)]


def test_contact_without_coupling_is_not_a_grasp():
    env = _Env(dt=0.1); s = _Script(); s.install(); t = G.GraspTracker(env)
    env.episode_length_buf[:] = 1
    # hand moves, object stays put (pushing along the fingers / sliding out)
    for x in (0.00, 0.01, 0.02, 0.03):
        grasped, _ = _tick(env, t, s, True, hand=[x, 0, 0], obj=[0.05, 0, 0], closure=0.6)
    assert grasped is False


def test_static_hand_holding_is_not_yet_confirmed_until_it_moves():
    env = _Env(dt=0.1); s = _Script(); s.install(); t = G.GraspTracker(env)
    env.episode_length_buf[:] = 1
    for _ in range(4):
        grasped, _ = _tick(env, t, s, True, hand=[0, 0, 0], obj=[0.05, 0, 0], closure=0.6)
    assert grasped is False
    for x in (0.01, 0.02):
        grasped, _ = _tick(env, t, s, True, hand=[x, 0, 0], obj=[x + 0.05, 0, 0], closure=0.6)
    assert grasped is True


def test_reset_clears_state_and_update_is_once_per_step():
    env = _Env(dt=0.1); s = _Script(); s.install(); t = G.GraspTracker(env)
    env.episode_length_buf[:] = 1
    for x in (0.00, 0.01, 0.02):
        _tick(env, t, s, True, hand=[x, 0, 0], obj=[x + 0.05, 0, 0], closure=0.6)
    assert t.grasped("banana", "gripper", env_id=0) is True          # same step, no extra update
    env.episode_length_buf[:] = 0; env.common_step_counter += 1      # reset
    assert t.grasped("banana", "gripper", env_id=0) is False


def test_open_hand_carry_is_towed_not_grasped():
    env = _Env(dt=0.1); s = _Script(); s.install(); t = G.GraspTracker(env)
    env.episode_length_buf[:] = 1
    evs = []
    for x in (0.00, 0.01, 0.02, 0.03):
        grasped, ev = _tick(env, t, s, True, hand=[x, 0, 0], obj=[x + 0.05, 0, 0], closure=0.02)  # hand OPEN (< 0.1)
        evs += ev
    assert grasped is False
    assert _kinds(evs) == [(0, "banana", "towed", None)]                 # flagged once
    assert t.towed_objects() == {0: {"banana"}}
    # closing the hand on the same carry → now a grasp
    grasped, _ = _tick(env, t, s, True, hand=[0.04, 0, 0], obj=[0.09, 0, 0], closure=0.6)
    assert grasped is True


def test_wide_object_grip_at_low_closure_is_a_grasp_not_a_tow():
    env = _Env(dt=0.1); s = _Script(); s.install(); t = G.GraspTracker(env)
    env.episode_length_buf[:] = 1
    evs = []
    for x in (0.00, 0.01, 0.02, 0.03):
        grasped, ev = _tick(env, t, s, True, hand=[x, 0, 0], obj=[x + 0.05, 0, 0], closure=0.23)  # orange: 23 % closed
        evs += ev
    assert grasped is True and evs == []


def test_centred_object_between_open_jaws_is_a_grip_not_a_tow():
    """A can as wide as the aperture: closure 0.00 but centred → grasp."""
    env = _Env(dt=0.1); s = _Script(); s.install(); s.offset = 0.0; t = G.GraspTracker(env)
    env.episode_length_buf[:] = 1
    evs = []
    for x in (0.00, 0.01, 0.02, 0.03):
        grasped, ev = _tick(env, t, s, True, hand=[x, 0, 0], obj=[x + 0.05, 0, 0], closure=0.0)
        evs += ev
    assert grasped is True and evs == []


def test_attempt_burst_is_one_line_with_a_count():
    env = _Env(dt=0.1); s = _Script(); s.install(); t = G.GraspTracker(env)   # burst window 2 s = 20 ticks
    env.episode_length_buf[:] = 1
    evs = []
    for _ in range(3):                                    # three touch-and-lose blips in a row
        evs += _tick(env, t, s, True, closure=0.6)[1]
        evs += _tick(env, t, s, False, closure=0.6)[1]
    assert evs == []                                      # nothing emitted while the burst is open
    for _ in range(25):                                   # quiet period > 2 s
        evs += _tick(env, t, s, False, closure=0.0)[1]
    assert _kinds(evs) == [(0, "banana", "attempt_failed", 3)]


def test_open_burst_is_flushed_at_episode_end():
    env = _Env(dt=0.1); s = _Script(); s.install(); t = G.GraspTracker(env)
    env.episode_length_buf[:] = 1
    _tick(env, t, s, True, closure=0.6); _tick(env, t, s, False, closure=0.6)
    t.flush_attempts()
    assert _kinds(t.pop_events()) == [(0, "banana", "attempt_failed", 1)]
