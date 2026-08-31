"""P78 — the grip is its own rung.

The reviewer, reviewing rc5 FoodPacking2Cans env0: "we start out with an object carried, an
object grabbed success flag, without the object ever being gripped for the first
time, so that's not great"; "it needs to be grip, carry, and then grab success".

So a successful pick now reads

    OBJECT_GRIPPED   the jaws closed on it          (neutral)
    OBJECT_CARRIED   it then travelled with the hand (neutral)
    OBJECT_GRABBED_SUCCESS   the ladder credits it   (green)

and a failed one reads OBJECT_GRIPPED then GRASP_ATTEMPT_FAILED. A carry with **no**
grip in front of it is the hand pushing the object along, which is what the reviewer labelled
a shove on the lizard_figurine clips.
"""
import torch

import robolab.core.task.grasp as G
from robolab.core.task.status import NEUTRAL_STATUS_CODES, StatusCode


class _ActionManager:
    def __init__(self): self.action = torch.tensor([[0.0] * 7 + [1.0]])
    def set_open(self, is_open): self.action[0, -1] = 0.0 if is_open else 1.0


class _Env:
    def __init__(self, n=1, dt=0.1):
        self.num_envs = n; self.device = "cpu"; self.step_dt = dt
        self.episode_length_buf = torch.zeros(n, dtype=torch.long)
        self.common_step_counter = 0
        self.action_manager = _ActionManager()


class _Script:
    def __init__(self):
        self.contact = False; self.hand = torch.zeros(3)
        self.obj = torch.tensor([0.05, 0.0, 0.0]); self.closure = 0.0

    def install(self):
        self.offset = 0.06
        G.jaw_offset = lambda env, obj: torch.tensor([self.offset])
        G.hand_contact = lambda env, obj, hand: torch.tensor([self.contact])
        G.hand_position = lambda env, hand: self.hand.clone().unsqueeze(0)
        G.object_position = lambda env, obj: self.obj.clone().unsqueeze(0)
        G.hand_closed = lambda env, hand, thr: torch.tensor([self.closure >= thr])


def _tick(env, tracker, script, contact, hand=None, obj=None, closure=None):
    env.common_step_counter += 1; env.episode_length_buf += 1
    script.contact = contact
    if hand is not None: script.hand = torch.tensor(hand, dtype=torch.float)
    if obj is not None: script.obj = torch.tensor(obj, dtype=torch.float)
    if closure is not None: script.closure = closure
    tracker.grasped("banana", "gripper", env_id=0)
    return [e[3] for e in tracker.pop_events()]


def _fresh():
    env = _Env(dt=0.1); s = _Script(); s.install()
    env.episode_length_buf[:] = 1
    return env, s, G.GraspTracker(env)


def test_a_pick_reads_grip_then_carry():
    env, s, t = _fresh()
    kinds = []
    for x in (0.00, 0.01, 0.02):
        kinds += _tick(env, t, s, True, hand=[x, 0, 0], obj=[x + 0.05, 0, 0], closure=0.6)
    assert kinds == ["gripped", "grabbed"], kinds
    assert kinds.index("gripped") < kinds.index("grabbed")


def test_a_grip_that_never_becomes_a_carry_is_reported_only_as_a_failure():
    """The reviewer's either/or: grip+carry+success, OR failed attempt. Not both — a 0.3 s
    fumble producing three lines is the noise P47 exists to prevent."""
    env, s, t = _fresh()
    kinds = _tick(env, t, s, True, closure=0.6)          # closes on it, never moves it
    assert kinds == []
    for _ in range(25):
        kinds += _tick(env, t, s, False, closure=0.0)
    assert kinds == ["attempt_failed"]


def test_a_burst_of_fumbles_adds_no_grip_lines():
    env, s, t = _fresh()
    kinds = []
    for _ in range(3):
        kinds += _tick(env, t, s, True, closure=0.6)
        kinds += _tick(env, t, s, False, closure=0.6)
    assert kinds == []
    for _ in range(25):
        kinds += _tick(env, t, s, False, closure=0.0)
    assert kinds == ["attempt_failed"]


def test_a_shove_produces_a_carry_with_no_grip_in_front_of_it():
    """The lizard_figurine case: contact on an OPEN hand, object riding along."""
    env, s, t = _fresh()
    kinds = []
    for x in (0.00, 0.01, 0.02, 0.03):
        kinds += _tick(env, t, s, True, hand=[x, 0, 0], obj=[x + 0.05, 0, 0], closure=0.0)
    assert "gripped" not in kinds
    assert "grabbed" in kinds, "the detector still calls this a carry — that is the finding"


def test_the_grip_line_is_emitted_once_per_carry_not_once_per_step():
    env, s, t = _fresh()
    kinds = []
    for i in range(6):
        kinds += _tick(env, t, s, True, hand=[i * 0.01, 0, 0], obj=[i * 0.01 + 0.05, 0, 0], closure=0.6)
    assert kinds.count("gripped") == 1 and kinds.count("grabbed") == 1


def test_the_grip_is_stamped_when_the_jaws_closed_not_when_the_carry_confirmed():
    """So the dashboard draws it ahead of the carry, which is the order the reviewer asked for."""
    env, s, t = _fresh()
    evs = []
    for i in range(4):
        env.common_step_counter += 1; env.episode_length_buf += 1
        s.contact = True; s.closure = 0.6
        s.hand = torch.tensor([i * 0.01, 0, 0], dtype=torch.float)
        s.obj = torch.tensor([i * 0.01 + 0.05, 0, 0], dtype=torch.float)
        t.grasped("banana", "gripper", env_id=0)
        evs += t.pop_events()
    grip = next(e for e in evs if e[3] == "gripped")
    carry = next(e for e in evs if e[3] == "grabbed")
    assert grip[4]["onset_step"] <= carry[4]["onset_step"]


def test_picking_the_same_object_up_twice_is_two_grip_lines():
    env, s, t = _fresh()
    kinds = []
    for i in range(3):                                   # grip and carry
        kinds += _tick(env, t, s, True, hand=[i * 0.01, 0, 0], obj=[i * 0.01 + 0.05, 0, 0], closure=0.6)
    for _ in range(3):                                   # let go
        kinds += _tick(env, t, s, False, closure=0.0)
    for i in range(3, 7):                                # grip and carry again
        kinds += _tick(env, t, s, True, hand=[i * 0.01, 0, 0], obj=[i * 0.01 + 0.05, 0, 0], closure=0.6)
    assert kinds.count("gripped") == 2, kinds


def test_an_open_hand_brushing_the_object_is_not_a_grip():
    env, s, t = _fresh()
    kinds = []
    for _ in range(4):
        kinds += _tick(env, t, s, True, closure=0.0)
    assert "gripped" not in kinds


def test_the_grip_is_neutral_not_a_failure():
    """The reviewer: "object carry should definitely not be red"—the same goes for its rung."""
    assert int(StatusCode.OBJECT_GRIPPED) in NEUTRAL_STATUS_CODES
    assert int(StatusCode.OBJECT_CARRIED) in NEUTRAL_STATUS_CODES


def test_a_grip_that_came_to_nothing_does_not_attach_to_a_later_carry():
    """rc6 FoodPacking2Cans env0: the jaws closed at 6.67 s, the object was not
    carried, and the grip line resurfaced on the carry at 14.27 s -- 7.6 s stale, so
    the dashboard drew it in front of the wrong carry. A grip belongs to the contact
    episode it happened in."""
    env, s, t = _fresh()
    kinds = []
    kinds += _tick(env, t, s, True, closure=0.6)          # jaws close on it...
    for _ in range(30):                                    # ...and it comes to nothing
        kinds += _tick(env, t, s, False, closure=0.0)
    assert "gripped" not in kinds
    # much later, a carry with an OPEN hand (a shove): it must NOT inherit that grip
    carry = []
    for i in range(4):
        carry += _tick(env, t, s, True, hand=[i * 0.01, 0, 0],
                       obj=[i * 0.01 + 0.05, 0, 0], closure=0.0)
    assert "grabbed" in carry and "gripped" not in carry, carry


def test_a_fresh_grip_still_attaches_to_the_carry_it_produced():
    """The reset must not throw away the grip we actually want."""
    env, s, t = _fresh()
    kinds = []
    for i in range(4):
        kinds += _tick(env, t, s, True, hand=[i * 0.01, 0, 0],
                       obj=[i * 0.01 + 0.05, 0, 0], closure=0.6)
    assert kinds.index("gripped") < kinds.index("grabbed")
