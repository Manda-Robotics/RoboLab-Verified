"""P64: a ladder rung that is already true at reset earns no credit.

Reproducers, both from `isaac60_robolab120_pi05` (Finn confirmed both on the
dashboard, 2026-08-26):

  BananasOutOfBinTask env0/env1 — ladder is
  [object_grabbed, object_outside_of, object_dropped]; `object_dropped` means
  "not in contact with the hand", which is true before the arm moves, and it is
  the LAST rung. Upstream's forward scan takes the highest satisfied rung, so the
  ladder completed on frame 1: "Completed subtask 'bananas_out_of_bin' 1/1" at
  0.07 s in both envs. The intended-object set then empties, and the task's own
  targets are logged as `Wrong object grabbed: 'banana_04' (target objects: [])`
  at 13.07 s / 16.40 s (env0) and 20.33 / 36.00 / 51.40 s (env1).

  BlackItemsInBinTask env0/env1 — `keyboard` spawns inside `grey_bin`, so
  `object_in_container(keyboard, grey_bin)` is true at reset and was credited at
  0.07 s in both envs.
"""
import sys
import types


def _stub(name, **attrs):
    m = types.ModuleType(name); m.__dict__.update(attrs); sys.modules[name] = m


# Isaac-free: the state machine only needs these three modules to exist.
_stub("robolab.core.task.predicate_logic", get_task_conditional_func=lambda *a, **k: None)
_stub("robolab.core.utils.function_loader",
      func_as_str=lambda f: getattr(f, "__name__", str(f)),
      get_callable_info=lambda f: (getattr(f, "__name__", str(f)), None))
_stub("robolab.core.world.world_state", WorldState=object, get_world=lambda e: None)

import robolab.constants  # noqa: E402
import robolab.core.task.conditionals_state_machine as CSM  # noqa: E402
from robolab.core.task.conditionals_state_machine import ConditionalsStateMachine  # noqa: E402
from robolab.core.task.subtask import Subtask  # noqa: E402


def _machine(subtask):
    # Bind directly on the module: another offline test may have stubbed
    # world_state first, so the import-time binding is not ours.
    CSM.get_world = lambda e: None
    return ConditionalsStateMachine(env=types.SimpleNamespace(), env_id=0, subtask=subtask)


# --- the BananasOutOfBin ladder, with a controllable world -------------------

class Bananas:
    """grabbed/outside are false at reset; dropped (= not in hand) is true."""

    def __init__(self):
        self.grabbed = False
        self.outside = False

    def cond_grabbed(self, env, env_id=None):
        return self.grabbed

    def cond_outside(self, env, env_id=None):
        return self.outside

    def cond_dropped(self, env, env_id=None):
        return not self.grabbed          # true at reset, exactly as upstream


def _bananas_subtask(w):
    return Subtask(
        name="bananas_out_of_bin",
        conditions={"banana_02": [w.cond_grabbed, w.cond_outside, w.cond_dropped]},
        logical="all",
        score=1.0,
    )


def test_spawn_true_last_rung_does_not_complete_the_ladder_on_frame_one():
    w = Bananas()
    sm = _machine(_bananas_subtask(w))
    done, info, code, all_codes = sm.step()
    assert done is False, "ladder completed before the policy acted (H-R9-9)"
    assert sm.total_score == 0.0
    assert not any("Completed subtask" in str(i) for i, _c in all_codes)


def test_the_ladder_still_completes_on_its_real_rungs():
    w = Bananas()
    sm = _machine(_bananas_subtask(w))
    sm.step()
    w.grabbed = True
    sm.step()
    w.outside = True
    w.grabbed = False                     # released outside the bin
    done, _info, _c, _a = sm.step()
    assert done is True, "the real ladder must still be completable"
    assert abs(sm.total_score - 1.0) < 1e-9, "excluded rungs must be renormalised away"


def test_excluded_rung_earns_no_score_on_its_own():
    """BlackItemsInBin: the group's only rung is true at spawn -> zero credit."""
    class Keyboard:
        def cond_in_bin(self, env, env_id=None):
            return True                   # keyboard spawns inside grey_bin

    k = Keyboard()
    sm = _machine(Subtask(name="black_items", conditions={"keyboard": [k.cond_in_bin]},
                          logical="all", score=1.0))
    done, _info, _code, all_codes = sm.step()
    assert sm.total_score == 0.0, "a scene-satisfied rung must not score"
    assert not any("Completed subtask" in str(i) for i, _c in all_codes)
    assert sm.spawn_excluded_rungs() == {"keyboard": [0]}


def test_kill_switch_restores_upstream_behaviour():
    w = Bananas()
    robolab.constants.SUBTASK_EXCLUDE_SPAWN_TRUE_RUNGS = False
    try:
        sm = _machine(_bananas_subtask(w))
        done, _i, _c, _a = sm.step()
        assert done is True, "upstream behaviour is the frame-1 completion"
    finally:
        robolab.constants.SUBTASK_EXCLUDE_SPAWN_TRUE_RUNGS = True


def test_a_rung_false_at_spawn_is_never_excluded():
    w = Bananas()
    sm = _machine(_bananas_subtask(w))
    sm.step()
    assert sm.spawn_excluded_rungs() == {"banana_02": [2]}, "only the dropped rung is spawn-true"
