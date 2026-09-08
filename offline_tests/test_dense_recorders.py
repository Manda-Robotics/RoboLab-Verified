"""Dense-annotation recorder helpers (R2/R4, docs/verified/dense_annotations.md §5):
legend, per-step condition evaluation with the subtask's logical mode, tracker
packing, and body force magnitudes. No simulator."""
import functools
import sys
import types

for n in ("isaaclab", "isaaclab.utils", "isaaclab.managers"):
    sys.modules.setdefault(n, types.ModuleType(n))
def _stub(name, **a):
    m = types.ModuleType(name); m.__dict__.update(a); sys.modules[name] = m
if not hasattr(sys.modules["isaaclab.managers"], "recorder_manager"):
    _stub("isaaclab.managers.recorder_manager", RecorderTerm=object, RecorderTermCfg=object)
if not hasattr(sys.modules["isaaclab.utils"], "configclass"):
    sys.modules["isaaclab.utils"].configclass = lambda c: c

import torch  # noqa: E402

from robolab.core.events import dense_recorders as D  # noqa: E402


class _Subtask:
    def __init__(self, conditions, logical="all", K=None, name="st"):
        self.conditions, self.logical, self.K, self.name = conditions, logical, K, name


def _pred(name):
    def f(env=None, env_id=None, **kw):
        return env[name][env_id]
    f.__name__ = name
    return f


def test_legend_and_keys_follow_recording_order():
    grabbed = functools.partial(_pred("grabbed"), object="banana", distance=0.05)
    placed = _pred("placed")
    st = [_Subtask({"g": [(grabbed, 0.5), (placed, 0.5)]}, name="pick_place"),
          _Subtask({"a": [(placed, 1.0)], "b": [(placed, 1.0)]}, logical="any")]
    keys = [k for k, *_ in D.condition_keys(st)]
    assert keys == ["s0", "s0_g_0", "s0_g_1", "s1", "s1_a_0", "s1_b_0"]
    legend = D.condition_legend(st)
    assert [r["key"] for r in legend] == keys
    assert legend[1]["func"] == "grabbed" and legend[1]["params"] == {"object": "banana", "distance": 0.05}
    assert legend[0]["name"] == "pick_place" and legend[3]["logical"] == "any"


def test_evaluate_conditions_applies_logical_modes_and_swallows_errors():
    env = {"grabbed": [True, False, True], "placed": [True, True, False]}
    def boom(env=None, env_id=None):
        raise RuntimeError("no sensor")
    boom.__name__ = "boom"
    st = [_Subtask({"g": [(_pred("grabbed"), 1.0), (_pred("placed"), 1.0)]}),                  # terminal rung: placed
          _Subtask({"a": [(_pred("grabbed"), 1.0)], "b": [(_pred("placed"), 1.0)]}, "any"),  # any group
          _Subtask({"a": [(_pred("grabbed"), 1.0)], "b": [(_pred("placed"), 1.0)]}, "choose", K=2),
          _Subtask({"x": [(boom, 1.0)]})]
    res = D.evaluate_conditions(st, 3, lambda f, eid: f(env=env, env_id=eid))
    assert res["s0"] == [True, True, False]          # the group's terminal condition, not all rungs at once
    assert res["s1"] == [True, True, True]
    assert res["s2"] == [True, False, False]
    assert res["s3"] == [False, False, False] and res["s3_x_0"] == [False, False, False]
    assert res["s0_g_0"] == env["grabbed"] and res["s0_g_1"] == env["placed"]


def test_pack_tracker_state_keys_and_columns():
    class _St:
        def __init__(self, g, a):
            self.grasped = torch.tensor(g); self.attempt_closed = torch.tensor(a)
    pairs = {("banana", "gripper"): _St([True, False], [True, True]),
             ("bowl", "left"): _St([False, False], [False, True])}
    out = D.pack_tracker_state(pairs)
    assert set(out) == {"banana", "bowl__left"}
    assert out["banana"].dtype == torch.uint8 and out["banana"].tolist() == [[1, 1], [0, 1]]
    assert out["bowl__left"].tolist() == [[0, 0], [0, 1]]


def test_body_force_magnitudes_per_object():
    fm = torch.zeros((2, 1, 3, 3))
    fm[0, 0, 1] = torch.tensor([3.0, 4.0, 0.0])      # env 0 touches object 1 with 5 N
    fm[1, 0, 2] = torch.tensor([0.0, 0.0, 1.0])
    out = D.body_force_magnitudes(fm, ["a", "b", "c"])
    assert out["a"].tolist() == [0.0, 0.0]
    assert out["b"].tolist() == [5.0, 0.0]
    assert out["c"].tolist() == [0.0, 1.0]
    try:
        D.body_force_magnitudes(fm, ["a", "b"])
    except ValueError:
        pass
    else:
        raise AssertionError("shape mismatch must raise")
