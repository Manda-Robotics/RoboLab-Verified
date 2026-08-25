"""A1 / B7: one flag per non-target entering a goal container; pre-placed objects never count."""
import torch
from robolab.core.task.collateral import CollateralTracker

T, F = True, False
def b(*v): return torch.tensor(v)

def test_pre_placed_never_counts_and_each_object_flags_once():
    c = CollateralTracker(2, "cpu")
    warm, cold = b(T, T), b(F, F)
    # warm-up: mug already inside in env0
    assert c.update({("mug", "bin"): b(T, F)}, warm, {}) == []
    assert c.update({("mug", "bin"): b(T, F)}, cold, {}) == []            # still inside, was there at reset
    # env1: mug enters later while recently held → placed
    ev = c.update({("mug", "bin"): b(T, T)}, cold, {"mug": b(F, T)})
    assert ev == [(1, "mug", "bin", "placed")]
    assert c.update({("mug", "bin"): b(T, T)}, cold, {"mug": b(F, T)}) == []   # no re-flag
    # env1: leaves and re-enters → still no second flag for the same object
    c.update({("mug", "bin"): b(T, F)}, cold, {})
    assert c.update({("mug", "bin"): b(T, T)}, cold, {}) == []
    assert c.collateral == {1: {"mug"}}

def test_pushed_vs_placed():
    c = CollateralTracker(1, "cpu"); cold = b(F)
    c.update({("orange", "bin"): b(F)}, b(T), {})
    assert c.update({("orange", "bin"): b(T)}, cold, {"orange": b(F)}) == [(0, "orange", "bin", "pushed")]

def test_reset_clears():
    c = CollateralTracker(1, "cpu"); cold = b(F)
    c.update({("orange", "bin"): b(F)}, b(T), {})
    c.update({("orange", "bin"): b(T)}, cold, {"orange": b(T)})
    c.reset_envs([0])
    c.update({("orange", "bin"): b(F)}, b(T), {})
    assert c.update({("orange", "bin"): b(T)}, cold, {"orange": b(T)}) == [(0, "orange", "bin", "placed")]
