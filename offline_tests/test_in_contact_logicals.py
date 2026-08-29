"""`in_contact` combines per-pair contact with `logical`/`K` (changes.md P06).

`predicate_logic` imports isaaclab at module level; we stub those imports so the
pure-Python logic can be exercised without a simulator.
"""
import sys
import types

import torch


def _stub(name, **attrs):
    m = types.ModuleType(name)
    m.__dict__.update(attrs)
    sys.modules[name] = m
    return m


for name in ("isaaclab", "isaaclab.utils"):
    sys.modules.setdefault(name, types.ModuleType(name))
_stub("isaaclab.utils.math", quat_apply=None, quat_apply_inverse=None)
_stub("robolab.core.task.hull_check", build_local_hull=None, point_in_hull=None)
_stub("robolab.core.utils.geometry_utils", spatial_condition_check_vector_based=None)
_stub("robolab.core.utils.transform_utils", transform_pose_from_w_to_b_vectorized=None)
_stub("robolab.core.world.world_state", get_world=None)

from robolab.core.task.predicate_logic import in_contact  # noqa: E402


class _World:
    def __init__(self, table):
        self.table = table  # (o1, o2) -> bool or tensor

    def in_contact(self, o1, o2, force_threshold, env_id=None):
        return self.table[(o1, o2)]


def test_scalar_default_is_all_pairs():
    w = _World({("a", "x"): True, ("b", "x"): False})
    assert in_contact(w, ["a", "b"], "x", env_id=0) is False


def test_scalar_any_and_choose():
    w = _World({("a", "x"): True, ("b", "x"): False})
    assert in_contact(w, ["a", "b"], "x", logical="any", env_id=0) is True
    assert in_contact(w, ["a", "b"], "x", logical="choose", K=1, env_id=0) is True
    assert in_contact(w, ["a", "b"], "x", logical="choose", K=2, env_id=0) is False


def test_vectorized_logicals():
    t = torch.tensor
    w = _World({("a", "x"): t([True, False, True]), ("b", "x"): t([False, False, True])})
    assert in_contact(w, ["a", "b"], "x").tolist() == [False, False, True]
    assert in_contact(w, ["a", "b"], "x", logical="any").tolist() == [True, False, True]
    assert in_contact(w, ["a", "b"], "x", logical="choose", K=1).tolist() == [True, False, False]
