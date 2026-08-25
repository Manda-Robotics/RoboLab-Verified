"""`object_grabbed` = contact AND closed hand (P31 / B1). The conditionals module
imports IsaacLab through predicate_logic/world_state; those are stubbed here."""
import sys
import types

import torch


def _stub(name, **attrs):
    m = types.ModuleType(name); m.__dict__.update(attrs); sys.modules[name] = m; return m


for n in ("isaaclab", "isaaclab.utils"):
    sys.modules.setdefault(n, types.ModuleType(n))
_stub("isaaclab.utils.math", quat_apply=None, quat_apply_inverse=None)
_stub("robolab.core.task.hull_check", build_local_hull=None, point_in_hull=None)
_stub("robolab.core.utils.geometry_utils", spatial_condition_check_vector_based=None)
_stub("robolab.core.utils.transform_utils", transform_pose_from_w_to_b_vectorized=None)
_stub("robolab.core.utils.function_loader", func_as_str=lambda f: getattr(f, "__name__", str(f)), get_callable_info=lambda f: (getattr(f, "__name__", "?"), {}))


class _World:
    """Two envs: env0 holds the banana with a closed hand, env1 only touches it (open)."""
    def __init__(self):
        self.env = types.SimpleNamespace(num_envs=2, device="cpu")
        self.contacts = {("banana", "gripper"): torch.tensor([True, True])}
        self.finger = torch.tensor([0.6, 0.05])          # finger_joint rad; closed ≈ 0.785

    def in_contact(self, a, b, force_threshold=0.1, env_id=None):
        v = self.contacts.get((a, b), torch.tensor([False, False]))
        return bool(v[env_id]) if env_id is not None else v

    def resolve_contact_bodies(self, name): return [name]
    def get_joint_names(self, robot): return ["j0", "finger_joint"]
    def get_joint_positions(self, robot, env_id=None):
        jp = torch.stack([torch.zeros(2), self.finger], dim=1)   # (N, J)
        return jp[env_id] if env_id is not None else jp


WORLD = _World()
_stub("robolab.core.world.world_state", get_world=lambda env: WORLD, WorldState=object)

# test_list_form_subtasks stubs predicate_logic with a bare fake; drop it (and
# anything that imported it) so the real module loads against our stubs.
for _n in list(sys.modules):
    if _n.startswith("robolab.core.task") and not hasattr(sys.modules[_n], "__file__"):
        del sys.modules[_n]
for _n in ("robolab.core.task.conditionals", "robolab.core.task.subtask", "robolab.core.task.subtask_utils"):
    sys.modules.pop(_n, None)

import robolab.constants as C  # noqa: E402
import robolab.core.task.conditionals as cond  # noqa: E402

# Other offline tests may have imported `conditionals` first with a different
# world stub bound; bind ours explicitly instead of relying on import order.
cond.get_world = lambda env: WORLD
object_grabbed = cond.object_grabbed

ENV = types.SimpleNamespace(num_envs=2, device="cpu", cfg=types.SimpleNamespace(gripper_closure_cfg=None))


def test_contact_with_open_hand_is_not_a_grab():
    C.GRAB_MIN_CLOSURE = 0.30
    assert object_grabbed(ENV, "banana").tolist() == [True, False]        # vectorized
    assert object_grabbed(ENV, "banana", env_id=0) is True                 # scalar
    assert object_grabbed(ENV, "banana", env_id=1) is False


def test_zero_threshold_restores_contact_only():
    C.GRAB_MIN_CLOSURE = 0.0
    assert object_grabbed(ENV, "banana").tolist() == [True, True]
    C.GRAB_MIN_CLOSURE = 0.30


def test_no_contact_is_never_a_grab():
    C.GRAB_MIN_CLOSURE = 0.30
    WORLD.contacts[("banana", "gripper")] = torch.tensor([False, False])
    try:
        assert object_grabbed(ENV, "banana").tolist() == [False, False]
    finally:
        WORLD.contacts[("banana", "gripper")] = torch.tensor([True, True])
