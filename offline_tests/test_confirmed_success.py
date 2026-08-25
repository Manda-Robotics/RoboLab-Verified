"""Confirmed success: predicate must hold for hold_s with targets at rest (P30 / A2)."""
import sys
import types

import torch

from robolab.core.task.confirm import SuccessConfirmer, _target_names, confirmed_success_term


class _World:
    def __init__(self, speeds):
        self.speeds = speeds            # name -> (N,) linear speed

    def get_velocity(self, name, env_id=None):
        s = self.speeds[name]
        v = torch.zeros(s.shape[0], 6); v[:, 0] = s
        return v


class _Env:
    def __init__(self, n, dt=0.1):
        self.num_envs = n; self.device = "cpu"; self.step_dt = dt
        self.episode_length_buf = torch.zeros(n, dtype=torch.long)
        self.world = None


def _install_world(world):
    m = types.ModuleType("robolab.core.world.world_state"); m.get_world = lambda env: world
    sys.modules["robolab.core.world.world_state"] = m


def test_holds_only_after_hold_s_with_targets_at_rest():
    env = _Env(2, dt=0.1)
    speeds = {"banana": torch.tensor([0.0, 0.5])}          # env1's banana is flying
    _install_world(_World(speeds))
    raw = torch.tensor([True, True])
    c = SuccessConfirmer(lambda env, **p: raw, hold_s=0.5, max_speed=0.02, targets=["banana"])
    for step in range(1, 5):                                 # 4 steps < 5 needed
        env.episode_length_buf[:] = step
        assert c(env).tolist() == [False, False]
    env.episode_length_buf[:] = 5
    assert c(env).tolist() == [True, False]                  # env0 confirmed, env1 still moving
    assert c.first_hold_step.tolist() == [1, 1] and c.confirmed_step.tolist() == [5, -1]
    speeds["banana"][1] = 0.0                                # env1 settles
    env.episode_length_buf[:] = 6
    assert c(env).tolist() == [True, True]


def test_streak_resets_when_predicate_drops():
    env = _Env(1, dt=0.1); _install_world(_World({"cube": torch.tensor([0.0])}))
    seq = [True, True, False, True, True, True]
    it = iter(seq)
    c = SuccessConfirmer(lambda env, **p: torch.tensor([next(it)]), hold_s=0.3, max_speed=0.02, targets=["cube"])
    out = []
    for step, _ in enumerate(seq, start=1):
        env.episode_length_buf[:] = step; out.append(bool(c(env)[0]))
    assert out == [False, False, False, False, False, True]   # 3-step streak only after the drop


def test_reset_clears_state():
    env = _Env(1, dt=0.1); _install_world(_World({"cube": torch.tensor([0.0])}))
    c = SuccessConfirmer(lambda env, **p: torch.tensor([True]), hold_s=0.2, max_speed=0.02, targets=["cube"])
    env.episode_length_buf[:] = 1; c(env); env.episode_length_buf[:] = 2; assert c(env)[0]
    env.episode_length_buf[:] = 1                            # new episode
    assert not c(env)[0] and c.first_hold_step.tolist() == [1]


def test_wrapper_keeps_signature_and_targets():
    def object_in_container(env, object, container, require_gripper_detached=False, env_id=None):
        return torch.tensor([True])
    term = types.SimpleNamespace(func=object_in_container, params={"object": ["a", "b"], "container": "bin"})
    t = confirmed_success_term(term, hold_s=1.0, max_speed=0.02)
    import inspect
    assert list(inspect.signature(t.func).parameters) == ["env", "object", "container", "require_gripper_detached", "env_id"]
    assert t.func.confirmer.targets == ["a", "b"]
    assert _target_names({"groups": [{"object": ["x"], "container": "c"}, {"object": "y"}]}) == ["x", "y"]
    assert confirmed_success_term(term, hold_s=0, max_speed=0.02) is term    # disabled → untouched
