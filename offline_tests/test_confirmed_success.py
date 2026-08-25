"""Confirmed success (P30 / A2): goal holds AND targets at rest for rest_s → success.
At rest already when the goal is reached → immediate; still moving → wait."""
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


def _install_world(world):
    m = types.ModuleType("robolab.core.world.world_state"); m.get_world = lambda env: world
    sys.modules["robolab.core.world.world_state"] = m


def test_at_rest_cuts_off_immediately_moving_waits():
    env = _Env(2, dt=0.1)
    speeds = {"banana": torch.tensor([0.0, 0.5])}          # env0 still, env1 flying
    _install_world(_World(speeds))
    raw = torch.tensor([False, False])
    c = SuccessConfirmer(lambda env, **p: raw, rest_s=0.2, max_speed=0.02, targets=["banana"])
    for step in range(1, 4):                                 # goal not yet reached; env0 accumulates rest
        env.episode_length_buf[:] = step; assert c(env).tolist() == [False, False]
    raw[:] = True                                            # goal reached in both envs at step 4
    env.episode_length_buf[:] = 4
    assert c(env).tolist() == [True, False]                  # env0: already at rest → done on this frame
    assert c.first_hold_step.tolist() == [4, 4] and c.confirmed_step.tolist() == [4, -1]
    speeds["banana"][1] = 0.0                                # env1 settles from step 5
    for step in (5, 6):
        env.episode_length_buf[:] = step; out = c(env).tolist()
    assert out == [True, True] and c.confirmed_step[1].item() == 6   # 2 rest ticks (0.2 s) after settling


def test_momentary_zero_speed_mid_bounce_is_not_rest():
    env = _Env(1, dt=0.1); speeds = {"cube": torch.tensor([0.3])}; _install_world(_World(speeds))
    c = SuccessConfirmer(lambda env, **p: torch.tensor([True]), rest_s=0.2, max_speed=0.02, targets=["cube"])
    seq = [0.3, 0.0, 0.3, 0.0, 0.0]                           # apex of a bounce, then settled
    out = []
    for step, v in enumerate(seq, start=1):
        speeds["cube"][0] = v; env.episode_length_buf[:] = step; out.append(bool(c(env)[0]))
    assert out == [False, False, False, False, True]


def test_reset_clears_state():
    env = _Env(1, dt=0.1); _install_world(_World({"cube": torch.tensor([0.0])}))
    c = SuccessConfirmer(lambda env, **p: torch.tensor([True]), rest_s=0.2, max_speed=0.02, targets=["cube"])
    env.episode_length_buf[:] = 1; c(env); env.episode_length_buf[:] = 2; assert c(env)[0]
    env.episode_length_buf[:] = 1                            # new episode
    assert not c(env)[0] and c.first_hold_step.tolist() == [1]


def test_wrapper_keeps_signature_and_targets():
    def object_in_container(env, object, container, require_gripper_detached=False, env_id=None):
        return torch.tensor([True])
    term = types.SimpleNamespace(func=object_in_container, params={"object": ["a", "b"], "container": "bin"})
    t = confirmed_success_term(term, rest_s=0.2, max_speed=0.02)
    import inspect
    assert list(inspect.signature(t.func).parameters) == ["env", "object", "container", "require_gripper_detached", "env_id"]
    assert t.func.confirmer.targets == ["a", "b"]
    assert _target_names({"groups": [{"object": ["x"], "container": "c"}, {"object": "y"}]}) == ["x", "y"]
    assert confirmed_success_term(term, rest_s=0, max_speed=0.02) is term    # disabled → untouched
