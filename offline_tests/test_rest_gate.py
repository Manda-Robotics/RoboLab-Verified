"""P46: a placement is credited only once the object has settled."""
import sys, types
import torch

for n in ("isaaclab", "isaaclab.utils"):
    sys.modules.setdefault(n, types.ModuleType(n))

import robolab.constants as C
from robolab.core.task import rest as R


class _Env:
    def __init__(self, n=2, dt=0.1):
        self.num_envs = n; self.device = "cpu"; self.step_dt = dt
        self.episode_length_buf = torch.zeros(n, dtype=torch.long)
        self.common_step_counter = 0


class _World:
    def __init__(self, speeds): self.speeds = speeds
    def get_velocity(self, obj, env_id=None):
        v = torch.zeros(self.speeds[obj].shape[0], 6); v[:, 0] = self.speeds[obj]; return v


def _install(world):
    m = types.ModuleType("robolab.core.world.world_state"); m.get_world = lambda env: world
    sys.modules["robolab.core.world.world_state"] = m


def test_moving_object_is_not_at_rest_until_the_streak_is_full():
    C.PLACEMENT_REST_S = 0.2                       # 2 ticks at dt=0.1
    env = _Env(); speeds = {"banana": torch.tensor([0.5, 0.0])}; _install(_World(speeds))
    env.episode_length_buf[:] = 2
    env.common_step_counter = 2
    assert R.at_rest(env, "banana").tolist() == [False, False]     # env1 has 1 tick so far
    env.common_step_counter = 3
    assert R.at_rest(env, "banana").tolist() == [False, True]      # env1 now has 2
    speeds["banana"][0] = 0.0
    for step in (4, 5):
        env.common_step_counter = step; out = R.at_rest(env, "banana")
    assert out.tolist() == [True, True]


def test_one_update_per_step_regardless_of_how_often_it_is_asked():
    C.PLACEMENT_REST_S = 0.2
    env = _Env(1); speeds = {"cube": torch.tensor([0.0])}; _install(_World(speeds))
    env.episode_length_buf[:] = 2; env.common_step_counter = 7
    for _ in range(5):
        R.at_rest(env, "cube")
    assert int(R.get_rest_tracker(env).streak["cube"][0]) == 1


def test_disabled_gate_is_transparent():
    C.PLACEMENT_REST_S = 0.0
    env = _Env(1); _install(_World({"cube": torch.tensor([9.9])}))
    assert R.at_rest(env, "cube", env_id=0) is True
    C.PLACEMENT_REST_S = 0.2
