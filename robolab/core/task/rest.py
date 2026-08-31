# SPDX-License-Identifier: Apache-2.0
"""Has this object come to rest?

P30 gates *success* on the target being at rest; the subtask **ladder** still
credited a placement on the first frame its predicate held, so an object that
passed through a bowl and bounced onto the rim scored 1.0 anyway (BananaInBowl
v_ run env 3, 2026-08-26: released at 20.87 s, "Completed subtask" at 20.93 s,
banana ends 8 cm off centre on the rim). Placement predicates therefore ask the
same question the success gate does: has the object been slower than
``max_speed`` for ``seconds`` in a row?

State is per env instance and updated at most once per (object, sim step), so
the many predicate evaluations inside one step share one streak.
"""
from __future__ import annotations

import torch

import robolab.constants


class RestTracker:
    def __init__(self, env):
        self.env = env
        self.streak: dict[str, torch.Tensor] = {}
        self.last_step: dict[str, int] = {}

    def _step(self) -> int:
        c = getattr(self.env, "common_step_counter", None)
        if c is not None:
            return int(c)
        return int(self.env.episode_length_buf.max().item())

    def update(self, obj: str) -> torch.Tensor:
        step = self._step()
        if self.last_step.get(obj) == step and obj in self.streak:
            return self.streak[obj]
        from robolab.core.world.world_state import get_world
        n = self.env.num_envs
        streak = self.streak.get(obj)
        if streak is None or streak.shape[0] != n:
            streak = torch.zeros(n, dtype=torch.long, device=self.env.device)
        try:
            v = get_world(self.env).get_velocity(obj, env_id=None)
            slow = torch.linalg.norm(v[:, :3], dim=-1) < float(getattr(robolab.constants, "SUCCESS_MAX_SPEED", 0.02))
        except Exception:
            slow = torch.ones(n, dtype=torch.bool, device=self.env.device)   # unknown → do not block
        fresh = self.env.episode_length_buf <= 1
        streak = torch.where(slow, streak + 1, torch.zeros_like(streak))
        streak = torch.where(fresh, torch.zeros_like(streak), streak)
        self.streak[obj] = streak
        self.last_step[obj] = step
        return streak


_TRACKERS: dict[int, RestTracker] = {}


def get_rest_tracker(env) -> RestTracker:
    t = _TRACKERS.get(id(env))
    if t is None or t.env is not env:
        t = _TRACKERS[id(env)] = RestTracker(env)
    return t


def rest_steps(env, seconds: float) -> int:
    dt = float(getattr(env, "step_dt", 0.0) or 0.0)
    return max(1, int(round(seconds / dt))) if dt > 0 else 1


def at_rest(env, obj: str, env_id: int | None = None, seconds: float | None = None):
    """True where ``obj`` has been slower than ``SUCCESS_MAX_SPEED`` for
    ``PLACEMENT_REST_S`` in a row. ``PLACEMENT_REST_S = 0`` disables the gate
    (upstream behaviour: credit on the first frame)."""
    s = float(getattr(robolab.constants, "PLACEMENT_REST_S", 0.0) or 0.0) if seconds is None else seconds
    if s <= 0:
        return True if env_id is not None else torch.ones(env.num_envs, dtype=torch.bool, device=env.device)
    streak = get_rest_tracker(env).update(obj)
    ok = streak >= rest_steps(env, s)
    return bool(ok[env_id]) if env_id is not None else ok
