# SPDX-License-Identifier: Apache-2.0
"""Confirmed success: the success predicate must hold continuously for a while,
with the task's target objects at rest, before the episode is scored.

Upstream ends the episode on the FIRST frame the success predicate holds. On the
Verified corpus every recorded success ended with an object still moving (median
8.9 cm/s at the final frame, 44 % above 10 cm/s, worst 1 m/s) — an object still
falling into the bin, still sliding on the plate, still attached to a finger — so
a human reviewer could not tell whether the placement would have survived
(VERIFIED_PLAN A2, F§3, H-R5-11, H-R6-7, H-R7-6, H-R8-16).

``confirmed_success_term`` wraps a task's ``success`` DoneTerm: per env it counts
consecutive steps on which the raw predicate holds, and reports success only once
that streak reaches ``hold_s`` **and** every target object named by the
predicate's parameters is slower than ``max_speed``. The policy keeps acting
during the hold — knocking the object back out within the hold window is a
failure to place, not a success. Nothing else changes: timeouts, subtask credit
and events are as before; the subtask ladder still marks the stage at the first
hold (so the dashboard shows both moments).
"""
from __future__ import annotations

import functools
import inspect
from typing import Callable

import torch

TARGET_KWARGS = ("object", "objects", "groups")


def _target_names(params: dict) -> list[str]:
    out: list[str] = []

    def add(v):
        if isinstance(v, str):
            out.append(v)
        elif isinstance(v, dict):
            for k in ("object", "objects"):
                add(v.get(k))
        elif isinstance(v, (list, tuple)):
            for x in v:
                add(x)

    for k in TARGET_KWARGS:
        add(params.get(k))
    return list(dict.fromkeys(out))


class SuccessConfirmer:
    """Stateful per-env hold counter for one success predicate (one per task env cfg)."""

    def __init__(self, func: Callable, hold_s: float, max_speed: float, targets: list[str]):
        self.func = func
        self.hold_s = float(hold_s)
        self.max_speed = float(max_speed)
        self.targets = targets
        self.count: torch.Tensor | None = None
        self.first_hold_step: torch.Tensor | None = None   # -1 until the raw predicate first holds
        self.confirmed_step: torch.Tensor | None = None

    def _init(self, env):
        n = env.num_envs
        self.count = torch.zeros(n, dtype=torch.long, device=env.device)
        self.first_hold_step = torch.full((n,), -1, dtype=torch.long, device=env.device)
        self.confirmed_step = torch.full((n,), -1, dtype=torch.long, device=env.device)

    def hold_steps(self, env) -> int:
        dt = float(getattr(env, "step_dt", 0.0) or 0.0)
        if dt <= 0:
            return 1
        return max(1, int(round(self.hold_s / dt)))

    def targets_at_rest(self, env) -> torch.Tensor:
        if not self.targets:
            return torch.ones(env.num_envs, dtype=torch.bool, device=env.device)
        from robolab.core.world.world_state import get_world
        world = get_world(env)
        ok = torch.ones(env.num_envs, dtype=torch.bool, device=env.device)
        for name in self.targets:
            try:
                v = world.get_velocity(name, env_id=None)          # (N, 6)
            except Exception:
                continue                                          # not a rigid body → no constraint
            ok &= torch.linalg.norm(v[:, :3], dim=-1) < self.max_speed
        return ok

    def __call__(self, env, **params) -> torch.Tensor:
        raw = self.func(env, **params)
        if not isinstance(raw, torch.Tensor):
            raw = torch.as_tensor(raw, dtype=torch.bool, device=env.device).reshape(-1)
        raw = raw.bool()
        if self.count is None or self.count.shape[0] != raw.shape[0]:
            self._init(env)
        step = env.episode_length_buf
        fresh = step <= 1                                        # env was just reset
        self.count = torch.where(fresh, torch.zeros_like(self.count), self.count)
        self.first_hold_step = torch.where(fresh, torch.full_like(self.first_hold_step, -1), self.first_hold_step)
        self.confirmed_step = torch.where(fresh, torch.full_like(self.confirmed_step, -1), self.confirmed_step)

        self.count = torch.where(raw, self.count + 1, torch.zeros_like(self.count))
        newly = raw & (self.first_hold_step < 0)
        self.first_hold_step = torch.where(newly, step, self.first_hold_step)

        confirmed = raw & (self.count >= self.hold_steps(env)) & self.targets_at_rest(env)
        self.confirmed_step = torch.where(confirmed & (self.confirmed_step < 0), step, self.confirmed_step)
        return confirmed


def confirmed_success_term(term, hold_s: float, max_speed: float):
    """Return ``term`` (an Isaac Lab ``DoneTerm``) with its ``func`` wrapped by a
    :class:`SuccessConfirmer`. The wrapper keeps the original signature so the
    termination manager's parameter validation sees the real predicate."""
    if term is None or hold_s <= 0:
        return term
    confirmer = SuccessConfirmer(term.func, hold_s, max_speed, _target_names(dict(term.params or {})))

    @functools.wraps(term.func)
    def wrapped(env, **params):
        return confirmer(env, **params)

    try:
        wrapped.__signature__ = inspect.signature(term.func)
    except (TypeError, ValueError):
        pass
    wrapped.confirmer = confirmer
    term.func = wrapped
    return term
