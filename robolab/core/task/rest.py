# SPDX-License-Identifier: Apache-2.0
"""Object settling for placement predicates.

P128 uses a contiguous sampled pose window: every pair of positions is within
2 mm and every pair of orientations within 2 degrees for 0.2 s by default.
This tolerates contact-solver velocity jitter without accepting sampled bounces
or motion exceeding those limits. It cannot observe motion between samples.
The P46 velocity streak is retained for explicit legacy comparisons.
State is per environment, sampled once per object/simulation step, with reset
and missing-step invalidation in pose mode.
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
    """True where the selected rest metric holds for ``PLACEMENT_REST_S``.
    Pose mode bounds the full sampled pose window; velocity mode is retained
    for legacy comparisons. ``PLACEMENT_REST_S = 0`` disables the gate
    (upstream behaviour: credit on the first frame)."""
    s = float(getattr(robolab.constants, "PLACEMENT_REST_S", 0.0) or 0.0) if seconds is None else seconds
    if s <= 0:
        return True if env_id is not None else torch.ones(env.num_envs, dtype=torch.bool, device=env.device)
    if robolab.constants.REST_METRIC == "pose":
        ok = pose_at_rest(env, obj, s)
    else:
        streak = get_rest_tracker(env).update(obj)
        ok = streak >= rest_steps(env, s)
    return bool(ok[env_id]) if env_id is not None else ok


class PoseRestTracker:
    """Contiguous pose window, sampled once per simulation step and reset per env.

    Pairwise position/rotation diameters reject return-to-start motion within
    the sampled window. This cannot observe motion between simulator samples.
    """
    def __init__(self, env):
        self.env = env
        self.history = {}
        self.last_step = {}
        self.valid = {}

    def stable(self, obj: str, seconds: float) -> torch.Tensor:
        import math
        from collections import deque
        from robolab.core.world.world_state import get_world
        env = self.env
        dt = float(env.step_dt)
        if dt <= 0:
            raise ValueError("Pose rest requires a positive simulation step_dt")
        samples = max(2, int(math.ceil(seconds / dt)) + 1)
        counter = getattr(env, "common_step_counter", None)
        step = int(counter) if counter is not None else int(env.episode_length_buf.max().item())
        key = (obj, samples)
        h = self.history.setdefault(key, deque(maxlen=samples))
        prev = self.last_step.get(key)
        if prev != step:
            if prev is not None and step != prev + 1:
                h.clear()
                self.valid.pop(key, None)
            pos, quat = get_world(env).get_pose(obj, env_id=None)
            pos, quat = pos.clone(), quat.clone()
            count = self.valid.get(key, torch.zeros(env.num_envs, dtype=torch.long, device=env.device))
            count = torch.where(env.episode_length_buf <= 1, torch.ones_like(count), count + 1)
            self.valid[key] = count.clamp(max=samples)
            h.append((pos, quat))
            self.last_step[key] = step
        if len(h) < samples:
            return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
        p = torch.stack([x[0] for x in h])
        q = torch.stack([x[1] for x in h])
        norm = torch.linalg.norm(q, dim=-1, keepdim=True)
        finite = torch.isfinite(p).all(dim=(0, 2)) & torch.isfinite(q).all(dim=(0, 2)) & (norm[..., 0] > 1e-8).all(dim=0)
        q = q / norm.clamp_min(1e-8)
        span = torch.linalg.norm(p[:, None] - p[None, :], dim=-1).amax(dim=(0, 1))
        dots = (q[:, None] * q[None, :]).sum(dim=-1).abs().clamp(max=1)
        limit = math.cos(math.radians(robolab.constants.REST_MAX_ANGLE_DEG) / 2)
        return (self.valid[key] >= samples) & finite & (span <= robolab.constants.REST_MAX_DISPLACEMENT_M) & (dots.amin(dim=(0, 1)) >= limit)


_POSE_TRACKERS: dict[int, PoseRestTracker] = {}


def pose_at_rest(env, obj: str, seconds: float):
    tracker = _POSE_TRACKERS.get(id(env))
    if tracker is None or tracker.env is not env:
        tracker = _POSE_TRACKERS[id(env)] = PoseRestTracker(env)
    return tracker.stable(obj, seconds)
