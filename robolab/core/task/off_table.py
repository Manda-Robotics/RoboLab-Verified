# SPDX-License-Identifier: Apache-2.0
"""Objects that leave the table, and the moment a task can no longer succeed.

Upstream only watched *non-target* objects for large displacements, so the
target can itself go over the edge without a single event (VERIFIED_PLAN
H-B17, H-R7-2 "it actually dropped the can off the table — that would have been
an interesting flag", H-R8-15/35). Rules (Finn, 2026-08-25):

* any object that falls ``OFF_TABLE_DROP_M`` below its starting height gets one
  ``OBJECT_FELL_OFF_TABLE`` event — a flag, not a verdict;
* the episode ends as a failure (``TARGET_LOST``) only when the success
  condition can no longer be met: for an ``all`` group every listed object is
  needed, for ``any`` one of them, for ``choose K`` at least K — so "put two of
  the four bananas in the crate" survives one banana on the floor.
"""
from __future__ import annotations

import torch

OFF_TABLE_DROP_M = 0.15


def _names(v) -> list[str]:
    if isinstance(v, str):
        return [v]
    if isinstance(v, (list, tuple)):
        out = []
        for x in v:
            out.extend(_names(x))
        return out
    return []


def required_groups(params: dict) -> list[tuple[list[str], str, int]]:
    """Object groups the success term needs, as ``(objects, logical, K)``.

    ``object_groups_in_containers`` carries one dict per group; every other
    predicate names its objects via ``object`` / ``objects`` with an optional
    ``logical`` / ``K``."""
    params = params or {}
    if isinstance(params.get("groups"), (list, tuple)):
        out = []
        for g in params["groups"]:
            if isinstance(g, dict):
                objs = _names(g.get("object")) + _names(g.get("objects"))
                if objs:
                    out.append((objs, str(g.get("logical", "all")), int(g.get("K", g.get("k", 1)) or 1)))
        return out
    objs = _names(params.get("object")) + _names(params.get("objects"))
    if not objs:
        return []
    return [(objs, str(params.get("logical", "all")), int(params.get("K", params.get("k", 1)) or 1))]


def group_lost(fallen: dict[str, torch.Tensor], objs: list[str], logical: str, K: int, n: int, device) -> torch.Tensor:
    """(N,) bool: this group can no longer be satisfied given per-object fallen masks."""
    cols = [fallen.get(o, torch.zeros(n, dtype=torch.bool, device=device)) for o in objs]
    if not cols:
        return torch.zeros(n, dtype=torch.bool, device=device)
    f = torch.stack(cols, dim=0)                      # (num_objs, N)
    if logical == "all":
        return f.any(dim=0)
    if logical == "any":
        return f.all(dim=0)
    # choose K: lost once fewer than K objects remain
    return (len(objs) - f.sum(dim=0)) < max(1, K)


def task_lost(fallen: dict[str, torch.Tensor], groups, n: int, device) -> torch.Tensor:
    lost = torch.zeros(n, dtype=torch.bool, device=device)
    for objs, logical, K in groups:
        lost |= group_lost(fallen, objs, logical, K, n, device)
    return lost


class OffTableMonitor:
    """Per-env bookkeeping: starting heights, who has fallen, who was flagged."""

    def __init__(self, env):
        self.env = env
        self.initial_z: dict[str, torch.Tensor] = {}
        self.flagged: dict[str, torch.Tensor] = {}
        self.lost_flagged = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)

    def fallen_masks(self, objects) -> dict[str, torch.Tensor]:
        from robolab.core.world.world_state import get_world
        world = get_world(self.env)
        fresh = self.env.episode_length_buf <= 1
        out = {}
        for o in objects:
            try:
                z = world.get_pose(o, env_id=None)[0][:, 2]
            except Exception:
                continue
            z0 = self.initial_z.get(o)
            if z0 is None:
                z0 = self.initial_z[o] = z.clone()
            else:
                z0 = torch.where(fresh, z, z0)
                self.initial_z[o] = z0
            if bool(fresh.any()):
                if o in self.flagged:
                    self.flagged[o][fresh] = False
                self.lost_flagged[fresh] = False
            out[o] = z < z0 - OFF_TABLE_DROP_M
        return out


_MONITORS = {}


def get_monitor(env) -> OffTableMonitor:
    m = _MONITORS.get(id(env))
    if m is None or m.env is not env:
        m = _MONITORS[id(env)] = OffTableMonitor(env)
    return m


def targets_lost(env, groups=None) -> torch.Tensor:
    """Isaac Lab termination term: the success condition can no longer be met."""
    if not groups:
        return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    mon = get_monitor(env)
    objs = sorted({o for g in groups for o in g[0]})
    return task_lost(mon.fallen_masks(objs), groups, env.num_envs, env.device)
