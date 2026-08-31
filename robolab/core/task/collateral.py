# SPDX-License-Identifier: Apache-2.0
"""Collateral placement: a non-target object ends up in a goal container.

Success has no exclusivity requirement — "put X in Y" scores 1.0 even if the
whole table is swept into Y (findings.md A1: BBQSauceInBin env 3 succeeded
with the mug, the mustard and the ranch also in the bin; B7: a deliberate
wrong-object placement only ever showed up as OBJECT_BUMPED). Scores are left
alone; this module raises one flag per object (The reviewer, 2026-08-25):

* ``WRONG_OBJECT_PLACED``    — a non-target the hand had held is released inside
                               a goal container;
* ``WRONG_OBJECT_PUSHED_IN`` — a non-target enters a goal container without
                               having been held (knocked / pushed in).

Objects already inside a goal container when the episode starts never count
(the warm-up window is used to learn who was inside at reset), and each object
flags at most once per episode. ``collateral_placed`` (distinct non-targets
that entered a goal container) is published for the results row.

Pure bookkeeping over boolean tensors so it can be unit-tested without a
simulator; the event tracker feeds it the predicate results.
"""
from __future__ import annotations

import torch


class CollateralTracker:
    def __init__(self, num_envs: int, device):
        self.n = num_envs
        self.device = device
        self._inside_prev: dict[tuple[str, str], torch.Tensor] = {}
        self._inside_at_reset: dict[tuple[str, str], torch.Tensor] = {}
        self._flagged: dict[str, torch.Tensor] = {}          # object -> (N,) flagged this episode
        self.collateral: dict[int, set[str]] = {}             # env -> objects that entered a goal container

    def _z(self):
        return torch.zeros(self.n, dtype=torch.bool, device=self.device)

    def reset_envs(self, env_ids) -> None:
        for i in list(env_ids):
            i = int(i)
            for d in (self._inside_prev, self._inside_at_reset):
                for t in d.values():
                    t[i] = False
            for t in self._flagged.values():
                t[i] = False
            self.collateral.pop(i, None)

    def update(self, inside: dict[tuple[str, str], torch.Tensor], warm: torch.Tensor,
               recently_held: dict[str, torch.Tensor]) -> list[tuple[int, str, str, str]]:
        """``inside[(obj, container)]`` (N,) bool for this step; ``warm`` (N,) bool =
        still in the reset warm-up; ``recently_held[obj]`` (N,) bool. Returns
        ``(env_id, obj, container, 'placed' | 'pushed')`` for new entries."""
        events: list[tuple[int, str, str, str]] = []
        for key, now in inside.items():
            obj, cont = key
            now = now.bool()
            at_reset = self._inside_at_reset.setdefault(key, self._z())
            # anything inside during the warm-up counts as "was there at reset"
            at_reset |= now & warm
            prev = self._inside_prev.get(key, self._z())
            flagged = self._flagged.setdefault(obj, self._z())
            entered = now & ~prev & ~warm & ~at_reset & ~flagged
            for eid in entered.nonzero(as_tuple=False).flatten().tolist():
                held = recently_held.get(obj)
                kind = "placed" if (held is not None and bool(held[eid])) else "pushed"
                events.append((eid, obj, cont, kind))
                flagged[eid] = True
                self.collateral.setdefault(eid, set()).add(obj)
            self._inside_prev[key] = now.clone()
        return events
