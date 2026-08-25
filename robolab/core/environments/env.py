# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""RoboLab environment runtime class.

This module contains the RobolabEnv class which extends ManagerBasedRLEnv
with custom recorder manager support and eval-specific behavior:
- Terminated envs are frozen (no auto-reset) so they hold their final state
- Per-env success/failure tracking
- Per-env recording export on termination
"""

import logging

import torch
from isaaclab.envs import ManagerBasedRLEnv

from robolab.core.logging.recorder_manager import RobolabRecorderManager
from robolab.core.world.world_state import get_world

logger = logging.getLogger(__name__)

# A termination within the first 2 steps is re-reset this many times before
# the episode is recorded as `pre_satisfied` (VERIFIED_PLAN H-B12).
MAX_EARLY_RESETS = 3


class RobolabEnv(ManagerBasedRLEnv):
    """Environment for RoboLab evaluation.

    Extends ManagerBasedRLEnv with:
    - Custom recorder manager (RobolabRecorderManager)
    - Frozen terminated envs: when an env terminates, it holds its final state
      instead of auto-resetting. Actions for frozen envs are zeroed out.
    - Per-env result tracking (success/truncated, termination step)
    """

    def __init__(self, cfg, **kwargs):
        super().__init__(cfg, **kwargs)
        self._frozen_envs = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._pre_step_frozen = self._frozen_envs.clone()  # snapshot before each step()
        self._env_results: dict[int, bool] = {}      # env_id -> True (success) / False (truncated)
        self._env_term_step: dict[int, int] = {}      # env_id -> episode step when terminated
        self._has_stepped = False                     # tracks whether step() has been called
        self._early_resets: dict[int, int] = {}       # env_id -> re-resets after a <=2-step termination

    def load_managers(self):
        """Load managers; replace upstream RecorderManager with the streaming-
        capable RobolabRecorderManager.

        super().load_managers() builds an upstream RecorderManager whose
        constructor eagerly creates the HDF5 file at ``cfg.dataset_filename``
        (default ``data.hdf5``) — we don't want that, because we replace the
        manager below with RobolabRecorderManager (which opens the real
        per-run file lazily via ``set_hdf5_file("run_N.hdf5")``). To suppress
        the eager file creation, temporarily set ``cfg.recorders`` to None
        so upstream's RecorderManager.__init__ takes its
        ``if not cfg: return`` early-exit path; then restore cfg before
        instantiating our manager.
        """
        recorders_cfg = self.cfg.recorders
        self.cfg.recorders = None
        try:
            super().load_managers()
        finally:
            self.cfg.recorders = recorders_cfg

        self.recorder_manager = RobolabRecorderManager(self.cfg.recorders, self)

    def step(self, action):
        """Step the environment. Zero out actions for frozen (terminated) envs."""
        self._has_stepped = True
        # Snapshot frozen state before step so recorder can detect newly-frozen envs
        self._pre_step_frozen = self._frozen_envs.clone()
        if self._frozen_envs.any():
            action = action.clone()
            action[self._frozen_envs] = 0.0
        return super().step(action)

    def _reset_idx(self, env_ids):
        """Override to freeze terminated envs instead of resetting them.

        On initial reset (before any stepping), all envs are reset normally.
        During stepping, when an env terminates:
        1. Mark it as frozen
        2. Record success/failure and termination step
        3. Export its recording data
        4. Skip the actual reset (env holds its final state)
        """
        if not self._has_stepped:
            # Initial reset — let all envs reset normally
            super()._reset_idx(env_ids)
            get_world(self).reset_predicate_state(env_ids)
            return

        # During stepping — freeze newly terminated envs
        for eid in env_ids.tolist():
            if not self._frozen_envs[eid]:
                ep_len = int(self.episode_length_buf[eid].item())
                if ep_len <= 2 and self._early_resets.get(eid, 0) < MAX_EARLY_RESETS:
                    # Terminated before the robot could act — usually a physics
                    # artifact at reset, sometimes a predicate that is already true
                    # in the authored scene. Reset this env for a clean start, but
                    # count it and give up after MAX_EARLY_RESETS so a pre-satisfied
                    # task is recorded instead of silently re-run forever.
                    n = self._early_resets[eid] = self._early_resets.get(eid, 0) + 1
                    logger.warning(
                        "env%d: terminated at step %d before the robot could act (early reset %d/%d)",
                        eid, ep_len, n, MAX_EARLY_RESETS,
                    )
                    artifact_ids = torch.tensor([eid], device=self.device, dtype=env_ids.dtype)
                    super()._reset_idx(artifact_ids)
                    get_world(self).reset_predicate_state(artifact_ids)
                    continue
                self._frozen_envs[eid] = True
                self._env_results[eid] = bool(self.termination_manager.terminated[eid])
                self._env_term_step[eid] = ep_len
                # Auto-export recording for this env
                if self.recorder_manager is not None:
                    try:
                        self.recorder_manager.export_episodes(env_ids=[eid])
                    except Exception:
                        logger.exception(
                            "Failed to export recording for env_id=%d at step=%d; episode data may be incomplete.",
                            eid, ep_len,
                        )

        # Only reset non-frozen envs (typically none in eval)
        mask = ~self._frozen_envs[env_ids]
        active_ids = env_ids[mask]
        if len(active_ids) > 0:
            super()._reset_idx(active_ids)
            get_world(self).reset_predicate_state(active_ids)

    @property
    def all_terminated(self) -> bool:
        """True when all envs have terminated."""
        return self._frozen_envs.all().item()

    @property
    def active_env_ids(self) -> list[int]:
        """List of env_ids that are still running."""
        return (~self._frozen_envs).nonzero(as_tuple=False).squeeze(-1).tolist()

    def get_env_results(self) -> list[dict]:
        """Get per-env results after termination."""
        results = []
        for eid in range(self.num_envs):
            step = self._env_term_step.get(eid)
            results.append({
                'env_id': eid,
                'success': self._env_results.get(eid),
                'step': step,
                # How often this env terminated within 2 steps and was re-reset
                # before this episode; `pre_satisfied` when it still did after
                # the last allowed re-reset (the predicate holds at reset).
                'early_resets': self._early_resets.get(eid, 0),
                'pre_satisfied': bool(step is not None and step <= 2),
                # A2 confirmed success: step the raw predicate first held / was confirmed
                'success_first_hold_step': self._confirm_step(eid, 'first_hold_step'),
                'success_confirmed_step': self._confirm_step(eid, 'confirmed_step'),
            })
        return results

    def _confirm_step(self, eid: int, which: str):
        """Per-env step from the success confirmer (None when not wrapped / not held)."""
        try:
            term = self.cfg.terminations.success
            conf = getattr(term.func, "confirmer", None)
            t = getattr(conf, which, None)
            if t is None:
                return None
            v = int(t[eid].item())
            return v if v >= 0 else None
        except Exception:
            return None

    def reset_eval_state(self):
        """Reset frozen state for next episode batch."""
        self._frozen_envs[:] = False
        self._pre_step_frozen[:] = False
        self._env_results.clear()
        self._env_term_step.clear()
        self._early_resets.clear()
        self._has_stepped = False
