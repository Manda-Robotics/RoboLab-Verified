"""Newton-only DROID gripper action; PhysX keeps its original binary action."""

import math

import torch
from isaaclab.utils import configclass

from robolab.core.utils.isaaclab_compat import as_torch
from robolab.robots.droid import BinaryJointPositionZeroToOneAction, BinaryJointPositionZeroToOneActionCfg


class RateLimitedGripperAction(BinaryJointPositionZeroToOneAction):
    """Retain binary commands, but slew joint targets at the physics update rate."""

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        if not math.isfinite(cfg.max_target_velocity) or cfg.max_target_velocity <= 0:
            raise ValueError("max_target_velocity must be finite and positive")
        self._limited_target = torch.zeros_like(self._processed_actions)
        self._needs_sync = torch.ones(self.num_envs, dtype=torch.bool, device=self.device)
        self._max_delta = cfg.max_target_velocity * env.cfg.sim.dt
        # Lab release/3.0 stores the action's joint IDs as a Warp array. Keep
        # that array for simulator writes and a Torch view for tensor indexing.
        self._read_joint_ids = self._joint_ids if isinstance(self._joint_ids, (list, tuple, slice)) else (
            as_torch(self._joint_ids).to(dtype=torch.long))

    def apply_actions(self):
        # Read the reset pose lazily: action reset may run before scene reset.
        if self._needs_sync.any():
            measured = as_torch(self._asset.data.joint_pos)[:, self._read_joint_ids]
            self._limited_target[self._needs_sync] = measured[self._needs_sync]
            self._needs_sync[:] = False
        delta = (self._processed_actions - self._limited_target).clamp(-self._max_delta, self._max_delta)
        self._limited_target.add_(delta)
        self._asset.set_joint_position_target_index(target=self._limited_target, joint_ids=self._joint_ids)

    def reset(self, env_ids=None):
        super().reset(env_ids)
        ids = slice(None) if env_ids is None else env_ids
        self._needs_sync[ids] = True
        self._processed_actions[ids] = self._open_command


@configclass
class RateLimitedGripperActionCfg(BinaryJointPositionZeroToOneActionCfg):
    class_type = RateLimitedGripperAction
    max_target_velocity: float = math.pi / 2  # 45 degrees in 0.5 seconds
