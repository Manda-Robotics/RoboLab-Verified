# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""A scripted client for the bimanual rigs.

There is no released checkpoint that drives two arms: pi05 is a single-arm DROID
policy emitting 8 numbers, and the bimanual action spaces want 16 (joint position)
or 14 (relative IK). Until such a policy exists, the rigs could not be *run* at all,
so nothing about them was verifiable at runtime — the registrations, the per-arm EE
channels, the wrist cameras and the event pipeline were all offline-only claims.

This client closes that gap without pretending to be a policy. It emits a slow,
deterministic motion around whatever pose the robot is already in, so the episode
exercises the full stack: reset, both arms actuated, both wrist cameras rendered,
per-arm metrics recorded, events emitted, HDF5 and video written.

It is a **smoke test, not an evaluation.** Success rates from it are meaningless and
the runner refuses to present them as anything else.
"""
from __future__ import annotations

import math

import numpy as np

from robolab.eval.base_client import InferenceClient

# Both rigs use 16 joint-space numbers, but they are cut up differently:
#   dual Franka  [left arm 7, left gripper 1, right arm 7, right gripper 1]
#   ALOHA/ViperX [left arm 6, left fingers 2, right arm 6, right fingers 2]
# so the segmentation is read off the observation rather than assumed. Getting this
# wrong is silent: an arm joint lands in a finger slot and the arm quietly bends.
JOINTPOS_DIM = 16
RELIK_DIM = 14


class ScriptedBimanualClient(InferenceClient):
    """Deterministic two-arm motion. No network, no checkpoint, no randomness.

    ``amplitude_rad`` is deliberately small. The point is to prove the stack turns,
    not to fling the arms: a large excursion would trip the collision and off-table
    flags and make the smoke test look like a failing evaluation.
    """

    open_loop_horizon = 1

    def __init__(self, action_space: str = "jointpos", amplitude_rad: float = 0.12,
                 period_s: float = 6.0, control_hz: float = 15.0,
                 finger_travel_m: float | None = None) -> None:
        super().__init__()
        if action_space not in ("jointpos", "rel_ik"):
            raise ValueError(f"action_space must be 'jointpos' or 'rel_ik', got {action_space!r}")
        self.action_space = action_space
        self.amplitude_rad = float(amplitude_rad)
        self.period_steps = max(1.0, float(period_s) * float(control_hz))
        # Rigs whose finger joints are commanded in metres from a single openness
        # observation (bimanual YAM: 0 closed .. -finger_travel_m open) get the grip
        # signal mapped onto both finger slots; None keeps the Franka/ALOHA behaviour.
        self.finger_travel_m = finger_travel_m
        self._t: dict[int, int] = {}

    def begin_episode(self, episode_idx: int) -> None:
        super().begin_episode(episode_idx)
        self._t.clear()

    # The base class's four hooks describe a query-then-step-chunk flow against an
    # inference server. There is no server here, so `infer` is overridden whole (the
    # base documents this as the third override level) and these are unreachable.
    # They exist because InferenceClient is an ABC, and they say why rather than
    # silently returning something wrong.
    def _unreachable(self, hook: str):
        raise NotImplementedError(
            f"{type(self).__name__} overrides infer() entirely and never calls {hook}(); "
            "there is no inference server behind this client.")

    def _extract_observation(self, raw_obs, *, env_id: int = 0):
        self._unreachable("_extract_observation")

    def _pack_request(self, extracted_obs, instruction: str):
        self._unreachable("_pack_request")

    def _query_server(self, request):
        self._unreachable("_query_server")

    def _unpack_response(self, response):
        self._unreachable("_unpack_response")

    @staticmethod
    def _term(raw_obs: dict, key: str, env_id: int) -> np.ndarray:
        v = raw_obs["proprio_obs"][key][env_id]
        return np.asarray(v.detach().cpu().numpy(), dtype=np.float32).reshape(-1)

    def infer(self, obs, instruction: str, *, env_id: int = 0) -> dict:
        """Return one action. Overridden whole: there is no server to query."""
        t = self._t.get(env_id, 0)
        self._t[env_id] = t + 1
        phase = 2.0 * math.pi * (t / self.period_steps)
        wobble = self.amplitude_rad * math.sin(phase)
        # Grippers close on the second half of the cycle, so a run exercises both the
        # grasp detector's closing edge and its release.
        grip = 1.0 if math.sin(phase) < 0 else 0.0

        if self.action_space == "rel_ik":
            # 14 = [left dpos 3, left drot 3, left gripper 1, right ...]; deltas, so a
            # zero-centred wobble is safe without reading the current pose.
            action = np.zeros(RELIK_DIM, dtype=np.float32)
            action[2] = wobble * 0.1          # left  ee z
            action[9] = -wobble * 0.1         # right ee z, mirrored
            action[6] = action[13] = grip
            return {"action": action, "viz": None}

        # Joint position is ABSOLUTE, so the action must be built from where the arm
        # already is. Emitting zeros here would command every joint to 0 rad and slam
        # the arms through the table on the first step.
        left = self._term(obs, "left_arm_joint_pos", env_id)
        right = self._term(obs, "right_arm_joint_pos", env_id)
        n_arm = len(left)
        n_fing = (JOINTPOS_DIM - 2 * n_arm) // 2
        if len(right) != n_arm or n_fing < 1:
            raise ValueError(
                f"cannot lay out a {JOINTPOS_DIM}-dim action from arms of "
                f"{n_arm} and {len(right)} joints")

        action = np.zeros(JOINTPOS_DIM, dtype=np.float32)
        action[0:n_arm] = left
        action[n_arm + n_fing:2 * n_arm + n_fing] = right
        for side, base in (("left", n_arm), ("right", 2 * n_arm + n_fing)):
            if n_fing == 1:
                # One binary 0..1 gripper channel (dual Franka).
                action[base] = grip
            elif self.finger_travel_m is not None:
                # Two finger joints in metres driven from one grip signal (bimanual YAM):
                # grip 1 = closed -> 0 m, grip 0 = open -> -travel.
                action[base:base + n_fing] = -self.finger_travel_m * (1.0 - grip)
            else:
                # Finger joints commanded in metres (ALOHA). A 0/1 here would be a
                # metre of travel, so hold them where they are and let the arms move.
                action[base:base + n_fing] = self._term(obs, f"{side}_gripper_pos", env_id)[:n_fing]

        # Wobble one elbow joint per arm — enough motion to move the wrist cameras and
        # the EE channels, far from any joint limit.
        elbow = min(3, n_arm - 1)
        action[elbow] += wobble
        action[n_arm + n_fing + elbow] += wobble
        return {"action": action, "viz": None}
