# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Run a pointing-capable VLM as a RoboLab policy (adoption unit A1, was "P36").

The model returns an image point and a phase; a geometric controller turns that into
metric motion. The same Gemini scored 0/4 driven as Cartesian deltas and 6/6 through
this path on BananaInBowl.

**Connector only.** The controller lives in its own package, ``vlm-pinpoint``
(Manda-Robotics/vlm-pinpoint) so it is reusable outside RoboLab and RoboLab does not
vendor a copy. This file does one job: translate RoboLab's nested, batched observation
into the harness's flat per-env observation, delegate, and hand the action chunk back
unchanged.

The import of ``vlm_pinpoint`` is deliberately lazy. RoboLab Verified must import,
test and run with the package absent -- it is an optional extra, not a dependency of
the benchmark.
"""
from __future__ import annotations

import numpy as np

# Franka flange origin to the fingertip, along the gripper's local +z. The controller
# aims the fingertip, not the flange; without this the arm stops a hand's width short.
FLANGE_TO_FINGERTIP_M = 0.1034


def quat_rotate(quat_wxyz, vec) -> np.ndarray:
    """Rotate ``vec`` by a (w, x, y, z) quaternion. Pure numpy so it is testable
    without Isaac, which is the only part of this connector worth unit-testing."""
    q = np.asarray(quat_wxyz, dtype=float)
    v = np.asarray(vec, dtype=float)
    w, xyz = q[0], q[1:]
    t = 2.0 * np.cross(xyz, v)
    return v + w * t + np.cross(xyz, t)


def fingertip_position(ee_pos, ee_quat_wxyz) -> np.ndarray:
    """Where the fingers actually are, given the recorded flange pose."""
    return np.asarray(ee_pos, dtype=float) + quat_rotate(
        ee_quat_wxyz, (0.0, 0.0, FLANGE_TO_FINGERTIP_M)
    )


def _first(x):
    """RoboLab hands observations back batched; the harness wants one env."""
    a = np.asarray(x)
    return a[0] if a.ndim > 1 or (a.ndim == 1 and a.shape[0] == 1 and a.dtype == object) else a


def to_harness_observation(obs: dict, env_id: int = 0) -> dict:
    """RoboLab's nested batched obs -> the harness's flat per-env observation.

    Keys the controller expects: scene, depth, K, cam_pos, cam_quat, tip_pos, ee_quat,
    gripper_pos. Anything missing is passed through as None so the controller can say
    what it needs rather than failing on a KeyError deep inside.
    """
    def pick(*path):
        node = obs
        for key in path:
            if not isinstance(node, dict) or key not in node:
                return None
            node = node[key]
        return node

    def first_of(*paths):
        """`a or b` is a ValueError on numpy arrays -- pick the first non-None."""
        for path in paths:
            v = pick(*path)
            if v is not None:
                return v
        return None

    def env_slice(v):
        if v is None:
            return None
        a = np.asarray(v)
        return a[env_id] if a.ndim >= 1 and a.shape[0] > env_id else a

    ee_pos = env_slice(first_of(("policy", "ee_pos"), ("ee_pos",)))
    ee_quat = env_slice(first_of(("policy", "ee_quat"), ("ee_quat",)))
    tip = fingertip_position(ee_pos, ee_quat) if ee_pos is not None and ee_quat is not None else None

    return {
        "scene": env_slice(first_of(("policy", "scene_rgb"), ("scene_rgb",))),
        "depth": env_slice(first_of(("policy", "scene_depth"), ("scene_depth",))),
        "K": env_slice(first_of(("policy", "scene_intrinsics"), ("scene_intrinsics",))),
        "cam_pos": env_slice(first_of(("policy", "scene_cam_pos"), ("scene_cam_pos",))),
        "cam_quat": env_slice(first_of(("policy", "scene_cam_quat"), ("scene_cam_quat",))),
        "tip_pos": tip,
        "ee_quat": ee_quat,
        "gripper_pos": env_slice(first_of(("policy", "gripper_pos"), ("gripper_pos",))),
    }


class RoboLabPointingClient:
    """Thin adapter: RoboLab observation in, controller action chunk out.

    ``vlm_pinpoint`` is imported on first use so this module can be imported, and the
    translation above unit-tested, with the package absent.
    """

    def __init__(self, instruction: str, env_id: int = 0, backend=None, **backend_kwargs):
        self.instruction = instruction
        self.env_id = env_id
        self._controller = None
        self._backend = backend
        self._backend_kwargs = backend_kwargs

    def _ensure(self):
        if self._controller is not None:
            return
        try:
            from vlm_pinpoint import GeminiBackend, PointingController
        except ImportError as exc:  # pragma: no cover - depends on an optional extra
            raise ImportError(
                "the pointing policy needs the 'vlm-pinpoint' package "
                "(pip install vlm-pinpoint); RoboLab carries only the connector"
            ) from exc
        self._controller = PointingController(self._backend or GeminiBackend(**self._backend_kwargs))

    def get_action(self, obs: dict):
        self._ensure()
        return self._controller.step(to_harness_observation(obs, self.env_id), self.instruction)
