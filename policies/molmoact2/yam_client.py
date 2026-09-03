# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""MolmoAct 2 bimanual-YAM client: RoboLab observations -> Ai2's ``/act`` server -> 16-dim actions.

Wire contract (``examples/yam/host_server_yam.py`` in allenai/molmoact2), unchanged here:

    POST <url>/act   json_numpy payload
        top_cam, left_cam, right_cam : uint8 (H, W, 3) RGB, trained at 360x640
        instruction : str
        state       : float32 (14,) = [L j1..j6, L grip, R j1..j6, R grip], grip 1 = open
        normalization_tag = "yam_dual_molmoact2", num_steps = 10 (flow solver steps)
    -> {"actions": float32 (30, 14)} absolute joint targets at 30 Hz, grip in [0, 1]

RoboLab's bimanual YAM env runs at 30 Hz with a 16-dim action
``[L arm 6, L fingers 2, R arm 6, R fingers 2]`` (finger joints in metres, 0 closed,
-0.04695 open); the client places the policy's gripper into both finger slots. Ai2's
ManiSkill harness plays the whole 30-step chunk before replanning; so does this client
(``open_loop_horizon``), and their real-robot launcher plays 25.
"""

from __future__ import annotations

import base64
import json
import urllib.error
import urllib.request

import numpy as np
from numpy.lib.format import descr_to_dtype, dtype_to_descr

from robolab.eval.base_client import InferenceClient

NORM_TAG = "yam_dual_molmoact2"
STATE_DIM = 14
ACTION_DIM = 14
ENV_ACTION_DIM = 16
# Finger joint travel of the asset (robolab/robots/bimanual_yam.py::FINGER_TRAVEL_M); repeated
# here so the client imports without Isaac Lab. offline_tests/test_bimanual_yam_asset.py pins it.
FINGER_TRAVEL_M = 0.04695
# Observation keys the bimanual YAM registration produces -> wire keys.
CAMERA_KEYS = {"top_cam": "top_cam", "left_wrist_cam": "left_cam", "right_wrist_cam": "right_cam"}


# ---------------------------------------------------------------- json_numpy (vendored, 20 lines)
def _np_default(o):
    if isinstance(o, (np.ndarray, np.generic)):
        a = np.ascontiguousarray(o)
        return {"__numpy__": base64.b64encode(a.tobytes()).decode(), "dtype": dtype_to_descr(a.dtype),
                "shape": a.shape}
    raise TypeError(f"not JSON serializable: {type(o).__name__}")


def _np_hook(d):
    if "__numpy__" in d:
        arr = np.frombuffer(base64.b64decode(d["__numpy__"]), descr_to_dtype(d["dtype"]))
        return arr.reshape(d["shape"]) if d["shape"] else arr[0]
    return d


def np_dumps(obj) -> bytes:
    return json.dumps(obj, default=_np_default).encode()


def np_loads(data: bytes):
    return json.loads(data, object_hook=_np_hook)


# ---------------------------------------------------------------- pure functions (tested offline)
def build_state(left_q, left_grip, right_q, right_grip) -> np.ndarray:
    """14-D MolmoAct 2 state from per-arm joint positions (rad) and gripper openness [0, 1]."""
    s = np.concatenate([np.asarray(left_q, np.float32).reshape(6), [float(left_grip)],
                        np.asarray(right_q, np.float32).reshape(6), [float(right_grip)]]).astype(np.float32)
    assert s.shape == (STATE_DIM,), s.shape
    return s


def expand_chunk(actions: np.ndarray) -> np.ndarray:
    """(N, 14) server actions -> (N, 16) env actions: gripper [0,1] (1 = open) -> both finger
    joints at ``-FINGER_TRAVEL_M * grip`` metres."""
    a = np.asarray(actions, np.float32)
    if a.ndim == 1:
        a = a[None]
    assert a.shape[-1] == ACTION_DIM, a.shape
    out = np.zeros((a.shape[0], ENV_ACTION_DIM), np.float32)
    out[:, 0:6] = a[:, 0:6]
    out[:, 8:14] = a[:, 7:13]
    lg = -FINGER_TRAVEL_M * np.clip(a[:, 6], 0.0, 1.0)
    rg = -FINGER_TRAVEL_M * np.clip(a[:, 13], 0.0, 1.0)
    out[:, 6] = out[:, 7] = lg
    out[:, 14] = out[:, 15] = rg
    return out


def _as_uint8_image(img) -> np.ndarray:
    a = np.asarray(img)
    if a.dtype != np.uint8:
        a = np.clip(a * 255.0 if a.max() <= 1.0 else a, 0, 255).astype(np.uint8)
    if a.ndim == 4:
        a = a[0]
    assert a.ndim == 3 and a.shape[-1] == 3, a.shape
    return np.ascontiguousarray(a)


# ---------------------------------------------------------------- the client
class MolmoAct2YamClient(InferenceClient):
    open_loop_horizon = 30          # one full chunk per replan, like Ai2's sim harness

    def __init__(self, server: str = "http://localhost:8202", open_loop_horizon: int | None = None,
                 num_steps: int = 10, timeout_s: float = 60.0):
        super().__init__()
        self.url = server if server.endswith("/act") else server.rstrip("/") + "/act"
        if open_loop_horizon:
            self.open_loop_horizon = open_loop_horizon
        self.num_steps = num_steps
        self.timeout_s = timeout_s

    # -- hooks -----------------------------------------------------------------
    def _extract_observation(self, raw_obs, *, env_id: int = 0) -> dict:
        images = {}
        for obs_key, wire_key in CAMERA_KEYS.items():
            img = self._find_obs_term(raw_obs, obs_key)
            if img is None:
                raise KeyError(f"observation has no camera '{obs_key}' (needed for MolmoAct 2 YAM)")
            images[wire_key] = _as_uint8_image(self._to_numpy(img, env_id))
        state = build_state(
            self._to_numpy(self._find_obs_term(raw_obs, "left_arm_joint_pos"), env_id),
            float(self._to_numpy(self._find_obs_term(raw_obs, "left_gripper_pos"), env_id).reshape(-1)[0]),
            self._to_numpy(self._find_obs_term(raw_obs, "right_arm_joint_pos"), env_id),
            float(self._to_numpy(self._find_obs_term(raw_obs, "right_gripper_pos"), env_id).reshape(-1)[0]),
        )
        return {"images": images, "state": state}

    def _pack_request(self, extracted_obs: dict, instruction: str) -> bytes:
        payload = {**extracted_obs["images"], "instruction": instruction, "state": extracted_obs["state"],
                   "normalization_tag": NORM_TAG, "num_steps": self.num_steps}
        return np_dumps(payload)

    def _query_server(self, request: bytes):
        req = urllib.request.Request(self.url, data=request, headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=self.timeout_s) as r:
                return np_loads(r.read())
        except urllib.error.HTTPError as e:
            raise RuntimeError(f"MolmoAct 2 server {self.url} returned {e.code}: {e.read()[:300]!r}") from e

    def _unpack_response(self, response) -> np.ndarray:
        actions = np.asarray(response["actions"], np.float32)
        if actions.ndim == 3 and actions.shape[0] == 1:
            actions = actions[0]
        if actions.ndim != 2 or actions.shape[1] != ACTION_DIM:
            raise ValueError(f"expected (N, {ACTION_DIM}) actions, got {actions.shape}")
        return actions

    def _postprocess_chunk(self, chunk: np.ndarray) -> np.ndarray:
        return expand_chunk(chunk)

    def _build_visualization(self, extracted_obs: dict):
        # top | left wrist | right wrist, the policy's own view, for the episode video.
        ims = [extracted_obs["images"][k] for k in ("top_cam", "left_cam", "right_cam")]
        return np.concatenate(ims, axis=1)
