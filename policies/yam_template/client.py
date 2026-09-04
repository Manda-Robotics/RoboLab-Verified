# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Template client for a bimanual YAM policy: copy this folder, replace the "model".

Written against the contract in docs/bimanual_yam.md and nothing else, then checked from a
fresh clone (2026-09-03). The stand-in model holds the current joint pose, wobbles one elbow
per arm and opens/closes both grippers on a slow cycle; ``_query_server`` is where your
model goes, and ``_postprocess_chunk`` maps whatever it emits onto the rig's 16-dim action.
"""
import math
import numpy as np
from robolab.eval.base_client import InferenceClient

FINGER_TRAVEL_M = 0.04695          # from the guide: 0 closed .. -0.04695 open
CHUNK = 15                         # actions per "inference"; 0.5 s at 30 Hz control


class TemplateYamClient(InferenceClient):
    open_loop_horizon = CHUNK

    def __init__(self):
        super().__init__()
        self._t = {}

    # hook 1: what the env gives us -> what "our model" wants
    def _extract_observation(self, raw_obs, *, env_id=0):
        get = lambda k: self._to_numpy(self._find_obs_term(raw_obs, k), env_id)
        return {
            "images": {k: get(k) for k in ("top_cam", "left_wrist_cam", "right_wrist_cam")},
            "left_q": get("left_arm_joint_pos"), "right_q": get("right_arm_joint_pos"),
            "left_open": float(get("left_gripper_pos").reshape(-1)[0]),
            "right_open": float(get("right_gripper_pos").reshape(-1)[0]),
            "env_id": env_id,
        }

    # hooks 2-4: no server here, the "model" is local
    def _pack_request(self, extracted_obs, instruction):
        return extracted_obs

    def _query_server(self, request):
        t0 = self._t.get(request["env_id"], 0)
        self._t[request["env_id"]] = t0 + CHUNK
        chunk = []
        for k in range(CHUNK):
            phase = 2 * math.pi * (t0 + k) / 180.0          # 6 s cycle at 30 Hz
            lq, rq = request["left_q"].copy(), request["right_q"].copy()
            lq[2] += 0.10 * math.sin(phase); rq[2] -= 0.10 * math.sin(phase)
            grip_open = 1.0 if math.sin(phase) >= 0 else 0.0
            chunk.append(np.concatenate([lq, [grip_open], rq, [grip_open]]).astype(np.float32))  # 14 = 6+1+6+1
        return np.stack(chunk)

    def _unpack_response(self, response):
        return np.asarray(response, np.float32)

    # map our 14-dim (6 + openness, twice) onto the env's 16-dim action from the guide's table
    def _postprocess_chunk(self, chunk):
        out = np.zeros((len(chunk), 16), np.float32)
        out[:, 0:6] = chunk[:, 0:6]; out[:, 8:14] = chunk[:, 7:13]
        out[:, 6] = out[:, 7] = -FINGER_TRAVEL_M * chunk[:, 6]
        out[:, 14] = out[:, 15] = -FINGER_TRAVEL_M * chunk[:, 13]
        return out
