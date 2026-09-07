# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""RoboLab client for Ai2's MolmoAct2-DROID policy server.

MolmoAct2-DROID is trained for absolute DROID joint-position control.  The
upstream FastAPI server accepts one exterior RGB image, one wrist RGB image,
and an eight-dimensional state ``[q1..q7, gripper_closedness]``.  It returns a
15-step chunk in the same action space, so this adapter deliberately performs
no Cartesian-frame conversion.
"""

from __future__ import annotations

from typing import Any

import json_numpy
import numpy as np
import requests

from robolab.eval.base_client import InferenceClient

ACTION_CHUNK_SIZE = 15
DEFAULT_TIMEOUT_SECONDS = 300.0


def _http_endpoint(host: str, port: int, remote_uri: str | None) -> str:
    value = remote_uri or host
    if not value.startswith(("http://", "https://")):
        value = f"http://{value}:{port}"
    value = value.rstrip("/")
    return value if value.endswith("/act") else f"{value}/act"


class MolmoAct2PolicyClient:
    """Minimal client for the upstream ``examples/droid`` FastAPI server."""

    def __init__(
        self,
        host: str = "localhost",
        port: int = 8000,
        *,
        remote_uri: str | None = None,
        timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
        session: requests.Session | None = None,
    ) -> None:
        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        self.endpoint = _http_endpoint(host, port, remote_uri)
        self.timeout_seconds = float(timeout_seconds)
        self.session = session or requests.Session()

        try:
            response = self.session.get(self.endpoint, timeout=self.timeout_seconds)
            response.raise_for_status()
            metadata = response.json()
        except Exception:
            self.close()
            raise
        if not isinstance(metadata, dict) or metadata.get("status") != "ok":
            self.close()
            raise RuntimeError(f"Unexpected MolmoAct2 health response: {metadata!r}")
        self.metadata = metadata

    def infer(self, request: dict[str, Any]) -> dict[str, Any]:
        payload = json_numpy.dumps(request).encode("utf-8")
        response = self.session.post(
            self.endpoint,
            data=payload,
            headers={"content-type": "application/json"},
            timeout=self.timeout_seconds,
        )
        try:
            decoded = json_numpy.loads(response.content.decode("utf-8"))
        except Exception as exc:
            response.raise_for_status()
            raise RuntimeError("MolmoAct2 server returned a non-json_numpy response") from exc
        if response.status_code >= 400:
            error = decoded.get("error", decoded) if isinstance(decoded, dict) else decoded
            raise RuntimeError(f"MolmoAct2 server error ({response.status_code}): {error}")
        if not isinstance(decoded, dict):
            raise RuntimeError(f"MolmoAct2 server returned {type(decoded).__name__}, expected a mapping")
        if "error" in decoded:
            raise RuntimeError(f"MolmoAct2 server error: {decoded['error']}")
        return decoded

    def close(self) -> None:
        self.session.close()


class MolmoAct2DroidClient(InferenceClient):
    """Adapt RoboLab's DROID observations to MolmoAct2-DROID."""

    def __init__(
        self,
        remote_host: str = "localhost",
        remote_port: int = 8000,
        *,
        remote_uri: str | None = None,
        open_loop_horizon: int = ACTION_CHUNK_SIZE,
        timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
        policy_client: MolmoAct2PolicyClient | None = None,
    ) -> None:
        super().__init__()
        if not (1 <= int(open_loop_horizon) <= ACTION_CHUNK_SIZE):
            raise ValueError(f"open_loop_horizon must be in [1, {ACTION_CHUNK_SIZE}], got {open_loop_horizon}")
        self.open_loop_horizon = int(open_loop_horizon)
        self.client = policy_client or MolmoAct2PolicyClient(
            host=remote_host,
            port=remote_port,
            remote_uri=remote_uri,
            timeout_seconds=timeout_seconds,
        )

    def _extract_observation(self, raw_obs: dict, *, env_id: int = 0) -> dict:
        image_obs = raw_obs["image_obs"]
        proprio_obs = raw_obs["proprio_obs"]
        return {
            "external_cam": self._to_numpy(image_obs["over_shoulder_left_camera"], env_id),
            "wrist_cam": self._to_numpy(image_obs["wrist_cam"], env_id),
            "joint_position": self._to_numpy(proprio_obs["arm_joint_pos"], env_id).astype(np.float32),
            "gripper_position": self._to_numpy(proprio_obs["gripper_pos"], env_id).astype(np.float32),
        }

    def _pack_request(self, extracted_obs: dict, instruction: str) -> dict:
        joint_position = np.asarray(extracted_obs["joint_position"], dtype=np.float32).reshape(7)
        gripper_position = np.asarray(extracted_obs["gripper_position"], dtype=np.float32).reshape(1)
        state = np.concatenate([joint_position, np.clip(gripper_position, 0.0, 1.0)]).astype(np.float32)
        return {
            "external_cam": self._as_rgb_uint8(extracted_obs["external_cam"]),
            "wrist_cam": self._as_rgb_uint8(extracted_obs["wrist_cam"]),
            "instruction": str(instruction),
            "state": state,
        }

    def _query_server(self, request: dict) -> dict:
        return self.client.infer(request)

    def _unpack_response(self, response: dict) -> np.ndarray:
        if "actions" not in response:
            raise ValueError("MolmoAct2 response is missing the 'actions' field")
        chunk = np.asarray(response["actions"], dtype=np.float32)
        if chunk.ndim == 3 and chunk.shape[0] == 1:
            chunk = chunk[0]
        if chunk.ndim != 2 or chunk.shape[1] != 8:
            raise ValueError(f"Expected MolmoAct2 actions shaped [T, 8], got {chunk.shape}")
        if chunk.shape[0] < self.open_loop_horizon:
            raise ValueError(
                f"MolmoAct2 returned {chunk.shape[0]} actions, fewer than open_loop_horizon={self.open_loop_horizon}"
            )
        if not np.isfinite(chunk).all():
            raise ValueError("MolmoAct2 returned non-finite actions")
        return chunk

    def _postprocess_chunk(self, chunk: np.ndarray) -> np.ndarray:
        result = np.asarray(chunk, dtype=np.float32).copy()
        # MolmoAct2 and RoboLab both use 0=open, 1=closed. RoboLab's gripper
        # action is binary, whereas the DROID checkpoint predicts a position.
        result[:, -1] = (result[:, -1] > 0.5).astype(np.float32)
        return result

    def _build_visualization(self, extracted_obs: dict) -> np.ndarray:
        external = self._as_rgb_uint8(extracted_obs["external_cam"])
        wrist = self._as_rgb_uint8(extracted_obs["wrist_cam"])
        if external.shape[:2] != wrist.shape[:2]:
            return external
        return np.concatenate([external, wrist], axis=1)

    @staticmethod
    def _as_rgb_uint8(image: np.ndarray) -> np.ndarray:
        array = np.asarray(image)
        if array.ndim != 3 or array.shape[-1] not in (3, 4):
            raise ValueError(f"Expected HWC RGB/RGBA image, got shape {array.shape}")
        if array.shape[-1] == 4:
            array = array[..., :3]
        if np.issubdtype(array.dtype, np.floating):
            max_value = float(np.nanmax(array)) if array.size else 0.0
            if max_value <= 1.0 + 1e-6:
                array = array * 255.0
        return np.ascontiguousarray(np.clip(array, 0, 255).astype(np.uint8, copy=False))

    def close(self) -> None:
        self.client.close()
