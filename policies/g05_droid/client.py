# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""RoboLab client for OpenGalaxea's G0.5-DROID policy server.

G0.5's reference DROID server advances an action chunk one step at a time and
only asks for a fresh observation when it needs to replan.  That state lives
on the WebSocket connection, so this adapter keeps one connection per RoboLab
environment instead of using :class:`InferenceClient`'s local chunk cache.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import msgpack
import numpy as np
import websockets.sync.client

from robolab.eval.base_client import InferenceClient

DROID_CONTROL_FREQUENCY = 15
DUMMY_WRIST_RIGHT_SHAPE = (3, 224, 224)
EMBODIMENT_TYPE = "Droid_Franka"
DEFAULT_TIMEOUT_SECONDS = 300.0


def _websocket_uri(host: str, port: int, remote_uri: str | None) -> str:
    if remote_uri:
        uri = remote_uri
    elif host.startswith(("ws://", "wss://", "http://", "https://")):
        uri = host
    else:
        uri = f"ws://{host}:{port}"
    if uri.startswith("http://"):
        uri = "ws://" + uri.removeprefix("http://")
    elif uri.startswith("https://"):
        uri = "wss://" + uri.removeprefix("https://")
    return uri.rstrip("/")


def _encode_numpy(value: Any) -> dict[str, Any]:
    """Encode NumPy values exactly as Galaxea's public DROID client does."""
    if isinstance(value, np.ndarray):
        if value.dtype.kind in ("V", "O", "c"):
            raise ValueError(f"Unsupported NumPy dtype: {value.dtype}")
        return {
            "__ndarray__": True,
            "data": value.tobytes(),
            "dtype": value.dtype.str,
            "shape": value.shape,
        }
    if isinstance(value, np.generic):
        return {
            "__npgeneric__": True,
            "data": value.item(),
            "dtype": value.dtype.str,
        }
    raise TypeError(f"Cannot msgpack value of type {type(value).__name__}")


def _decode_numpy(value: dict[Any, Any]) -> Any:
    ndarray_key = "__ndarray__" if "__ndarray__" in value else b"__ndarray__"
    generic_key = "__npgeneric__" if "__npgeneric__" in value else b"__npgeneric__"
    if ndarray_key in value:
        data_key = "data" if "data" in value else b"data"
        dtype_key = "dtype" if "dtype" in value else b"dtype"
        shape_key = "shape" if "shape" in value else b"shape"
        return np.ndarray(
            buffer=value[data_key],
            dtype=np.dtype(value[dtype_key]),
            shape=tuple(value[shape_key]),
        ).copy()
    if generic_key in value:
        data_key = "data" if "data" in value else b"data"
        dtype_key = "dtype" if "dtype" in value else b"dtype"
        return np.dtype(value[dtype_key]).type(value[data_key])
    return value


def pack_message(value: Any) -> bytes:
    return msgpack.packb(value, default=_encode_numpy)


def unpack_message(value: bytes) -> Any:
    return msgpack.unpackb(
        value,
        object_hook=_decode_numpy,
        raw=False,
        strict_map_key=False,
    )


class G05PolicyClient:
    """One stateful connection to the official G0.5 inference server."""

    def __init__(
        self,
        host: str = "localhost",
        port: int = 8000,
        *,
        remote_uri: str | None = None,
        timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
    ) -> None:
        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        self.uri = _websocket_uri(host, port, remote_uri)
        self.timeout_seconds = float(timeout_seconds)
        try:
            self._socket = websockets.sync.client.connect(
                self.uri,
                compression=None,
                max_size=None,
                open_timeout=self.timeout_seconds,
            )
        except TypeError:
            # Isaac Sim may bundle an older websockets without open_timeout.
            self._socket = websockets.sync.client.connect(
                self.uri,
                compression=None,
                max_size=None,
            )

        try:
            metadata = self._receive()
            if not isinstance(metadata, dict):
                raise RuntimeError(f"Unexpected G0.5 handshake: {metadata!r}")
            action_steps = int(metadata.get("action_steps", 0))
            if action_steps <= 0:
                raise RuntimeError(f"G0.5 handshake has no valid action_steps: {metadata!r}")
            self.metadata = metadata
            self.action_steps = action_steps
        except Exception:
            self.close()
            raise

    def _receive(self) -> Any:
        try:
            message = self._socket.recv(timeout=self.timeout_seconds)
        except TypeError:
            message = self._socket.recv()
        if isinstance(message, str):
            raise RuntimeError(f"G0.5 server returned text instead of msgpack: {message}")
        return unpack_message(message)

    def infer(self, request: dict[str, Any]) -> dict[str, Any]:
        self._socket.send(pack_message(request))
        response = self._receive()
        if not isinstance(response, dict):
            raise RuntimeError(f"G0.5 server returned {type(response).__name__}, expected a mapping")
        if "error" in response:
            raise RuntimeError(f"G0.5 server error: {response['error']}")
        return response

    def reset(self) -> None:
        response = self.infer({"__reset__": True})
        if not response.get("__reset__"):
            raise RuntimeError(f"Unexpected G0.5 reset response: {response}")

    def close(self) -> None:
        socket = getattr(self, "_socket", None)
        if socket is not None:
            try:
                socket.close()
            finally:
                self._socket = None


class G05DroidClient(InferenceClient):
    """Adapt RoboLab DROID observations to G0.5's stateful protocol."""

    open_loop_horizon = 1

    def __init__(
        self,
        remote_host: str = "localhost",
        remote_port: int = 8000,
        *,
        remote_uri: str | None = None,
        timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
        policy_client: G05PolicyClient | None = None,
        policy_client_factory: Callable[[int], G05PolicyClient] | None = None,
    ) -> None:
        super().__init__()
        if policy_client is not None and policy_client_factory is not None:
            raise ValueError("Pass policy_client or policy_client_factory, not both")

        if policy_client_factory is None:
            if policy_client is not None:
                policy_client_factory = lambda _env_id: policy_client
            else:
                policy_client_factory = lambda _env_id: G05PolicyClient(
                    host=remote_host,
                    port=remote_port,
                    remote_uri=remote_uri,
                    timeout_seconds=timeout_seconds,
                )
        self._policy_client_factory = policy_client_factory
        self._clients: dict[int, G05PolicyClient] = {}
        self._need_observation: dict[int, bool] = {}
        self._last_cot: dict[int, str] = {}

        # Connect eagerly for env 0 so configuration/server failures surface
        # before the simulator starts an episode. Other envs connect lazily.
        self._clients[0] = self._policy_client_factory(0)

    def _client_for(self, env_id: int) -> G05PolicyClient:
        if env_id not in self._clients:
            self._clients[env_id] = self._policy_client_factory(env_id)
        return self._clients[env_id]

    def infer(self, obs: Any, instruction: str, *, env_id: int = 0) -> dict:
        extracted = self._extract_observation(obs, env_id=env_id)
        if self._need_observation.get(env_id, True):
            request = self._pack_request(extracted, instruction)
        else:
            # The server is advancing its cached chunk and explicitly does not
            # want another image/state payload on this step.
            request = {}

        response = self._client_for(env_id).infer(request)
        action = self._action_from_response(response, extracted)
        self._need_observation[env_id] = bool(response.get("need_obs", True))
        if response.get("cot_text") is not None:
            self._last_cot[env_id] = str(response["cot_text"])
        return {"action": action, "viz": self._build_visualization(extracted)}

    def _extract_observation(self, raw_obs: dict, *, env_id: int = 0) -> dict:
        image_obs = raw_obs["image_obs"]
        proprio_obs = raw_obs["proprio_obs"]
        return {
            "exterior_image": self._to_numpy(image_obs["over_shoulder_left_camera"], env_id),
            "wrist_image": self._to_numpy(image_obs["wrist_cam"], env_id),
            "joint_position": self._to_numpy(proprio_obs["arm_joint_pos"], env_id).astype(np.float32),
            "gripper_position": self._to_numpy(proprio_obs["gripper_pos"], env_id).astype(np.float32),
        }

    def _pack_request(self, extracted_obs: dict, instruction: str) -> dict:
        gripper_closedness = np.clip(
            np.asarray(extracted_obs["gripper_position"], dtype=np.float32).reshape(1),
            0.0,
            1.0,
        )
        return {
            "images": {
                "exterior_image": self._as_chw_uint8(extracted_obs["exterior_image"]),
                "wrist_image": self._as_chw_uint8(extracted_obs["wrist_image"]),
                "dummy_wrist_right": np.zeros(DUMMY_WRIST_RIGHT_SHAPE, dtype=np.uint8),
            },
            "state": {
                "right_arm": np.asarray(extracted_obs["joint_position"], dtype=np.float32).reshape(7),
                # G0.5 uses 1=open while DROID/RoboLab uses 1=closed.
                "right_gripper": np.asarray(1.0 - gripper_closedness, dtype=np.float32),
            },
            "task": str(instruction),
            "frequency": DROID_CONTROL_FREQUENCY,
            "embodiment_type": EMBODIMENT_TYPE,
        }

    def _query_server(self, request: dict) -> dict:
        return self._client_for(0).infer(request)

    def _unpack_response(self, response: dict) -> np.ndarray:
        action, has_gripper = self._extract_server_action(response)
        if not has_gripper:
            raise ValueError("G0.5 response is missing a gripper action")
        return action.reshape(1, 8)

    def _action_from_response(self, response: dict, extracted_obs: dict) -> np.ndarray:
        action, has_gripper = self._extract_server_action(response)
        if has_gripper:
            # Convert G0.5's 1=open value back to RoboLab's 1=closed value.
            closedness = float(np.clip(1.0 - action[-1], 0.0, 1.0))
        else:
            # The reference client holds the observed gripper if the server
            # omits that key for a cached-chunk step.
            closedness = float(np.clip(np.asarray(extracted_obs["gripper_position"]).reshape(-1)[0], 0.0, 1.0))
        action[-1] = float(closedness > 0.5)
        if not np.isfinite(action).all():
            raise ValueError("G0.5 returned non-finite actions")
        return action.astype(np.float32, copy=False)

    @staticmethod
    def _extract_server_action(response: dict) -> tuple[np.ndarray, bool]:
        payload = response.get("action")
        if not isinstance(payload, dict):
            raise ValueError("G0.5 response is missing the 'action' mapping")

        arm_value = payload.get("right_arm")
        if arm_value is None:
            arm_value = payload.get("joint_position")
        if arm_value is None:
            raise ValueError("G0.5 response is missing the right-arm action")
        arm = np.asarray(arm_value, dtype=np.float32).reshape(-1)
        if arm.shape != (7,):
            raise ValueError(f"Expected a 7-dimensional G0.5 arm action, got {arm.shape}")

        gripper_value = payload.get("right_gripper")
        if gripper_value is None:
            gripper_value = payload.get("gripper")
        if gripper_value is None:
            return np.concatenate([arm, np.zeros(1, dtype=np.float32)]), False
        gripper = np.asarray(gripper_value, dtype=np.float32).reshape(-1)
        if gripper.shape != (1,):
            raise ValueError(f"Expected a scalar G0.5 gripper action, got {gripper.shape}")
        return np.concatenate([arm, gripper]), True

    def _build_visualization(self, extracted_obs: dict) -> np.ndarray:
        exterior = self._as_hwc_uint8(extracted_obs["exterior_image"])
        wrist = self._as_hwc_uint8(extracted_obs["wrist_image"])
        if exterior.shape[:2] != wrist.shape[:2]:
            return exterior
        return np.concatenate([exterior, wrist], axis=1)

    @classmethod
    def _as_chw_uint8(cls, image: np.ndarray) -> np.ndarray:
        return np.ascontiguousarray(cls._as_hwc_uint8(image).transpose(2, 0, 1))

    @staticmethod
    def _as_hwc_uint8(image: np.ndarray) -> np.ndarray:
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

    def reset(self, *, env_id: int | None = None) -> None:
        if env_id is None:
            targets = list(self._clients)
        elif env_id in self._clients:
            targets = [env_id]
        else:
            targets = []
        for target in targets:
            self._clients[target].reset()

        if env_id is None:
            self._need_observation.clear()
            self._last_cot.clear()
        else:
            self._need_observation.pop(env_id, None)
            self._last_cot.pop(env_id, None)
        super().reset(env_id=env_id)

    def close(self) -> None:
        # The injected single-client form may intentionally appear under only
        # env 0 today, but de-duplicate defensively before closing transports.
        clients = {id(client): client for client in self._clients.values()}
        for client in clients.values():
            client.close()
        self._clients.clear()
