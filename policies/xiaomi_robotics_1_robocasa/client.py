# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""RoboLab inference client for Xiaomi-Robotics-1-RoboCasa.

The model and its exact Transformers stack live in a separate server process.
This client sends three resized RGB views, Franka joint state, and language;
then converts the checkpoint's normalized RoboCasa OSC action into RoboLab's
relative differential-IK action contract.
"""

from __future__ import annotations

from typing import Any

import cv2
import numpy as np
import zmq

from robolab.eval.base_client import InferenceClient

from .protocol import PROTOCOL_VERSION, ROBOT_TYPE, MsgSerializer

IMAGE_SIZE = 256
CROP_RATIO = 0.95
ACTION_CHUNK_SIZE = 10

# RoboCasa's OSC_POSE controller maps normalized actions to these physical
# deltas. RoboLab's DroidRelIKActionCfg applies a scalar 0.5 to all six arm
# dimensions, so the client compensates per dimension before execution.
ROBOCASA_TRANSLATION_DELTA_M = 0.05
ROBOCASA_ROTATION_DELTA_RAD = 0.5
ROBOLAB_REL_IK_SCALE = 0.5


def center_crop_resize_rgb(
    image: np.ndarray,
    *,
    size: int = IMAGE_SIZE,
    crop_ratio: float = CROP_RATIO,
) -> np.ndarray:
    """Center-crop an HWC RGB image to a square and resize it.

    Xiaomi's reference evaluation applies a 0.95 center crop to square
    256x256 RoboCasa images, then resizes with bilinear interpolation.
    RoboLab's cameras are 16:9, so the central square is resized to the target
    size first and Xiaomi's crop is then reproduced at that resolution.
    """
    array = np.asarray(image)
    if array.ndim != 3 or array.shape[-1] not in (3, 4):
        raise ValueError(f"Expected HWC RGB/RGBA image, got shape {array.shape}")
    if not (0.0 < crop_ratio <= 1.0):
        raise ValueError(f"crop_ratio must be in (0, 1], got {crop_ratio}")
    if size <= 0:
        raise ValueError(f"size must be positive, got {size}")

    if array.shape[-1] == 4:
        array = array[..., :3]
    if np.issubdtype(array.dtype, np.floating):
        max_value = float(np.nanmax(array)) if array.size else 0.0
        if max_value <= 1.0 + 1e-6:
            array = array * 255.0
    array = np.clip(array, 0, 255).astype(np.uint8, copy=False)

    height, width = array.shape[:2]
    square_size = min(height, width)
    top = (height - square_size) // 2
    left = (width - square_size) // 2
    square = array[top : top + square_size, left : left + square_size]
    interpolation = cv2.INTER_AREA if square_size >= size else cv2.INTER_LINEAR
    resized = cv2.resize(square, (size, size), interpolation=interpolation)

    if crop_ratio < 1.0:
        crop_size = max(1, int(size * crop_ratio))
        offset = (size - crop_size) // 2
        cropped = resized[offset : offset + crop_size, offset : offset + crop_size]
        resized = cv2.resize(cropped, (size, size), interpolation=cv2.INTER_LINEAR)
    return resized.astype(np.uint8, copy=False)


class XR1RoboCasaPolicyClient:
    """Minimal ZMQ REQ client for :mod:`.server`."""

    def __init__(self, host: str = "localhost", port: int = 10086, timeout_seconds: float = 300.0):
        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        self._context = zmq.Context()
        self._socket = self._context.socket(zmq.REQ)
        timeout_ms = int(timeout_seconds * 1000)
        self._socket.setsockopt(zmq.RCVTIMEO, timeout_ms)
        self._socket.setsockopt(zmq.SNDTIMEO, timeout_ms)
        self._socket.setsockopt(zmq.LINGER, 0)
        self._address = f"tcp://{host}:{port}"
        self._socket.connect(self._address)

    def infer(self, request: dict[str, Any]) -> dict[str, Any]:
        try:
            self._socket.send(MsgSerializer.to_bytes(request))
            response = MsgSerializer.from_bytes(self._socket.recv())
        except zmq.error.Again as exc:
            raise TimeoutError(f"Timed out waiting for XR-1 server at {self._address}") from exc
        if not isinstance(response, dict):
            raise RuntimeError(f"XR-1 server returned {type(response).__name__}, expected a mapping")
        if "error" in response:
            raise RuntimeError(f"XR-1 server error: {response['error']}")
        return response

    def ping(self) -> dict[str, Any]:
        return self.infer({"protocol_version": PROTOCOL_VERSION, "endpoint": "ping"})

    def close(self) -> None:
        self._socket.close(linger=0)
        self._context.term()


class XR1RoboCasaClient(InferenceClient):
    """Adapt RoboLab DROID observations to XR-1's RoboCasa checkpoint."""

    def __init__(
        self,
        remote_host: str = "localhost",
        remote_port: int = 10086,
        open_loop_horizon: int = ACTION_CHUNK_SIZE,
        timeout_seconds: float = 300.0,
        image_size: int = IMAGE_SIZE,
        crop_ratio: float = CROP_RATIO,
        translation_delta_m: float = ROBOCASA_TRANSLATION_DELTA_M,
        rotation_delta_rad: float = ROBOCASA_ROTATION_DELTA_RAD,
        robolab_ik_scale: float = ROBOLAB_REL_IK_SCALE,
        gripper_open_position: float = 0.04,
        gripper_closed_position: float = 0.0,
        request_seed: int = 42,
        policy_client: XR1RoboCasaPolicyClient | None = None,
    ) -> None:
        super().__init__()
        if not (1 <= int(open_loop_horizon) <= ACTION_CHUNK_SIZE):
            raise ValueError(f"open_loop_horizon must be in [1, {ACTION_CHUNK_SIZE}], got {open_loop_horizon}")
        if robolab_ik_scale <= 0:
            raise ValueError("robolab_ik_scale must be positive")
        if translation_delta_m <= 0 or rotation_delta_rad <= 0:
            raise ValueError("RoboCasa action delta scales must be positive")

        self.open_loop_horizon = int(open_loop_horizon)
        self.image_size = int(image_size)
        self.crop_ratio = float(crop_ratio)
        self.translation_factor = float(translation_delta_m) / float(robolab_ik_scale)
        self.rotation_factor = float(rotation_delta_rad) / float(robolab_ik_scale)
        self.gripper_open_position = float(gripper_open_position)
        self.gripper_closed_position = float(gripper_closed_position)
        self.request_seed = int(request_seed)
        if policy_client is None:
            self.client = XR1RoboCasaPolicyClient(
                host=remote_host,
                port=remote_port,
                timeout_seconds=timeout_seconds,
            )
            try:
                metadata = self.client.ping()
                if metadata.get("protocol_version") != PROTOCOL_VERSION:
                    raise RuntimeError(
                        "XR-1 server protocol mismatch: "
                        f"server={metadata.get('protocol_version')!r}, client={PROTOCOL_VERSION}"
                    )
                if metadata.get("robot_type") != ROBOT_TYPE:
                    raise RuntimeError(f"XR-1 server does not serve {ROBOT_TYPE!r}: metadata={metadata}")
            except Exception:
                self.client.close()
                raise
        else:
            self.client = policy_client

    def _extract_observation(self, raw_obs: dict, *, env_id: int = 0) -> dict:
        image_obs = raw_obs["image_obs"]
        proprio_obs = raw_obs["proprio_obs"]
        joint_position = self._to_numpy(proprio_obs["arm_joint_pos"], env_id).astype(np.float32)
        closedness = float(self._to_numpy(proprio_obs["gripper_pos"], env_id).astype(np.float32).reshape(-1)[0])
        closedness = float(np.clip(closedness, 0.0, 1.0))
        # Xiaomi's reference adapter feeds the first raw Panda gripper joint.
        # PandaGripper uses 0.04 m when open and 0.0 m when closed.
        gripper_joint_position = self.gripper_open_position + closedness * (
            self.gripper_closed_position - self.gripper_open_position
        )
        return {
            "base_left": self._to_numpy(image_obs["over_shoulder_left_camera"], env_id),
            "base_right": self._to_numpy(image_obs["over_shoulder_right_camera"], env_id),
            "wrist": self._to_numpy(image_obs["wrist_cam"], env_id),
            "joint_position": joint_position.reshape(7),
            "gripper_joint_position": np.asarray([gripper_joint_position], dtype=np.float32),
        }

    def _pack_request(self, extracted_obs: dict, instruction: str) -> dict:
        state = np.concatenate([extracted_obs["joint_position"], extracted_obs["gripper_joint_position"]]).astype(
            np.float32
        )
        return {
            "protocol_version": PROTOCOL_VERSION,
            "endpoint": "infer",
            "robot_type": ROBOT_TYPE,
            "images": {
                name: center_crop_resize_rgb(extracted_obs[name], size=self.image_size, crop_ratio=self.crop_ratio)
                for name in ("base_left", "base_right", "wrist")
            },
            "state": state,
            "instruction": str(instruction),
            "seed": self.request_seed,
        }

    def _query_server(self, request: dict) -> dict:
        return self.client.infer(request)

    def _unpack_response(self, response: dict) -> np.ndarray:
        if response.get("protocol_version") != PROTOCOL_VERSION:
            raise ValueError(
                "XR-1 response protocol mismatch: "
                f"server={response.get('protocol_version')!r}, client={PROTOCOL_VERSION}"
            )
        if "actions" not in response:
            raise ValueError("XR-1 response is missing the 'actions' field")
        chunk = np.asarray(response["actions"], dtype=np.float32)
        if chunk.ndim == 3 and chunk.shape[0] == 1:
            chunk = chunk[0]
        if chunk.ndim != 2 or chunk.shape[1] < 7:
            raise ValueError(f"Expected XR-1 actions shaped [T, >=7], got {chunk.shape}")
        if chunk.shape[0] < self.open_loop_horizon:
            raise ValueError(
                f"XR-1 returned {chunk.shape[0]} actions, fewer than open_loop_horizon={self.open_loop_horizon}"
            )
        return chunk[:, :7]

    def _postprocess_chunk(self, chunk: np.ndarray) -> np.ndarray:
        normalized_arm = np.clip(np.asarray(chunk[:, :6], dtype=np.float32), -1.0, 1.0)
        mapped = np.empty((chunk.shape[0], 7), dtype=np.float32)
        mapped[:, :3] = normalized_arm[:, :3] * self.translation_factor
        mapped[:, 3:6] = normalized_arm[:, 3:6] * self.rotation_factor
        # RoboCasa/robosuite: negative=open, positive=close. RoboLab: 0=open, 1=close.
        mapped[:, 6] = (chunk[:, 6] > 0.0).astype(np.float32)
        return mapped

    def _build_visualization(self, extracted_obs: dict) -> np.ndarray:
        images = [
            center_crop_resize_rgb(extracted_obs[name], size=self.image_size, crop_ratio=self.crop_ratio)
            for name in ("base_left", "base_right", "wrist")
        ]
        return np.concatenate(images, axis=1)

    def close(self) -> None:
        self.client.close()
