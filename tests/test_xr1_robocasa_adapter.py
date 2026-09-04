# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Contract tests for the XR-1 RoboCasa cross-benchmark adapter."""

from __future__ import annotations

from contextlib import nullcontext
from types import SimpleNamespace

import numpy as np
import pytest

from policies.xiaomi_robotics_1_robocasa.client import (
    XR1RoboCasaClient,
    center_crop_resize_rgb,
)
from policies.xiaomi_robotics_1_robocasa.protocol import MsgSerializer
from policies.xiaomi_robotics_1_robocasa.server import XR1RoboCasaModel


class _Tensor:
    def __init__(self, value):
        self.value = np.asarray(value)

    def __getitem__(self, index):
        return _Tensor(self.value[index])

    def detach(self):
        return self

    def cpu(self):
        return self

    def numpy(self):
        return self.value


class _PolicyClient:
    def __init__(self, actions=None):
        self.actions = np.zeros((10, 7), dtype=np.float32) if actions is None else np.asarray(actions, dtype=np.float32)
        self.requests = []
        self.closed = False

    def infer(self, request):
        self.requests.append(request)
        return {"protocol_version": 1, "actions": self.actions}

    def close(self):
        self.closed = True


class _ModelTensor:
    def __init__(self, value):
        self.value = np.asarray(value)

    def is_floating_point(self):
        return np.issubdtype(self.value.dtype, np.floating)

    def to(self, **_kwargs):
        return self

    def detach(self):
        return self

    def cpu(self):
        return self

    def float(self):
        self.value = self.value.astype(np.float32)
        return self

    def numpy(self):
        return self.value


class _Torch:
    @staticmethod
    def is_tensor(value):
        return isinstance(value, _ModelTensor)

    @staticmethod
    def inference_mode():
        return nullcontext()


class _Processor:
    def __init__(self):
        self.messages = None
        self.template_kwargs = None

    def apply_chat_template(self, messages, **kwargs):
        self.messages = messages
        self.template_kwargs = kwargs
        return {"input_ids": _ModelTensor(np.asarray([[1, 2]], dtype=np.int64))}

    @staticmethod
    def decode_action(actions, *, robot_type):
        assert robot_type == "robocasa_mg"
        return actions


class _Model:
    def __init__(self, actions):
        self.actions = actions
        self.inputs = None

    def __call__(self, **inputs):
        self.inputs = inputs
        return SimpleNamespace(actions=_ModelTensor(self.actions))


def _obs(*, closedness=0.0):
    height, width = 12, 20
    return {
        "image_obs": {
            "over_shoulder_left_camera": _Tensor(np.zeros((1, height, width, 3), dtype=np.uint8)),
            "over_shoulder_right_camera": _Tensor(np.full((1, height, width, 3), 64, dtype=np.uint8)),
            "wrist_cam": _Tensor(np.full((1, height, width, 3), 128, dtype=np.uint8)),
        },
        "proprio_obs": {
            "arm_joint_pos": _Tensor(np.arange(7, dtype=np.float32)[None]),
            "gripper_pos": _Tensor(np.asarray([[closedness]], dtype=np.float32)),
        },
    }


def test_center_crop_resize_rgb_contract():
    image = np.zeros((10, 20, 4), dtype=np.uint8)
    image[..., 0] = np.arange(20, dtype=np.uint8)[None, :]

    result = center_crop_resize_rgb(image, size=8, crop_ratio=1.0)

    assert result.shape == (8, 8, 3)
    assert result.dtype == np.uint8
    # The central square excludes the far left and right of the 2:1 source.
    assert 3 <= int(result[4, 0, 0]) <= 7
    assert 12 <= int(result[4, -1, 0]) <= 16


@pytest.mark.parametrize("closedness, expected_joint", [(0.0, 0.04), (1.0, 0.0), (0.25, 0.03)])
def test_request_maps_robolab_closedness_to_panda_gripper(closedness, expected_joint):
    transport = _PolicyClient()
    client = XR1RoboCasaClient(policy_client=transport, image_size=8, crop_ratio=1.0)

    client.infer(_obs(closedness=closedness), "pick the fruit")

    request = transport.requests[0]
    np.testing.assert_array_equal(request["state"][:7], np.arange(7, dtype=np.float32))
    np.testing.assert_allclose(request["state"][7], expected_joint, atol=1e-7)
    assert request["instruction"] == "pick the fruit"
    assert set(request["images"]) == {"base_left", "base_right", "wrist"}
    assert all(image.shape == (8, 8, 3) for image in request["images"].values())


def test_action_mapping_matches_robocasa_and_robolab_scales():
    actions = np.zeros((10, 7), dtype=np.float32)
    actions[0] = [2.0, -2.0, 0.5, 2.0, -2.0, 0.5, -0.1]
    actions[1, 6] = 0.1
    client = XR1RoboCasaClient(policy_client=_PolicyClient(actions))

    mapped = client._postprocess_chunk(actions)

    # Clipping to normalized [-1, 1], then compensating for RoboLab's 0.5 IK scale.
    np.testing.assert_allclose(mapped[0, :3], [0.1, -0.1, 0.05])
    np.testing.assert_allclose(mapped[0, 3:6], [1.0, -1.0, 0.5])
    assert mapped[0, 6] == 0.0  # negative -> open
    assert mapped[1, 6] == 1.0  # positive -> close


def test_inference_consumes_ten_action_chunk_before_replanning():
    transport = _PolicyClient()
    client = XR1RoboCasaClient(policy_client=transport, image_size=8, crop_ratio=1.0)

    for _ in range(10):
        result = client.infer(_obs(), "task")
        assert result["action"].shape == (7,)
    assert len(transport.requests) == 1

    client.infer(_obs(), "task")
    assert len(transport.requests) == 2


def test_numeric_protocol_roundtrip_and_object_array_rejection():
    payload = {"array": np.arange(6, dtype=np.float32).reshape(2, 3), "name": "xr1"}
    decoded = MsgSerializer.from_bytes(MsgSerializer.to_bytes(payload))

    np.testing.assert_array_equal(decoded["array"], payload["array"])
    assert decoded["name"] == "xr1"
    with pytest.raises(TypeError, match="object-dtype"):
        MsgSerializer.to_bytes({"bad": np.asarray([object()], dtype=object)})


def test_server_builds_official_no_cot_prompt_and_padded_state():
    processor = _Processor()
    decoded_actions = np.arange(70, dtype=np.float32).reshape(1, 10, 7)
    backend = _Model(decoded_actions)
    model = XR1RoboCasaModel.__new__(XR1RoboCasaModel)
    model.processor = processor
    model.model = backend
    model.device = "cuda"
    model.dtype = "bfloat16"
    model.torch = _Torch()

    state = np.arange(8, dtype=np.float32)
    actions = model.infer(
        {
            "protocol_version": 1,
            "robot_type": "robocasa_mg",
            "images": {
                "base_left": np.zeros((8, 8, 3), dtype=np.uint8),
                "base_right": np.ones((8, 8, 3), dtype=np.uint8),
                "wrist": np.full((8, 8, 3), 2, dtype=np.uint8),
            },
            "state": state,
            "instruction": "place the mug in the cabinet",
            "seed": 7,
        }
    )

    np.testing.assert_array_equal(actions, decoded_actions[0])
    assert processor.messages[0]["content"][0]["text"].endswith("# Base View\n")
    assert processor.messages[0]["content"][3]["text"] == "\n# Left-Wrist View\n"
    assert processor.messages[0]["content"][-1]["text"].endswith("place the mug in the cabinet /no_cot")
    assert processor.messages[1]["content"][0]["text"] == "<cot></cot>"
    padded_state = processor.template_kwargs["state"]
    assert padded_state.shape == (1, 1, 60)
    np.testing.assert_array_equal(padded_state[0, 0, :8], state)
    np.testing.assert_array_equal(padded_state[0, 0, 8:], np.zeros(52, dtype=np.float32))
    assert processor.template_kwargs["robot_type"] == "robocasa_mg"
    assert backend.inputs["task_id"] == "robocasa_mg"
    assert backend.inputs["seed"] == 7


def test_close_releases_transport():
    transport = _PolicyClient()
    client = XR1RoboCasaClient(policy_client=transport)

    client.close()

    assert transport.closed
