# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Protocol and action-space tests for the MolmoAct2 and G0.5 adapters."""

from __future__ import annotations

import numpy as np
import pytest

from policies.g05_droid import client as g05_client_module
from policies.g05_droid.client import G05DroidClient, G05PolicyClient, pack_message, unpack_message
from policies.molmoact2_droid.client import MolmoAct2DroidClient


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


def _obs(*, closedness=(0.25,), num_envs=1):
    height, width = 12, 20
    closedness = np.asarray(closedness, dtype=np.float32).reshape(num_envs, 1)
    joints = np.stack([np.arange(7, dtype=np.float32) + 10 * env_id for env_id in range(num_envs)])
    exterior = np.stack([np.full((height, width, 3), env_id / 2, dtype=np.float32) for env_id in range(num_envs)])
    wrist = np.stack([np.full((height, width, 3), 64 + env_id, dtype=np.uint8) for env_id in range(num_envs)])
    return {
        "image_obs": {
            "over_shoulder_left_camera": _Tensor(exterior),
            "wrist_cam": _Tensor(wrist),
        },
        "proprio_obs": {
            "arm_joint_pos": _Tensor(joints),
            "gripper_pos": _Tensor(closedness),
        },
    }


class _MolmoTransport:
    def __init__(self, actions=None):
        if actions is None:
            actions = np.zeros((15, 8), dtype=np.float32)
        self.actions = np.asarray(actions, dtype=np.float32)
        self.requests = []
        self.closed = False

    def infer(self, request):
        self.requests.append(request)
        return {"actions": self.actions, "dt_ms": 1.0}

    def close(self):
        self.closed = True


class _G05Transport:
    def __init__(self, responses):
        self.responses = list(responses)
        self.requests = []
        self.reset_count = 0
        self.closed = False

    def infer(self, request):
        self.requests.append(request)
        return self.responses.pop(0)

    def reset(self):
        self.reset_count += 1

    def close(self):
        self.closed = True


def _g05_response(arm_offset=0.0, *, gripper=0.25, need_obs=True, include_gripper=True):
    action = {"right_arm": np.arange(7, dtype=np.float32) + arm_offset}
    if include_gripper:
        action["right_gripper"] = np.asarray([gripper], dtype=np.float32)
    return {"action": action, "need_obs": need_obs}


def test_molmo_request_and_fifteen_step_chunk_contract():
    actions = np.zeros((15, 8), dtype=np.float32)
    actions[:, :7] = np.arange(15, dtype=np.float32)[:, None]
    actions[:, 7] = np.linspace(0.0, 1.0, 15)
    transport = _MolmoTransport(actions)
    client = MolmoAct2DroidClient(policy_client=transport)

    results = [client.infer(_obs(), "pick up the fruit") for _ in range(15)]

    assert len(transport.requests) == 1
    request = transport.requests[0]
    assert set(request) == {"external_cam", "wrist_cam", "instruction", "state"}
    assert request["external_cam"].shape == (12, 20, 3)
    assert request["external_cam"].dtype == np.uint8
    np.testing.assert_array_equal(request["external_cam"], np.zeros((12, 20, 3), dtype=np.uint8))
    np.testing.assert_array_equal(request["wrist_cam"], np.full((12, 20, 3), 64, dtype=np.uint8))
    np.testing.assert_allclose(request["state"], [0, 1, 2, 3, 4, 5, 6, 0.25])
    assert request["state"].dtype == np.float32
    assert request["instruction"] == "pick up the fruit"
    assert [int(result["action"][-1]) for result in results] == [0] * 8 + [1] * 7

    client.infer(_obs(), "pick up the fruit")
    assert len(transport.requests) == 2


def test_molmo_rejects_short_or_nonfinite_chunks():
    client = MolmoAct2DroidClient(policy_client=_MolmoTransport(np.zeros((14, 8), dtype=np.float32)))
    with pytest.raises(ValueError, match="fewer than open_loop_horizon"):
        client.infer(_obs(), "task")

    actions = np.zeros((15, 8), dtype=np.float32)
    actions[0, 0] = np.nan
    client = MolmoAct2DroidClient(policy_client=_MolmoTransport(actions))
    with pytest.raises(ValueError, match="non-finite"):
        client.infer(_obs(), "task")


def test_g05_full_request_camera_order_state_and_gripper_polarity():
    transport = _G05Transport([_g05_response(gripper=0.2)])
    client = G05DroidClient(policy_client=transport)

    result = client.infer(_obs(closedness=(0.25,)), "put the banana in the bowl")

    request = transport.requests[0]
    assert request["task"] == "put the banana in the bowl"
    assert request["frequency"] == 15
    assert request["embodiment_type"] == "Droid_Franka"
    assert set(request["images"]) == {"exterior_image", "wrist_image", "dummy_wrist_right"}
    assert request["images"]["exterior_image"].shape == (3, 12, 20)
    assert request["images"]["wrist_image"].shape == (3, 12, 20)
    assert request["images"]["dummy_wrist_right"].shape == (3, 224, 224)
    assert request["images"]["dummy_wrist_right"].dtype == np.uint8
    np.testing.assert_array_equal(request["state"]["right_arm"], np.arange(7, dtype=np.float32))
    assert request["state"]["right_gripper"].shape == (1,)
    assert request["state"]["right_gripper"].dtype == np.float32
    np.testing.assert_allclose(request["state"]["right_gripper"], 0.75)
    np.testing.assert_array_equal(result["action"][:7], np.arange(7, dtype=np.float32))
    assert result["action"][7] == 1.0  # G0.5 0.2=open amount -> RoboLab closed.


def test_g05_advances_server_chunk_without_observation_and_holds_missing_gripper():
    transport = _G05Transport(
        [
            _g05_response(need_obs=False),
            _g05_response(arm_offset=10, include_gripper=False, need_obs=True),
            _g05_response(arm_offset=20, gripper=0.8, need_obs=True),
        ]
    )
    client = G05DroidClient(policy_client=transport)

    client.infer(_obs(closedness=(0.1,)), "task")
    cached = client.infer(_obs(closedness=(0.9,)), "task")
    client.infer(_obs(closedness=(0.1,)), "task")

    assert transport.requests[1] == {}
    assert "images" in transport.requests[0]
    assert "images" in transport.requests[2]
    np.testing.assert_array_equal(cached["action"][:7], np.arange(7, dtype=np.float32) + 10)
    assert cached["action"][7] == 1.0  # Missing key holds the current observed closed gripper.


def test_g05_parallel_envs_use_independent_connections_and_resets():
    transports = {
        0: _G05Transport([_g05_response(need_obs=False), _g05_response()]),
        1: _G05Transport([_g05_response(arm_offset=10)]),
    }
    client = G05DroidClient(policy_client_factory=transports.__getitem__)

    result = client.infer_batch(_obs(closedness=(0.1, 0.9), num_envs=2), "task", env_ids=[0, 1])

    assert set(result) == {0, 1}
    assert len(transports[0].requests) == 1
    assert len(transports[1].requests) == 1
    np.testing.assert_allclose(transports[0].requests[0]["state"]["right_gripper"], 0.9, atol=1e-7)
    np.testing.assert_allclose(transports[1].requests[0]["state"]["right_gripper"], 0.1, atol=1e-7)

    client.reset(env_id=0)
    assert transports[0].reset_count == 1
    assert transports[1].reset_count == 0
    client.infer(_obs(closedness=(0.1, 0.9), num_envs=2), "task", env_id=0)
    assert "images" in transports[0].requests[1]


def test_g05_numeric_msgpack_roundtrip_and_object_array_rejection():
    payload = {
        "array": np.arange(6, dtype=np.float32).reshape(2, 3),
        "scalar": np.float32(0.5),
        "name": "g05",
    }

    decoded = unpack_message(pack_message(payload))

    np.testing.assert_array_equal(decoded["array"], payload["array"])
    assert decoded["array"].dtype == np.float32
    assert decoded["scalar"] == np.float32(0.5)
    assert decoded["name"] == "g05"
    with pytest.raises(ValueError, match="Unsupported NumPy dtype"):
        pack_message({"bad": np.asarray([object()], dtype=object)})


def test_g05_transport_handshake_inference_and_reset_ack(monkeypatch):
    class _Socket:
        def __init__(self):
            self.frames = [
                pack_message({"action_steps": 16}),
                pack_message(_g05_response()),
                pack_message({"__reset__": True}),
            ]
            self.sent = []
            self.closed = False

        def recv(self, timeout=None):
            assert timeout == 12.0
            return self.frames.pop(0)

        def send(self, frame):
            self.sent.append(unpack_message(frame))

        def close(self):
            self.closed = True

    socket = _Socket()

    def _connect(uri, **kwargs):
        assert uri == "ws://policy.example:8123"
        assert kwargs == {
            "compression": None,
            "max_size": None,
            "open_timeout": 12.0,
        }
        return socket

    monkeypatch.setattr(g05_client_module.websockets.sync.client, "connect", _connect)
    client = G05PolicyClient(host="http://policy.example:8123", timeout_seconds=12.0)

    assert client.action_steps == 16
    response = client.infer({"observation": np.arange(3, dtype=np.float32)})
    assert response["need_obs"] is True
    client.reset()
    client.close()

    np.testing.assert_array_equal(socket.sent[0]["observation"], np.arange(3, dtype=np.float32))
    assert socket.sent[1] == {"__reset__": True}
    assert socket.closed


def test_adapter_close_releases_transports():
    molmo_transport = _MolmoTransport()
    g05_transport = _G05Transport([])
    molmo_client = MolmoAct2DroidClient(policy_client=molmo_transport)
    g05_client = G05DroidClient(policy_client=g05_transport)

    molmo_client.close()
    g05_client.close()

    assert molmo_transport.closed
    assert g05_transport.closed
