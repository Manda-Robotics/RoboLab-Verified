# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Offline tests for the MolmoAct 2 bimanual-YAM client: wire format, state/action mapping,
and a full infer() round trip against a fake ``/act`` server. No Isaac, no network."""
import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import numpy as np
import pytest

from policies.molmoact2 import yam_client as yc


def test_json_numpy_round_trip_matches_reference_format():
    img = (np.arange(2 * 3 * 3) % 255).astype(np.uint8).reshape(2, 3, 3)
    state = np.linspace(-1, 1, 14, dtype=np.float32)
    raw = yc.np_dumps({"top_cam": img, "state": state, "instruction": "x"})
    d = json.loads(raw)
    assert set(d["top_cam"]) == {"__numpy__", "dtype", "shape"}
    assert d["top_cam"]["dtype"] == "|u1" and d["top_cam"]["shape"] == [2, 3, 3]
    assert d["state"]["dtype"] == "<f4"
    back = yc.np_loads(raw)
    assert np.array_equal(back["top_cam"], img) and np.array_equal(back["state"], state)


def test_build_state_order_and_gripper_convention():
    s = yc.build_state(np.arange(6), 1.0, np.arange(10, 16), 0.0)
    assert s.shape == (14,) and s.dtype == np.float32
    assert list(s[:6]) == [0, 1, 2, 3, 4, 5] and s[6] == 1.0
    assert list(s[7:13]) == [10, 11, 12, 13, 14, 15] and s[13] == 0.0


def test_expand_chunk_puts_gripper_in_both_finger_slots():
    a = np.zeros((2, 14), np.float32)
    a[0, :6] = 0.1; a[0, 6] = 1.0          # left arm, left gripper fully open
    a[1, 7:13] = 0.2; a[1, 13] = 0.5       # right arm, right gripper half open
    e = yc.expand_chunk(a)
    assert e.shape == (2, 16)
    assert np.allclose(e[0, :6], 0.1) and np.allclose(e[0, 6:8], -yc.FINGER_TRAVEL_M)
    assert np.allclose(e[0, 8:16], 0.0)
    assert np.allclose(e[1, 8:14], 0.2) and np.allclose(e[1, 14:16], -0.5 * yc.FINGER_TRAVEL_M)
    assert np.allclose(e[1, :8], 0.0)


def test_expand_chunk_clips_gripper_outside_unit_interval():
    a = np.zeros((1, 14), np.float32); a[0, 6] = 1.7; a[0, 13] = -0.3
    e = yc.expand_chunk(a)
    assert np.allclose(e[0, 6:8], -yc.FINGER_TRAVEL_M) and np.allclose(e[0, 14:16], 0.0)


class _FakeAct(BaseHTTPRequestHandler):
    seen = []

    def do_POST(self):
        body = self.rfile.read(int(self.headers["Content-Length"]))
        payload = yc.np_loads(body)
        _FakeAct.seen.append(payload)
        acts = np.zeros((30, 14), np.float32)
        acts[:, 0] = np.linspace(0, 1, 30)   # left joint1 ramps
        acts[:, 6] = 1.0                      # left gripper open
        out = yc.np_dumps({"actions": acts, "dt_ms": 1.0})
        self.send_response(200); self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(out))); self.end_headers(); self.wfile.write(out)

    def log_message(self, *a):
        pass


@pytest.fixture
def fake_server():
    srv = HTTPServer(("127.0.0.1", 0), _FakeAct)
    t = threading.Thread(target=srv.serve_forever, daemon=True); t.start()
    yield f"http://127.0.0.1:{srv.server_port}"
    srv.shutdown()


def _raw_obs(n_env=2):
    img = np.zeros((n_env, 360, 640, 3), np.uint8)
    return {
        "image_obs": {"top_cam": img, "left_wrist_cam": img + 1, "right_wrist_cam": img + 2},
        "proprio_obs": {
            "left_arm_joint_pos": np.tile(np.arange(6, dtype=np.float32), (n_env, 1)),
            "left_gripper_pos": np.ones((n_env, 1), np.float32),
            "right_arm_joint_pos": np.zeros((n_env, 6), np.float32),
            "right_gripper_pos": np.zeros((n_env, 1), np.float32),
        },
    }


def test_infer_round_trip_against_fake_server(fake_server):
    _FakeAct.seen.clear()
    client = yc.MolmoAct2YamClient(server=fake_server)
    out = client.infer(_raw_obs(), "put everything into the box", env_id=1)
    action = out["action"]
    assert action.shape == (16,)
    assert action[0] == 0.0 and np.allclose(action[6:8], -yc.FINGER_TRAVEL_M)
    sent = _FakeAct.seen[-1]
    assert sent["normalization_tag"] == yc.NORM_TAG and sent["num_steps"] == 10
    assert sent["top_cam"].shape == (360, 640, 3) and sent["top_cam"].dtype == np.uint8
    assert sent["left_cam"][0, 0, 0] == 1 and sent["right_cam"][0, 0, 0] == 2   # env row 1, camera mapping
    assert sent["state"].shape == (14,) and sent["state"][6] == 1.0
    # Whole chunk plays before the next request (open_loop_horizon = 30).
    for i in range(1, 30):
        a = client.infer(_raw_obs(), "x", env_id=1)["action"]
        assert np.isclose(a[0], i / 29)
    assert len(_FakeAct.seen) == 1
    client.infer(_raw_obs(), "x", env_id=1)
    assert len(_FakeAct.seen) == 2
    assert out["viz"].shape == (360, 3 * 640, 3)
