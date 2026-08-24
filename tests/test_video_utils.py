# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest


def _video_utils_module():
    path = Path(__file__).resolve().parents[1] / "robolab/core/utils/video_utils.py"
    spec = importlib.util.spec_from_file_location("video_utils_under_test", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class _FakeImageioWriter:
    def __init__(self):
        self.frames = []
        self.closed = False

    def append_data(self, frame):
        self.frames.append(frame)

    def close(self):
        self.closed = True


def test_default_video_profile_is_browser_ready_and_storage_efficient(monkeypatch):
    video_utils = _video_utils_module()
    fake = _FakeImageioWriter()
    captured = {}

    def get_writer(path, **kwargs):
        captured.update({"path": path, **kwargs})
        return fake

    monkeypatch.setattr(video_utils.imageio, "get_writer", get_writer)
    writer = video_utils.VideoWriter("episode.mp4", 15)
    frame = np.zeros((360, 1280, 3), dtype=np.uint8)
    writer.write(frame)
    writer.release()

    params = captured["ffmpeg_params"]
    assert params[params.index("-preset") + 1] == "veryfast"
    assert params[params.index("-crf") + 1] == "30"
    assert params[params.index("-vf") + 1] == "scale='min(960,iw)':-2"
    assert params[params.index("-movflags") + 1] == "+faststart"
    assert captured["codec"] == "libx264"
    assert captured["pixelformat"] == "yuv420p"
    assert len(fake.frames) == 1
    assert fake.frames[0] is frame
    assert fake.closed


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"fps": 0}, "fps must be positive"),
        ({"fps": 15, "crf": 52}, "crf must be between"),
        ({"fps": 15, "preset": "instant"}, "Unsupported libx264 preset"),
        ({"fps": 15, "max_width": 959}, "max_width must be an even integer"),
    ],
)
def test_video_profile_rejects_invalid_encoder_settings(kwargs, message):
    video_utils = _video_utils_module()
    with pytest.raises(ValueError, match=message):
        video_utils.VideoWriter("episode.mp4", **kwargs)
