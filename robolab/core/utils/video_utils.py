# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import atexit
import logging

import imageio.v2 as imageio
import numpy as np

logger = logging.getLogger(__name__)

_X264_PRESETS = {
    "ultrafast",
    "superfast",
    "veryfast",
    "faster",
    "fast",
    "medium",
    "slow",
    "slower",
    "veryslow",
}


class VideoWriter:
    """Streaming H.264 video writer.

    Pipes RGB frames to an ffmpeg subprocess (via imageio-ffmpeg) which encodes
    libx264/yuv420p with +faststart so the output plays in Chrome/Safari/Firefox.
    Memory is bounded by the encoder's lookahead, not by video length.
    """

    def __init__(
        self,
        video_path: str,
        fps: int | float,
        *,
        crf: int = 30,
        preset: str = "veryfast",
        max_width: int | None = 960,
    ):
        """Initialize video writer.

        Args:
            video_path: Path to output video file.
            fps: Frames per second.
            crf: H.264 constant-rate-factor quality. Higher values make smaller,
                lower-fidelity files; 30 is intended for visual inspection.
            preset: libx264 speed/compression preset. ``veryfast`` balances
                parallel evaluation throughput with substantially smaller files
                than ``ultrafast``.
            max_width: Downscale wider frames to this even width while preserving
                aspect ratio. Pass ``None`` to keep the source resolution.
        """
        # Keep partially initialized instances safe if validation raises and
        # Python subsequently invokes ``__del__``.
        self._writer = None
        if fps <= 0:
            raise ValueError("fps must be positive")
        if not 0 <= crf <= 51:
            raise ValueError("crf must be between 0 and 51")
        if preset not in _X264_PRESETS:
            raise ValueError(f"Unsupported libx264 preset: {preset}")
        if max_width is not None and (max_width < 2 or max_width % 2):
            raise ValueError("max_width must be an even integer >= 2 or None")
        self.video_path = video_path
        self.fps = fps
        self.crf = crf
        self.preset = preset
        self.max_width = max_width
        atexit.register(self.release)

    def write(self, frame: np.ndarray):
        if frame is None:
            print(f"No frame to write to video writer; nothing written to file '{self.video_path}'")
            return

        if self._writer is None:
            # macro_block_size=2 pads odd-dimensioned frames so libx264/yuv420p
            # (even-only) is safe. threads=1 prevents parallel environments from
            # multiplying each encoder across every CPU core.
            ffmpeg_params = [
                "-preset",
                self.preset,
                "-threads",
                "1",
                "-crf",
                str(self.crf),
            ]
            if self.max_width is not None:
                ffmpeg_params.extend(["-vf", f"scale='min({self.max_width},iw)':-2"])
            ffmpeg_params.extend(["-movflags", "+faststart"])
            self._writer = imageio.get_writer(
                self.video_path,
                fps=self.fps,
                codec="libx264",
                pixelformat="yuv420p",
                macro_block_size=2,
                ffmpeg_params=ffmpeg_params,
            )

        self._writer.append_data(frame)

    def release(self):
        if self._writer is not None:
            try:
                self._writer.close()
            except Exception:
                logger.exception("Failed to close video writer for '%s'", self.video_path)
            self._writer = None

    def __del__(self):
        self.release()
