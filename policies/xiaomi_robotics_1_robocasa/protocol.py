# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Dependency-light wire protocol shared by the XR-1 client and model server."""

from __future__ import annotations

import functools
from typing import Any

import msgpack
import msgpack_numpy as mnp
import numpy as np

PROTOCOL_VERSION = 1
ROBOT_TYPE = "robocasa_mg"


class MsgSerializer:
    """Serialize numeric numpy payloads while rejecting object arrays."""

    @staticmethod
    def to_bytes(data: Any) -> bytes:
        default = functools.partial(MsgSerializer._safe_encode, chain=lambda obj: obj)
        return msgpack.packb(data, default=default, use_bin_type=True)

    @staticmethod
    def from_bytes(data: bytes) -> Any:
        object_hook = functools.partial(MsgSerializer._safe_decode, chain=lambda obj: obj)
        return msgpack.unpackb(data, object_hook=object_hook, raw=False, strict_map_key=False)

    @staticmethod
    def _safe_encode(obj: Any, chain=None) -> Any:
        if isinstance(obj, np.ndarray) and obj.dtype.kind == "O":
            raise TypeError(
                f"Refusing to encode object-dtype ndarray (shape={obj.shape}); "
                "convert it to a concrete numeric dtype first."
            )
        return mnp.encode(obj, chain=chain)

    @staticmethod
    def _safe_decode(obj: Any, chain=None) -> Any:
        if isinstance(obj, dict):
            nd_value = obj.get(b"nd", obj.get("nd"))
            kind_value = obj.get(b"kind", obj.get("kind"))
            if nd_value and kind_value in (b"O", "O"):
                raise ValueError("Refusing to decode object-dtype ndarray payload.")
        return mnp.decode(obj, chain=chain)
