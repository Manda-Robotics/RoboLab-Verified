# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Raw-observation model server for Xiaomi-Robotics-1-RoboCasa.

Run this module in Xiaomi's pinned PyTorch/Transformers environment. The
RoboLab process only needs the lightweight client and never imports remote
Hugging Face model code.
"""

from __future__ import annotations

import argparse
import traceback
from typing import Any

import numpy as np
import zmq
from PIL import Image

from .protocol import PROTOCOL_VERSION, ROBOT_TYPE, MsgSerializer

STATE_DIM = 60
ACTION_DIM = 7
DEFAULT_MODEL = "XiaomiRobotics/Xiaomi-Robotics-1-RoboCasa"


def build_messages(images: dict[str, Image.Image], instruction: str) -> list[dict[str, Any]]:
    """Reproduce Xiaomi's official RoboCasa no-CoT prompt exactly."""
    return [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": "The following observations are captured from multiple views.\n# Base View\n",
                },
                {"type": "image", "image": images["base_left"]},
                {"type": "image", "image": images["base_right"]},
                {"type": "text", "text": "\n# Left-Wrist View\n"},
                {"type": "image", "image": images["wrist"]},
                {
                    "type": "text",
                    "text": f"\nGenerate robot actions for the task:\n{instruction} /no_cot",
                },
            ],
        },
        {
            "role": "assistant",
            "content": [{"type": "text", "text": "<cot></cot>"}],
        },
    ]


class XR1RoboCasaModel:
    """Load the checkpoint, construct processor inputs, and decode actions."""

    def __init__(
        self,
        model_path: str,
        *,
        revision: str | None,
        device: str,
        dtype: str,
        attn_implementation: str,
    ) -> None:
        import torch
        from transformers import AutoModel, AutoProcessor

        if device.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError(f"CUDA device requested ({device}) but CUDA is unavailable")
        dtype_map = {
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
            "float32": torch.float32,
        }
        torch_dtype = dtype_map[dtype]
        common_kwargs: dict[str, Any] = {
            "trust_remote_code": True,
        }
        if revision:
            common_kwargs["revision"] = revision

        self.model_path = model_path
        self.processor = AutoProcessor.from_pretrained(
            model_path,
            use_fast=False,
            **common_kwargs,
        )
        robot_types = list(self.processor.list_robot_types())
        if ROBOT_TYPE not in robot_types:
            raise RuntimeError(f"Checkpoint does not provide robot_type={ROBOT_TYPE!r}; available={robot_types}")
        self.model = (
            AutoModel.from_pretrained(
                model_path,
                attn_implementation=attn_implementation,
                dtype=torch_dtype,
                **common_kwargs,
            )
            .to(device)
            .eval()
        )
        self.device = torch.device(device)
        self.dtype = torch_dtype
        self.torch = torch

    @staticmethod
    def _validate_image(value: Any, name: str) -> Image.Image:
        array = np.asarray(value)
        if array.ndim != 3 or array.shape[-1] != 3:
            raise ValueError(f"images[{name!r}] must be HWC RGB, got {array.shape}")
        return Image.fromarray(np.clip(array, 0, 255).astype(np.uint8))

    def infer(self, request: dict[str, Any]) -> np.ndarray:
        if request.get("protocol_version") != PROTOCOL_VERSION:
            raise ValueError(
                f"Unsupported protocol_version={request.get('protocol_version')!r}; expected {PROTOCOL_VERSION}"
            )
        robot_type = request.get("robot_type")
        if robot_type != ROBOT_TYPE:
            raise ValueError(f"Expected robot_type={ROBOT_TYPE!r}, got {robot_type!r}")

        raw_images = request.get("images")
        if not isinstance(raw_images, dict):
            raise ValueError("Request field 'images' must be a mapping")
        images = {name: self._validate_image(raw_images[name], name) for name in ("base_left", "base_right", "wrist")}
        state = np.asarray(request.get("state"), dtype=np.float32).reshape(-1)
        if state.shape != (8,):
            raise ValueError(f"Expected state=[7 joints, 1 gripper] with shape (8,), got {state.shape}")
        padded_state = np.zeros(STATE_DIM, dtype=np.float32)
        padded_state[: state.size] = state

        model_inputs = self.processor.apply_chat_template(
            build_messages(images, str(request.get("instruction", ""))),
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
            state=padded_state.reshape(1, 1, STATE_DIM),
            robot_type=robot_type,
        )
        data = dict(model_inputs)
        data["task_id"] = robot_type
        data["seed"] = int(request.get("seed", 42))
        data = {
            key: (
                value.to(device=self.device, dtype=self.dtype)
                if self.torch.is_tensor(value) and value.is_floating_point()
                else value.to(device=self.device)
                if self.torch.is_tensor(value)
                else value
            )
            for key, value in data.items()
        }

        with self.torch.inference_mode():
            outputs = self.model(**data)
            decoded = self.processor.decode_action(outputs.actions, robot_type=robot_type)
        actions = decoded.detach().cpu().float().numpy()
        if actions.ndim != 3 or actions.shape[0] != 1 or actions.shape[2] < ACTION_DIM:
            raise RuntimeError(f"Unexpected decoded XR-1 action shape: {actions.shape}")
        return actions[0, :, :ACTION_DIM].astype(np.float32)


def serve(model: XR1RoboCasaModel, *, host: str, port: int) -> None:
    context = zmq.Context()
    socket = context.socket(zmq.REP)
    socket.setsockopt(zmq.LINGER, 0)
    address = f"tcp://{host}:{port}"
    socket.bind(address)
    print(f"[XR1RoboCasaServer] Listening on {address}", flush=True)
    try:
        while True:
            try:
                request = MsgSerializer.from_bytes(socket.recv())
                if not isinstance(request, dict):
                    raise ValueError("Request must decode to a mapping")
                endpoint = request.get("endpoint")
                if endpoint == "ping":
                    response = {
                        "protocol_version": PROTOCOL_VERSION,
                        "model": model.model_path,
                        "robot_type": ROBOT_TYPE,
                    }
                elif endpoint == "infer":
                    response = {
                        "protocol_version": PROTOCOL_VERSION,
                        "actions": model.infer(request),
                    }
                else:
                    raise ValueError(f"Unknown endpoint: {endpoint!r}")
            except Exception as exc:
                traceback.print_exc()
                response = {"error": f"{type(exc).__name__}: {exc}"}
            socket.send(MsgSerializer.to_bytes(response))
    except KeyboardInterrupt:
        print("\n[XR1RoboCasaServer] Shutting down.", flush=True)
    finally:
        socket.close(linger=0)
        context.term()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Serve Xiaomi-Robotics-1-RoboCasa for RoboLab.")
    parser.add_argument(
        "--model-path", default=DEFAULT_MODEL, help="Hugging Face model id or local checkpoint directory."
    )
    parser.add_argument(
        "--revision", default=None, help="Optional Hugging Face commit revision; recommended for reproducibility."
    )
    parser.add_argument("--host", default="127.0.0.1", help="Bind host (default: loopback only).")
    parser.add_argument("--port", type=int, default=10086)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", choices=["bfloat16", "float16", "float32"], default="bfloat16")
    parser.add_argument("--attn-implementation", default="flash_attention_2")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model = XR1RoboCasaModel(
        args.model_path,
        revision=args.revision,
        device=args.device,
        dtype=args.dtype,
        attn_implementation=args.attn_implementation,
    )
    serve(model, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
