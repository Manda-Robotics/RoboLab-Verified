"""GA camera/reset compatibility without starting Kit (run with tests/newton)."""
import ast
import importlib.util
from pathlib import Path
from types import SimpleNamespace as NS

import pytest
import torch

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("ga_compat", ROOT / "robolab/core/utils/isaaclab_compat.py")
compat = importlib.util.module_from_spec(spec)
spec.loader.exec_module(compat)


@pytest.mark.parametrize("ids,expected", [
    (None, [0, 1, 2, 3]), (slice(None), [0, 1, 2, 3]),
    (slice(1, 4, 2), [1, 3]), (slice(3, 1), []),
    (slice(-2, None), [2, 3]), ([3, 1], [3, 1]), ([], []),
])
def test_reset_selectors_preserve_selected_environments(ids, expected):
    actual = compat.env_ids_tensor(ids, 4, "cpu")
    assert actual.dtype == torch.long and actual.tolist() == expected


def test_reset_tensor_does_not_copy_and_rejects_reverse_slice():
    indices = torch.tensor([3, 1])
    assert compat.env_ids_tensor(indices, 4, "cpu") is indices
    with pytest.raises(ValueError, match="positive step"):
        compat.env_ids_tensor(slice(None, None, -1), 4, "cpu")


def image_functions(mdp):
    # Execute production camera functions with only the Isaac type annotations
    # substituted. Sensor buffers use the same ProxyArray.torch contract as GA.
    tree = ast.parse((ROOT / "robolab/core/observations/observation_utils.py").read_text())
    names = {"_legacy_image_observation_func", "_image_observation_func", "image_safe"}
    nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names]
    namespace = {"torch": torch, "mdp": mdp, "Any": object, "as_torch": compat.as_torch,
                 "SceneEntityCfg": lambda name: NS(name=name)}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), "<camera functions>", "exec"), namespace)
    return namespace


def camera_env(output):
    camera = NS(cfg=NS(height=2, width=2), data=NS(output=output))
    return NS(scene=NS(sensors={"camera": camera}), num_envs=2, device="cpu")


def test_ga_rgb_preserves_policy_format_and_owns_storage():
    funcs = image_functions(NS())
    assert funcs["_image_observation_func"]() is funcs["image_safe"]
    rgba = torch.arange(32, dtype=torch.uint8).reshape(2, 2, 2, 4)
    env = camera_env({"rgb": NS(torch=rgba)})
    raw = funcs["image_safe"](env, NS(name="camera"), normalize=False)
    assert raw.shape == (2, 2, 2, 3) and raw.dtype == torch.uint8
    torch.testing.assert_close(raw, rgba[..., :3])
    raw.zero_()  # downstream corruption/recording must not modify sensor output
    assert rgba.sum() > 0
    normalized = funcs["image_safe"](env, NS(name="camera"))
    expected = rgba[..., :3].float() / 255
    expected -= expected.mean(dim=(1, 2), keepdim=True)
    torch.testing.assert_close(normalized, expected)


def test_ga_depth_stays_metric_and_does_not_modify_sensor_buffer():
    funcs = image_functions(NS())
    depth = torch.tensor([0.2, float("inf"), float("nan"), 3.5]).reshape(1, 2, 2, 1).repeat(2, 1, 1, 1)
    env = camera_env({"depth": NS(torch=depth)})
    raw = funcs["image_safe"](env, NS(name="camera"), "depth", False)
    torch.testing.assert_close(raw, depth, equal_nan=True)
    actual = funcs["image_safe"](env, NS(name="camera"), "depth", True)
    assert actual.flatten()[:4].tolist() == pytest.approx([0.2, 0, 0, 3.5])
    assert torch.isinf(depth).any() and torch.isnan(depth).any()


@pytest.mark.parametrize("data_type,normalize,dtype,channels", [
    ("rgb", False, torch.uint8, 3), ("rgb", True, torch.float32, 3),
    ("depth", False, torch.float32, 1),
])
def test_before_first_render_has_correct_observation_shape(data_type, normalize, dtype, channels):
    image = image_functions(NS())["image_safe"](camera_env({}), NS(name="camera"), data_type, normalize)
    assert image.shape == (2, 2, 2, channels) and image.dtype == dtype
    assert torch.count_nonzero(image) == 0


@pytest.mark.parametrize("nested", [True, False])
def test_legacy_isaaclab_image_api_is_retained(nested):
    calls = []
    def legacy(*args, **kwargs):
        calls.append(kwargs)
        return "legacy"
    mdp = NS(observations=NS(image=legacy)) if nested else NS(image=legacy)
    funcs = image_functions(mdp)
    assert funcs["_image_observation_func"]() is legacy
    assert funcs["image_safe"](camera_env({"rgb": object()}), NS(name="camera"), normalize=False) == "legacy"
    assert calls[0]["normalize"] is False
