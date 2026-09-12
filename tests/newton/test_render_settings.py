"""RTX settings reach the correct config on old wheels and the Lab 3 source tree."""

import importlib.util
from pathlib import Path
from types import SimpleNamespace as NS

import pytest

spec = importlib.util.spec_from_file_location("render_settings_under_test",
    Path(__file__).resolve().parents[2] / "robolab/core/environments/render_settings.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_legacy_quality_and_carb_settings_are_preserved():
    target = NS(carb_settings={"existing": 7}, rendering_mode=None)
    cfg = NS(sim=NS(render=target))
    module.apply_render_settings(cfg, "pathtracing", "quality")
    assert target.rendering_mode == "quality"
    assert target.carb_settings == {"existing": 7, "/rtx/rendermode": "PathTracing"}


def test_modern_settings_reach_every_camera_and_reject_removed_presets():
    left, right = NS(carb_settings={"existing": 7}), NS(carb_settings=None)
    cfg = NS(sim=NS(), scene=NS(
        left=NS(renderer_cfg=NS(renderer_type="isaac_rtx", global_settings=left)),
        right=NS(renderer_cfg=NS(renderer_type="isaac_rtx", global_settings=right)),
        robot=object()))
    module.apply_render_settings(cfg, "pathtracing", None)
    assert left.carb_settings == {"existing": 7, "/rtx/rendermode": "PathTracing"}
    assert right.carb_settings == {"/rtx/rendermode": "PathTracing"}
    with pytest.raises(ValueError, match="removed rendering quality presets"):
        module.apply_render_settings(cfg, "realtime", "balanced")
    with pytest.raises(ValueError, match="Unknown renderer"):
        module.apply_render_settings(cfg, "unsupported", None)
