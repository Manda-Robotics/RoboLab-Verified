"""Shared seed override parsing does not require simulator startup."""

import argparse
import importlib.util
from pathlib import Path
import sys
from types import ModuleType

import pytest


@pytest.mark.parametrize("legacy", [False, True])
def test_launch_flags_survive_new_and_old_lab(monkeypatch, legacy):
    from types import SimpleNamespace
    path = Path(__file__).resolve().parents[2] / "robolab/core/utils/app_launcher.py"
    spec = importlib.util.spec_from_file_location("app_args_under_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    def upstream(parser):
        if legacy:
            parser.add_argument("--headless", action="store_true")
            parser.add_argument("--enable_cameras", action="store_true")
        parser.add_argument("--device", default="cuda:0")
    app = ModuleType("isaaclab.app")
    if legacy:
        app.AppLauncher = SimpleNamespace(add_app_launcher_args=upstream)
    else:
        app.add_launcher_args = upstream
    monkeypatch.setitem(sys.modules, "isaaclab.app", app)
    parser = argparse.ArgumentParser()
    module.add_app_launcher_args(parser)
    args = parser.parse_args(["--headless", "--enable_cameras"])
    assert args.headless and args.enable_cameras
    assert args.device == "cuda:0"
    assert not parser.parse_args([]).headless


def test_environment_seed_override_is_optional_and_accepts_zero(monkeypatch):
    constants = ModuleType("robolab.constants")
    constants.DEFAULT_TASK_SUBFOLDERS = ["benchmark"]
    monkeypatch.setitem(sys.modules, "robolab.constants", constants)
    path = Path(__file__).resolve().parents[2] / "robolab/eval/runner.py"
    spec = importlib.util.spec_from_file_location("eval_args_under_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    parser = argparse.ArgumentParser()
    module.add_common_eval_args(parser)
    assert parser.parse_args([]).seed is None
    assert not parser.parse_args([]).newton_disable_cuda_graph
    assert parser.parse_args(["--newton-disable-cuda-graph"]).newton_disable_cuda_graph
    assert parser.parse_args(["--seed", "0"]).seed == 0
    assert parser.parse_args(["--seed", "7"]).seed == 7
    assert parser.parse_args([]).newton_convex_decomposition_threshold is None
    assert parser.parse_args(["--newton-convex-decomposition-threshold", ".05"]).newton_convex_decomposition_threshold == .05
    for invalid in ["0", "-0.1", "1", "nan", "inf"]:
        with pytest.raises(SystemExit):
            parser.parse_args(["--newton-convex-decomposition-threshold", invalid])
