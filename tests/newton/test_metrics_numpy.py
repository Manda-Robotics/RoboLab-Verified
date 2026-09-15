"""Reporting must support both Isaac Sim 5's NumPy and NumPy without trapz."""
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

spec = importlib.util.spec_from_file_location("trajectory_metrics_under_test",
    Path(__file__).resolve().parents[2] / "robolab/core/metrics/trajectory_metrics.py")
metrics = importlib.util.module_from_spec(spec)
spec.loader.exec_module(metrics)


@pytest.mark.parametrize("name", ["trapezoid", "trapz"])
def test_trapezoid_api_compatibility(monkeypatch, name):
    calls = []
    def integrate(values, **kwargs):
        calls.append((values, kwargs))
        return 7.5
    monkeypatch.setattr(metrics, "np", SimpleNamespace(**{name: integrate}))
    assert metrics._trapezoid([1, 2], dx=.5, axis=0) == 7.5
    assert calls == [([1, 2], {"dx": .5, "axis": 0})]


@pytest.mark.parametrize("name", [
    "compute_joint_isj_from_velocity", "compute_joint_isj_per_joint_from_velocity",
    "compute_joint_isj_from_position", "compute_joint_isj_per_joint_from_position",
    "compute_ee_isj_from_position", "compute_ee_isj_from_velocity",
])
def test_stationary_jerk_metrics(name):
    np.testing.assert_array_equal(getattr(metrics, name)(np.zeros((8, 3)), .1), 0)
