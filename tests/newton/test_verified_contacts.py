"""Verified's dense contact channels must keep object identity under Newton.

Execute the real sensor builder and recorder with CPU tensors; Isaac/Kit is not
imported. GPU contact generation itself still requires the runtime probes.
"""
import ast
import logging
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace
from typing import Any

import numpy as np
import pytest

torch = pytest.importorskip("torch")
ROOT = Path(__file__).resolve().parents[2]


def definitions(relative, names, namespace):
    path = ROOT / relative
    tree = ast.parse(path.read_text())
    selected = [n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.ClassDef)) and n.name in names]
    assert len(selected) == len(names)
    exec(compile(ast.Module(body=selected, type_ignores=[]), str(path), "exec"), namespace)


class RecorderTerm:
    def __init__(self, cfg, env):
        self._env = env


@pytest.fixture
def code(monkeypatch):
    ns = dict(torch=torch, np=np, Any=Any, RecorderTerm=RecorderTerm,
              logger=logging.getLogger(__name__), ContactSensorCfg=SimpleNamespace)
    definitions("robolab/core/utils/isaaclab_compat.py", {"as_torch"}, ns)
    compat = ModuleType("robolab.core.utils.isaaclab_compat")
    compat.as_torch = ns["as_torch"]
    monkeypatch.setitem(sys.modules, compat.__name__, compat)
    definitions("robolab/core/sensors/contact_sensor_utils.py",
                {"create_contact_sensor_cfg", "configure_newton_extra_contacts"}, ns)
    definitions("robolab/core/events/dense_recorders.py",
                {"body_force_magnitudes", "PostStepExtraContactRecorder"}, ns)
    return ns


def config(enabled=True):
    return SimpleNamespace(
        physics_backend="newton_mjwarp", contact_extra_bodies={"wrist": "/robot/wrist"},
        contact_object_list=["cube", "bowl", "table"],
        scene=SimpleNamespace(wrist__all_objs=object() if enabled else None,
                              **{n: SimpleNamespace(prim_path=f"/scene/{n}") for n in ("cube", "bowl", "table")}))


def test_newton_splits_only_enabled_extra_body_sensors(code):
    cfg = config()
    code["configure_newton_extra_contacts"](cfg)
    assert cfg.scene.wrist__all_objs is None
    for name in cfg.contact_object_list:
        sensor = getattr(cfg.scene, f"wrist__{name}")
        assert sensor.prim_path == "/robot/wrist"
        assert sensor.filter_prim_paths_expr == [f"/scene/{name}"]
    disabled = config(False)
    code["configure_newton_extra_contacts"](disabled)
    assert not hasattr(disabled.scene, "wrist__cube")


def sensor(tensor):
    # Isaac Lab may return a ProxyArray; the actual as_torch helper unwraps it.
    return SimpleNamespace(data=SimpleNamespace(force_matrix_w=SimpleNamespace(torch=tensor)))


def test_newton_records_named_objects_across_envs_and_filter_bodies(code):
    cfg = config()
    cube = torch.zeros(2, 1, 2, 3)
    cube[0, 0, 0] = torch.tensor([3., 4., 0.])
    cube[0, 0, 1] = torch.tensor([0., 0., 2.])
    bowl = torch.zeros(2, 1, 1, 3)
    bowl[1, 0, 0, 2] = 11
    sensors = {"wrist__bowl": sensor(bowl), "wrist__cube": sensor(cube),
               "wrist__table": sensor(torch.zeros(2, 1, 1, 3)),
               # This batch is deliberately wrong; Newton must not read it.
               "wrist__all_objs": sensor(torch.full((2, 1, 3, 3), 100.))}
    env = SimpleNamespace(cfg=cfg, scene=SimpleNamespace(sensors=sensors))
    recorder = code["PostStepExtraContactRecorder"](None, env)
    key, out = recorder.record_post_step()
    assert key == "contact_body"
    assert out["wrist__cube"].tolist() == [7., 0.]
    assert out["wrist__bowl"].tolist() == [0., 11.]
    assert out["wrist__table"].tolist() == [0., 0.]
    assert all(v.dtype == torch.float32 for v in out.values())
    cube.zero_()
    assert recorder.record_post_step()[1]["wrist__cube"].tolist() == [0., 0.]


def test_physx_keeps_batch_channel_names_and_values(code):
    cfg = config()
    cfg.physics_backend = "physx"
    forces = torch.zeros(2, 1, 3, 3)
    forces[0, 0, 1, 2] = 5
    forces[1, 0, 2, 0] = 3
    env = SimpleNamespace(cfg=cfg, scene=SimpleNamespace(sensors={"wrist__all_objs": sensor(forces)}))
    key, out = code["PostStepExtraContactRecorder"](None, env).record_post_step()
    assert key == "contact_body"
    assert out["wrist__cube"].tolist() == [0., 0.]
    assert out["wrist__bowl"].tolist() == [5., 0.]
    assert out["wrist__table"].tolist() == [0., 3.]
