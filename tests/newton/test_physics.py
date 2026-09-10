"""CPU-only tests: pytest --confcutdir=tests/newton tests/newton."""

import importlib.util
import json
import sys
from contextlib import contextmanager
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("physics_under_test", ROOT / "robolab/core/environments/physics.py")
physics = importlib.util.module_from_spec(spec)
spec.loader.exec_module(physics)


def config():
    return SimpleNamespace(
        sim=SimpleNamespace(physics="original", dt=1 / 120, device="cuda:0"),
        scene=SimpleNamespace(robot=SimpleNamespace(spawn=SimpleNamespace(
            usd_path=str(ROOT / "assets/robots/franka_robotiq_2f_85_isaac60.usd")
        ))),
    )


def install_module(monkeypatch, name, **attrs):
    module = ModuleType(name)
    module.__dict__.update(attrs)
    monkeypatch.setitem(sys.modules, name, module)


def test_physx_does_not_import_newton_or_modify_asset():
    cfg = config()
    original = cfg.scene.robot.spawn.usd_path
    physics.configure_physics(cfg)
    assert cfg.sim.physics == "original"
    assert cfg.scene.robot.spawn.usd_path == original


def test_newton_configuration_preserves_timing_and_selects_overlay(monkeypatch):
    install_module(monkeypatch, "isaaclab_newton.physics", NewtonCfg=SimpleNamespace, MJWarpSolverCfg=SimpleNamespace)
    install_module(monkeypatch, "robolab.constants", ROBOTS_DIR=str(ROOT / "assets/robots"))
    cfg = config()
    physics.configure_physics(cfg, "newton_mjwarp")
    assert cfg.sim.dt == 1 / 120
    assert cfg.sim.physics.solver_cfg.nconmax == 2048
    assert Path(cfg.scene.robot.spawn.usd_path).is_file()
    cfg.scene.robot.spawn.usd_path = "unsupported.usd"
    with pytest.raises(ValueError, match="DROID"):
        physics.configure_physics(cfg, "newton_mjwarp")


def test_rejects_old_isaaclab_and_unknown_backend():
    cfg = config()
    del cfg.sim.physics
    with pytest.raises(RuntimeError, match="Isaac Lab 3"):
        physics.configure_physics(cfg, "newton_mjwarp")
    with pytest.raises(ValueError, match="Unknown"):
        physics.configure_physics(cfg, "typo")


def test_output_cannot_mix_backends(tmp_path):
    physics.ensure_output_backend(tmp_path, "newton_mjwarp")
    physics.ensure_output_backend(tmp_path, "newton_mjwarp")
    with pytest.raises(ValueError, match="choose a new"):
        physics.ensure_output_backend(tmp_path, "physx")
    assert json.loads((tmp_path / "physics_backend.json").read_text())["physics_backend"] == "newton_mjwarp"


def test_legacy_results_are_not_assumed_to_be_newton(tmp_path):
    (tmp_path / "results.json").write_text("{}")
    with pytest.raises(ValueError, match="fresh output"):
        physics.ensure_output_backend(tmp_path, "newton_mjwarp")
    physics.ensure_output_backend(tmp_path, "physx")


@pytest.mark.parametrize("fail", [False, True])
def test_runtime_encloses_environment_and_closes_on_error(monkeypatch, fail):
    events = []
    cfg = config()
    env = SimpleNamespace(close=lambda: events.append("env_close"))
    install_module(monkeypatch, "robolab.core.environments.config", parse_env_cfg=lambda *a, **k: cfg)
    install_module(monkeypatch, "robolab.core.environments.runtime",
                   create_env=lambda *a, **k: (env, cfg))
    monkeypatch.setattr(physics, "configure_physics", lambda *a: events.append("configure"))

    @contextmanager
    def runtime(*args):
        events.append("runtime_enter")
        try:
            yield
        finally:
            events.append("runtime_exit")

    monkeypatch.setattr(physics, "physics_context", runtime)
    try:
        with physics.managed_env("task", physics="newton_mjwarp"):
            events.append("step")
            if fail:
                raise ValueError("step failed")
    except ValueError:
        assert fail
    assert events == ["configure", "runtime_enter", "step", "env_close", "runtime_exit"]


def test_overlay_has_valid_inertia_and_preserves_geometry():
    from pxr import Usd, UsdPhysics

    source = Usd.Stage.Open(str(ROOT / "assets/robots/franka_robotiq_2f_85_isaac60.usd"))
    overlay = Usd.Stage.Open(str(ROOT / "assets/robots/franka_robotiq_2f_85_newton.usda"))
    assert source and overlay
    assert source.GetDefaultPrim().GetPath() == overlay.GetDefaultPrim().GetPath()
    assert [p.GetPath() for p in source.Traverse()] == [p.GetPath() for p in overlay.Traverse()]
    allowed = {"physics:mass", "physics:centerOfMass", "physics:diagonalInertia", "physics:principalAxes"}
    for prim in overlay.Traverse():
        original = source.GetPrimAtPath(prim.GetPath())
        for attr in original.GetAttributes():
            if attr.GetName() not in allowed:
                assert attr.Get() == prim.GetAttribute(attr.GetName()).Get(), attr.GetPath()
        for rel in original.GetRelationships():
            assert rel.GetTargets() == prim.GetRelationship(rel.GetName()).GetTargets()
        if str(prim.GetPath()).startswith("/panda/") and prim.HasAPI(UsdPhysics.RigidBodyAPI):
            api = UsdPhysics.MassAPI(prim)
            assert api.GetMassAttr().Get() > 0, prim.GetPath()
            diagonal = sorted(api.GetDiagonalInertiaAttr().Get())
            assert diagonal[0] > 0 and diagonal[2] <= diagonal[0] + diagonal[1] + 1e-10
            assert api.GetPrincipalAxesAttr().Get().GetLength() == pytest.approx(1)
