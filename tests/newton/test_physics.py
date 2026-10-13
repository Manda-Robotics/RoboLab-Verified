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
        actions=SimpleNamespace(finger_joint="original"),
        scene=SimpleNamespace(robot=SimpleNamespace(actuators={"gripper": SimpleNamespace(stiffness=None, damping=None)}, spawn=SimpleNamespace(
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
    install_module(monkeypatch, "robolab.robots.newton_gripper", RateLimitedGripperActionCfg=SimpleNamespace)
    cfg = config()
    physics.configure_physics(cfg, "newton_mjwarp")
    assert cfg.sim.dt == 1 / 120
    assert cfg.sim.physics.solver_cfg.nconmax == 65536
    assert cfg.sim.physics.solver_cfg.njmax == 131072
    assert cfg.sim.physics.use_cuda_graph is False
    assert cfg.newton_contact_settings["enable_multiccd"] is True
    assert cfg.newton_contact_settings["mesh_compatibility"] is True
    assert cfg.newton_contact_settings["compact_keyboard_colliders"] is True
    assert cfg.newton_contact_settings["compact_shelf_colliders"] is True
    assert cfg.newton_contact_settings["compact_contact_storage"] is True
    assert cfg.newton_contact_settings["timeconst"] == .0045
    assert cfg.newton_contact_settings["filter_immovable_contacts"] is True
    assert cfg.newton_contact_settings["support_rolling_friction"] == .001
    assert cfg.sim.physics.num_substeps == 4
    assert cfg.scene.robot.actuators["gripper"].stiffness == 10
    assert cfg.scene.robot.actuators["gripper"].damping == .5
    assert cfg.actions.finger_joint.max_target_velocity == pytest.approx(3.141592653589793)
    assert cfg.actions.finger_joint.close_command_expr == {"finger_joint": pytest.approx(0.7853981633974483)}
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


def test_verified_friction_override_rejected_before_newton_import():
    cfg = config()
    cfg.friction = {"mode": "uniform", "spec": "0.5"}
    with pytest.raises(ValueError, match="only --friction upstream"):
        physics.configure_physics(cfg, "newton_mjwarp")
    # The existing PhysX override path remains supported.
    physics.configure_physics(cfg, "physx")


@pytest.mark.parametrize("overrides", [
    {"enable_multiccd": "yes"}, {"compact_keyboard_colliders": "yes"}, {"compact_shelf_colliders": "yes"}, {"compact_contact_storage": "yes"}, {"mesh_compatibility": "yes"}, {"contact_capacity": 0}, {"constraint_capacity": 1.5},
    {"timeconst": .001}, {"impedance": (.999, .99)},
    {"gripper_damping": -1}, {"typo": 1}, {"filter_immovable_contacts": "yes"},
    {"support_rolling_friction": -1}, {"support_rolling_friction": float("nan")},
    {"support_rolling_friction": True},
])
def test_invalid_contact_settings_rejected(monkeypatch, overrides):
    install_module(monkeypatch, "isaaclab_newton.physics", NewtonCfg=SimpleNamespace, MJWarpSolverCfg=SimpleNamespace)
    install_module(monkeypatch, "robolab.constants", ROBOTS_DIR=str(ROOT / "assets/robots"))
    cfg = config()
    cfg.newton_contact_settings = overrides
    with pytest.raises(ValueError):
        physics.configure_physics(cfg, "newton_mjwarp")


@pytest.mark.parametrize("disabled", [True, False, None])
def test_gravity_overlay_respects_robot_setting(monkeypatch, disabled):
    install_module(monkeypatch, "isaaclab_newton.physics", NewtonCfg=SimpleNamespace, MJWarpSolverCfg=SimpleNamespace)
    install_module(monkeypatch, "robolab.constants", ROBOTS_DIR=str(ROOT / "assets/robots"))
    install_module(monkeypatch, "robolab.robots.newton_gripper", RateLimitedGripperActionCfg=SimpleNamespace)
    cfg = config()
    cfg.scene.robot.spawn.rigid_props = SimpleNamespace(disable_gravity=disabled)
    physics.configure_physics(cfg, "newton_mjwarp")
    assert ("nogravity" in cfg.scene.robot.spawn.usd_path) == bool(disabled)


def test_gravity_overlay_only_changes_robot_gravity():
    from pxr import Usd, UsdPhysics
    base = Usd.Stage.Open(str(ROOT / "assets/robots/franka_robotiq_2f_85_newton.usda"))
    compensated = Usd.Stage.Open(str(ROOT / "assets/robots/franka_robotiq_2f_85_newton_nogravity.usda"))
    count = 0
    assert [p.GetPath() for p in base.Traverse()] == [p.GetPath() for p in compensated.Traverse()]
    for prim in compensated.Traverse():
        original = base.GetPrimAtPath(prim.GetPath())
        for attr in original.GetAttributes():
            assert attr.Get() == prim.GetAttribute(attr.GetName()).Get()
        for rel in original.GetRelationships():
            assert rel.GetTargets() == prim.GetRelationship(rel.GetName()).GetTargets()
        if str(prim.GetPath()).startswith("/panda/") and prim.HasAPI(UsdPhysics.RigidBodyAPI):
            assert prim.GetAttribute("mjc:gravcomp").Get() == 1
            count += 1
        else:
            assert not prim.GetAttribute("mjc:gravcomp")
    assert count == 18


def test_eager_diagnostic_disables_only_graph_and_rejects_physx(monkeypatch):
    cfg = config()
    cfg.sim.physics = SimpleNamespace(use_cuda_graph=True, num_substeps=4)
    args = SimpleNamespace(newton_disable_cuda_graph=True)
    @contextmanager
    def launcher(active_cfg, launch_args):
        assert active_cfg.sim.physics.use_cuda_graph is False
        assert active_cfg.sim.physics.num_substeps == 4
        yield
    install_module(monkeypatch, "isaaclab.app.sim_launcher", launch_simulation=launcher)
    with physics.physics_context(cfg, "newton_mjwarp", args):
        pass
    with pytest.raises(ValueError, match="requires --physics"):
        with physics.physics_context(config(), "physx", args):
            pass


def test_mesh_contact_options_enclose_runtime_and_validate_timestep(monkeypatch):
    # Exercise the legacy branch even on a host with modern Lab installed.
    monkeypatch.setitem(sys.modules, "isaaclab.app.sim_launcher", None)
    events = []
    cfg = config()
    cfg.sim.physics = SimpleNamespace(num_substeps=4)
    cfg.newton_convex_decomposition_threshold = .05
    cfg.newton_contact_timeconst = .006
    @contextmanager
    def mesh_context(threshold, *, contact_timeconst):
        assert (threshold, contact_timeconst) == (.05, .006)
        events.append("mesh_enter")
        yield
        events.append("mesh_exit")
    @contextmanager
    def launcher(*args):
        events.append("launch")
        yield
    install_module(monkeypatch, "isaaclab_tasks.utils.sim_launcher", launch_simulation=launcher)
    install_module(monkeypatch, "robolab.core.environments.newton_meshes", mesh_decomposition_context=mesh_context)
    with physics.physics_context(cfg, "newton_mjwarp"):
        events.append("run")
    assert events == ["mesh_enter", "launch", "run", "mesh_exit"]
    for invalid in [0, -.006, .001, float("nan"), float("inf")]:
        cfg.newton_contact_timeconst = invalid
        with pytest.raises(ValueError, match="solver timestep"):
            with physics.physics_context(cfg, "newton_mjwarp"):
                pass
    cfg.newton_contact_timeconst = .006
    cfg.newton_convex_decomposition_threshold = None
    with pytest.raises(ValueError, match="mesh-import"):
        with physics.physics_context(cfg, "newton_mjwarp"):
            pass


def test_output_cannot_mix_backends(tmp_path):
    physics.ensure_output_backend(tmp_path, "newton_mjwarp")
    physics.ensure_output_backend(tmp_path, "newton_mjwarp")
    with pytest.raises(ValueError, match="choose a new"):
        physics.ensure_output_backend(tmp_path, "physx")
    assert json.loads((tmp_path / "physics_backend.json").read_text())["physics_backend"] == "newton_mjwarp"


def test_modern_launcher_preserves_solver_and_does_not_patch_meshes(monkeypatch):
    from argparse import Namespace
    cfg = config()
    original = cfg.sim.physics
    args = Namespace(physics="newton_mjwarp", headless=True)
    @contextmanager
    def launcher(received_cfg, received_args):
        assert received_cfg is cfg
        assert received_args.physics is None
        assert received_args.headless
        assert received_args.require_kit
        assert cfg.sim.physics is original
        yield
    install_module(monkeypatch, "isaaclab.app.sim_launcher", launch_simulation=launcher)
    with physics.physics_context(cfg, "newton_mjwarp", args):
        pass
    assert args.physics == "newton_mjwarp"
    for field in ("newton_convex_decomposition_threshold", "newton_contact_timeconst"):
        setattr(cfg, field, .006)
        with pytest.raises(ValueError, match="native authored mesh import"):
            with physics.physics_context(cfg, "newton_mjwarp", args):
                pass
        delattr(cfg, field)


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
    from pxr import UsdGeom
    promoted = 0
    for prim in source.Traverse():
        if not prim.IsA(UsdGeom.Mesh) or "/Gripper/" not in str(prim.GetPath()):
            continue
        parent = prim.GetParent()
        if parent.HasAPI(UsdPhysics.CollisionAPI):
            collider = UsdPhysics.CollisionAPI(overlay.GetPrimAtPath(prim.GetPath()))
            assert collider and collider.GetCollisionEnabledAttr().Get()
            promoted += 1
    assert promoted >= 11
    for name in ("left_inner_finger_joint", "left_inner_finger_knuckle_joint",
                 "right_outer_knuckle_joint", "right_inner_finger_joint", "right_inner_finger_knuckle_joint"):
        prim = overlay.GetPrimAtPath(f"/panda/Gripper/Robotiq_2F_85/Joints/{name}")
        assert prim.GetAttribute("mjc:damping").Get() == pytest.approx(0.05)
        assert list(prim.GetAttribute("mjc:solref").Get()) == pytest.approx([0.006, 1])
        assert list(prim.GetAttribute("mjc:solimp").Get()) == pytest.approx([.99, .99, .001, .5, 2])
        reference = next(r for r in prim.GetRelationships()
                         if r.GetName().startswith("physxMimicJoint:") and r.GetName().endswith(":referenceJoint"))
        assert prim.GetRelationship("mjc:target").GetTargets() == reference.GetTargets()
        prefix = reference.GetName().removesuffix(":referenceJoint")
        assert prim.GetAttribute("mjc:coef1").Get() == -prim.GetAttribute(prefix + ":gearing").Get()
