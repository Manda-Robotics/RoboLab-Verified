"""CPU Torch tests with Isaac imports stubbed; no simulator startup required."""

import importlib.util
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace

import pytest

torch = pytest.importorskip("torch")


@pytest.fixture
def action_class(monkeypatch):
    def install(name, **attrs):
        module = ModuleType(name)
        module.__dict__.update(attrs)
        monkeypatch.setitem(sys.modules, name, module)

    class BaseAction:
        def __init__(self, cfg, env):
            self.cfg, self._asset = cfg, env.asset
            self.device, self.num_envs, self._joint_ids = "cpu", 2, getattr(env, "joint_ids", [0])
            self._processed_actions = torch.zeros(2, 1)
            self._open_command = torch.zeros(1)

        def reset(self, env_ids=None):
            pass

    install("isaaclab.utils", configclass=lambda cls: cls)
    install("robolab.core.utils.isaaclab_compat", as_torch=lambda x: x)
    install("robolab.robots.droid", BinaryJointPositionZeroToOneAction=BaseAction,
            BinaryJointPositionZeroToOneActionCfg=object)
    path = Path(__file__).resolve().parents[2] / "robolab/robots/newton_gripper.py"
    spec = importlib.util.spec_from_file_location("gripper_under_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.RateLimitedGripperAction


def make_action(cls, velocity=1.5, joint_ids=None):
    writes = []
    asset = SimpleNamespace(data=SimpleNamespace(joint_pos=torch.tensor([[0.2], [0.4]])),
                            set_joint_position_target_index=lambda **kw: writes.append(kw["target"].clone()))
    env = SimpleNamespace(asset=asset, cfg=SimpleNamespace(sim=SimpleNamespace(dt=0.01)))
    if joint_ids is not None:
        env.joint_ids = joint_ids
    return cls(SimpleNamespace(max_target_velocity=velocity), env), asset, writes


def test_slew_limit_and_no_overshoot(action_class):
    action, _, writes = make_action(action_class)
    action._processed_actions[:] = torch.tensor([[0.7], [0.405]])
    action.apply_actions()
    assert torch.allclose(writes[-1], torch.tensor([[0.215], [0.405]]))
    action.apply_actions()
    assert torch.allclose(writes[-1], torch.tensor([[0.230], [0.405]]))


def test_array_joint_indices_keep_reset_synchronization(action_class):
    ids = torch.tensor([0], dtype=torch.int32)
    action, _, writes = make_action(action_class, joint_ids=ids)
    action._processed_actions[:] = .7
    action.apply_actions()
    assert action._joint_ids is ids
    assert action._read_joint_ids.dtype == torch.long
    assert torch.allclose(writes[-1], torch.tensor([[0.215], [0.415]]))


def test_partial_reset_resynchronizes_only_reset_environment(action_class):
    action, asset, writes = make_action(action_class)
    action._processed_actions[:] = 0.7
    action.apply_actions()
    action.reset([1])
    # Simulate scene reset occurring after action reset.
    asset.data.joint_pos[1] = 0.05
    action.apply_actions()
    assert torch.allclose(writes[-1], torch.tensor([[0.230], [0.035]]))


@pytest.mark.parametrize("velocity", [0, -1, float("inf"), float("nan")])
def test_invalid_rate_rejected(action_class, velocity):
    with pytest.raises(ValueError):
        make_action(action_class, velocity)
