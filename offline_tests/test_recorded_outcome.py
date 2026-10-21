"""Exercise production outcome methods without booting Isaac.

The pasta target-loss regression had terminated=True but canonical success=False.
Extract the methods into lightweight classes so reset ordering and filtering are
covered without replacing the behavior being tested.
"""
import ast
from collections.abc import Sequence
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]


def load_method_class(path, class_name, method, base=object):
    tree = ast.parse((ROOT / path).read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == class_name)
    cls.body = [n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == method]
    cls.bases = [ast.Name(id="Base", ctx=ast.Load())]
    ns = {"Base": base, "torch": torch, "Sequence": Sequence}
    exec(compile(ast.fix_missing_locations(ast.Module(body=[cls], type_ignores=[])),
                 str(path), "exec"), ns)
    return ns[class_name]


class UpstreamRecorder:
    def record_pre_reset(self, env_ids, force_export_or_skip=None):
        self.upstream_called = True
        self.recorded = {eid: False for eid in env_ids}

    def set_success_to_episodes(self, env_ids, success):
        assert self.upstream_called  # override after upstream, before export
        self.recorded.update(zip(env_ids, success.tolist()))


Env = load_method_class("robolab/core/environments/env.py", "RobolabEnv", "_success_of")
Recorder = load_method_class("robolab/core/logging/recorder_manager.py",
                            "RobolabRecorderManager", "record_pre_reset", UpstreamRecorder)


@pytest.mark.parametrize("ids", [None, [0, 1, 2], torch.tensor([0, 1, 2])])
def test_target_loss_does_not_become_recorded_success(ids):
    env = Env()
    env.device, env.num_envs = "cpu", 3
    env.termination_manager = SimpleNamespace(
        terminated=torch.tensor([True, True, False]),
        get_term=lambda name: torch.tensor([False, True, False]),
    )
    recorder = Recorder()
    recorder._env = env
    recorder.active_terms = ["actions"]
    recorder._active_env_ids = lambda: [0, 1, 2]
    recorder.record_pre_reset(ids)
    assert recorder.recorded == {0: False, 1: True, 2: False}
    assert recorder.recorded == {i: env._success_of(i) for i in range(3)}


def test_frozen_episodes_are_not_regraded():
    env = Env()
    env.device, env.num_envs = "cpu", 3
    env.termination_manager = SimpleNamespace(
        terminated=torch.tensor([True, True, False]),
        get_term=lambda name: torch.tensor([False, True, False]),
    )
    recorder = Recorder()
    recorder._env = env
    recorder.active_terms = ["actions"]
    recorder._active_env_ids = lambda: [1]
    recorder.record_pre_reset([0, 1, 2])
    assert recorder.recorded == {1: True}


def test_no_named_success_preserves_canonical_fallback():
    def missing(name):
        raise ValueError(name)

    env = Env()
    env.device, env.num_envs = "cpu", 2
    env.termination_manager = SimpleNamespace(
        terminated=torch.tensor([True, False]), get_term=missing)
    recorder = Recorder()
    recorder._env = env
    recorder.active_terms = ["actions"]
    recorder._active_env_ids = lambda: [0, 1]
    recorder.record_pre_reset(None)
    assert recorder.recorded == {0: True, 1: False}
