# SPDX-License-Identifier: Apache-2.0
"""P126: recorded success must agree with the canonical evaluation result.

Execute the production reset hook and result selector with a small upstream
recorder double; Isaac Sim is unnecessary for termination-result bookkeeping.
"""
import ast
from collections.abc import Sequence
from pathlib import Path
from types import SimpleNamespace, MethodType

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]


def _method(path, class_name, method_name):
    tree = ast.parse((ROOT / path).read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == class_name)
    return next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == method_name)


class _UpstreamRecorder:
    def record_pre_reset(self, env_ids, force_export_or_skip=None):
        self.upstream_ids = list(env_ids)
        self.upstream_export = force_export_or_skip
        # Upstream cannot recognize success_<object> tasks. The override must
        # correct that case without turning target_lost into a success.
        self.set_success_to_episodes(env_ids, torch.zeros(len(env_ids), dtype=torch.bool))

    def set_success_to_episodes(self, env_ids, success):
        for eid, value in zip(env_ids, success):
            self.recorded[eid] = bool(value)


def _recorder(success, terminated, active=None):
    namespace = {"torch": torch, "Sequence": Sequence, "_UpstreamRecorder": _UpstreamRecorder}
    reset = _method("robolab/core/logging/recorder_manager.py", "RobolabRecorderManager", "record_pre_reset")
    cls = ast.ClassDef(name="Recorder", bases=[ast.Name(id="_UpstreamRecorder", ctx=ast.Load())],
                       keywords=[], body=[reset], decorator_list=[])
    selector = _method("robolab/core/environments/env.py", "RobolabEnv", "_success_of")
    exec(compile(ast.fix_missing_locations(ast.Module(body=[cls, selector], type_ignores=[])),
                 "<production recorder methods>", "exec"), namespace)

    def get_term(name):
        assert name == "success"
        if success is None:
            raise KeyError(name)
        return torch.tensor(success, dtype=torch.bool)

    tm = SimpleNamespace(get_term=get_term, terminated=torch.tensor(terminated, dtype=torch.bool))
    env = SimpleNamespace(termination_manager=tm, num_envs=len(terminated), device="cpu")
    env._success_of = MethodType(namespace["_success_of"], env)
    rec = namespace["Recorder"]()
    rec._env = env
    rec.active_terms = ["states"]
    rec._active_env_ids = lambda: list(range(env.num_envs)) if active is None else active
    rec.recorded = {}
    return rec


@pytest.mark.parametrize("success,terminated,expected", [
    ([False, True, False], [True, True, False], [False, True, False]),  # lost target, success, timeout
    (None, [True, False, True], [True, False, True]),  # per-object success terms
])
def test_hdf5_success_matches_canonical_result(success, terminated, expected):
    rec = _recorder(success, terminated)
    rec.record_pre_reset(torch.tensor([0, 1, 2]), force_export_or_skip=False)
    assert rec.recorded == dict(enumerate(expected))
    assert list(rec.recorded.values()) == [rec._env._success_of(i) for i in range(3)]
    assert rec.upstream_export is False


def test_partial_reset_keeps_order_and_never_overwrites_frozen_episode():
    rec = _recorder([True, True, False], [True, True, True], active=[0, 2])
    rec.recorded[1] = True
    rec.record_pre_reset([2, 1, 0])
    assert rec.upstream_ids == [2, 0]
    assert rec.recorded == {1: True, 2: False, 0: True}


def test_no_active_episodes_does_not_record():
    rec = _recorder([False], [True], active=[])
    rec.record_pre_reset(None)
    assert rec.recorded == {}
    assert not hasattr(rec, "upstream_ids")


@pytest.mark.parametrize("ids,expected", [(slice(None), [0, 2]), (slice(1, 3), [2])])
def test_ga_slice_reset_preserves_success_and_frozen_episodes(ids, expected):
    rec = _recorder([True, True, False], [True, True, True], active=[0, 2])
    rec.recorded[1] = True
    rec.record_pre_reset(ids)
    assert rec.upstream_ids == expected
    assert rec.recorded[1] is True
    assert rec.recorded[2] is False
