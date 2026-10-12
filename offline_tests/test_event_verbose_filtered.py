"""Filtered container contacts must neither crash nor reprint a previous event."""
import ast
from pathlib import Path
from types import SimpleNamespace
import torch
from robolab.core.task.status import StatusCode


def test_verbose_filtered_container_events(capsys):
    path = Path(__file__).parents[1] / 'robolab/core/task/event_tracker.py'
    tree = ast.parse(path.read_text())
    loop = next(n for n in ast.walk(tree) if isinstance(n, ast.For)
                and isinstance(n.iter, ast.Call) and isinstance(n.iter.func, ast.Attribute)
                and n.iter.func.attr == 'pop_events')
    code = compile(ast.Module(body=[loop], type_ignores=[]), str(path), 'exec')
    events = []
    pending = [(0, 'crate', 'gripper', 'gripped', {}),
               (0, 'crate', 'gripper', 'attempt_failed', {}),
               (0, 'banana', 'gripper', 'released', {}),
               (0, 'crate', 'gripper', 'attempt_failed', {})]
    scope = dict(events=events, tracker=SimpleNamespace(pop_events=lambda: pending),
                 frozen_mask=torch.tensor([False]), torch=torch,
                 self=SimpleNamespace(num_envs=1, device='cpu'),
                 per_env_containers=[{'crate'}], step_dt=1/15,
                 StatusCode=StatusCode, verbose=True)
    exec(code, scope)
    assert len(events) == 1
    assert capsys.readouterr().out == "[EventTracker] env0: 'banana' released (hand opened)\n"
