"""H-B3 / P34: the ladder re-judged on the final frame (pure aggregation)."""
import sys, types
for n in ("isaaclab", "isaaclab.utils"):
    sys.modules.setdefault(n, types.ModuleType(n))
def _stub(name, **a):
    m = types.ModuleType(name); m.__dict__.update(a); sys.modules[name] = m
_stub("isaaclab.utils.math", quat_apply=None, quat_apply_inverse=None)
_stub("robolab.core.task.hull_check", build_local_hull=None, point_in_hull=None)
_stub("robolab.core.utils.geometry_utils", spatial_condition_check_vector_based=None)
_stub("robolab.core.utils.transform_utils", transform_pose_from_w_to_b_vectorized=None)
_stub("robolab.core.world.world_state", get_world=None, WorldState=object)
_stub("robolab.core.utils.function_loader", func_as_str=lambda f: "f", get_callable_info=lambda f: ("f", {}))
sys.modules.setdefault("isaaclab.managers", types.ModuleType("isaaclab.managers"))
_stub("isaaclab.managers.recorder_manager", RecorderManagerBaseCfg=object, RecorderTerm=object, RecorderTermCfg=object)
_stub("isaaclab.utils", configclass=lambda c: c)
_stub("robolab.core.task.event_tracker", EventTracker=object)
for _n in list(sys.modules):
    if _n.startswith("robolab.core.task") and not hasattr(sys.modules[_n], "__file__"):
        del sys.modules[_n]
for _n in ("robolab.core.events.subtask_recorder",): sys.modules.pop(_n, None)

from robolab.core.events.subtask_recorder import judge_ladder_final  # noqa: E402


class _Stage:
    def __init__(self, score): self.score = score


class _CSM:
    def __init__(self, s): self.total_score = s; self.stepped = False
    def step(self): self.stepped = True


def test_first_stage_undone_second_kept():
    stages = [_Stage(1.0), _Stage(1.0)]
    finals = {0: 0.0, 1: 1.0}                    # mug_01 gone, mug still on the shelf
    assert judge_ladder_final(stages, 2.0, lambda i, st: _CSM(finals[i])) == 0.5


def test_weights_follow_stage_scores():
    stages = [_Stage(0.5), _Stage(1.5)]
    assert abs(judge_ladder_final(stages, 2.0, lambda i, st: _CSM(1.0 if i == 1 else 0.0)) - 0.75) < 1e-9


def test_empty_ladder_and_clamping():
    assert judge_ladder_final([], 0.0, lambda i, st: _CSM(1.0)) == 1.0
    assert judge_ladder_final([_Stage(1.0)], 1.0, lambda i, st: _CSM(1.7)) == 1.0


# Exercise the recorder entry point with real ladder machines, not only the
# aggregation helper: completed groups latch during rollout but must be judged
# afresh at termination.
def _terminal_fixture(monkeypatch):
    import robolab.constants as constants
    import robolab.core.task.conditionals_state_machine as conditionals
    from robolab.core.task.subtask import Subtask
    from robolab.core.task.subtask_state_machine import SubtaskStateMachine
    from robolab.core.events import subtask_recorder
    monkeypatch.setattr(conditionals, 'get_world', lambda env: None)
    monkeypatch.setattr(constants, 'VERBOSE', False)
    monkeypatch.setattr(constants, 'DEBUG', False)
    monkeypatch.setattr(constants, 'SUBTASK_EXCLUDE_SPAWN_TRUE_RUNGS', True)
    state = {'goal': False}
    stage = Subtask(name='terminal regression', conditions={
        'object': [(lambda **kw: state['goal'], 1.0)]}, score=1.0)
    sm = SubtaskStateMachine(env=None, subtasks=[stage])
    recorder = types.SimpleNamespace(subtask_state_machines=[sm], _env=None)
    judge = lambda: subtask_recorder.SubtaskCompletionRecorderTerm.judge_final_score(recorder, 0)
    return state, sm, judge


def test_completed_ladder_can_lose_terminal_credit(monkeypatch):
    state, sm, judge = _terminal_fixture(monkeypatch)
    sm.step()  # goal false at stage activation
    state['goal'] = True
    sm.step()
    assert sm.is_complete()
    assert judge() == 1.0  # terminal truth must not be re-probed as spawn
    state['goal'] = False
    assert judge() == 0.0  # BananasInCrate: complete at 875, undone at 900
    assert sm.get_total_score() == 1.0  # historical rollout score unchanged


def test_terminal_judgment_preserves_spawn_exclusion(monkeypatch):
    state, sm, judge = _terminal_fixture(monkeypatch)
    state['goal'] = True
    sm.step()
    assert judge() == 0.0
    original = sm.stage_state_machines[0]
    assert original._excluded['object'] == [True]
    sm.reset()
    assert sm.stage_state_machines[0] is not original
    assert not sm.stage_state_machines[0]._spawn_probed


def test_unvisited_stage_is_final_scene_diagnostic(monkeypatch):
    state, sm, judge = _terminal_fixture(monkeypatch)
    from robolab.core.task.subtask import Subtask
    # The first stage is still false. The unvisited second stage happens to be
    # true at termination; final-scene credit does not certify sequential success.
    sm.subtasks.append(Subtask(name='unvisited', conditions={
        'other': [(lambda **kw: True, 1.0)]}, score=1.0))
    sm.unnormalized_total_score = 2.0
    sm.step()
    assert not sm.is_complete()
    assert judge() == 0.5
    assert sm.get_total_score() == 0.0


def test_mustard_surface_placement_earns_final_progress(monkeypatch):
    """P127: an on-top placement must not require containment inside the box."""
    import ast
    from pathlib import Path
    import robolab.core.task.conditionals as conditions
    import robolab.core.task.conditionals_state_machine as conditionals
    from robolab.core.task.subtask_state_machine import SubtaskStateMachine
    from robolab.core.events.subtask_recorder import SubtaskCompletionRecorderTerm
    import robolab.constants as constants
    monkeypatch.setattr(conditionals, 'get_world', lambda env: None)
    monkeypatch.setattr(constants, 'VERBOSE', False)
    monkeypatch.setattr(constants, 'DEBUG', False)
    state = {'on_top': False}
    monkeypatch.setattr(conditions, 'object_grabbed', lambda **kw: False)
    monkeypatch.setattr(conditions, 'object_in_container', lambda **kw: False)
    monkeypatch.setattr(conditions, 'object_on_top', lambda **kw: state['on_top'])
    path = Path(__file__).resolve().parents[1] / 'robolab/tasks/benchmark/mustard_above_raisin_task.py'
    task = next(n for n in ast.parse(path.read_text()).body
                if isinstance(n, ast.ClassDef) and n.name == 'MustardAboveRaisinTask')
    assignment = next(n for n in task.body if isinstance(n, ast.Assign)
                      and any(isinstance(t, ast.Name) and t.id == 'subtasks' for t in n.targets))
    stages = eval(compile(ast.Expression(assignment.value), str(path), 'eval'), {
        'pick_and_place': conditions.pick_and_place,
        'pick_and_place_on_surface': conditions.pick_and_place_on_surface,
    })
    sm = SubtaskStateMachine(None, subtasks=stages)
    sm.step()
    state['on_top'] = True
    sm.step()
    recorder = types.SimpleNamespace(subtask_state_machines=[sm], _env=None)
    assert sm.is_complete()
    assert SubtaskCompletionRecorderTerm.judge_final_score(recorder, 0) == 1.0
