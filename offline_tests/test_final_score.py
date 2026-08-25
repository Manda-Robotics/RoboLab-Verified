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
