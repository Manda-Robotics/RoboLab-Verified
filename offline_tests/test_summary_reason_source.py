"""Final-frame subtask diagnostics must not masquerade as termination evidence."""
import ast
from pathlib import Path
from types import SimpleNamespace

path = Path(__file__).resolve().parents[1] / "robolab/eval/summarize.py"
tree = ast.parse(path.read_text())
fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef)
          and n.name == "build_run_summary")
namespace = {}
exec(compile(ast.Module(body=[fn], type_ignores=[]), str(path), "exec"), namespace)
build = namespace["build_run_summary"]


def summary(final_info, success=False):
    return build(env_result={"success": success, "step": 900}, env_id=0,
                 run_idx=0, num_envs=1, run_name="test", task_env="Task",
                 env_cfg=SimpleNamespace(_task_name="Task", instruction="transfer",
                                         _task_attributes=[]),
                 policy="test", dt=1/15, traj_metrics=None, events={},
                 events_list=[{"code": 101, "info": "object was grabbed", "score": 1}],
                 final_info=final_info, enable_subtask_progress=True)


def test_released_tool_diagnostic_is_labeled_not_used_as_grade():
    result = summary({"info": "Condition not satisfied: object_grabbed (step 1/2)"})
    assert result["success"] is False
    assert result["reason_source"] == "subtask_final_state"
    assert result["subtask_diagnostic"] == result["reason"]


def test_empty_final_info_does_not_relabel_event_reason():
    result = summary({})
    assert result["reason_source"] == "event_log"
    assert result["subtask_diagnostic"] is None


def test_success_is_not_replaced_by_final_ladder_regression():
    result = summary({"info": "no longer grabbed"}, success=True)
    assert result["success"] is True
    assert result["reason"] == "object was grabbed"
    assert result["reason_source"] == "event_log"
