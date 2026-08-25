"""P38: an off-table target ends the episode only when the task can no longer succeed."""
import torch
from robolab.core.task.off_table import group_lost, required_groups, task_lost

def m(*v): return torch.tensor(v)

def test_required_groups_from_success_params():
    assert required_groups({"object": "banana", "container": "bowl"}) == [(["banana"], "all", 1)]
    assert required_groups({"object": ["a", "b"], "logical": "choose", "K": 2}) == [(["a", "b"], "choose", 2)]
    g = required_groups({"groups": [{"object": ["coffee_can"], "container": "bin_a06", "logical": "all"},
                                    {"object": ["mustard", "sugar_box"], "container": "bin_b03", "logical": "all"}]})
    assert g == [(["coffee_can"], "all", 1), (["mustard", "sugar_box"], "all", 1)]
    assert required_groups({"objects": ["red", "blue"], "order": None}) == [(["red", "blue"], "all", 1)]

def test_all_any_choose_semantics():
    n, dev = 3, "cpu"
    fallen = {"a": m(True, False, True), "b": m(False, False, True), "c": m(False, False, False)}
    assert group_lost(fallen, ["a", "b", "c"], "all", 1, n, dev).tolist() == [True, False, True]     # any one lost → lost
    assert group_lost(fallen, ["a", "b", "c"], "any", 1, n, dev).tolist() == [False, False, False]   # one still on the table
    assert group_lost(fallen, ["a", "b", "c"], "choose", 2, n, dev).tolist() == [False, False, True] # need 2 of 3: env2 has only c

def test_task_lost_is_or_over_groups():
    n, dev = 2, "cpu"
    fallen = {"can": m(True, False), "mustard": m(False, False)}
    groups = [(["can"], "all", 1), (["mustard", "sugar_box"], "all", 1)]
    assert task_lost(fallen, groups, n, dev).tolist() == [True, False]
