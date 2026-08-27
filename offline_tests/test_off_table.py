"""P38: an off-table target ends the episode only when the task can no longer succeed."""
import torch
from robolab.core.task.off_table import group_lost, required_groups, task_lost

def m(*v): return torch.tensor(v)

def test_required_groups_from_success_params():
    # P76: a single-group success term also requires its DESTINATION to stay on the
    # table -- if the bowl is gone, "banana in bowl" can no longer be achieved.
    assert required_groups({"object": "banana", "container": "bowl"}) == [
        (["banana"], "all", 1), (["bowl"], "all", 1)]
    assert required_groups({"object": ["a", "b"], "logical": "choose", "K": 2}) == [(["a", "b"], "choose", 2)]
    g = required_groups({"groups": [{"object": ["coffee_can"], "container": "bin_a06", "logical": "all"},
                                    {"object": ["mustard", "sugar_box"], "container": "bin_b03", "logical": "all"}]})
    # multi-group terms are deliberately left alone: one container per group, and a
    # lost container must be attributed to ITS group, which this structure cannot say.
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


def test_p76_a_lost_destination_dooms_a_single_group_task():
    """PutMugsOnShelf, every rc3 episode: the policy tips the rack off the table,
    OBJECT_FELL_OFF_TABLE fires for the rack, and the episode then runs its full
    180 s with nothing left to achieve."""
    groups = required_groups({"object": ["mug", "ceramic_mug"], "container": "rack_l04"})
    n, dev = 2, "cpu"
    fallen = {"mug": m(False, False), "ceramic_mug": m(False, False), "rack_l04": m(True, False)}
    lost = task_lost(fallen, groups, n, dev)
    assert lost.tolist() == [True, False], "losing the shelf must end the episode"


def test_p76_surface_counts_as_a_destination():
    groups = required_groups({"object": ["drill"], "surface": "table_top"})
    assert (["table_top"], "all", 1) in groups
