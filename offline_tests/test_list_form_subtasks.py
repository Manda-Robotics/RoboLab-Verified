"""List-form `Subtask(conditions=[a, b, c])` is a sequence under logical='all'
and a set of alternatives under 'any'/'choose' (changes.md P28 / H-B1)."""
import sys
import types
from functools import partial


def _stub(name, **attrs):
    m = types.ModuleType(name); m.__dict__.update(attrs); sys.modules[name] = m


_stub("robolab.core.task.predicate_logic", get_task_conditional_func=lambda *a, **k: None)
_stub("robolab.core.utils.function_loader", func_as_str=lambda f: getattr(f, "__name__", str(f)), get_callable_info=lambda f: {})

from robolab.core.task.subtask import Subtask  # noqa: E402
from robolab.core.task.subtask_utils import sanitize_subtask_conditions  # noqa: E402


def grabbed(env, object, env_id=None): ...
def left_of(env, object, reference_object, env_id=None): ...
def dropped(env, object, env_id=None): ...


def test_list_all_is_one_sequential_group_named_after_the_object():
    st = Subtask(name="cube", conditions=[
        partial(grabbed, object="rubiks_cube"),
        partial(left_of, object="rubiks_cube", reference_object="bowl"),
        partial(dropped, object="rubiks_cube"),
    ], logical="all")
    assert st.group_names == ["rubiks_cube"]
    funcs = [c[0].func for c in st.get_group("rubiks_cube")]
    assert funcs == [grabbed, left_of, dropped]          # order preserved
    assert all(abs(c[1] - 1 / 3) < 1e-9 for c in st.get_group("rubiks_cube"))


def test_list_any_stays_parallel_alternatives():
    st = Subtask(name="bagel", conditions=[
        partial(grabbed, object="bagel_1"), partial(grabbed, object="bagel_2"), partial(grabbed, object="bagel_3"),
    ], logical="any")
    assert st.group_names == ["group1", "group2", "group3"]
    assert all(len(st.get_group(g)) == 1 for g in st.group_names)


def test_list_of_tuples_all_is_sequential_with_given_scores():
    d = sanitize_subtask_conditions([(partial(grabbed, object="mug"), 0.0), (partial(dropped, object="mug"), 1.0)], logical="all")
    assert list(d) == ["mug"] and [w for _, w in d["mug"]] == [0.0, 1.0]


def test_mixed_objects_get_generic_name():
    d = sanitize_subtask_conditions([partial(grabbed, object="a"), partial(grabbed, object="b")], logical="all")
    assert list(d) == ["sequence"] and len(d["sequence"]) == 2


def test_dict_form_unchanged():
    d = sanitize_subtask_conditions({"mug": [partial(grabbed, object="mug"), partial(dropped, object="mug")]}, logical="all")
    assert list(d) == ["mug"] and len(d["mug"]) == 2
