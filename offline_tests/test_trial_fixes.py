"""Fixes from the first external hands-on trial of the published repo (2026-08-29).

Each test names the report item it closes. The trial ran six tasks outside the set the
fork had been verified on and read the outputs the way a new user would: the results
row, the event log, the verifier, the analysis scripts.
"""
import importlib.util
import json
import os
import pathlib
import subprocess
import sys

import pytest
import torch

ROOT = pathlib.Path(__file__).resolve().parents[1]
ENV = {**os.environ, "PYTHONPATH": str(ROOT)}


def _load(name, rel):
    spec = importlib.util.spec_from_file_location(name, ROOT / rel)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


# --------------------------------------------------------------- BUG-2: events tally

def test_events_tally_counts_ladder_and_tracker_lines_too():
    """t1 BananaInBowl: the jsonl showed {GRIPPER_HIT_TABLE, OBJECT_GRIPPED, OBJECT_RELEASED}
    while the log also held OBJECT_CARRIED, OBJECT_GRABBED_SUCCESS and SUBTASK_COMPLETED;
    t6 env1 read as two releases and no grasp."""
    from robolab.eval.summarize import _tally_events
    events = [{"code": 255, "name": "GRIPPER_HIT_TABLE"}, {"code": 283, "name": "OBJECT_CARRIED"},
              {"code": 284, "name": "OBJECT_GRIPPED"}, {"code": 139, "name": "OBJECT_GRABBED_SUCCESS"},
              {"code": 267, "name": "OBJECT_RELEASED"}, {"code": 190, "name": "SUBTASK_COMPLETED"},
              {"code": 250, "name": "WRONG_OBJECT_GRABBED_FAILURE", "info": "Wrong object grabbed: 'mug' (target objects: [])"},
              {"code": 0, "name": "OK", "info": "not an event"}]
    out = _tally_events(events)
    assert out["OBJECT_CARRIED"] == 1 and out["OBJECT_GRABBED_SUCCESS"] == 1 and out["SUBTASK_COMPLETED"] == 1
    assert out["OBJECT_RELEASED"] == 1 and out["GRIPPER_HIT_TABLE"] == 1
    assert out["WRONG_OBJECT_GRABBED"] == 1 and out["wrong_objects_grabbed"] == ["mug"]
    assert "OK" not in out


# --------------------------------------------------------------- BUG-4: P72 by task role

def _run_with_cfg(tmp_path, attempts_on, subtasks_text):
    run = tmp_path / "rc_X"; task = run / "XTask"; task.mkdir(parents=True)
    (task / "env_cfg.json").write_text(json.dumps({"subtasks": subtasks_text}))
    events = [{"step": 10 * (i + 1), "code": 266, "name": "GRASP_ATTEMPT_FAILED",
               "info": f"Grasp attempt on '{o}' failed (contact lost before a carry was established)"}
              for i, o in enumerate(attempts_on)]
    json.dump({"task": "XTask", "env_id": 0, "run": 0, "dt": 0.1, "success": False, "final_step": 100, "events": events},
              open(task / "log_0_env0.json", "w"))
    return str(run)


def test_p72_a_bowl_that_is_the_task_target_is_not_a_container(tmp_path):
    """t3 BowlStackingLeftOnRight: three attempts on the pick target bowl_2 were reported
    as attempts on a container, and --summary exited 1."""
    vp = _load("verify_patches_trial", "scripts/verify_patches.py")
    stacking = "functools.partial(object_in_container, object=['bowl_2'], container='bowl_1', gripper_name='gripper')"
    run = _run_with_cfg(tmp_path, ["bowl_2", "bowl_2", "bowl_2"], stacking)
    r = vp.p72_no_attempts_on_containers(vp.load(run))
    assert r.verdict == "PASS", r
    # ...while an attempt on the destination bowl is still caught
    run2 = _run_with_cfg(tmp_path / "b", ["bowl_1"], stacking)
    assert vp.p72_no_attempts_on_containers(vp.load(run2)).verdict == "FAIL"


def test_p72_bins_are_still_containers_and_products_still_are_not(tmp_path):
    vp = _load("verify_patches_trial2", "scripts/verify_patches.py")
    packing = "functools.partial(object_in_container, object=['tomato_soup_can', 'tuna_can'], container='bin_a06')"
    assert vp.p72_no_attempts_on_containers(vp.load(_run_with_cfg(tmp_path / "a", ["bin_a06"], packing))).verdict == "FAIL"
    assert vp.p72_no_attempts_on_containers(vp.load(_run_with_cfg(tmp_path / "b", ["sugar_box"], packing))).verdict == "PASS"
    # no env_cfg.json at all (an older recording): the name heuristics apply
    run = _run_with_cfg(tmp_path / "c", ["grey_bin"], packing)
    os.remove(os.path.join(run, "XTask", "env_cfg.json"))
    assert vp.p72_no_attempts_on_containers(vp.load(run)).verdict == "FAIL"


# --------------------------------------------------------------- BUG-3: check_results

def test_check_results_runs_over_a_run_directory(tmp_path):
    """`NameError: hdf5_path` in both code paths since upstream v0.1.0; and a relative
    path that already contained output/ was prefixed again."""
    run = tmp_path / "output" / "t1_X"; task = run / "XTask"; task.mkdir(parents=True)
    # the fixture is written in a subprocess: another test module leaves a stub `h5py`
    # in sys.modules, and check_results itself runs as a subprocess anyway
    made = subprocess.run([sys.executable, "-c",
                           "import h5py,sys; f=h5py.File(sys.argv[1],'w'); g=f.create_group('data/demo_0'); "
                           "g.create_dataset('actions', data=[[0.0]*8]*5); f.close()", str(task / "run_0.hdf5")],
                          capture_output=True, text=True)
    if made.returncode != 0:
        pytest.skip("h5py not installed in this interpreter")
    with open(run / "episode_results.jsonl", "w") as f:
        f.write(json.dumps({"env_name": "XTask", "episode": 0, "env_id": 0, "run": 0, "success": True}) + "\n")
    for args in ([str(run)], ["--diagnose", str(run)]):
        p = subprocess.run([sys.executable, str(ROOT / "analysis" / "check_results.py"), *args],
                           capture_output=True, text=True, cwd=tmp_path, env=ENV)
        assert "NameError" not in p.stderr and "Traceback" not in p.stderr, p.stderr[-800:]
        assert "Folder not found" not in p.stdout
    # a cwd-relative path is used as given, not prefixed with output/ a second time
    p = subprocess.run([sys.executable, str(ROOT / "analysis" / "check_results.py"), "output/t1_X"],
                       capture_output=True, text=True, cwd=tmp_path, env=ENV)
    assert "Folder not found" not in p.stdout and "Traceback" not in p.stderr


# --------------------------------------------------------------- BUG-6: by instruction type

def test_by_instruction_type_is_a_pivot_not_a_stub(capsys):
    from robolab.core.logging.results import instruction_type_pivot, summarize_experiments_by_instruction_type
    rows = [{"task_name": "GrabAFruitTask", "instruction_type": "vague", "success": False},
            {"task_name": "GrabAFruitTask", "instruction_type": "vague", "success": True},
            {"task_name": "GrabAFruitTask", "instruction_type": "default", "success": True},
            {"task_name": "BananaInBowlTask", "success": True}]
    types, table = instruction_type_pivot(rows)
    assert types == ["default", "vague"]
    assert table["GrabAFruitTask"] == {"vague": (1, 2), "default": (1, 1)}
    assert table["BananaInBowlTask"] == {"default": (1, 1)}
    summarize_experiments_by_instruction_type(rows)
    out = capsys.readouterr().out
    assert "not yet implemented" not in out and "1/2 (50%)" in out and "pooled" in out
    summarize_experiments_by_instruction_type(rows, csv=True, csv_compact=True)
    assert "GrabAFruitTask,1/1,1/2" in capsys.readouterr().out


# --------------------------------------------------------------- glob nit

def test_bare_glob_resolves_under_output_before_the_cwd(tmp_path, monkeypatch):
    """`read_results.py "t*"` matched the repo's tests/ directory in the cwd."""
    from robolab.core.utils.file_utils import expand_folder_patterns
    (tmp_path / "tests").mkdir(); (tmp_path / "output" / "t1_run").mkdir(parents=True)
    monkeypatch.chdir(tmp_path)
    folders, expanded = expand_folder_patterns(["t*"], base_dir=str(tmp_path / "output"))
    assert [os.path.basename(f) for f in folders] == ["t1_run"] and expanded
    # a literal cwd path still wins, and a glob with a directory component is cwd-relative
    (tmp_path / "output" / "tests").mkdir()
    assert os.path.basename(expand_folder_patterns(["tests"], base_dir=str(tmp_path / "output"))[0][0]) == "tests"
    assert expand_folder_patterns(["output/t*"], base_dir=str(tmp_path / "output"))[0][0].endswith("output/t1_run")


# --------------------------------------------------------------- onset order

class _ActionManager:
    def __init__(self): self.action = torch.tensor([[0.0] * 7 + [1.0]])


class _Env:
    def __init__(self, n=1, dt=0.1):
        self.num_envs = n; self.device = "cpu"; self.step_dt = dt
        self.episode_length_buf = torch.zeros(n, dtype=torch.long)
        self.common_step_counter = 0
        self.action_manager = _ActionManager()


def test_a_carry_never_starts_before_the_grip(monkeypatch):
    """t1: OBJECT_CARRIED's onset (99) preceded OBJECT_GRIPPED's (103) although the README
    describes grip -> carry -> credit."""
    import robolab.core.task.grasp as G
    state = {"contact": False, "hand": torch.zeros(3), "obj": torch.tensor([0.05, 0.0, 0.0]), "closure": 0.0}
    monkeypatch.setattr(G, "jaw_offset", lambda env, obj: torch.tensor([0.0]))
    monkeypatch.setattr(G, "hand_contact", lambda env, obj, hand: torch.tensor([state["contact"]]))
    monkeypatch.setattr(G, "hand_position", lambda env, hand: state["hand"].clone().unsqueeze(0))
    monkeypatch.setattr(G, "object_position", lambda env, obj: state["obj"].clone().unsqueeze(0))
    monkeypatch.setattr(G, "hand_closed", lambda env, hand, thr: torch.tensor([state["closure"] >= thr]))
    env = _Env(); env.episode_length_buf[:] = 1; tracker = G.GraspTracker(env)
    events = []

    def tick(contact, x, closure):
        env.common_step_counter += 1; env.episode_length_buf += 1
        state["contact"] = contact; state["closure"] = closure
        state["hand"] = torch.tensor([x, 0.0, 0.0]); state["obj"] = torch.tensor([x + 0.05, 0.0, 0.0])
        tracker.grasped("banana", "gripper", env_id=0)
        events.extend(tracker.pop_events())

    tick(True, 0.00, 0.0)      # contact begins with the hand open (step 1)
    tick(True, 0.00, 0.0)
    tick(True, 0.00, 0.6)      # jaws close on it (step 3)
    for i in range(1, 6):      # hand moves, object coupled -> a carry
        tick(True, 0.01 * i, 0.6)
    kinds = {e[3]: e[4]["onset_step"] for e in events}
    assert "gripped" in kinds and "grabbed" in kinds
    assert kinds["grabbed"] >= kinds["gripped"] == 4


# --------------------------------------------------------------- friction summary

def test_friction_applied_carries_a_per_object_summary():
    from robolab.core.physics.friction import summarise_rows
    rows = [[0.5, 0.5, 0.1]] * 256
    assert summarise_rows(rows) == {"static": 0.5, "dynamic": 0.5, "restitution": 0.1, "shapes": 256, "uniform": True}
    mixed = summarise_rows([[0.5, 0.5, 0.1], [2.0, 2.0, 0.1]])
    assert mixed["uniform"] is False and mixed["shapes"] == 2 and len(mixed["distinct"]) == 2
    assert summarise_rows("NOT A BODY") == {"shapes": 0}


# --------------------------------------------------------------- reason names the furthest object

def test_reason_for_an_any_ladder_names_the_object_that_progressed():
    src = (ROOT / "robolab/core/task/conditionals_state_machine.py").read_text()
    assert "max(incomplete_objects, key=lambda t: t[2])" in src


# --------------------------------------------------------------- pyproject resolves off Linux

def test_torch_index_is_linux_only():
    """`uv sync` could not resolve on macOS: the CUDA index has no macOS wheels and the
    source had no platform marker, so the no-simulator suite could not be installed on a
    laptop the documented way."""
    import tomllib
    src = tomllib.load(open(ROOT / "pyproject.toml", "rb"))["tool"]["uv"]["sources"]
    for pkg in ("torch", "torchvision"):
        entries = src[pkg] if isinstance(src[pkg], list) else [src[pkg]]
        assert any("linux" in e.get("marker", "") for e in entries), pkg
