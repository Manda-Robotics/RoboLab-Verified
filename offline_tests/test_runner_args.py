"""Every `args_cli.<name>` a runner reads must be an argument someone declared.

This is a GPU-only failure otherwise: argparse is happy, Isaac boots, the scene loads,
and ~40 s in the run dies on `AttributeError: 'Namespace' object has no attribute
'record_image_data'`. It cost a pod launch on the bimanual runner, which read three
options that `add_common_eval_args` does not define -- each runner declares its own.

Parsed statically on purpose: importing a runner starts Isaac Sim.
"""
import pathlib
import re

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
RUNNERS = sorted(ROOT.glob("policies/*/run.py"))

# Set by AppLauncher.add_app_launcher_args(), which we cannot parse without importing
# isaaclab, plus the ones a runner assigns to itself before use.
APP_LAUNCHER_ARGS = {"enable_cameras", "headless", "device", "livestream", "kit_args"}


def declared_in(path: pathlib.Path) -> set[str]:
    text = path.read_text()
    return {m.replace("-", "_") for m in re.findall(r'add_argument\(\s*"--([A-Za-z0-9_-]+)"', text)}


COMMON = declared_in(ROOT / "robolab" / "eval" / "runner.py")


@pytest.mark.parametrize("runner", RUNNERS, ids=lambda p: p.parent.name)
def test_every_option_the_runner_reads_is_declared(runner):
    used = set(re.findall(r"args_cli\.([A-Za-z_][A-Za-z0-9_]*)", runner.read_text()))
    missing = sorted(used - declared_in(runner) - COMMON - APP_LAUNCHER_ARGS)
    assert not missing, f"{runner.relative_to(ROOT)} reads undeclared options: {missing}"


def test_the_check_would_catch_the_bug_it_was_written_for(tmp_path):
    """A runner that reads --record-image-data without declaring it must fail."""
    bad = tmp_path / "run.py"
    bad.write_text('parser.add_argument("--policy")\n'
                   'x = args_cli.record_image_data\n')
    used = set(re.findall(r"args_cli\.([A-Za-z_][A-Za-z0-9_]*)", bad.read_text()))
    assert sorted(used - declared_in(bad) - COMMON - APP_LAUNCHER_ARGS) == ["record_image_data"]


def test_there_is_a_runner_to_check():
    assert RUNNERS, "no policies/*/run.py found — the glob is wrong"
