# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# isort: skip_file

"""Pytest configuration for RoboLab install-verification tests.

Boots Isaac Sim once at conftest load so test files can freely import
isaaclab/robolab modules at their top level. Auto-accepts the Omniverse
EULA so a fresh install works headless without any prompts.
"""

import os

# Accept the Omniverse EULA non-interactively. Must be set before any
# isaaclab import. setdefault → user can still override.
os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "Y")

import cv2  # noqa: E402, F401  must be imported before isaaclab

from isaaclab.app import AppLauncher  # noqa: E402

# Launch Isaac Sim once for the whole pytest session. Carb's logger is told to
# only emit warnings or worse so install-verification output isn't drowned in
# Isaac Sim's [Info] startup/shutdown chatter.
_launcher = AppLauncher(
    headless=True,
    enable_cameras=True,
    carb_settings={
        "/log/level": "warn",
        "/log/outputStreamLevel": "warn",
        "/log/fileLogLevel": "warn",
    },
)
simulation_app = _launcher.app


def pytest_addoption(parser):
    parser.addoption(
        "--task",
        default="BananaInBowlTask",
        help="Task class name for test_run_empty (default: BananaInBowlTask). "
             "The test runs the first registered env matching this task.",
    )
    parser.addoption(
        "--env-name",
        default=None,
        help="Full registered env name for test_run_empty (e.g. BananaInBowlTaskHomeOffice). "
             "If set, overrides --task.",
    )


import pytest


@pytest.fixture
def task_arg(request):
    return request.config.getoption("--task")


@pytest.fixture
def env_name_arg(request):
    return request.config.getoption("--env-name")


_EXIT_STATUS = 0


def pytest_sessionfinish(session, exitstatus):
    """Remember pytest's verdict; the app is closed in ``pytest_unconfigure`` below."""
    global _EXIT_STATUS
    _EXIT_STATUS = int(getattr(exitstatus, "value", exitstatus))


def pytest_unconfigure(config):
    """Close Isaac Sim without losing pytest's exit status.

    Closing the app inside ``pytest_sessionfinish`` made every run — failures
    included — exit 0: Kit's shutdown ends the process on its own terms and
    pytest never gets to return its status, so CI could not see red.
    ``pytest_unconfigure`` runs after the terminal summary. On a green run we
    close the app normally; on a red run we skip Kit's shutdown entirely and
    exit with pytest's status ourselves (the process is ending either way).
    """
    import sys

    sys.stdout.flush()
    sys.stderr.flush()
    if _EXIT_STATUS != 0:
        os._exit(_EXIT_STATUS)
    try:
        simulation_app.close()
    except Exception:
        pass
