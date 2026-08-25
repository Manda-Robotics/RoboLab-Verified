# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os
from datetime import datetime

# Get the robolab package root directory (repo root)
PACKAGE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) # robolab (repo root)

# Get source directory (the robolab package itself)
SOURCE_DIR = os.path.dirname(os.path.abspath(__file__)) # robolab/robolab

# Get children of package directory
DEFAULT_OUTPUT_DIR = os.path.join(PACKAGE_DIR, "output")
ASSET_DIR = os.path.join(PACKAGE_DIR, "assets")
BACKGROUND_ASSET_DIR = os.path.join(ASSET_DIR, "backgrounds")
OBJECT_DIR = os.path.join(ASSET_DIR, "objects")
FIXTURE_DIR = os.path.join(ASSET_DIR, "fixtures")
SCENE_DIR = os.path.join(ASSET_DIR, "scenes")
ROBOTS_DIR = os.path.join(ASSET_DIR, "robots")

# Get children of source directory
TASK_DIR = os.path.join(SOURCE_DIR, "tasks")
DEFAULT_TASK_SUBFOLDERS = [
    'benchmark',
]


# Object catalog
OBJECT_CATALOG_PATH = os.path.join(OBJECT_DIR, "object_catalog.json")


def resolve_catalog_path(relative_path: str) -> str:
    """
    Resolve a relative path from object_catalog.json to an absolute path.

    The catalog stores paths relative to PACKAGE_DIR (e.g., 'assets/objects/ycb/banana.usd').
    This function converts them to absolute paths.

    Args:
        relative_path: Path relative to PACKAGE_DIR

    Returns:
        Absolute path string
    """
    # If already absolute, return as-is
    if os.path.isabs(relative_path):
        return relative_path

    return os.path.join(PACKAGE_DIR, relative_path)

# Output directory
_output_dir = None

def set_output_dir(path: str):
    """Set the global output directory."""
    global _output_dir
    _output_dir = path
    if _output_dir is not None:
        os.makedirs(_output_dir, exist_ok=True)

def clear_output_dir():
    set_output_dir(None)

def get_output_dir() -> str:
    """Get the global output directory. Returns DEFAULT_OUTPUT_DIR if not set."""
    if _output_dir is None:
        set_output_dir(DEFAULT_OUTPUT_DIR)
    return _output_dir

def get_timestamp():
    return datetime.now().strftime("%Y-%m-%d_%H-%M-%S")


DEBUG = False
VERBOSE = False
VISUALIZE = False
ENABLE_SUBTASK_PROGRESS_CHECKING = True
# Confirmed success (VERIFIED_PLAN A2): an episode is scored a success when the
# success predicate holds AND every target object has been at rest (slower than
# SUCCESS_MAX_SPEED m/s) for SUCCESS_REST_S seconds in a row. An object already at
# rest when the goal is reached ends the episode immediately; a moving one makes
# the episode wait until it settles. 0 restores upstream's first-frame termination.
SUCCESS_REST_S = 0.2
SUCCESS_MAX_SPEED = 0.02
# A grasp is a carry, not a touch (VERIFIED_PLAN B1; robolab/core/task/grasp.py):
# the object must stay in contact for GRASP_HOLD_S with its offset to the hand
# changing < GRASP_COUPLING_M while the hand moves >= GRASP_HAND_MOVE_M. A contact
# that ends earlier with the hand >= GRASP_ATTEMPT_CLOSURE closed is one
# GRASP_ATTEMPT_FAILED; after a grasp, losing contact is OBJECT_RELEASED when the
# hand is below GRASP_RELEASE_CLOSURE (opening) and OBJECT_DROPPED otherwise.
GRASP_HOLD_S = 0.2
GRASP_COUPLING_M = 0.005
GRASP_HAND_MOVE_M = 0.01
GRASP_ATTEMPT_CLOSURE = 0.3
GRASP_RELEASE_CLOSURE = 0.1
# Scene settling (VERIFIED_PLAN B12): object motion during the first
# SETTLE_WARMUP_S of an episode, for objects the hand is not touching, is the
# scene settling — reported once per env as SCENE_SETTLING, not as OBJECT_BUMPED
# / OBJECT_MOVED on the robot's account.
SETTLE_WARMUP_S = 1.0
RECORD_IMAGE_DATA = False
DEVICE = "cuda:0"

# Difficulty scoring constants (authoritative source for compute_difficulty_score in subtask_utils.py)
SKILL_WEIGHTS: dict[str, int] = {
    'color': 0, 'semantics': 0, 'size': 0, 'conjunction': 0, 'vague': 0,
    'spatial': 1,
    'counting': 2, 'sorting': 2, 'stacking': 2, 'affordance': 2,
    'reorientation': 3,
}
DIFFICULTY_THRESHOLDS = (2, 4)  # simple <= 2, moderate <= 4, complex > 4

# Task category remap: maps fine-grained attributes to higher-level categories
BENCHMARK_TASK_CATEGORIES = {
    'size': 'visual',
    'color': 'visual',
    'semantics': 'visual',
    'spatial': 'relational',
    'conjunction': 'relational',
    'counting': 'relational',
    'stacking': 'procedural',
    'sorting': 'procedural',
    'reorientation': 'procedural',
    'affordance': 'procedural',
}
