# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""P67: per-arm end-effector channels, and a loud failure when there are none.

The bimanual rigs declare `ee_recorder_bodies = {"left_ee_pose": …,
"right_ee_pose": …}` (robolab/robots/bimanual_franka.py:190), so a bimanual demo
has no `ee_pose` group. Upstream read `demo["ee_pose"]["position"]`, which raised
KeyError inside a bare `except Exception`, so `load_demo_data` returned None and
every bimanual run silently produced no metrics.

`read_ee_channels` takes any mapping, so this needs no h5py.
"""
import sys
import types

import pytest

# compute_metrics imports h5py at module level; read_ee_channels itself takes any
# mapping, so the offline suite stubs the module out (same pattern as the isaaclab
# stubs elsewhere in offline_tests/).
sys.modules.setdefault("h5py", types.ModuleType("h5py"))

from robolab.core.metrics.compute_metrics import read_ee_channels  # noqa: E402


class _Arr(list):
    """Stands in for an h5py dataset: supports [:]."""
    def __getitem__(self, k):
        return list(self) if isinstance(k, slice) else super().__getitem__(k)


def _group(p, o, v=None):
    g = {"position": _Arr(p), "orientation": _Arr(o)}
    if v is not None:
        g["linear_velocity"] = _Arr(v)
    return g


def test_single_arm_demo_is_unchanged():
    demo = {"actions": None, "ee_pose": _group([1, 2], [3, 4], [5, 6])}
    out = read_ee_channels(demo)
    assert list(out["ee_channels"]) == ["ee_pose"]
    assert out["ee_position"] == [1, 2]
    assert out["ee_orientation"] == [3, 4]
    assert out["ee_linear_velocity"] == [5, 6]


def test_bimanual_demo_yields_both_arms():
    demo = {"left_ee_pose": _group([1], [2]), "right_ee_pose": _group([9], [8])}
    out = read_ee_channels(demo)
    assert set(out["ee_channels"]) == {"left_ee_pose", "right_ee_pose"}
    # legacy flat keys point at the first channel alphabetically
    assert out["ee_position"] == [1]


def test_missing_velocity_is_optional():
    out = read_ee_channels({"ee_pose": _group([1], [2])})
    assert "ee_linear_velocity" not in out


def test_no_channel_fails_loudly():
    """The upstream bug was a KeyError swallowed into a silent None."""
    with pytest.raises(KeyError) as e:
        read_ee_channels({"actions": None, "states": None}, "demo_0")
    assert "no ee_pose channel" in str(e.value)
