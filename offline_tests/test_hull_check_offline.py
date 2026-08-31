# Offline tests for the open-top containment cap (findings.md H-B6).
# No Isaac, no robolab package import — runs anywhere torch+scipy do:
#
#     python robolab/core/task/test_hull_check_offline.py
#
# (Deliberately not under tests/: that suite's conftest boots Isaac Sim.
# The module is loaded by file path so `import robolab` never runs.)
#
# H-B6: open-top containment used to be an infinite +z column — an object on a
# shelf above the bin, or one released and still falling, counted as "in the
# container". Seen live: H-R5-8 (success with the spatula centroid above the
# rim, still moving), H-R7-6 (bowl hovering above the target bowl scored as
# stacked). The fix caps the column at rim + margin.

import importlib.util
import sys
from pathlib import Path

import numpy as np
import torch

_spec = importlib.util.spec_from_file_location(
    "hull_check", Path(__file__).resolve().parents[1] / "robolab" / "core" / "task" / "hull_check.py")
hull_check = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(hull_check)

build_local_hull = hull_check.build_local_hull
point_in_hull = hull_check.point_in_hull
open_top_planes = hull_check.open_top_planes


def _box_points(w=0.2, d=0.2, h=0.1):
    """An open-top-container stand-in: a w×d×h box with rim at z=h."""
    xs, ys, zs = [-w / 2, w / 2], [-d / 2, d / 2], [0.0, h]
    return np.array([[x, y, z] for x in xs for y in ys for z in zs])


def _pt(x, y, z):
    return torch.tensor([[x, y, z]], dtype=torch.float32)


def test_closed_hull_unchanged():
    hull = build_local_hull(_box_points())
    assert point_in_hull(_pt(0, 0, 0.05), hull.planes_full).item()
    assert not point_in_hull(_pt(0, 0, 0.15), hull.planes_full).item()
    assert not point_in_hull(_pt(0.3, 0, 0.05), hull.planes_full).item()


def test_uncapped_column_is_the_old_bug():
    # Documents the historical semantics (and keeps the opt-out honest):
    # margin=None -> a point 10 m above the bin is "in the container".
    hull = build_local_hull(_box_points(), open_top_cap_margin=None)
    assert point_in_hull(_pt(0, 0, 10.0), hull.planes_open_top).item()


def test_capped_column():
    hull = build_local_hull(_box_points(h=0.1), open_top_cap_margin=0.05)
    # inside the box proper
    assert point_in_hull(_pt(0, 0, 0.05), hull.planes_open_top).item()
    # within the margin band above the rim (banana resting in a shallow bowl)
    assert point_in_hull(_pt(0, 0, 0.13), hull.planes_open_top).item()
    # hovering above the cap (H-R7-6's bowl) -> out
    assert not point_in_hull(_pt(0, 0, 0.25), hull.planes_open_top).item()
    # far above (object on a shelf over the bin) -> out
    assert not point_in_hull(_pt(0, 0, 10.0), hull.planes_open_top).item()
    # below the bottom and lateral misses stay out
    assert not point_in_hull(_pt(0, 0, -0.05), hull.planes_open_top).item()
    assert not point_in_hull(_pt(0.3, 0, 0.05), hull.planes_open_top).item()
    # the cap must not clip the rim itself
    assert point_in_hull(_pt(0, 0, 0.1), hull.planes_open_top).item()


def test_cap_follows_rim_height():
    tall = build_local_hull(_box_points(h=0.3), open_top_cap_margin=0.05)
    assert point_in_hull(_pt(0, 0, 0.33), tall.planes_open_top).item()
    assert not point_in_hull(_pt(0, 0, 0.4), tall.planes_open_top).item()


def test_open_top_planes_cap_plane():
    planes = torch.tensor([
        [0.0, 0.0, 1.0, -0.1],    # rim face (up) — dropped
        [0.0, 0.0, -1.0, 0.0],    # bottom — kept
        [1.0, 0.0, 0.0, -0.1],    # side — kept
    ])
    out = open_top_planes(planes, cap=0.15)
    assert out.shape == (3, 4)                       # 2 kept + 1 cap
    assert torch.allclose(out[-1], torch.tensor([0.0, 0.0, 1.0, -0.15]))
    out_none = open_top_planes(planes, cap=None)
    assert out_none.shape == (2, 4)


def test_batched_points():
    hull = build_local_hull(_box_points(), open_top_cap_margin=0.05)
    pts = torch.tensor([[[0.0, 0.0, 0.05]], [[0.0, 0.0, 0.5]]])   # (2, 1, 3)
    mask = point_in_hull(pts, hull.planes_open_top)
    assert mask.shape == (2, 1)
    assert mask[0, 0].item() and not mask[1, 0].item()


def _main():
    mod = sys.modules[__name__]
    tests = [getattr(mod, n) for n in dir(mod) if n.startswith("test_")]
    failed = 0
    for t in tests:
        try:
            t()
            print(f"  ok    {t.__name__}")
        except AssertionError as e:
            failed += 1
            print(f"  FAIL  {t.__name__}: {e}")
    print(f"{len(tests) - failed}/{len(tests)} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(_main())
