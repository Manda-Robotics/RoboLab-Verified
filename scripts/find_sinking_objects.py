#!/usr/bin/env python3
"""Find objects that sink into their support during the first second of an episode.

Reads recorded HDF5 only -- no simulator. An object that loses height while the
robot has not touched it is either interpenetrating its support at spawn or
resting on a collider that does not match its visual mesh. Both read to a
reviewer as "it sank through the shelf" (VERIFIED_PLAN H-R7-3).

    python scripts/find_sinking_objects.py output/isaac60_robolab120_pi05
"""
from __future__ import annotations

import argparse
import glob
import os
import sys

SINK_MM = 5.0        # report a drop larger than this
SETTLE_STEPS = 15    # ~1 s at 15 Hz


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("roots", nargs="+", help="run directories to scan")
    ap.add_argument("--sink-mm", type=float, default=SINK_MM)
    args = ap.parse_args()

    try:
        import h5py
        import numpy as np
    except ImportError:
        print("needs h5py + numpy", file=sys.stderr)
        return 2

    hits = []
    scanned = 0
    for root in args.roots:
        for path in sorted(glob.glob(os.path.join(root, "**", "*.hdf5"), recursive=True)):
            task = os.path.basename(os.path.dirname(path))
            try:
                with h5py.File(path) as h:
                    for demo in h["data"]:
                        g = h[f"data/{demo}/states/rigid_object"]
                        for obj in g:
                            z = g[obj]["root_pose"][:, 2]
                            if len(z) <= SETTLE_STEPS:
                                continue
                            scanned += 1
                            drop = (float(z[0]) - float(z[SETTLE_STEPS])) * 1000.0
                            if drop > args.sink_mm:
                                hits.append((drop, task, obj, demo, float(z[0])))
            except (OSError, KeyError):
                continue

    by_pair: dict[tuple[str, str], list[float]] = {}
    for drop, task, obj, _demo, _z in hits:
        by_pair.setdefault((task, obj), []).append(drop)

    print(f"scanned {scanned} (episode, object) pairs")
    print(f"\nobjects losing more than {args.sink_mm:.0f} mm in the first second — {len(by_pair)}")
    for (task, obj), drops in sorted(by_pair.items(), key=lambda kv: -max(kv[1])):
        arr = np.array(drops)
        print(f"  {task:32s} {obj:18s} {arr.mean():6.1f} mm mean over {len(drops)} episodes (max {arr.max():.1f})")
    return 1 if by_pair else 0


if __name__ == "__main__":
    sys.exit(main())
