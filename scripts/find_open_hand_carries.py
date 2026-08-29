#!/usr/bin/env python3
"""Find objects carried while the gripper is essentially OPEN ("stuck to one finger").

findings.md H-R6-2 / H-R7-1: reviewers repeatedly saw an object move with the
hand while the hand was visibly not closed -- "it looks almost like the orange is
magnetic". The suspected cause is C1: friction is authored at mu = 2.0 on 289 of
312 objects, which can make single-finger contact a stable hold.

The plan asks for a detector. This is it, offline, from recorded HDF5 -- it
replicates the runtime tow rule (robolab/core/task/grasp.py) without a simulator:

    carried  : object-to-hand offset stays within COUPLING_M over the window
               and the hand itself moved at least HAND_MOVE_M
    open     : gripper closure below TOW_CLOSURE (joint 7 of 13, 0 -> 0.785 rad)
    lifted   : the object rose at least TOW_LIFT_M during the window

Contact is not recorded, so "which finger" cannot be established here -- that is
what P62 would add. Everything else is checkable.

    python scripts/find_open_hand_carries.py output/isaac60_robolab120_pi05
"""
from __future__ import annotations

import argparse
import glob
import os
import sys

GRIPPER_JOINT = 7
GRIPPER_CLOSED_RAD = 0.785
TOW_CLOSURE = 0.10      # fraction closed; above this it is a real grip
COUPLING_M = 0.005      # offset must stay this stable to count as carried
HAND_MOVE_M = 0.01      # the hand must actually travel
TOW_LIFT_M = 0.02       # and the object must come clear of its support
WIN = 8                 # window length in steps (~0.5 s at 15 Hz)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("roots", nargs="+")
    ap.add_argument("--min-duration-s", type=float, default=0.3)
    args = ap.parse_args()

    try:
        import h5py
        import numpy as np
    except ImportError:
        print("needs h5py + numpy", file=sys.stderr)
        return 2

    hits = []
    for root in args.roots:
        for path in sorted(glob.glob(os.path.join(root, "**", "*.hdf5"), recursive=True)):
            task = os.path.basename(os.path.dirname(path))
            run = path.split(os.sep)[-3] if len(path.split(os.sep)) > 2 else "?"
            try:
                with h5py.File(path) as h:
                    for ei, demo in enumerate(h["data"]):
                        d = h[f"data/{demo}"]
                        ee = d["ee_pose/position"][:]
                        jp = d["states/articulation/robot/joint_position"][:]
                        closure = jp[:, GRIPPER_JOINT] / GRIPPER_CLOSED_RAD
                        n = min(len(ee), len(closure))
                        for obj in d["states/rigid_object"]:
                            p = d[f"states/rigid_object/{obj}/root_pose"][:, :3]
                            m = min(n, len(p))
                            rel = p[:m] - ee[:m]
                            run_start = None
                            for i in range(m - WIN):
                                j = i + WIN
                                stable = np.linalg.norm(rel[j] - rel[i]) < COUPLING_M
                                moved = np.linalg.norm(ee[j] - ee[i]) >= HAND_MOVE_M
                                lifted = (p[j, 2] - p[i, 2]) >= TOW_LIFT_M
                                openhand = closure[i:j].max() < TOW_CLOSURE
                                if stable and moved and lifted and openhand:
                                    run_start = i if run_start is None else run_start
                                elif run_start is not None:
                                    dur = (i - run_start) / 15.0
                                    if dur >= args.min_duration_s:
                                        hits.append((dur, task, obj, ei, run_start / 15.0,
                                                     float(closure[run_start:i].max()), run))
                                    run_start = None
            except (OSError, KeyError, IndexError):
                continue

    hits.sort(reverse=True)
    print(f"open-hand carries found: {len(hits)}\n")
    for dur, task, obj, ei, t0, cl, run in hits[:25]:
        print(f"  {task:30s} env {ei}  t={t0:6.2f}s  {dur:.1f}s  '{obj}' "
              f"(max closure {cl*100:.0f}% ) [{run}]")
    return 0


if __name__ == "__main__":
    sys.exit(main())
