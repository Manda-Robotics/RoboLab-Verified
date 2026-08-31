#!/usr/bin/env python3
"""Which recorded carries had a grip in front of them, and which did not?

P78 makes the grip its own rung, so a pick reads grip -> carry -> the ladder's
success line. A carry with **no** grip in front of it is the hand pushing the object
along rather than holding it — the reviewer's "shove", labelled by eye on the lizard_figurine
and cheez_it clips.

The patched detector cannot be run against an old recording without a GPU, so this
reconstructs the grip from what the recording already holds: the commanded gripper
channel (`actions[:, -1]`, 1.0 = closed) and P77's per-pad contact force. A grip is
the jaws commanded closed while the pads are loaded on that object, in the window
before the carry was established.

    scripts/grip_before_carry.py output/rc4_* output/rc5_*
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import numpy as np

CARRIED, RELEASED, DROPPED = 283, 267, 268
LOOKBACK_S = 2.0          # how far before the carry a grip may have been taken


def object_of(event: dict) -> str | None:
    info = event.get("info", "")
    return info.split("'")[1] if "'" in info else None


def scan(run_dir: str):
    import h5py
    rows = []
    for h5path in sorted(glob.glob(os.path.join(run_dir, "*", "run_*.hdf5"))):
        task_dir = os.path.dirname(h5path)
        try:
            fh = h5py.File(h5path, "r")
        except OSError as exc:
            # A run cut short by the pod stopping leaves a truncated file behind. Say so
            # and move on: one dead recording must not take the whole sweep with it.
            print(f"  skipping {h5path}: {str(exc).split(chr(40))[0].strip()}", file=sys.stderr)
            continue
        with fh as f:
            for demo in sorted(f["data"], key=lambda k: int(k.split("_")[1])):
                env = int(demo.split("_")[1])
                logs = os.path.join(task_dir, f"log_0_env{env}.json")
                if not os.path.exists(logs):
                    continue
                log = json.load(open(logs))
                dt = log["dt"]
                g = f["data"][demo]
                contact = g.get("contact")
                if contact is None or "actions" not in g:
                    continue
                closed = np.array(g["actions"])[:, -1] > 0.5
                back = max(1, round(LOOKBACK_S / dt))
                for e in log["events"]:
                    if e["code"] != CARRIED:
                        continue
                    obj = object_of(e)
                    if obj is None or obj not in contact:
                        continue
                    pads = np.array(contact[obj])
                    onset = e["step"]
                    lo, hi = max(0, onset - back), min(len(pads) - 1, onset)
                    touching = (pads[lo:hi + 1] > 0).any(axis=1)
                    gripping = touching & closed[lo:hi + 1]
                    rows.append({
                        "run": os.path.basename(run_dir), "env": env, "obj": obj,
                        "t": onset * dt, "grip_frames": int(gripping.sum()),
                        "grip": bool(gripping.any()),
                    })
    return rows


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run_dirs", nargs="+")
    ap.add_argument("--only-missing", action="store_true", help="list only the carries with no grip")
    args = ap.parse_args()

    rows = [r for d in args.run_dirs for r in scan(d)]
    if not rows:
        print("no carries found — do these runs have P77 contact columns and actions?")
        return 1
    missing = [r for r in rows if not r["grip"]]
    print(f"{len(rows)} carries; {len(missing)} with NO grip in the {LOOKBACK_S:.0f}s before them "
          f"({100 * len(missing) / len(rows):.0f}%)\n")
    show = missing if args.only_missing else rows
    print(f"{'run':24s} {'env':>3s} {'object':20s} {'t(s)':>7s} {'grip':>6s} {'frames':>6s}")
    for r in sorted(show, key=lambda r: (r["grip"], r["run"], r["env"], r["t"])):
        print(f"{r['run'][4:]:24s} {r['env']:>3d} {r['obj']:20s} {r['t']:7.2f} "
              f"{'yes' if r['grip'] else 'NO':>6s} {r['grip_frames']:6d}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
