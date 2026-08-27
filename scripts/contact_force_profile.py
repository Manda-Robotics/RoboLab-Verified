#!/usr/bin/env python3
"""What does a real grasp look like in the pad forces P77 records?

The tow question — is the hand carrying the object, or is the object stuck to it —
could not be answered from boolean contact: a drag, a "magnetic" mug and an ordinary
grasp all read as both-pads-touching. P77 records the force instead. This script asks
whether the force actually separates them, using the one labelling we get for free:
the tracker's own verdict.

  * a **carry** window is the span from an `OBJECT_CARRIED` onset to its release/drop.
    The tracker confirmed object-to-hand coupling over that span, so it is a grasp.
  * a **failed-attempt** window is contact that ended without a carry — the hand
    touched the object and lost it.

If the two separate in force, the same statistic can be turned on the tow candidates.
If they do not, the tow question needs a different signal, and this says so.

    scripts/contact_force_profile.py output/rc4_*
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import numpy as np

CARRIED, RELEASED, DROPPED, ATTEMPT_FAILED = 283, 267, 268, 266


def object_of(event: dict) -> str | None:
    info = event.get("info", "")
    return info.split("'")[1] if "'" in info else None


def windows(events: list, dt: float, n_steps: int):
    """(object, kind, start, end) spans, in HDF5 step indices."""
    carries, attempts = [], []
    open_carry: dict[str, int] = {}
    for e in sorted(events, key=lambda x: x["step"]):
        obj, step = object_of(e), e["step"]
        if obj is None:
            continue
        if e["code"] == CARRIED:
            open_carry[obj] = step
        elif e["code"] in (RELEASED, DROPPED) and obj in open_carry:
            carries.append((obj, open_carry.pop(obj), step))
        elif e["code"] == ATTEMPT_FAILED:
            # The attempt line is stamped at the onset of the burst (P61) and the burst
            # is at most GRASP_ATTEMPT_BURST_S long; take that as the window.
            attempts.append((obj, step, min(n_steps - 1, step + int(round(2.0 / dt)))))
    for obj, start in open_carry.items():          # still held at the end of the episode
        carries.append((obj, start, n_steps - 1))
    return carries, attempts


def sample(pads: np.ndarray, start: int, end: int) -> dict | None:
    """Force statistics over one window, from the steps where the pads touch at all."""
    seg = pads[start:end + 1]
    touching = seg[(seg > 0).any(axis=1)]
    if len(touching) < 2:
        return None
    both = (touching > 0).all(axis=1)
    return {
        "steps": len(touching),
        "peak": float(touching.max()),
        "median": float(np.median(touching[touching > 0])),
        "both_pads_frac": float(both.mean()),
        # A real grip squeezes from both sides at once; a dragged object leans on one.
        "symmetry": float(np.median(np.minimum(touching[:, 0], touching[:, 1]) /
                                    np.maximum(touching.max(axis=1), 1e-6))),
    }


def collect(run_dir: str):
    import h5py
    rows = []
    for h5path in sorted(glob.glob(os.path.join(run_dir, "*", "run_*.hdf5"))):
        task_dir = os.path.dirname(h5path)
        with h5py.File(h5path, "r") as f:
            for demo in sorted(f["data"], key=lambda k: int(k.split("_")[1])):
                env = int(demo.split("_")[1])
                logs = os.path.join(task_dir, f"log_0_env{env}.json")
                if not os.path.exists(logs):
                    continue
                log = json.load(open(logs))
                contact = f["data"][demo].get("contact")
                if contact is None:
                    continue
                pads = {k: np.array(contact[k]) for k in contact if contact[k].ndim == 2}
                n = len(next(iter(pads.values()))) if pads else 0
                carries, attempts = windows(log["events"], log["dt"], n)
                for kind, spans in (("carry", carries), ("failed_attempt", attempts)):
                    for obj, a, b in spans:
                        if obj not in pads:
                            continue
                        st = sample(pads[obj], a, b)
                        if st:
                            rows.append(dict(run=os.path.basename(run_dir), env=env, obj=obj,
                                             kind=kind, t=a * log["dt"], **st))
    return rows


def describe(rows, key):
    v = np.array([r[key] for r in rows], dtype=float)
    if not len(v):
        return "        (none)"
    q = np.percentile(v, [10, 50, 90])
    return f"n={len(v):3d}  p10={q[0]:8.3f}  median={q[1]:8.3f}  p90={q[2]:8.3f}"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run_dirs", nargs="+")
    ap.add_argument("--csv", help="write the per-window rows here for labelling")
    args = ap.parse_args()

    rows = [r for d in args.run_dirs for r in collect(d)]
    if not rows:
        print("no windows found — do these runs have P77 force columns?")
        return 1

    for kind in ("carry", "failed_attempt"):
        sub = [r for r in rows if r["kind"] == kind]
        print(f"\n{kind}  ({len(sub)} windows)")
        for key in ("peak", "median", "both_pads_frac", "symmetry", "steps"):
            print(f"  {key:16s}{describe(sub, key)}")

    carry = [r for r in rows if r["kind"] == "carry"]
    fail = [r for r in rows if r["kind"] == "failed_attempt"]
    if carry and fail:
        print("\nseparation (higher |AUC-0.5| = the statistic tells them apart):")
        for key in ("peak", "median", "both_pads_frac", "symmetry", "steps"):
            a = np.array([r[key] for r in carry], float)
            b = np.array([r[key] for r in fail], float)
            # Mann-Whitney AUC: P(a random carry scores above a random failed attempt).
            auc = float((np.greater.outer(a, b) + 0.5 * np.equal.outer(a, b)).mean())
            print(f"  {key:16s} AUC={auc:.2f}")

    if args.csv:
        import csv
        with open(args.csv, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0]))
            w.writeheader(); w.writerows(rows)
        print(f"\nwrote {len(rows)} rows to {args.csv}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
