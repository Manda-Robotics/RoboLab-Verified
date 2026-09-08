#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Threshold sensitivity of the dense annotation (docs/verified/dense_annotations.md, H4).

    scripts/phase_sensitivity.py output/rc7_upstream_* ../TESTING/trial_output/t?_* [--out table.md]

For every threshold the annotator introduces, halve it and double it, and report over the
episodes given: the share of episodes whose L2 segment set changes (a label, a result, or a
boundary moved by more than 0.5 s), the share of steps whose L1 label changes, and the mean
boundary shift. The tracker replay is done once per episode; only the numpy layers re-run.

A threshold that changes nothing either way is not doing work; one that changes many episodes
is what the gold set must pin down. Constants imported from robolab.constants (the tracker's
own) are not varied here: they are the event log's, not the annotator's.
"""
from __future__ import annotations

import argparse
import collections
import glob
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from robolab.eval import phases as P  # noqa: E402
from robolab.eval.tracker_replay import load_recording, replay  # noqa: E402

VARY = ["V_MOVE", "V_LIFT", "LIFT_M", "NEAR_M", "APPROACH_M", "CLOSING_RATE", "DISP_M", "SLIP_V",
        "SMOOTH_W", "MIN_RUN", "RELEASE_S", "GAP_NCS_S", "NCS_BREAKER_SHARE", "PICK_MIN_HOLD_S", "BREAK_RUN_S",
        "CONTACT_PROXY_M"]


def episodes(run_dirs):
    for run in run_dirs:
        for td in sorted(d for d in glob.glob(os.path.join(run, "*")) if os.path.isdir(d)):
            for lp in sorted(glob.glob(os.path.join(td, "log_0_env*.json"))):
                yield td, int(lp.split("env")[-1].split(".")[0])


def prepare(td, env):
    """Everything that does not depend on the varied thresholds."""
    h5py = P._real_h5py()
    dt = P.resolve_dt(td)
    targets, dests = P.task_roles(td)
    kind = P.task_kind(td)
    with h5py.File(P.find_hdf5(td, 0), "r") as f:
        rec = load_recording(f["data"][f"demo_{env}"])
    rec.dt = dt
    proxy = not rec.pads
    if proxy:
        rec.pads = P.proxy_pads(rec, dt)
    rr = replay(rec, dt, containers=dests)     # the expensive part; invariant to the annotator's thresholds
    return rec, dt, targets, dests, kind, proxy, rr


def run_variant(rec, dt, targets, dests, kind, proxy, rr, redo_replay=False):
    if proxy and redo_replay:
        rec.pads = P.proxy_pads(rec, dt)       # CONTACT_PROXY_M changed
        rr = replay(rec, dt, containers=dests)
    ch = P.compute_channels(rec, dt, targets, dests, rr, "proxy" if proxy else "pads")
    labels, objs = P.label_steps(ch)
    labels, objs = P.smooth_labels(labels, objs, P.SMOOTH_W)
    phases, _ = P.segment(labels, objs, P.MIN_RUN)
    segs = P.attempts(rec, ch, phases, targets, dests, kind, [])
    return labels, [(s["start"] * dt, s["end"] * dt, s["label"], s["result"]) for s in segs]


def l2_changed(a, b, tol=0.5):
    if len(a) != len(b):
        return True, None
    shifts = []
    for x, y in zip(a, b):
        if x[2] != y[2] or x[3] != y[3]:
            return True, None
        shifts += [abs(x[0] - y[0]), abs(x[1] - y[1])]
    return (max(shifts) > tol if shifts else False), (sum(shifts) / len(shifts) if shifts else 0.0)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--out", default=None)
    ap.add_argument("--factors", nargs="*", type=float, default=[0.5, 2.0])
    args = ap.parse_args(argv)

    eps = list(episodes(args.runs))
    base = {}
    prepared = {}
    for td, env in eps:
        try:
            prepared[(td, env)] = prepare(td, env)
            base[(td, env)] = run_variant(*prepared[(td, env)])
        except Exception as exc:
            print(f"[sensitivity] {td} env{env}: {type(exc).__name__}: {exc}", file=sys.stderr)
    n = len(base)
    rows = []
    for name in VARY:
        default = getattr(P, name)
        for f in args.factors:
            value = default * f
            if name in ("SMOOTH_W", "MIN_RUN"):
                value = max(1, int(round(value)))
            setattr(P, name, value)
            changed = 0; l1 = 0.0; shifts = []
            for key, (labels0, segs0) in base.items():
                if name == "CONTACT_PROXY_M" and not prepared[key][5]:
                    continue                       # pads recordings do not use it
                labels1, segs1 = run_variant(*prepared[key], redo_replay=(name == "CONTACT_PROXY_M"))
                c, sh = l2_changed(segs0, segs1)
                changed += int(c)
                if sh is not None:
                    shifts.append(sh)
                l1 += sum(1 for x, y in zip(labels0, labels1) if x != y) / max(1, len(labels0))
            setattr(P, name, default)
            denom = sum(1 for key in base if not (name == "CONTACT_PROXY_M" and not prepared[key][5])) or 1
            rows.append((name, default, f, value, changed / denom, l1 / denom, (sum(shifts) / len(shifts)) if shifts else 0.0, denom))
            print(f"{name:18s} x{f:<4g} {value!s:>10s}  L2 changed {changed / denom:6.1%}  L1 steps changed {l1 / denom:6.1%}  mean shift {rows[-1][6]:.2f}s  (n={denom})", flush=True)
    if args.out:
        with open(args.out, "w") as fh:
            fh.write(f"| threshold | default | factor | value | episodes with a changed L2 | L1 steps changed | mean boundary shift (s) | n |\n|---|---|---|---|---|---|---|---|\n")
            for name, default, f, value, c, l1, sh, denom in rows:
                fh.write(f"| `{name}` | {default} | {f:g} | {value} | {c:.1%} | {l1:.1%} | {sh:.2f} | {denom} |\n")
        print(f"table written to {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
