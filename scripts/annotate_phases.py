#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Dense phase annotation of recorded runs (docs/verified/dense_annotations.md).

    scripts/annotate_phases.py output/<run> [...]           # write phases_<run>_env<env>.json next to each log
    scripts/annotate_phases.py --summary output/rc7_*       # one table across runs, nothing written
    scripts/annotate_phases.py --check-events output/<run>  # plan H2: do log events land in the phase that names them
    scripts/annotate_phases.py --check-replay output/<run>  # plan H5: does the offline GraspTracker reproduce the log
    scripts/annotate_phases.py --show output/<run> --env 1  # print one episode's phases and attempts

Reads the run directory only (``run_*.hdf5``, ``log_*.json``, ``env_cfg.json``). Needs
``h5py``, ``numpy`` and, for the tracker replay, ``torch``; without torch the in-hand
channel comes from geometry and every attempt is flagged as such.
"""
from __future__ import annotations

import argparse
import collections
import glob
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from robolab.eval.phases import annotate, check_events, write  # noqa: E402


def episodes(run_dir: str):
    """Yield (task_dir, run_index, env_id, log_path) for every log in a run directory."""
    for task_dir in sorted(d for d in glob.glob(os.path.join(run_dir, "*")) if os.path.isdir(d)):
        for lp in sorted(glob.glob(os.path.join(task_dir, "log_*_env*.json"))):
            base = os.path.basename(lp)[len("log_"):-len(".json")]
            run_s, env_s = base.split("_env")
            yield task_dir, int(run_s), int(env_s), lp


def load_log(lp: str) -> list[dict]:
    lg = json.load(open(lp))
    return lg.get("events", []) if isinstance(lg, dict) else []


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("runs", nargs="+", help="run directories (each holds <Task>/log_*.json)")
    ap.add_argument("--summary", action="store_true", help="table only; write nothing")
    ap.add_argument("--check-events", action="store_true", help="H2 consistency of log events with phases")
    ap.add_argument("--check-replay", action="store_true", help="H5 faithfulness of the offline tracker replay")
    ap.add_argument("--show", action="store_true", help="print phases and attempts of one episode (--env)")
    ap.add_argument("--env", type=int, default=0)
    ap.add_argument("--run-index", type=int, default=0)
    ap.add_argument("--no-replay", action="store_true", help="skip the GraspTracker replay (geometry only)")
    ap.add_argument("--json", help="write the aggregate summary to this file")
    args = ap.parse_args(argv)

    if args.show:
        run = args.runs[0]
        task_dirs = [d for d in glob.glob(os.path.join(run, "*")) if os.path.isdir(d)]
        td = task_dirs[0]
        a = annotate(td, args.env, args.run_index, use_replay=not args.no_replay)
        d = a.doc
        print(f"{d['task']} env{d['env_id']}  roles={d['roles']}  kind={d['task_kind']}  warnings={d['warnings']}")
        for p in d["phases"]:
            print(f"{p['start']:5d}-{p['end']:5d} {p['start'] * d['dt']:6.1f}-{p['end'] * d['dt']:6.1f}s  {p['label']:14s} {p['object'] or '':18s} {p['role'] or ''}")
        print("attempts:")
        for s in d["attempts"]:
            print(f"{s['start']:5d}-{s['end']:5d}  {s['label']:22s} {s['result']:7s} {s['description']}"
                  + (f"   FLAGS {s['flags']}" if s["flags"] else ""))
        print("summary:", json.dumps(d["summary"]))
        return 0

    agg = collections.Counter()
    fam_time = collections.Counter()
    flags = collections.Counter()
    h2 = collections.Counter()
    h2_miss = collections.Counter()
    h5 = collections.Counter()
    rows = []
    t0 = time.time()
    for run in args.runs:
        for td, ri, env, lp in episodes(run):
            log = load_log(lp)
            try:
                a = annotate(td, env, ri, log_events=log, use_replay=not args.no_replay)
            except Exception as exc:
                print(f"[annotate_phases] {run} env{env}: {type(exc).__name__}: {exc}", file=sys.stderr)
                agg["errors"] += 1
                continue
            d = a.doc
            # H1 invariants
            ph = d["phases"]
            ok = ph and ph[0]["start"] == 1 and ph[-1]["end"] == d["num_steps"] and all(
                ph[i]["end"] + 1 == ph[i + 1]["start"] for i in range(len(ph) - 1))
            at = d["attempts"]
            ok_l2 = all(at[i]["end"] < at[i + 1]["start"] for i in range(len(at) - 1))
            agg["episodes"] += 1
            agg["h1_violations"] += int(not ok) + int(not ok_l2)
            if not ok:
                print(f"[annotate_phases] H1 L1 violation: {run} {d['task']} env{env}", file=sys.stderr)
            if not ok_l2:
                bad = [(at[i]["start"], at[i]["end"], at[i]["label"], at[i + 1]["start"], at[i + 1]["end"], at[i + 1]["label"])
                       for i in range(len(at) - 1) if not at[i]["end"] < at[i + 1]["start"]]
                print(f"[annotate_phases] H1 L2 violation: {run} {d['task']} env{env} T={d['num_steps']} {bad}", file=sys.stderr)
            agg["phases"] += len(ph)
            agg["attempts"] += len(at)
            agg["steps"] += d["num_steps"]
            for s in at:
                for f in s["flags"]:
                    flags[f.split(" ")[0] if f.startswith("held_disagree") else f] += 1
            for k, v in d["summary"]["time_by_family_s"].items():
                fam_time[k] += v
            if args.check_events:
                c = check_events(d, log, a.channels)
                h2["checked"] += c["checked"]; h2["consistent"] += c["consistent"]
                for name, step, lab, good in c["rows"]:
                    if not good:
                        h2_miss[(name, lab)] += 1
            if args.check_replay and a.channels.replay_ok:
                from robolab.eval.tracker_replay import compare_events
                c = compare_events(a.channels.replay_events, log, tol_steps=1)
                for k in ("logged", "replayed", "matched"):
                    h5[k] += c[k]
            rows.append((os.path.basename(run), d["task"], env, d["summary"]))
            if not (args.summary or args.check_events or args.check_replay):
                write(d, td)
    el = time.time() - t0

    if rows:
        print(f"{'run':34s} env  picks pass wrong drops placed  first_s  in_hand_s table_s flags")
        for run, task, env, s in rows:
            print(f"{run[:34]:34s} {env:3d}  {s['n_picks']:5d} {s['n_picks_pass']:4d} {s['n_picks_wrong_object']:5d} "
                  f"{s['n_drops']:5d} {s['n_places_pass']:6d}  {str(s['first_contact_s']):>7s}  {s['time_in_hand_s']:9.1f} "
                  f"{s['time_press_table_s']:7.1f} {s['n_flags']:5d}")
    n = max(1, agg["episodes"])
    print(f"\n{agg['episodes']} episodes, {agg['errors']} errors, {el:.1f}s ({el / n:.2f}s per episode)")
    print(f"H1: {agg['h1_violations']} coverage/order violations")
    print(f"phases per episode {agg['phases'] / n:.1f}, attempts per episode {agg['attempts'] / n:.1f}")
    tot = sum(fam_time.values()) or 1
    print("time by family: " + ", ".join(f"{k} {v / tot:.0%}" for k, v in fam_time.most_common()))
    if flags:
        print("flags: " + ", ".join(f"{k} x{v}" for k, v in flags.most_common()))
    if args.check_events:
        print(f"H2: {h2['consistent']}/{h2['checked']} log events land in a phase that names them "
              f"({h2['consistent'] / max(1, h2['checked']):.1%})")
        for (name, lab), v in h2_miss.most_common(12):
            print(f"   miss x{v}: {name} in {lab}")
    if args.check_replay:
        print(f"H5: {h5['matched']}/{h5['logged']} logged tracker lines reproduced at ±1 step "
              f"({h5['matched'] / max(1, h5['logged']):.1%}); {h5['replayed']} replayed")
    if args.json:
        json.dump({"agg": dict(agg), "flags": dict(flags), "h2": dict(h2), "h5": dict(h5),
                   "family_time": dict(fam_time), "seconds": el}, open(args.json, "w"), indent=1)
    return 1 if agg["h1_violations"] else 0


if __name__ == "__main__":
    sys.exit(main())
