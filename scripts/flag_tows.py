#!/usr/bin/env python
# SPDX-License-Identifier: Apache-2.0
"""Flag towing artifacts in recorded runs (docs/verified/towing.md, P124).

Writes ``tows_<run>_env<env>.json`` next to every episode log and prints a prevalence table.

    python scripts/flag_tows.py output/cli_pi05_robolab120                 # every task dir under a run
    python scripts/flag_tows.py output/rc3_BananaInBowlTask/BananaInBowlTask  # one task dir
    python scripts/flag_tows.py output/cli_* --summary                     # table only, from existing tows_*.json
    python scripts/flag_tows.py output/cli_* --csv tows.csv                # every tow segment for review
    python scripts/flag_tows.py output/cli_* --workers 8

Needs h5py and usd-core (the object meshes come from the scene USD). No simulator.
"""
import argparse
import collections
import csv
import glob
import json
import os
import sys
from multiprocessing import Pool

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from robolab.eval.tows import tows_path, write_tows  # noqa: E402


def task_dirs(paths):
    out = []
    for p in paths:
        p = p.rstrip("/")
        if os.path.isfile(os.path.join(p, "env_cfg.json")):
            out.append(p)
        else:
            out += sorted(d.rstrip("/") for d in glob.glob(os.path.join(p, "*", "")) if os.path.isfile(os.path.join(d, "env_cfg.json")))
    return out


def episodes(td):
    for h5 in sorted(glob.glob(os.path.join(td, "run_*.hdf5"))):
        ri = int(os.path.basename(h5)[4:-5])
        for lg in sorted(glob.glob(os.path.join(td, f"log_{ri}_env*.json"))):
            yield ri, int(lg.rsplit("env", 1)[1][:-5])


def process(td):
    docs = []
    for ri, eid in episodes(td):
        try:
            docs.append(write_tows(td, eid, ri))
        except Exception as exc:  # noqa: BLE001
            docs.append({"task_dir": td, "env_id": eid, "run_index": ri, "error": f"{type(exc).__name__}: {exc}", "segments": [], "summary": {"tow_tier": None}})
    return docs


def load(td):
    docs = []
    for ri, eid in episodes(td):
        p = tows_path(td, eid, ri)
        if os.path.isfile(p):
            with open(p) as f:
                docs.append(json.load(f))
    return docs


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("paths", nargs="+", help="run directories or task directories")
    ap.add_argument("--summary", action="store_true", help="only read existing tows_*.json and print the table")
    ap.add_argument("--csv", help="write every tow segment (tier A/B/C) to this CSV")
    ap.add_argument("--workers", type=int, default=1)
    args = ap.parse_args()

    dirs = task_dirs(args.paths)
    if not dirs:
        sys.exit("no task directories with env_cfg.json found")
    if args.summary:
        per_dir = [load(td) for td in dirs]
    elif args.workers > 1:
        with Pool(args.workers) as pool:
            per_dir = pool.map(process, dirs)
    else:
        per_dir = [process(td) for td in dirs]

    by_run = collections.defaultdict(lambda: collections.Counter())
    rows = []
    for td, docs in zip(dirs, per_dir):
        run = os.path.basename(os.path.dirname(td))
        for d in docs:
            c = by_run[run]
            c["episodes"] += 1
            tier = d["summary"].get("tow_tier")
            if tier == "A":
                c["tier_A"] += 1
            if tier in ("A", "B"):
                c["tier_AB"] += 1
            if d.get("error"):
                c["errors"] += 1
            for s in d["segments"]:
                if s["cls"] == "tow":
                    rows.append([s["tier"], run, os.path.basename(td), d["env_id"], d["run_index"], s["object"], s["pad"], s["t_start"], s["t_end"],
                                 s["duration_s"], s["depth_med_mm"], s["depth_max_mm"], s["finger_min"], s["finger_max"], s["rise_m"], s["path_m"]])

    print(f"{'run':40s} {'episodes':>8s} {'tier A':>8s} {'A share':>8s} {'A or B':>8s} {'AB share':>9s} {'errors':>6s}")
    for run in sorted(by_run):
        c = by_run[run]
        n = max(c["episodes"], 1)
        print(f"{run:40s} {c['episodes']:8d} {c['tier_A']:8d} {100 * c['tier_A'] / n:7.1f}% {c['tier_AB']:8d} {100 * c['tier_AB'] / n:8.1f}% {c['errors']:6d}")
    if args.csv:
        with open(args.csv, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["tier", "run", "task", "env", "run_index", "object", "pad", "t_start", "t_end", "duration_s",
                        "depth_med_mm", "depth_max_mm", "finger_min", "finger_max", "rise_m", "path_m"])
            w.writerows(sorted(rows, key=lambda r: (str(r[0]), r[1], r[2], r[3], r[7])))
        print(f"wrote {len(rows)} tow segments to {args.csv}")


if __name__ == "__main__":
    main()
