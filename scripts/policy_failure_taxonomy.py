#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Per-policy failure taxonomy from the dense annotation (docs/verified/dense_annotations.md §9.14).

    scripts/policy_failure_taxonomy.py ../RoboLab/output/cli_*_robolab120 [--out table.md] [--tasks-common]

Reads every ``phases_<run>_env<env>.json`` under the runs given and the run's
``episode_results.jsonl`` for the success bit, and prints one row per policy: success rate, picks
per episode and per success, pick pass rate, wrong-object picks, regrasps, the release causes
of every place, the share of episode time in ``no_completed_subtask``, time in hand, time
pressing the table. ``--tasks-common`` restricts every policy to the tasks all of them ran.

The question it answers: do policies with similar success rates fail in different ways?
"""
from __future__ import annotations

import argparse
import collections
import glob
import json
import os
import statistics
import sys


def load_runs(run_dirs):
    eps = []
    for run in run_dirs:
        success = {}
        er = os.path.join(run, "episode_results.jsonl")
        if os.path.exists(er):
            for line in open(er):
                if line.strip():
                    r = json.loads(line)
                    success[(r["task_name"], int(r["env_id"]), int(r.get("run", 0)))] = (bool(r["success"]), r.get("policy"))
        for pf in glob.glob(os.path.join(run, "*", "phases_*_env*.json")):
            task = os.path.basename(os.path.dirname(pf))
            base = os.path.basename(pf)[7:-5]          # "<ri>_env<env>"
            ri, env = base.split("_env")
            try:
                doc = json.load(open(pf))
            except Exception:
                continue
            s, pol = success.get((task, int(env), int(ri)), (None, None))
            eps.append({"run": os.path.basename(os.path.normpath(run)), "task": task, "env": int(env), "ri": int(ri),
                        "success": s, "policy": pol or os.path.basename(os.path.normpath(run)), "doc": doc})
    return eps


def row(eps):
    n = len(eps)
    succ = [e for e in eps if e["success"] is True]
    att = [a for e in eps for a in e["doc"]["attempts"]]
    picks = [a for a in att if a["label"] == "pick"]
    places = [a for a in att if a["label"] == "place"]
    ncs = [a for a in att if a["label"] == "no_completed_subtask"]
    dur = sum(e["doc"]["num_steps"] * e["doc"]["dt"] for e in eps) or 1.0
    causes = collections.Counter(a.get("cause") or "none" for a in places)
    n_pl = len(places) or 1
    def share(k): return causes[k] / n_pl
    regrasp = sum(1 for a in picks if any(x.startswith("regrasp") for x in a["attributes"]))
    wrong = sum(1 for a in picks if any(x.startswith("wrong_object") for x in a["attributes"]))
    in_hand = sum(e["doc"]["summary"].get("time_in_hand_s", 0) for e in eps)
    press = sum(e["doc"]["summary"].get("time_press_table_s", 0) for e in eps)
    first = [e["doc"]["summary"]["first_contact_s"] for e in eps if e["doc"]["summary"].get("first_contact_s") is not None]
    return {
        "episodes": n, "success": len(succ) / n if n else 0.0,
        "picks/ep": len(picks) / n if n else 0.0,
        "picks/success": len(picks) / len(succ) if succ else float("inf"),
        "pick pass": sum(1 for a in picks if a["result"] == "pass") / (len(picks) or 1),
        "wrong obj": wrong / (len(picks) or 1), "regrasp": regrasp / (len(picks) or 1),
        "places/ep": len(places) / n if n else 0.0, "place pass": sum(1 for a in places if a["result"] == "pass") / n_pl,
        "released": share("released"), "knocked": share("knocked"), "slipped": share("slipped"), "unclear": share("unclear"),
        "ncs time": sum((a["end"] - a["start"] + 1) * e["doc"]["dt"] for e in eps for a in e["doc"]["attempts"] if a["label"] == "no_completed_subtask") / dur,
        "in hand": in_hand / dur, "press table": press / dur,
        "first contact s": statistics.median(first) if first else float("nan"),
        "no contact": sum(1 for e in eps if e["doc"]["summary"].get("first_contact_s") is None) / n if n else 0.0,
    }


COLS = ["episodes", "success", "picks/ep", "picks/success", "pick pass", "wrong obj", "regrasp", "places/ep", "place pass",
        "released", "knocked", "slipped", "unclear", "ncs time", "in hand", "press table", "first contact s", "no contact"]
PCT = {"success", "pick pass", "wrong obj", "regrasp", "place pass", "released", "knocked", "slipped", "unclear", "ncs time", "in hand", "press table", "no contact"}


def fmt(k, v):
    if k in PCT: return f"{v:.0%}"
    if k == "episodes": return str(v)
    return "∞" if v == float("inf") else f"{v:.2f}" if k != "first contact s" else f"{v:.1f}"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--out", default=None)
    ap.add_argument("--tasks-common", action="store_true")
    args = ap.parse_args(argv)
    eps = load_runs(args.runs)
    by_pol = collections.defaultdict(list)
    for e in eps:
        by_pol[e["policy"]].append(e)
    if args.tasks_common:
        common = set.intersection(*[{e["task"] for e in v} for v in by_pol.values()])
        by_pol = {p: [e for e in v if e["task"] in common] for p, v in by_pol.items()}
        print(f"{len(common)} tasks common to all policies")
    rows = {p: row(v) for p, v in sorted(by_pol.items())}
    lines = ["| metric | " + " | ".join(rows) + " |", "|---|" + "---|" * len(rows)]
    for k in COLS:
        lines.append(f"| {k} | " + " | ".join(fmt(k, r[k]) for r in rows.values()) + " |")
    table = "\n".join(lines)
    print(table)
    if args.out:
        open(args.out, "w").write(table + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
