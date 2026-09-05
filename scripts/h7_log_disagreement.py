#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""H7: does the dense annotation disagree with the run's own event log where the ledger says the
log is wrong? (docs/verified/dense_annotations.md §6 phase A, §9.15)

    scripts/h7_log_disagreement.py ../RoboLab/output/cli_*_robolab120 [--out analysis/h7.md]
        [--tol-s 1.0] [--burst-s 2.0] [--top 10] [--episodes analysis/h7_episodes.jsonl]

Reads every ``phases_<run>_env<env>.json`` (L1 ``phases`` and L2 ``attempts``), the
``log_<run>_env<env>.json`` next to it and the run's ``episode_results.jsonl`` success bit.

The log's vocabulary is upstream's (``OBJECT_GRABBED_SUCCESS`` / ``TARGET_OBJECT_DROPPED`` /
``*_SUCCESS`` ladder conditions), not the fork tracker's (``OBJECT_CARRIED``); the
``no OBJECT_CARRIED in log`` flags the annotator wrote on these corpora are vacuous and are only
counted here. Log grab events are de-flickered first: events on one object closer than ``burst``
are one grab run (P97: two thirds of the corpus's log lines are that flicker).

Two kinds of finding are kept apart, because they mean different things:

**Contradictions** (the machine and the log both commit, and disagree):

* C1  a machine ``place … fail`` on a place task whose log credits the object (a non-grab
      ``*_SUCCESS`` naming it near the segment, or the success bit with no other passing place):
      the disputed-success class of P89 / P96;
* C2  a machine ``place … pass`` on an episode the log scores a failure and never credits;
* C3  a machine ``pick … pass`` on an object the log never scored as grabbed within ``tol``;
* C4  a log grab run with nothing from the machine on that object at all: no pick (pass or fail)
      and no L1 grasp / in-hand phase overlapping it (the blind class);

**Abstentions and threshold splits** (one side does not commit):

* A1  the episode is a success and the machine's last place on the credited object is
      ``unknown`` because the recording ends before the object comes to rest (upstream terminates
      on success, so the settle never lands on disk);
* A2  a log grab run the machine saw as a failed pick or as a grasp / in-hand phase but not as a
      passing pick (a carry shorter than PICK_MIN_HOLD_S, or the tracker's coupling test failing
      where upstream's grasp condition held).

Tasks are ranked by contradictions per episode; the hypothesis (plan §6, H7) is that the P89 /
P90 / P96 / P97 tasks land in the top 10 at least 6 times. Success-bit provenance: the log's own
``success`` field when ``episode_results.jsonl`` has no row.
"""
from __future__ import annotations

import argparse
import collections
import glob
import json
import os
import re
import sys

# the tasks the ledger disputes (private clone docs/VERIFIED_PATCHES.md P89, P90, P96, P97)
LEDGER_TASKS = {
    "BowlStackingRightOnLeftTask", "BowlStackingLeftOnRightTask", "BlockStackingSpecifiedOrderTask",
    "Stack3RubiksCubeTask", "PickDrillTask", "BlockStackingOrderAgnosticTask",
    "GreenSpoonsInPotTask", "GrabAFruitTask",
}

GRAB_EVENTS = {"OBJECT_GRABBED_SUCCESS", "WRONG_OBJECT_GRABBED_FAILURE"}
NON_CREDIT_SUCCESS = {"OBJECT_GRABBED_SUCCESS"}
MACHINE_CONTACT = {"lift", "lower", "transport", "hold", "slip", "close_on", "pinch_no_lift", "release"}
CONTRA = ("C1", "C2", "C3", "C4")
ABST = ("A1", "A2")
NAMES = {"C1": "machine place fail, log credits it", "C2": "machine place pass, log scores a failure",
         "C3": "machine pick pass, log never grabbed it", "C4": "log grab run, machine saw no contact",
         "A1": "success, machine place unknown: recording ends before rest",
         "A2": "log grab run, machine saw contact or a failed pick only"}

_OBJ_EQ = re.compile(r"object=([A-Za-z0-9_]+)")
_OBJ_Q = re.compile(r"'([^']+)'")


def obj_of(e: dict) -> str | None:
    info = e.get("info", "") or ""
    m = _OBJ_EQ.search(info) or _OBJ_Q.search(info)
    return m.group(1) if m else None


def load_episodes(run_dirs):
    eps = []
    for run in run_dirs:
        run = os.path.normpath(run)
        success = {}
        er = os.path.join(run, "episode_results.jsonl")
        if os.path.exists(er):
            for line in open(er):
                if line.strip():
                    r = json.loads(line)
                    success[(r["task_name"], int(r["env_id"]), int(r.get("run", 0)))] = (bool(r["success"]), r.get("policy"))
        for pf in sorted(glob.glob(os.path.join(run, "*", "phases_*_env*.json"))):
            task = os.path.basename(os.path.dirname(pf))
            base = os.path.basename(pf)[7:-5]
            ri, env = base.split("_env")
            lf = os.path.join(os.path.dirname(pf), f"log_{ri}_env{env}.json")
            if not os.path.exists(lf):
                continue
            try:
                doc = json.load(open(pf))
                log = json.load(open(lf))
            except Exception:
                continue
            s, pol = success.get((task, int(env), int(ri)), (None, None))
            if s is None and isinstance(log, dict) and "success" in log:
                s = bool(log["success"])
            eps.append({"run": os.path.basename(run), "policy": pol or os.path.basename(run), "task": task,
                        "env": int(env), "ri": int(ri), "success": s, "doc": doc,
                        "events": log.get("events", []) if isinstance(log, dict) else []})
    return eps


def grab_runs(events, burst):
    by_obj = collections.defaultdict(list)
    for e in events:
        if e.get("name") in GRAB_EVENTS:
            o = obj_of(e)
            if o:
                by_obj[o].append(int(e["step"]))
    runs = []
    for o, steps in by_obj.items():
        steps.sort()
        a = b = steps[0]
        for s in steps[1:]:
            if s - b <= burst:
                b = s
            else:
                runs.append((o, a, b))
                a = b = s
        runs.append((o, a, b))
    return runs


def credits(events):
    """Non-grab success events as (step, object or None); 'Completed subtask' lines carry no object."""
    return [(int(e["step"]), obj_of(e)) for e in events
            if e.get("name", "").endswith("_SUCCESS") and e.get("name") not in NON_CREDIT_SUCCESS]


def overlaps(a, b, c, d, tol):
    return a - tol <= d and c <= b + tol


def judge(ep, tol, burst):
    doc, ev = ep["doc"], ep["events"]
    att, phases, T = doc["attempts"], doc.get("phases", []), doc["num_steps"]
    kind = doc.get("task_kind", "place")
    succ = ep["success"] is True
    picks = [s for s in att if s["label"] == "pick"]
    picks_pass = [s for s in picks if s["result"] == "pass"]
    places = [s for s in att if s["label"] == "place"]
    places_pass = [s for s in places if s["result"] == "pass"]
    runs = grab_runs(ev, burst)
    cred = credits(ev)
    d = collections.Counter()
    notes = []

    def credited(s):
        return any((co is None or co == s["object"]) and s["start"] - tol <= st <= s["end"] + 3 * tol for st, co in cred)

    # C3: machine pick pass, no log grab on that object nearby
    for s in picks_pass:
        if not any(o == s["object"] and overlaps(s["start"], s["end"], a, b, tol) for o, a, b in runs):
            d["C3"] += 1
    # C4 / A2: log grab run against the machine
    for o, a, b in runs:
        if any(s["object"] == o and overlaps(s["start"], s["end"], a, b, tol) for s in picks_pass):
            continue
        seen = any(s["object"] == o and overlaps(s["start"], s["end"], a, b, tol) for s in picks) or \
            any(p["object"] == o and p["label"] in MACHINE_CONTACT and overlaps(p["start"], p["end"], a, b, tol) for p in phases)
        d["A2" if seen else "C4"] += 1
    if kind == "place":
        for s in places:
            if s["result"] == "fail" and (credited(s) or (succ and not places_pass and s is places[-1])):
                d["C1"] += 1
                notes.append(f"C1 place {s['object']} {s['start']}-{s['end']} fail; cause={s.get('cause')}")
            if s["result"] == "pass" and not succ and not credited(s):
                d["C2"] += 1
            if s["result"] == "unknown" and succ and s is places[-1] and (
                    s["end"] >= T - tol or any("episode end" in str(x) or "final steps" in str(x) for x in s.get("attributes", []))):
                d["A1"] += 1
    else:
        if succ and picks and not picks_pass:
            d["C1"] += 1
    stale = sum(1 for s in att for f in s.get("flags", [])
                if "OBJECT_CARRIED" in f or "GRASP_ATTEMPT_FAILED" in f or "SUBTASK_COMPLETED" in f)
    info = {"grab_runs": len(runs), "grab_events": sum(1 for e in ev if e.get("name") in GRAB_EVENTS),
            "credits": len(cred), "stale_flags": stale, "picks_pass": len(picks_pass), "places_pass": len(places_pass),
            "places_unknown": sum(1 for s in places if s["result"] == "unknown"), "kind": kind, "notes": notes}
    return d, info


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--tol-s", type=float, default=1.0)
    ap.add_argument("--burst-s", type=float, default=2.0)
    ap.add_argument("--top", type=int, default=10)
    ap.add_argument("--out")
    ap.add_argument("--episodes", help="one json line per episode with its counts")
    args = ap.parse_args()

    eps = load_episodes(args.runs)
    if not eps:
        sys.exit("no episodes with both phases_*.json and log_*.json")
    keys = CONTRA + ABST
    new = lambda: {"n": 0, "contra": 0, "D": collections.Counter(), "succ": 0, "succ_unknown": 0}
    per_task = collections.defaultdict(new)
    per_policy = collections.defaultdict(new)
    totals = collections.Counter()
    ep_out = open(args.episodes, "w") if args.episodes else None
    for ep in eps:
        dt = ep["doc"]["dt"]
        tol, burst = int(round(args.tol_s / dt)), int(round(args.burst_s / dt))
        d, info = judge(ep, tol, burst)
        totals["episodes"] += 1
        totals["grab_events"] += info["grab_events"]
        totals["grab_runs"] += info["grab_runs"]
        totals["stale"] += info["stale_flags"]
        contra = any(d[k] for k in CONTRA)
        for key, agg in ((ep["task"], per_task), (ep["policy"], per_policy)):
            agg[key]["n"] += 1
            agg[key]["contra"] += int(contra)
            agg[key]["D"].update(d)
            if ep["success"] is True:
                agg[key]["succ"] += 1
                agg[key]["succ_unknown"] += int(d["A1"] > 0)
        for k in keys:
            if d[k]:
                totals[k] += d[k]
                totals[k + "_eps"] += 1
        if ep_out:
            ep_out.write(json.dumps({"run": ep["run"], "policy": ep["policy"], "task": ep["task"], "env": ep["env"],
                                     "ri": ep["ri"], "success": ep["success"], **info, "counts": dict(d)}) + "\n")
    if ep_out:
        ep_out.close()

    n = totals["episodes"]
    L = []
    P = L.append
    P(f"# H7: dense annotation against the upstream event log ({n} episodes, {len(per_task)} tasks, {len(per_policy)} policies)")
    P("")
    P(f"tolerance {args.tol_s} s, grab flicker burst {args.burst_s} s. Log grab events {totals['grab_events']:,} -> "
      f"{totals['grab_runs']:,} de-flickered grab runs ({totals['grab_events'] / max(1, totals['grab_runs']):.1f} per run). "
      f"Stale fork-vocabulary flags on these files: {totals['stale']:,} (vacuous, not used).")
    P("")
    P("| kind | count | episodes | share of episodes |")
    P("|---|---|---|---|")
    for k in keys:
        P(f"| {k} {NAMES[k]} | {totals[k]:,} | {totals[k + '_eps']:,} | {totals[k + '_eps'] / n:.1%} |")
    ce = sum(v["contra"] for v in per_task.values())
    P(f"| any contradiction (C1-C4) | | {ce:,} | {ce / n:.1%} |")
    P("")
    P("## by policy")
    P("")
    P("| policy | episodes | any contradiction | " + " | ".join(keys) + " | successes | successes ending unsettled |")
    P("|---|---|---|" + "---|" * len(keys) + "---|---|")
    for pol, v in sorted(per_policy.items()):
        P(f"| {pol} | {v['n']} | {v['contra'] / v['n']:.1%} | " + " | ".join(str(v["D"][k]) for k in keys)
          + f" | {v['succ']} | {v['succ_unknown']} |")
    P("")
    ranked = sorted(per_task.items(), key=lambda kv: (-sum(kv[1]["D"][k] for k in CONTRA) / kv[1]["n"], kv[0]))
    top = [t for t, _ in ranked[:args.top]]
    hit = [t for t in top if t in LEDGER_TASKS]
    P(f"## top {args.top} tasks by contradictions per episode")
    P("")
    P("| task | episodes | contradictions / ep | episodes with one | " + " | ".join(CONTRA) + " | A1 | A2 | successes | in ledger |")
    P("|---|---|---|---|" + "---|" * (len(CONTRA) + 2) + "---|---|")
    for t, v in ranked[:args.top]:
        P(f"| {t} | {v['n']} | {sum(v['D'][k] for k in CONTRA) / v['n']:.2f} | {v['contra'] / v['n']:.0%} | "
          + " | ".join(str(v["D"][k]) for k in CONTRA) + f" | {v['D']['A1']} | {v['D']['A2']} | {v['succ']} | {'yes' if t in LEDGER_TASKS else ''} |")
    P("")
    P(f"ledger tasks in the top {args.top}: {len(hit)} of {len(LEDGER_TASKS)} ({', '.join(hit) or 'none'}). H7 asks for >= 6 of 10.")
    P("")
    P("## the ledger tasks, wherever they rank")
    P("")
    P("| task | rank of 120 | contradictions / ep | " + " | ".join(CONTRA) + " | A1 | A2 | successes |")
    P("|---|---|---|" + "---|" * (len(CONTRA) + 2) + "---|")
    for rank, (t, v) in enumerate(ranked, 1):
        if t in LEDGER_TASKS:
            P(f"| {t} | {rank} | {sum(v['D'][k] for k in CONTRA) / v['n']:.2f} | " + " | ".join(str(v["D"][k]) for k in CONTRA)
              + f" | {v['D']['A1']} | {v['D']['A2']} | {v['succ']} |")
    P("")
    P("## C1 by task: scored successes the machine judged a failed place")
    P("")
    P("| task | C1 | successes |")
    P("|---|---|---|")
    for t, v in sorted(per_task.items(), key=lambda kv: -kv[1]["D"]["C1"])[:20]:
        if v["D"]["C1"]:
            P(f"| {t} | {v['D']['C1']} | {v['succ']} |")
    P("")
    P("## A1 by task: successes whose recording ends before the object is at rest")
    P("")
    P("| task | A1 | successes |")
    P("|---|---|---|")
    for t, v in sorted(per_task.items(), key=lambda kv: -kv[1]["D"]["A1"])[:15]:
        if v["D"]["A1"]:
            P(f"| {t} | {v['D']['A1']} | {v['succ']} |")
    P("")
    text = "\n".join(L)
    print(text)
    if args.out:
        with open(args.out, "w") as f:
            f.write(text + "\n")


if __name__ == "__main__":
    main()
