#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Pick the episodes for the dense-annotation gold set (docs/verified/dense_annotations.md, phase B).

    scripts/select_gold_episodes.py --out analysis/gold_set.jsonl \\
        --contact-runs output/rc3_* output/rc4_* output/rc5_* output/rc6_* output/rc6b_* output/rc7_* \\
        --other-runs output/cli_*_robolab120 [--exclude output/t?_*] [--n 50] [--seed 7] [--port 8080]

Stratified, seeded, at most two episodes per task and stratum, read from
``episode_results.jsonl`` and ``env_cfg.json`` only. Strata over the fork's own recordings
(``--contact-runs``: pad forces recorded, tracker replay possible):

    clean_success   success, no failed attempt, no drop
    messy_success   success with a failed attempt or a drop on the way
    grasped_failed  failure after at least one grip / carry
    never_grasped   failure with no grip or carry at all
    crowded         any outcome, scene with >= 12 contact objects

and over other policies (``--other-runs``: recordings without the contact group, annotated in
proxy mode), half successes and half failures per policy. The fixture episodes used for tuning
(``--exclude``) never enter the set. Prints one markdown line per episode with its dashboard
deep link; writes one JSON line per episode to ``--out``.
"""
from __future__ import annotations

import argparse
import collections
import glob
import json
import os
import random
import sys

GRASP_EVENTS = {"OBJECT_GRIPPED", "OBJECT_CARRIED", "OBJECT_GRABBED_SUCCESS", "WRONG_OBJECT_GRABBED",
                "TARGET_OBJECT_DROPPED", "OBJECT_DROPPED", "OBJECT_RELEASED"}
MESSY_EVENTS = {"GRASP_ATTEMPT_FAILED", "OBJECT_DROPPED", "TARGET_OBJECT_DROPPED"}
STRATA_N = {"clean_success": 7, "messy_success": 7, "grasped_failed": 8, "never_grasped": 7, "crowded": 6}


def episodes(run_dir: str):
    p = os.path.join(run_dir, "episode_results.jsonl")
    if not os.path.exists(p):
        return
    for line in open(p):
        line = line.strip()
        if not line:
            continue
        r = json.loads(line)
        task = r.get("task_name") or r.get("env_name")
        env = int(r.get("env_id", 0))
        ri = int(r.get("run", 0))
        td = os.path.join(run_dir, task)
        if not os.path.exists(os.path.join(td, f"log_{ri}_env{env}.json")):
            continue
        n_obj = 0
        cfg_p = os.path.join(td, "env_cfg.json")
        if os.path.exists(cfg_p):
            try:
                n_obj = len(json.load(open(cfg_p)).get("contact_object_list") or [])
            except Exception:
                n_obj = 0
        ev = {k: v for k, v in (r.get("events") or {}).items() if isinstance(v, int)}
        yield {
            "run": os.path.basename(os.path.normpath(run_dir)), "task": task, "env": env, "run_index": ri,
            "policy": r.get("policy") or "", "success": bool(r.get("success")),
            "duration_s": float(r.get("duration") or 0), "n_objects": n_obj, "events": ev,
        }


def stratum(e: dict) -> str:
    ev = set(e["events"])
    if e["success"]:
        return "messy_success" if ev & MESSY_EVENTS else "clean_success"
    return "grasped_failed" if ev & GRASP_EVENTS else "never_grasped"


def pick(pool: list[dict], n: int, rng: random.Random, per_task: int = 2, taken: set | None = None) -> list[dict]:
    taken = taken if taken is not None else set()
    rng.shuffle(pool)
    out, per = [], collections.Counter()
    for e in pool:
        key = (e["run"], e["task"], e["env"], e["run_index"])
        if key in taken or per[e["task"]] >= per_task:
            continue
        out.append(e); taken.add(key); per[e["task"]] += 1
        if len(out) >= n:
            break
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--contact-runs", nargs="*", default=[], help="runs recorded by this fork (contact/ group present)")
    ap.add_argument("--other-runs", nargs="*", default=[], help="runs of other policies (no contact/ group)")
    ap.add_argument("--exclude", nargs="*", default=[], help="runs to keep out (the tuning fixture)")
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--n-other", type=int, default=15, help="how many of --n come from --other-runs")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--port", type=int, default=8080)
    ap.add_argument("--out", default="analysis/gold_set.jsonl")
    args = ap.parse_args(argv)
    rng = random.Random(args.seed)

    excluded = {os.path.basename(os.path.normpath(r)) for pat in args.exclude for r in glob.glob(pat)}
    contact = [e for pat in args.contact_runs for r in sorted(glob.glob(pat)) for e in episodes(r) if e["run"] not in excluded]
    other = [e for pat in args.other_runs for r in sorted(glob.glob(pat)) for e in episodes(r) if e["run"] not in excluded]
    for e in contact + other:
        e["stratum"] = stratum(e)
    taken: set = set()
    chosen: list[dict] = []
    n_contact = args.n - args.n_other
    scale = n_contact / sum(STRATA_N.values())
    for name, base_n in STRATA_N.items():
        want = max(1, round(base_n * scale))
        if name == "crowded":
            pool = [e for e in contact if e["n_objects"] >= 12]
        else:
            pool = [e for e in contact if e["stratum"] == name]
        got = pick(pool, want, rng, taken=taken)
        for e in got:
            e["selected_as"] = name; e["mode"] = "pads"
        chosen += got
        if len(got) < want:
            print(f"[gold] stratum {name}: wanted {want}, found {len(got)}", file=sys.stderr)
    # other policies: per policy, half successes, half failures
    by_policy = collections.defaultdict(list)
    for e in other:
        by_policy[e["policy"]].append(e)
    if by_policy:
        per_pol = max(1, args.n_other // len(by_policy))
        for pol, pool in sorted(by_policy.items()):
            succ = [e for e in pool if e["success"]]
            fail = [e for e in pool if not e["success"]]
            got = pick(succ, per_pol - per_pol // 2, rng, taken=taken) + pick(fail, per_pol // 2, rng, taken=taken)
            for e in got:
                e["selected_as"] = f"other_policy:{e['stratum']}"; e["mode"] = "proxy"
            chosen += got
    chosen = chosen[: args.n]
    rng.shuffle(chosen)                      # labelling order independent of stratum

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as fh:
        for i, e in enumerate(chosen, 1):
            e["gold_id"] = f"G{i:02d}"
            e["url"] = f"http://127.0.0.1:{args.port}/#/results/run/{e['run']}/task/{e['task']}/ep/{e['env']}/{e['run_index']}"
            fh.write(json.dumps({k: e[k] for k in ("gold_id", "selected_as", "mode", "run", "task", "env", "run_index",
                                                    "policy", "success", "duration_s", "n_objects", "url")}) + "\n")
    print(f"# Gold set: {len(chosen)} episodes (seed {args.seed}); written to {args.out}\n")
    print("| id | stratum | policy | task | env | ok | s | objects | link |")
    print("|---|---|---|---|---|---|---|---|---|")
    for e in chosen:
        print(f"| {e['gold_id']} | {e['selected_as']} | {e['policy']} | {e['task']} | {e['env']} | {int(e['success'])} | "
              f"{e['duration_s']:.0f} | {e['n_objects']} | [{e['run']}]({e['url']}) |")
    counts = collections.Counter(e["selected_as"].split(":")[0] for e in chosen)
    print("\n" + ", ".join(f"{k} {v}" for k, v in counts.items()))
    return 0


if __name__ == "__main__":
    sys.exit(main())
