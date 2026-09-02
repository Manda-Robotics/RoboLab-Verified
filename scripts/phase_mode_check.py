#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Do the machine's phase statistics pick out the reviewer's failure modes? (plan phase E, H7)

    scripts/phase_mode_check.py --notes <episode_notes.jsonl> --sources output [--features out.jsonl]

The notes file is one reviewed episode per line (``run, task, env, run_index, modes[]``; the
private review corpus, not in this tree). Each episode is annotated (proxy mode where the
recording has no contact group) and reduced to a few numbers; then, for every reviewer mode
with a plausible machine counterpart, precision and recall of a fixed rule are reported.
Everything is stated per rule so a weak row names what the phases cannot see.
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

from robolab.eval.phases import annotate  # noqa: E402

# mode -> (rule name, predicate over the feature dict)
RULES = {
    "grinding_on_table": ("press_table_share >= 0.25", lambda f: f["press_table_share"] >= 0.25),
    "stuck_on_table": ("press_table_share >= 0.15", lambda f: f["press_table_share"] >= 0.15),
    "grip_never_established": ("no pick passed", lambda f: f["n_picks_pass"] == 0),
    "gripper_never_closes": ("no closing on anything", lambda f: f["n_picks"] == 0 and f["close_empty_s"] < 0.5),
    "drops_during_transport": ("a drop segment", lambda f: f["n_drops"] >= 1),
    "recovers_after_drop": ("drop then a later pick pass", lambda f: f["drop_then_pass"]),
    "wrong_object_first": ("first pick on a non-target", lambda f: f["first_pick_wrong"]),
    "wrong_object_lock_in": (">= 2 picks on non-targets", lambda f: f["n_picks_wrong"] >= 2),
    "test_gripping": ("close_empty >= 2 runs", lambda f: f["close_empty_runs"] >= 2),
    "hover_no_commit": ("still share >= 0.15 and no pick pass", lambda f: f["still_share"] >= 0.15 and f["n_picks_pass"] == 0),
    "no_carry_attempt": ("no pick at all", lambda f: f["n_picks"] == 0),
    "target_never_attempted": ("no pick on a target", lambda f: f["n_picks_target"] == 0),
    "clean_first_try": ("one pick, passed, no fail, no drop", lambda f: f["n_picks"] == 1 and f["n_picks_pass"] == 1 and f["n_drops"] == 0),
    "clean_transport": ("pick pass and place pass, no drop", lambda f: f["n_picks_pass"] >= 1 and f["n_places_pass"] >= 1 and f["n_drops"] == 0),
    "towed_not_grasped": ("towed / open-hand carry attribute", lambda f: f["towed"]),
    "carries_with_empty_gripper": ("close_empty >= 1.0 s", lambda f: f["close_empty_s"] >= 1.0),
    "multi_object_progress": (">= 2 place passes", lambda f: f["n_places_pass"] >= 2),
    "no_regrasp_from_new_angle": (">= 3 failed picks on one object", lambda f: f["max_fails_one_object"] >= 3),
    "regrasp_from_new_angle": ("pick pass after >= 1 failed pick", lambda f: f["pass_after_fail"]),
    "ran_out_of_time": ("still in hand / no place pass at the cap", lambda f: f["n_picks_pass"] >= 1 and f["n_places_pass"] == 0),
    "tips_object_over": ("a disturb phase >= 0.5 s", lambda f: f["disturb_s"] >= 0.5),
    "placement_incomplete": ("a place that failed", lambda f: f["n_places_fail"] >= 1),
}


def features(doc: dict) -> dict:
    dt = doc["dt"]; T = doc["num_steps"]
    fam = collections.Counter()
    for p in doc["phases"]:
        fam[p["label"]] += p["end"] - p["start"] + 1
    at = doc["attempts"]
    picks = [a for a in at if a["label"] == "pick"]
    fails_by_obj = collections.Counter(a["object"] for a in picks if a["result"] == "fail")
    wrong = [a for a in picks if any(x.startswith("wrong_object") or x.startswith("destination_as") for x in a["attributes"])]
    order = [(a["label"], a["result"]) for a in at]
    drop_then_pass = any(order[i][0] == "drop" and any(o == ("pick", "pass") for o in order[i + 1:]) for i in range(len(order)))
    pass_after_fail = any(order[i] == ("pick", "fail") and any(o == ("pick", "pass") for o in order[i + 1:]) for i in range(len(order)))
    return {
        "press_table_share": fam["press_table"] / T,
        "still_share": (fam["hover"] + fam["idle"]) / T,
        "disturb_s": fam["disturb"] * dt,
        "close_empty_s": fam["close_empty"] * dt,
        "close_empty_runs": sum(1 for p in doc["phases"] if p["label"] == "close_empty"),
        "n_picks": len(picks), "n_picks_pass": sum(1 for a in picks if a["result"] == "pass"),
        "n_picks_target": sum(1 for a in picks if doc["roles"].get(a["object"]) == "target"),
        "n_picks_wrong": len(wrong), "first_pick_wrong": bool(picks) and picks[0] in wrong,
        "n_drops": sum(1 for a in at if a["label"] == "drop"),
        "n_places_pass": sum(1 for a in at if a["label"] == "place" and a["result"] == "pass"),
        "n_places_fail": sum(1 for a in at if a["label"] == "place" and a["result"] == "fail"),
        "max_fails_one_object": max(fails_by_obj.values()) if fails_by_obj else 0,
        "drop_then_pass": drop_then_pass, "pass_after_fail": pass_after_fail,
        "towed": any("towed" in a["attributes"] or "open_hand_carry" in a["attributes"] for a in at),
        "n_flags": sum(len(a["flags"]) for a in at),
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--notes", required=True)
    ap.add_argument("--sources", nargs="*", default=["output"])
    ap.add_argument("--features", default=None, help="write one JSON line per episode with its features")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args(argv)
    notes = [json.loads(l) for l in open(args.notes) if l.strip()]
    if args.limit:
        notes = notes[: args.limit]
    out = open(args.features, "w") if args.features else None
    rows = []
    t0 = time.time()
    for i, n in enumerate(notes):
        td = next((p for src in args.sources for p in glob.glob(os.path.join(src, n["run"], n["task"])) if os.path.isdir(p)), None)
        if td is None:
            continue
        try:
            doc = annotate(td, int(n["env"]), int(n.get("run_index", 0))).doc
        except Exception as exc:
            print(f"[mode_check] {n['run']}/{n['task']} env{n['env']}: {type(exc).__name__}: {exc}", file=sys.stderr)
            continue
        f = features(doc)
        f.update(run=n["run"], task=n["task"], env=int(n["env"]), policy=n.get("policy"), success=n.get("success"),
                 modes=n.get("modes", []), proxy=doc["annotator"]["contact_source"] == "proxy")
        rows.append(f)
        if out:
            out.write(json.dumps(f) + "\n"); out.flush()
        if (i + 1) % 50 == 0:
            print(f"  {i + 1}/{len(notes)} ({time.time() - t0:.0f}s)", file=sys.stderr, flush=True)
    print(f"{len(rows)} reviewed episodes annotated ({time.time() - t0:.0f}s)\n")
    print("| reviewer mode | n | machine rule | precision | recall | rule fires |")
    print("|---|---|---|---|---|---|")
    for mode, (name, pred) in RULES.items():
        has = [r for r in rows if mode in r["modes"]]
        fires = [r for r in rows if pred(r)]
        tp = sum(1 for r in has if pred(r))
        if not has:
            continue
        prec = tp / len(fires) if fires else 0.0
        rec = tp / len(has)
        print(f"| `{mode}` | {len(has)} | {name} | {prec:.2f} | {rec:.2f} | {len(fires)} |")
    return 0


if __name__ == "__main__":
    sys.exit(main())
