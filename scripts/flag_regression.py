#!/usr/bin/env python3
"""Score re-annotated episodes against the reviewer's labelled verdicts.

`analysis/flag_labels.jsonl` records what a human said the right answer was for a
specific (run, task, env, time, flag). This replays the current rules over the
recorded episodes and reports, per label:

    wrong          -> the flag must be GONE
    missing        -> the flag must now be PRESENT
    correct        -> the flag must still be there
    correct-absent -> the flag must still NOT be there
    ambiguous      -> reported, never scored (the reviewer's "unclear" maps here)

The value is the diff over time: run it after each patch and it says which flag a
change fixed, and which one it broke. A label whose signal was never recorded is
reported as "cannot check" instead of counting as a pass.

    python scripts/flag_regression.py [--labels analysis/flag_labels.jsonl] [--output-dir ...]
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from reflag import reflag  # noqa: E402

TOL_S = 3.0            # a label's timestamp is a human's scrub position, not a tick
# Need contact FORCE (P77). Recordings made before P77 hold a per-pad boolean, which
# cannot separate a tow from an ordinary grasp -- so these stay uncheckable until a
# run is made with P77 in the build, and are never quietly passed.
UNRECORDED = {"TOWED_WITHOUT_GRASP", "DRAG"}
NEEDS_RUN = {"OBJECT_IN_CONTAINER_SUCCESS", "TARGET_LOST"}  # need the ladder / termination re-stepped


def load_episode(output_dir, run, env):
    for p in glob.glob(os.path.join(output_dir, run, "*", f"log_*_env{env}.json")):
        return json.load(open(p))
    return None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", default="analysis/flag_labels.jsonl")
    ap.add_argument("--output-dir", default="output")
    args = ap.parse_args()

    labels = [json.loads(l) for l in open(args.labels) if l.strip()]
    counts = {"pass": 0, "fail": 0, "cannot-check": 0, "no-episode": 0, "ambiguous": 0}
    rows = []

    for lb in labels:
        d = load_episode(args.output_dir, lb["run"], lb["env"])
        if d is None:
            counts["no-episode"] += 1
            rows.append(("no-episode", lb, "episode not on disk"))
            continue
        if lb["verdict"] == "ambiguous":
            counts["ambiguous"] += 1
            rows.append(("ambiguous", lb, lb["note"]))
            continue
        if lb["flag"] in UNRECORDED:
            counts["cannot-check"] += 1
            rows.append(("cannot-check", lb, "needs contact force — this run predates P77"))
            continue
        if lb["flag"] in NEEDS_RUN:
            counts["cannot-check"] += 1
            rows.append(("cannot-check", lb, "needs the ladder re-stepped — needs a run"))
            continue

        new, _removed = reflag(d["events"], d["dt"])
        present = any(
            e.get("name") == lb["flag"]
            and (lb.get("object") is None or lb["object"] in (e.get("info") or ""))
            and (lb.get("t") is None or abs(e["step"] * d["dt"] - lb["t"]) <= TOL_S)
            for e in new
        )
        want = {"wrong": False, "missing": True, "correct": True, "correct-absent": False}[lb["verdict"]]
        ok = (present == want)
        counts["pass" if ok else "fail"] += 1
        rows.append(("PASS" if ok else "FAIL", lb,
                     f"{'present' if present else 'absent'}, wanted {'present' if want else 'absent'}"))

    for status, lb, why in rows:
        t = f"{lb['t']:.2f}s" if lb.get("t") is not None else "  --  "
        print(f"  {status:12s} {lb['task'][:26]:26s} env{lb['env']} {t:>8s} {lb['flag'][:26]:26s} {why}")
    print("\n" + "  ".join(f"{k}={v}" for k, v in counts.items()))
    return 1 if counts["fail"] else 0


if __name__ == "__main__":
    sys.exit(main())
