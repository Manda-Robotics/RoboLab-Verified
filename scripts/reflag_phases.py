#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Recompute the log-disagreement flags on existing ``phases_*.json`` files without re-annotating.

    scripts/reflag_phases.py ../RoboLab/output/cli_*_robolab120 [--dry-run]

P123: the flags were written against the fork tracker's vocabulary only, so on a corpus logged by
upstream (``OBJECT_GRABBED_SUCCESS``) every passing pick carried ``no OBJECT_CARRIED in log``.
This strips the old log flags (both spellings) from every segment, re-runs
``_flag_against_log`` over the stored segments and the log next to the file, and rewrites the
file in place. Nothing else in the file changes.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from types import SimpleNamespace

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from robolab.eval.phases import _flag_against_log  # noqa: E402

OLD = ("in log", "SUBTASK_COMPLETED", "carry shorter than PICK_MIN_HOLD_S", "log credits a completion")


def reflag(pf: str, dry: bool) -> tuple[int, int]:
    doc = json.load(open(pf))
    base = os.path.basename(pf)[7:-5]
    ri, env = base.split("_env")
    lf = os.path.join(os.path.dirname(pf), f"log_{ri}_env{env}.json")
    if not os.path.exists(lf):
        return 0, 0
    log = json.load(open(lf))
    events = log.get("events", []) if isinstance(log, dict) else []
    before = 0
    for s in doc["attempts"]:
        keep = [f for f in s.get("flags", []) if not any(k in f for k in OLD)]
        before += len(s.get("flags", [])) - len(keep)
        s["flags"] = keep
    ch = SimpleNamespace(replay_ok=doc.get("annotator", {}).get("tracker_replay", True))
    _flag_against_log(doc["attempts"], ch, events, doc["dt"], doc.get("task_kind", "place"))
    after = sum(1 for s in doc["attempts"] for f in s["flags"] if any(k in f for k in OLD))
    for s in doc["attempts"]:
        head = s["description"].split(" [")[0]
        s["description"] = head + (" [" + "; ".join(s["attributes"]) + "]" if s.get("attributes") else "")
    if "summary" in doc:
        doc["summary"]["n_flags"] = sum(len(s["flags"]) for s in doc["attempts"])
    if not dry:
        tmp = pf + ".tmp"
        with open(tmp, "w") as f:
            json.dump(doc, f)
        os.replace(tmp, pf)
    return before, after


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    files = [pf for run in args.runs for pf in sorted(glob.glob(os.path.join(run, "*", "phases_*_env*.json")))]
    tb = ta = 0
    for pf in files:
        b, a = reflag(pf, args.dry_run)
        tb += b
        ta += a
    print(f"{len(files)} files: {tb:,} old log flags removed, {ta:,} log flags written{' (dry run)' if args.dry_run else ''}")


if __name__ == "__main__":
    main()
