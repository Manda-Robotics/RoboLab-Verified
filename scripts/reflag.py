#!/usr/bin/env python3
"""Re-annotate a recorded episode with the CURRENT flag rules, without a simulator.

The point is regression testing on real episodes: we now have many instances where
The reviewer has said what the right flag would be, and re-running the pod to find out
whether a patch helped costs hours and has twice produced nothing usable.

What can be re-derived from a recorded episode, and what cannot:

  YES  rules that are pure functions of the event list or the recorded state --
       renaming or folding lines, dropping lines by object class, suppressing a line
       given its neighbours, and anything driven by object poses or the commanded
       gripper channel (all recorded).
  NO   rules that change WHEN a predicate is evaluated (P74 moves the spawn probe,
       so the ladder must actually be stepped again), and anything needing a signal
       we never wrote to disk. Before P62 that was all contact; after P62 it is
       object-to-object contact and contact FORCE.

Anything in the second class is reported as "needs a run" rather than silently
skipped, so a green regression report can never mean "we did not check".

    python scripts/reflag.py output/rc3_FoodPacking2CansTask [--json out.json]
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
import sys

CONTAINER_HINTS = ("bin", "crate", "rack", "shelf", "pail", "box", "table", "plate")
NEEDS_A_RUN = {
    "P74": "moves the spawn probe to the end of the settle warm-up — the ladder must be re-stepped",
    "P62+": "tow / drag: the pad-force discriminator (recorded since P77) is not yet a rule this replay applies",
}


def _obj(e) -> str | None:
    m = re.search(r"'(\w+)'", e.get("info", "") or "")
    return m.group(1) if m else None


def _is_container(name: str | None) -> bool:
    return bool(name) and any(h in name.lower() for h in CONTAINER_HINTS)


def reflag(events: list[dict], dt: float) -> tuple[list[dict], list[dict]]:
    """Apply the post-processable rules. Returns (new_events, removed)."""
    out: list[dict] = []
    removed: list[dict] = []
    for i, e in enumerate(events):
        name = e.get("name", "")
        obj = _obj(e)

        # P71: the detector's line is a neutral physical observation, distinct from
        # the ladder's progress line, which keeps OBJECT_GRABBED_SUCCESS.
        if name == "OBJECT_GRABBED_SUCCESS" and "grasped (carry" in (e.get("info") or ""):
            out.append({**e, "name": "OBJECT_CARRIED",
                        "info": (e["info"].replace("grasped (carry established)",
                                                   "carried (grasp established)"))})
            continue

        if name == "GRASP_ATTEMPT_FAILED":
            # P72: a brush against a bin or a shelf is not a grasp attempt.
            if _is_container(obj):
                removed.append({**e, "why": "P72 container"})
                continue
            # P73: contact flickers as an object leaves the hand.
            if any(_obj(y) == obj and y.get("name") in ("OBJECT_RELEASED", "OBJECT_DROPPED")
                   and 0 <= (e["step"] - y["step"]) * dt <= 2.0 for y in events[:i]):
                removed.append({**e, "why": "P73 after release"})
                continue

        # P75: PLACED_WITHOUT_LIFT is retired.
        if name == "PLACED_WITHOUT_LIFT":
            removed.append({**e, "why": "P75 retired"})
            continue

        out.append(e)
    return out, removed


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("--json", default=None, help="write the re-annotated events here")
    args = ap.parse_args()

    logs = sorted(glob.glob(os.path.join(args.run_dir, "*", "log_*_env*.json")))
    if not logs:
        print(f"no episode logs under {args.run_dir}", file=sys.stderr)
        return 2

    payload = {}
    tot_before = tot_after = 0
    for path in logs:
        d = json.load(open(path))
        new, removed = reflag(d["events"], d["dt"])
        tot_before += len(d["events"])
        tot_after += len(new)
        key = f"env{d['env_id']}"
        payload[key] = {"events": new, "removed": removed}
        print(f"  env {d['env_id']}: {len(d['events']):3d} -> {len(new):3d} events")
        for r in removed:
            print(f"      - {r['step']*d['dt']:6.2f}s [{r['why']:17s}] {r['info'][:52]}")

    print(f"\n{os.path.basename(args.run_dir)}: {tot_before} -> {tot_after} events")
    print("not applied here (needs a run):")
    for k, v in NEEDS_A_RUN.items():
        print(f"   {k}: {v}")
    if args.json:
        json.dump(payload, open(args.json, "w"), indent=1)
        print(f"\nwrote {args.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
