#!/usr/bin/env python3
"""The friction sensitivity table (P79 / plan item C1): the same policy on the same tasks
at several ``--friction`` settings, side by side.

    scripts/friction_sweep_report.py output/rc7_*            # markdown to stdout
    scripts/friction_sweep_report.py output/rc7_* --md docs/friction_sweep.md

Run directories are grouped by the ``rc7_<mode>_<Task>`` naming the sweep chain uses
(``--pattern`` to change it). Per (mode, task) and per mode it reports success rate,
carries / failed attempts / drops / tows per episode, the share of carries that never
load the second pad (the one-pad "tow or push" signature from ``contact_force_profile.py``,
needs h5py), and the P79 verdict of ``verify_patches.py`` for that run -- so a condition
whose materials did not actually land is marked instead of silently compared.

Nothing here decides what the default should be. It puts the numbers next to each other.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
import sys
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

OBJECT_CARRIED = 283
GRASP_ATTEMPT_FAILED = 266
OBJECT_RELEASED = 267
OBJECT_DROPPED = 268
TOWED_WITHOUT_GRASP = 275
OBJECT_FELL_OFF_TABLE = 272
DEFAULT_PATTERN = r"^rc7_(?P<mode>[^_]+)_(?P<task>.+)$"


def load_run(run_dir: str) -> dict:
    """Counts for one run directory. Missing pieces are reported, not guessed."""
    rows = []
    results = os.path.join(run_dir, "episode_results.jsonl")
    if os.path.exists(results):
        rows = [json.loads(l) for l in open(results) if l.strip()]
    counts = defaultdict(int)
    n_logs = 0
    for f in sorted(glob.glob(os.path.join(run_dir, "*", "log_*.json"))):
        d = json.load(open(f))
        n_logs += 1
        for ev in d.get("events", []):
            counts[ev["code"]] += 1
    return {
        "dir": run_dir,
        "episodes": len(rows),
        "successes": sum(1 for r in rows if r.get("success")),
        "artifacts": sum(1 for r in rows if r.get("physics_artifact")),
        "score": (sum(float(r.get("score") or 0.0) for r in rows) / len(rows)) if rows else None,
        "logs": n_logs,
        "events": sum(counts.values()),
        "carried": counts[OBJECT_CARRIED],
        "failed": counts[GRASP_ATTEMPT_FAILED],
        "dropped": counts[OBJECT_DROPPED],
        "released": counts[OBJECT_RELEASED],
        "towed": counts[TOWED_WITHOUT_GRASP],
        "fell": counts[OBJECT_FELL_OFF_TABLE],
        "complete": os.path.exists(os.path.join(run_dir, "run_complete.json")),
    }


def one_pad_share(run_dir: str) -> tuple[int, int] | None:
    """(carries that never load the second pad, carries measured), or None without h5py."""
    try:
        import h5py  # noqa: F401
        import contact_force_profile as cfp
    except ImportError:
        return None
    if not os.path.exists(os.path.join(run_dir, "run_complete.json")):
        return None           # the HDF5 of a run still being written (or synced) is not readable
    try:
        rows = [r for r in cfp.collect(run_dir) if r["kind"] == "carry"]
    except (OSError, KeyError, ValueError):
        return None
    if not rows:
        return (0, 0)
    return (sum(1 for r in rows if r["both_pads_frac"] == 0.0), len(rows))


def p79_verdict(run_dir: str) -> str:
    try:
        import verify_patches as vp
    except ImportError:
        return "?"
    return vp.p79_friction_applied(run_dir).verdict


def group_runs(run_dirs: list[str], pattern: str) -> dict[str, dict[str, str]]:
    """{mode: {task: run_dir}} from the directory names."""
    rx = re.compile(pattern)
    out: dict[str, dict[str, str]] = defaultdict(dict)
    for d in run_dirs:
        m = rx.match(os.path.basename(os.path.normpath(d)))
        if m:
            out[m.group("mode")][m.group("task")] = d
    return out


def _sr(s: int, n: int) -> str:
    return f"{s}/{n} ({100.0 * s / n:.0f}%)" if n else "—"


def _per_ep(x: int, n: int) -> str:
    return f"{x / n:.1f}" if n else "—"


def render(groups: dict[str, dict[str, str]], mode_order: list[str] | None = None) -> str:
    modes = [m for m in (mode_order or []) if m in groups] + sorted(m for m in groups if m not in (mode_order or []))
    tasks = sorted({t for g in groups.values() for t in g})
    stats = {m: {t: load_run(d) for t, d in groups[m].items()} for m in modes}
    verdicts = {m: {t: p79_verdict(d) for t, d in groups[m].items()} for m in modes}
    pads = {m: {t: one_pad_share(d) for t, d in groups[m].items()} for m in modes}

    lines = ["| task | " + " | ".join(modes) + " |", "|---|" + "---|" * len(modes)]
    for t in tasks:
        cells = []
        for m in modes:
            s = stats[m].get(t)
            if s is None:
                cells.append("—")
                continue
            flag = "" if s["complete"] else " ⏳"
            v = verdicts[m][t]
            mark = "" if v in ("PASS", "N/A") else f" **{v}**"
            cells.append(_sr(s["successes"], s["episodes"]) + flag + mark)
        lines.append(f"| {t} | " + " | ".join(cells) + " |")

    lines += ["", "| per mode | " + " | ".join(modes) + " |", "|---|" + "---|" * len(modes)]

    def total(key):
        return {m: sum(s[key] for s in stats[m].values()) for m in modes}
    n, succ = total("episodes"), total("successes")
    lines.append("| success rate | " + " | ".join(_sr(succ[m], n[m]) for m in modes) + " |")
    for label, key in (("carries / episode", "carried"), ("failed attempts / episode", "failed"),
                       ("drops (closed hand) / episode", "dropped"), ("events / episode", "events")):
        tot = total(key)
        lines.append(f"| {label} | " + " | ".join(_per_ep(tot[m], n[m]) for m in modes) + " |")
    tows, arts, fell = total("towed"), total("artifacts"), total("fell")
    lines.append("| tows (`TOWED_WITHOUT_GRASP`) / physics-artifact episodes | "
                 + " | ".join(f"{tows[m]} / {arts[m]}" for m in modes) + " |")
    lines.append("| objects off the table | " + " | ".join(str(fell[m]) for m in modes) + " |")

    def pad_cell(m):
        vals = [p for p in pads[m].values() if p is not None]
        if not vals:
            return "n/a (h5py)"
        one, tot_ = sum(v[0] for v in vals), sum(v[1] for v in vals)
        return f"{one}/{tot_} ({100.0 * one / tot_:.0f}%)" if tot_ else "0 carries"
    lines.append("| one-pad carries (never load pad 2) | " + " | ".join(pad_cell(m) for m in modes) + " |")

    def verdict_cell(m):
        vs = list(verdicts[m].values())
        return ", ".join(f"{v}×{vs.count(v)}" for v in sorted(set(vs)))
    lines.append("| P79 verdict per run | " + " | ".join(verdict_cell(m) for m in modes) + " |")
    lines.append("| runs complete | " + " | ".join(
        f"{sum(1 for s in stats[m].values() if s['complete'])}/{len(stats[m])}" for m in modes) + " |")
    return "\n".join(lines) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run_dirs", nargs="+")
    ap.add_argument("--pattern", default=DEFAULT_PATTERN, help="regex with <mode> and <task> groups")
    ap.add_argument("--modes", default="upstream,1.0,0.5,realistic", help="column order")
    ap.add_argument("--md", help="also write the markdown here")
    a = ap.parse_args()
    groups = group_runs(a.run_dirs, a.pattern)
    if not groups:
        print(f"no run directory matched {a.pattern!r}", file=sys.stderr)
        return 2
    text = render(groups, a.modes.split(","))
    print(text)
    if a.md:
        with open(a.md, "w") as f:
            f.write(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
