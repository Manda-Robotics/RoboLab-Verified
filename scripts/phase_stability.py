#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Per-segment stability of the dense annotation under threshold jitter, and whether it
predicts human agreement (docs/verified/dense_annotations.md §9.13).

    scripts/phase_stability.py --labelled --labels analysis/phase_labels.jsonl \\
        --sources ../RoboLab/output ../TESTING/trial_output ../TESTING/dense_output \\
        --samples 40 --spread 1.5 --out analysis/phase_stability.jsonl
    scripts/phase_stability.py --runs ../RoboLab/output/rc3_* ... --out analysis/phase_stability_corpus.jsonl
    scripts/phase_stability.py --report analysis/phase_stability.jsonl --labels ... --sources ...

Every threshold in ``robolab.eval.phases.THRESHOLDS`` is a knife edge: a pick whose object
cleared ``HELD_LIFT_M`` by 12 cm and one that cleared it by 2 mm come out identical. Here each
episode is prepared once (recording, pad fill, tracker replay) and re-derived ``--samples``
times with every threshold multiplied by an independent log-uniform factor in
[1/spread, spread]. A base segment's **stability** is the share of samples in which a segment
with the same label overlaps it at IoU >= 0.5 (``stab``); ``stab_bounds`` also asks both edges
within 0.5 s, ``stab_result`` the same result. No human is involved.

The report joins those scores to the human record: a review verdict on the same machine
segment where one exists (matched by IoU, not seg_index, so a segmentation that has since
moved still finds its row), else the narrated gold via IoU matching. If stable segments agree
with the human far more often than unstable ones, stability is a confidence score that can
route review to the uncertain segments and state a precision for the rest.
"""
from __future__ import annotations

import argparse
import collections
import glob
import json
import math
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from robolab.eval import phases as P  # noqa: E402

# what is varied: the annotator's thresholds, minus geometry it measured (TCP_OFFSET), the
# prepare-stage proxy radius, and the two booleans (they are conventions, not knife edges)
SKIP = {"TCP_OFFSET", "CONTACT_PROXY_M", "DROP_ENDS_AT_LANDING", "RETREAT_BREAKS_APPROACH"}
INTS = {"SMOOTH_W", "MIN_RUN"}
VARY = [k for k in P.THRESHOLDS if k not in SKIP and isinstance(getattr(P, k), (int, float)) and not isinstance(getattr(P, k), bool)]
LABELS = ("pick", "place", "no_completed_subtask")


def iou(a, b):
    inter = max(0.0, min(a[1], b[1]) - max(a[0], b[0]))
    union = max(a[1], b[1]) - min(a[0], b[0])
    return inter / union if union > 0 else 0.0


def seg_tuples(segs, dt):
    return [((s["start"] - 1) * dt, s["end"] * dt, s["label"], s.get("object") or "", s.get("result") or "", s.get("cause"))
            for s in segs if s["label"] in LABELS]


def load_events(td, env, ri):
    lp = os.path.join(td, f"log_{ri}_env{env}.json")
    if not os.path.exists(lp):
        return []
    lg = json.load(open(lp))
    return lg.get("events", lg if isinstance(lg, list) else [])


def find_task_dir(sources, run, task):
    for src in sources:
        for cand in glob.glob(os.path.join(src, run, task)) + glob.glob(os.path.join(src, "*", run, task)):
            if os.path.isdir(cand):
                return cand
    return None


def episodes_from_labels(path, sources):
    seen = []
    for line in open(path):
        if not line.strip():
            continue
        r = json.loads(line)
        if r.get("deleted"):
            continue
        key = (r["run"], r["task"], int(r["env"]), int(r.get("run_index", 0)))
        if key not in seen:
            seen.append(key)
    out = []
    for run, task, env, ri in seen:
        td = find_task_dir(sources, run, task)
        if td:
            out.append((run, task, env, ri, td))
        else:
            print(f"[stability] {run}/{task} not under --sources; skipped", file=sys.stderr)
    return out


def episodes_from_runs(run_dirs):
    out = []
    for run in run_dirs:
        for td in sorted(d for d in glob.glob(os.path.join(run, "*")) if os.path.isdir(d)):
            for lp in sorted(glob.glob(os.path.join(td, "log_*_env*.json"))):
                base = os.path.basename(lp)[4:-5]              # "<ri>_env<env>"
                ri, env = base.split("_env")
                out.append((os.path.basename(os.path.normpath(run)), os.path.basename(td), int(env), int(ri), td))
    return out


def stability_of(td, env, ri, samples, spread, rng):
    prep = P.prepare_recording(td, env, ri)
    dt = prep["dt"]
    events = load_events(td, env, ri)
    defaults = {k: getattr(P, k) for k in VARY}
    base = seg_tuples(P.derive(prep, events)[1], dt)
    hits = np.zeros((len(base), 3))
    ln = math.log(spread)
    try:
        for _ in range(samples):
            for k, d in defaults.items():
                f = math.exp(rng.uniform(-ln, ln))
                v = d * f
                if k in INTS:
                    v = max(1, int(round(v)))
                setattr(P, k, v)
            segs = seg_tuples(P.derive(prep, events)[1], dt)
            for i, b in enumerate(base):
                best = max(((iou(b, s), s) for s in segs if s[2] == b[2]), key=lambda x: x[0], default=(0.0, None))
                if best[0] >= 0.5:
                    s = best[1]
                    hits[i, 0] += 1
                    hits[i, 1] += abs(s[0] - b[0]) <= 0.5 and abs(s[1] - b[1]) <= 0.5
                    hits[i, 2] += s[4] == b[4]
    finally:
        for k, d in defaults.items():
            setattr(P, k, d)
    return base, hits / max(1, samples), prep["contact_source"]


def compute(args):
    eps = episodes_from_labels(args.labels, args.sources) if args.labelled else episodes_from_runs(args.runs)
    rng = np.random.default_rng(args.seed)
    n_seg = 0
    t0 = time.time()
    with open(args.out, "w") as fh:
        for n, (run, task, env, ri, td) in enumerate(eps, 1):
            try:
                base, stab, src = stability_of(td, env, ri, args.samples, args.spread, rng)
            except Exception as exc:
                print(f"[stability] {run}/{task} env{env}: {type(exc).__name__}: {exc}", file=sys.stderr)
                continue
            for i, (b, st) in enumerate(zip(base, stab)):
                fh.write(json.dumps({"run": run, "task": task, "env": env, "run_index": ri, "seg_index": i,
                                     "t_start": round(b[0], 3), "t_end": round(b[1], 3), "label": b[2], "object": b[3],
                                     "result": b[4], "cause": b[5], "contact_source": src,
                                     "stab": round(float(st[0]), 3), "stab_bounds": round(float(st[1]), 3),
                                     "stab_result": round(float(st[2]), 3)}) + "\n")
            fh.flush()
            n_seg += len(base)
            print(f"[{n}/{len(eps)}] {run}/{task} env{env}: {len(base)} segments, "
                  f"mean stab {stab[:, 0].mean() if len(base) else float('nan'):.2f}  ({time.time() - t0:.0f}s)", flush=True)
    print(f"wrote {n_seg} segments to {args.out}")


# ---- report: stability against the human record ------------------------------------------

def human_record(labels_path, sources):
    """episode -> (review rows latest per seg_index, gold segments from score_gold's reader)."""
    import importlib.util
    spec = importlib.util.spec_from_file_location("score_gold", os.path.join(os.path.dirname(__file__), "score_gold.py"))
    SG = importlib.util.module_from_spec(spec); spec.loader.exec_module(SG)
    rows = collections.defaultdict(list)
    for r in SG.live_labels(SG.load_jsonl(labels_path)):
        rows[SG.key(r)].append(r)
    out = {}
    for ep, rs in rows.items():
        latest = {}
        for r in rs:
            if r.get("kind") == "review" and r.get("seg_index") is not None:
                latest[int(r["seg_index"])] = r
        SG.REVIEW_STATS.clear()
        gold, _ = SG.segments_of(rs)
        out[ep] = (list(latest.values()), gold)
    return out


def judge(seg, reviews, gold):
    """(label_ok, result_ok, bounds_ok) from the human record for one machine segment, or None.
    A review verdict on the same machine segment (IoU >= 0.5) is used directly; otherwise the
    narrated gold: matched at IoU >= 0.5 with the same label, both edges within 0.5 s."""
    span = (seg["t_start"], seg["t_end"])
    best = max(((iou(span, (r["t_start"], r["t_end"])), r) for r in reviews), key=lambda x: x[0], default=(0.0, None))
    if best[0] >= 0.5:
        # The verdict was given against the machine *of that day*; the rules have moved since
        # (HELD_LIFT_M, the landing rule), so where the reviewer corrected a value, judge the
        # current segment against the corrected value, not against the stale verdict.
        r = best[1]; v = r.get("verdict") or {}; c = r.get("corrected") or {}
        lab_ok = bool(v.get("label", True)) or (c.get("label") or "").replace("drop", "place") == seg["label"]
        res_ok = bool(v.get("result", True)) or (c.get("result") or "") == (seg["result"] or "")
        if v.get("bounds", True):
            bnd_ok = True
        elif c.get("t_start") is not None and c.get("t_end") is not None and c["t_end"] > c["t_start"]:
            bnd_ok = abs(c["t_start"] - span[0]) <= 0.5 and abs(c["t_end"] - span[1]) <= 0.5
        else:
            bnd_ok = False
        return lab_ok, res_ok, bnd_ok, "review"
    if not gold:
        return None
    lo, hi = gold[0][0], gold[-1][1]
    if max(0.0, min(span[1], hi) - max(span[0], lo)) < 0.5 * (span[1] - span[0]):
        return None                                    # outside what the human annotated
    best = max(((iou(span, (g[0], g[1])), g) for g in gold), key=lambda x: x[0], default=(0.0, None))
    if best[0] < 0.5:
        return False, False, False, "gold"             # no gold segment here at all
    g = best[1]
    return (g[2] == seg["label"], (g[4] or "unknown") == (seg["result"] or "unknown"),
            abs(g[0] - span[0]) <= 0.5 and abs(g[1] - span[1]) <= 0.5, "gold")


def report(args):
    segs = [json.loads(l) for l in open(args.report) if l.strip()]
    rec = human_record(args.labels, args.sources)
    judged = []
    for s in segs:
        ep = (s["run"], s["task"], int(s["env"]), int(s["run_index"]))
        if ep not in rec:
            continue
        j = judge(s, *rec[ep])
        if j is not None:
            judged.append((s, j))
    print(f"{len(segs)} machine segments, {len(judged)} with a human record "
          f"({sum(1 for _, j in judged if j[3] == 'review')} review verdicts, {sum(1 for _, j in judged if j[3] == 'gold')} narrated)")

    def agg(rows, name):
        if not rows:
            return
        n = len(rows)
        lab = sum(j[0] for _, j in rows) / n; res = sum(j[1] for _, j in rows) / n; bnd = sum(j[2] for _, j in rows) / n
        all3 = sum(j[0] and j[1] and j[2] for _, j in rows) / n
        print(f"  {name:<22} n={n:3d}  label {lab:.3f}  result {res:.3f}  bounds {bnd:.3f}  all three {all3:.3f}")

    for metric in ("stab", "stab_bounds"):
        print(f"\n## human agreement by {metric}")
        bins = [(1.0, 1.01, "= 1.00"), (0.9, 1.0, "[0.90, 1.00)"), (0.7, 0.9, "[0.70, 0.90)"), (0.5, 0.7, "[0.50, 0.70)"), (0.0, 0.5, "< 0.50")]
        for lo, hi, name in bins:
            agg([(s, j) for s, j in judged if lo <= s[metric] < hi], name)
        print(f"\n## coverage / precision if only segments with {metric} >= tau are shipped")
        print(f"  {'tau':>5}  {'coverage':>8}  {'label':>6}  {'bounds':>6}  {'all three':>9}   n")
        for tau in (1.0, 0.975, 0.95, 0.9, 0.8, 0.7, 0.5, 0.0):
            keep = [(s, j) for s, j in judged if s[metric] >= tau]
            if not keep:
                continue
            n = len(keep)
            print(f"  {tau:5.3f}  {n / len(judged):8.1%}  {sum(j[0] for _, j in keep) / n:6.3f}  {sum(j[2] for _, j in keep) / n:6.3f}  "
                  f"{sum(j[0] and j[1] and j[2] for _, j in keep) / n:9.3f}   {n}")
    # the least stable segments, for the eye
    print("\n## least stable segments with a human record")
    for s, j in sorted(judged, key=lambda x: x[0]["stab"])[:12]:
        print(f"  stab {s['stab']:.2f} bounds {s['stab_bounds']:.2f}  {s['run']}/{s['task']} env{s['env']}  "
              f"[{s['t_start']:6.2f},{s['t_end']:6.2f}] {s['label']:<20} {s['result']:<5} human: label {'ok' if j[0] else 'NO'} result {'ok' if j[1] else 'NO'} bounds {'ok' if j[2] else 'NO'} ({j[3]})")
    # stability distribution over everything (coverage at scale, no human needed)
    print("\n## stability distribution over all segments in the file")
    for metric in ("stab", "stab_bounds"):
        v = np.array([s[metric] for s in segs])
        print(f"  {metric:<12} mean {v.mean():.3f}  share = 1.00: {(v >= 1.0).mean():.1%}  >= 0.90: {(v >= 0.9).mean():.1%}  < 0.50: {(v < 0.5).mean():.1%}")
    by_label = collections.defaultdict(list)
    for s in segs:
        by_label[s["label"]].append(s["stab"])
    for lab, v in sorted(by_label.items()):
        v = np.array(v)
        print(f"  {lab:<22} n={len(v):4d}  mean stab {v.mean():.3f}  share = 1.00: {(v >= 1.0).mean():.1%}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--labelled", action="store_true", help="the episodes in --labels")
    ap.add_argument("--runs", nargs="*", default=[], help="run directories (every episode in them)")
    ap.add_argument("--labels", default="analysis/phase_labels.jsonl")
    ap.add_argument("--sources", nargs="*", default=["output"])
    ap.add_argument("--samples", type=int, default=40)
    ap.add_argument("--spread", type=float, default=1.5, help="each threshold x a log-uniform factor in [1/spread, spread]")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="analysis/phase_stability.jsonl")
    ap.add_argument("--report", default=None, help="a stability file to report on (skips compute)")
    args = ap.parse_args(argv)
    if args.report:
        report(args)
    else:
        compute(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
