#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Score the dense-annotation gold set (docs/verified/dense_annotations.md, H6 and H8).

    scripts/score_gold.py --gold analysis/gold_set.jsonl --labels analysis/phase_labels.jsonl \\
        [--labels2 analysis/phase_labels_<name>.jsonl] --sources output ../TESTING/trial_output

Two comparisons, both over the segment layer (L2) only:

* **H6, annotator vs annotator** (when ``--labels2`` is given): boundary agreement at
  ±0.25 / 0.5 / 1 / 2 s on internal boundaries, label agreement on IoU-matched segments.
* **H8, machine vs gold**: the machine's ``attempts`` (a ``phases_*.json`` next to the log, else
  annotated on the fly) against the consensus of the annotators (or the single one): segment F1
  at IoU 0.5, boundary recall at the four tolerances, label and result accuracy on matched
  segments, recall by segment duration (0-1, 1-2, 2-4, 4-8, 8+ s).

Scoring follows the labeler's ``eval/metrics.py``: greedy one-to-one matching by descending
IoU, internal boundaries only (the first start and the last end are dictated by the episode),
counts pooled over the set. A ``boundary`` mark counts as a boundary but not as a segment.
Labels ``pick`` / ``place`` / ``drop`` / ``no_completed_subtask``; the machine's ``carry``
(still held at the cap) is scored as unlabeled tail and ignored.
"""
from __future__ import annotations

import argparse
import collections
import glob
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

TOLS = (0.25, 0.5, 1.0, 2.0)
BUCKETS = ((0, 1), (1, 2), (2, 4), (4, 8), (8, 1e9))
LABELS = ("pick", "place", "drop", "no_completed_subtask")


def load_jsonl(path):
    if not path or not os.path.exists(path):
        return []
    rows = []
    for line in open(path):
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def live_labels(rows):
    dead = {r["id"] for r in rows if r.get("deleted")}
    return [r for r in rows if not r.get("deleted") and r.get("id") not in dead]


def key(r):
    return (r["run"], r["task"], int(r["env"]), int(r.get("run_index", 0)))


REVIEW_STATS = collections.Counter()


def review_segments(rows):
    """Review rows (a verdict per machine segment, latest per seg_index wins) -> gold segments:
    the machine's values where the verdict says they are right, the corrected ones where not;
    a corrected label "none" removes the segment."""
    latest = {}
    for r in rows:
        if r.get("kind") == "review" and r.get("seg_index") is not None:
            latest[int(r["seg_index"])] = r
    out = []
    for r in latest.values():
        v = r.get("verdict") or {}
        c = r.get("corrected") or {}
        REVIEW_STATS["n"] += 1
        for k in ("label", "result", "bounds"):
            REVIEW_STATS[k + "_ok"] += bool(v.get(k, True))
        lab = r.get("label") if v.get("label", True) else c.get("label")
        obj = r.get("object") if v.get("label", True) else c.get("object")
        res = r.get("result") if v.get("result", True) else c.get("result")
        t0, t1 = (r["t_start"], r["t_end"]) if v.get("bounds", True) else (c.get("t_start"), c.get("t_end"))
        if lab in (None, "", "none") or t0 is None or t1 is None:
            REVIEW_STATS["removed"] += 1
            continue
        out.append((float(t0), float(t1), "place" if lab == "drop" else lab, obj or "", res or ""))
    return out


def segments_of(rows):
    """label rows -> sorted [(start, end, label, object, result)] and boundary set. Free-form
    segment marks and review verdicts both count; boundary marks add boundaries only."""
    segs, bounds = list(review_segments(rows)), set()
    for r in rows:
        if r.get("kind") == "review":
            continue
        if r.get("kind") == "boundary" or r.get("t_end") is None:
            bounds.add(round(float(r["t_start"]), 3))
            continue
        lab = r.get("label") or ""
        segs.append((float(r["t_start"]), float(r["t_end"]), "place" if lab == "drop" else lab, r.get("object") or "", r.get("result") or ""))
    segs.sort()
    for a, b, *_ in segs:
        bounds.add(round(a, 3)); bounds.add(round(b, 3))
    return segs, bounds


def internal(bounds, segs):
    if not segs:
        return sorted(bounds)
    first, last = segs[0][0], segs[-1][1]
    return sorted(b for b in bounds if first < b < last)


def iou(a, b):
    inter = max(0.0, min(a[1], b[1]) - max(a[0], b[0]))
    union = max(a[1], b[1]) - min(a[0], b[0])
    return inter / union if union > 0 else 0.0


def match(pred, gold, thr=0.5):
    pairs = sorted(((iou(p, g), i, j) for i, p in enumerate(pred) for j, g in enumerate(gold)), reverse=True)
    used_p, used_g, out = set(), set(), []
    for v, i, j in pairs:
        if v < thr:
            break
        if i in used_p or j in used_g:
            continue
        used_p.add(i); used_g.add(j); out.append((i, j, v))
    return out


def boundary_hits(pred_b, gold_b, tol):
    hit = 0
    used = set()
    for g in gold_b:
        for k, p in enumerate(pred_b):
            if k not in used and abs(p - g) <= tol:
                used.add(k); hit += 1
                break
    return hit


def reviewed_indices(rows):
    """seg_index of every machine segment that has a review verdict (None if the episode was
    labelled from scratch): an unreviewed machine segment is unknown, not a false positive."""
    idx = {int(r["seg_index"]) for r in rows if r.get("kind") == "review" and r.get("seg_index") is not None}
    return idx or None


def machine_segments(task_dir, env, run_index, keep=None):
    pf = os.path.join(task_dir, f"phases_{run_index}_env{env}.json")
    if os.path.exists(pf):
        doc = json.load(open(pf))
    else:
        from robolab.eval.phases import annotate
        doc = annotate(task_dir, env, run_index).doc
    dt = doc["dt"]
    segs = [((a["start"] - 1) * dt, a["end"] * dt, a["label"], a.get("object") or "", a.get("result") or "")
            for i, a in enumerate(doc["attempts"]) if a["label"] in LABELS and (keep is None or i in keep)]
    bounds = set()
    for a, b, *_ in segs:
        bounds.add(round(a, 3)); bounds.add(round(b, 3))
    return segs, bounds


def score(pred_by_ep, gold_by_ep, name):
    c = collections.Counter()
    by_bucket = collections.Counter()
    tol_hit = collections.Counter(); tol_n = 0
    label_ok = result_ok = matched = 0
    for ep, (gsegs, gbounds) in gold_by_ep.items():
        psegs, pbounds = pred_by_ep.get(ep, ([], set()))
        m = match(psegs, gsegs)
        c["pred"] += len(psegs); c["gold"] += len(gsegs); c["matched"] += len(m)
        for i, j, v in m:
            matched += 1
            label_ok += psegs[i][2] == gsegs[j][2]
            result_ok += (psegs[i][4] or "unknown") == (gsegs[j][4] or "unknown")
        matched_g = {j for _, j, _ in m}
        for j, g in enumerate(gsegs):
            dur = g[1] - g[0]
            for lo, hi in BUCKETS:
                if lo <= dur < hi:
                    by_bucket[(lo, hi, "n")] += 1
                    by_bucket[(lo, hi, "hit")] += j in matched_g
        gi = internal(gbounds, gsegs); pi = internal(pbounds, psegs)
        tol_n += len(gi)
        for tol in TOLS:
            tol_hit[tol] += boundary_hits(pi, gi, tol)
    P = c["matched"] / c["pred"] if c["pred"] else 0.0
    R = c["matched"] / c["gold"] if c["gold"] else 0.0
    F1 = 2 * P * R / (P + R) if P + R else 0.0
    print(f"\n## {name}")
    print(f"episodes {len(gold_by_ep)}, gold segments {c['gold']}, predicted {c['pred']}, matched (IoU>=0.5) {c['matched']}")
    print(f"segment precision {P:.3f}  recall {R:.3f}  F1 {F1:.3f}")
    if matched:
        print(f"label accuracy on matched {label_ok / matched:.3f}   result accuracy {result_ok / matched:.3f}")
    if tol_n:
        print("internal boundary recall: " + "  ".join(f"±{t}s {tol_hit[t] / tol_n:.3f}" for t in TOLS) + f"   (n={tol_n})")
    print("recall by gold duration: " + "  ".join(
        f"{lo}-{'' if hi > 100 else hi}s {by_bucket[(lo, hi, 'hit')] / by_bucket[(lo, hi, 'n')]:.2f} (n={by_bucket[(lo, hi, 'n')]})"
        for lo, hi in BUCKETS if by_bucket[(lo, hi, "n")]))
    return {"P": P, "R": R, "F1": F1, "n_gold": c["gold"], "n_pred": c["pred"]}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gold", default="analysis/gold_set.jsonl")
    ap.add_argument("--labels", default="analysis/phase_labels.jsonl", help="annotator 1")
    ap.add_argument("--labels2", default=None, help="annotator 2 (enables H6)")
    ap.add_argument("--sources", nargs="*", default=["output"], help="directories that hold the runs")
    ap.add_argument("--no-machine", action="store_true", help="skip H8")
    args = ap.parse_args(argv)

    gold_eps = [json.loads(l) for l in open(args.gold) if l.strip()] if os.path.exists(args.gold) else []
    wanted = {(g["run"], g["task"], int(g["env"]), int(g.get("run_index", 0))) for g in gold_eps}
    a1 = collections.defaultdict(list); a2 = collections.defaultdict(list)
    for r in live_labels(load_jsonl(args.labels)):
        a1[key(r)].append(r)
    for r in live_labels(load_jsonl(args.labels2)) if args.labels2 else []:
        a2[key(r)].append(r)
    labelled = set(a1) | set(a2)
    if wanted:
        print(f"gold set: {len(wanted)} episodes; labelled by annotator 1: {len(set(a1) & wanted)}, "
              f"by annotator 2: {len(set(a2) & wanted)}; labelled outside the set: {len(labelled - wanted)}")
    g1 = {ep: segments_of(rows) for ep, rows in a1.items()}
    if REVIEW_STATS["n"]:
        n = REVIEW_STATS["n"]
        print(f"review verdicts (annotator 1): {n} machine segments reviewed; label right {REVIEW_STATS['label_ok'] / n:.3f}, "
              f"result right {REVIEW_STATS['result_ok'] / n:.3f}, bounds within tolerance {REVIEW_STATS['bounds_ok'] / n:.3f}, "
              f"segments removed {REVIEW_STATS['removed']}")
    REVIEW_STATS.clear()
    g2 = {ep: segments_of(rows) for ep, rows in a2.items()}

    if g2:
        both = {ep: g1[ep] for ep in g1 if ep in g2}
        score({ep: g2[ep] for ep in both}, both, "H6: annotator 2 against annotator 1")

    if not args.no_machine and g1:
        def find_task_dir(run, task):
            for src in args.sources:
                for cand in glob.glob(os.path.join(src, run, task)) + glob.glob(os.path.join(src, "*", run, task)):
                    if os.path.isdir(cand):
                        return cand
            return None
        pred = {}
        missing = []
        for ep in g1:
            run, task, env, ri = ep
            td = find_task_dir(run, task)
            if td is None:
                missing.append(ep); continue
            try:
                pred[ep] = machine_segments(td, env, ri, keep=reviewed_indices(a1.get(ep, [])))
            except Exception as exc:
                print(f"[score_gold] {run}/{task} env{env}: {type(exc).__name__}: {exc}", file=sys.stderr)
                missing.append(ep)
        if missing:
            print(f"{len(missing)} labelled episode(s) not found under --sources; skipped")
        gold = g1
        if g2:
            # consensus: annotator 1's segments where annotator 2 agrees (IoU >= 0.5 and same label)
            gold = {}
            for ep, (s1, b1) in g1.items():
                if ep not in g2:
                    gold[ep] = (s1, b1); continue
                s2, _ = g2[ep]
                keep = [s1[i] for i, j, _ in match(s1, s2) if s1[i][2] == s2[j][2]]
                keep.sort()
                bounds = set()
                for a, b, *_ in keep:
                    bounds.add(round(a, 3)); bounds.add(round(b, 3))
                gold[ep] = (keep, bounds)
        score({ep: pred[ep] for ep in gold if ep in pred}, {ep: v for ep, v in gold.items() if ep in pred},
              "H8: machine against " + ("the consensus" if g2 else "annotator 1"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
