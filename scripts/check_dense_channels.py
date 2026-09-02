#!/usr/bin/env python3
"""Check the dense-annotation channels of a recording made with the R1/R2/R4 recorder terms
(docs/verified/dense_annotations.md §5) against the offline derivations they replace.

    scripts/check_dense_channels.py <run dir> [<run dir> ...] [--json out.json]

Per episode, and summed:
  replay vs recorded tracker   H5 as a per-step number: the offline GraspTracker replay against
                               the state the live tracker wrote (tracker/<obj>), plus the replayed
                               lines against the log (compare_events, by detection step).
  finger geometry              pad-midpoint TCP against the old base_link + offset guess, and the
                               midpoint-to-object distance while the tracker says "carried".
  conditions vs log            the step the ladder's last subtask first reads true against the
                               log's SUBTASK_COMPLETED step, and the final value against success.
  extra-body contact           per label: steps with force on any object, and steps where an
                               extra body touches an object while neither pad does (arm pushes).

Reads only the run directory (h5py + numpy; torch for the replay), like annotate_phases.py.
"""
from __future__ import annotations

import argparse
import collections
import glob
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from robolab.eval import phases as P                                   # noqa: E402
from robolab.eval.tracker_replay import compare_events, compare_state, load_recording, quat_rotmat, replay  # noqa: E402


def episodes(run_dir: str):
    for task_dir in sorted(glob.glob(os.path.join(run_dir, "*"))):
        if not os.path.isdir(task_dir) or not os.path.exists(os.path.join(task_dir, "env_cfg.json")):
            continue
        for h5 in sorted(glob.glob(os.path.join(task_dir, "run_*.hdf5"))):
            run_index = int(os.path.basename(h5)[4:-5])
            import h5py
            with h5py.File(h5, "r") as f:
                keys = [k for k in f["data"].keys() if k.startswith("demo_")]
            for k in keys:
                env_id = int(k.split("_")[-1])
                yield task_dir, run_index, env_id, h5, k


def load_log(task_dir, run_index, env_id):
    lp = os.path.join(task_dir, f"log_{run_index}_env{env_id}.json")
    if not os.path.exists(lp):
        return [], None
    lg = json.load(open(lp))
    events = lg.get("events", lg if isinstance(lg, list) else [])
    return events, lg


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("sources", nargs="+")
    ap.add_argument("--json", default=None)
    args = ap.parse_args()
    import h5py

    tot = collections.Counter()
    onset_g, onset_a, tcp_gap, carry_dist, cond_lag = [], [], [], [], []
    body_touch = collections.Counter(); body_only = collections.Counter(); body_steps = 0
    rows = []
    for src in args.sources:
        for task_dir, run_index, env_id, h5, key in episodes(src):
            dt = P.resolve_dt(task_dir)
            targets, dests = P.task_roles(task_dir)
            with h5py.File(h5, "r") as f:
                rec = load_recording(f["data"][key])
            rec.dt = dt
            rec.joint_names = P.robot_joint_names(task_dir)
            events, lg = load_log(task_dir, run_index, env_id)
            row = {"task": os.path.basename(task_dir), "run": run_index, "env": env_id, "T": rec.T,
                   "has_finger": rec.finger_left is not None, "has_tracker": bool(rec.tracker_state),
                   "has_conditions": bool(rec.conditions), "has_body": bool(rec.body_forces)}
            rr = replay(rec, dt, containers=dests)
            cs = compare_state(rr, rec)
            if cs:
                row["state"] = {"steps": cs["steps"], "grasped_agree": cs["grasped_agree"], "attempt_agree": cs["attempt_agree"]}
                tot["state_steps"] += cs["steps"]; tot["grasped_agree"] += cs["grasped_agree"]; tot["attempt_agree"] += cs["attempt_agree"]
                onset_g += cs["grasped_onset_diffs"]; onset_a += cs["attempt_onset_diffs"]
                for o, d in cs["objects"].items():
                    if d["grasped_steps_rec"] or d["grasped_steps_rep"]:
                        tot["carry_pairs"] += 1
                        if d["grasped_agree"] == d["steps"]:
                            tot["carry_pairs_exact"] += 1
            ce = compare_events(rr.events, events, tol_steps=2, by="detected")
            row["events"] = {k: ce[k] for k in ce if isinstance(ce[k], int)}
            for k, v in row["events"].items():
                tot["ev_" + k] += v
            if rec.finger_left is not None:
                R = quat_rotmat(rec.ee_quat)
                old = rec.ee_pos + np.einsum("tij,j->ti", R, np.asarray(P.TCP_OFFSET))
                gap = np.linalg.norm(rec.finger_mid - rec.ee_pos, axis=1)
                tcp_gap.append((float(np.median(gap)), float(gap.max())))
                for o, st in rec.tracker_state.items():
                    g = st[:, 0].astype(bool)
                    if g.any() and o in rec.obj_pos:
                        carry_dist += list(np.linalg.norm(old[g] - rec.obj_pos[o][g], axis=1))
            if rec.conditions:
                # aggregate from the per-condition columns: terminal rung of each group of the
                # last subtask (the recorded s<i> of the first pod tasks used "all rungs at once")
                idx = sorted({int(k[1:].split("_")[0]) for k in rec.conditions})
                last = f"s{idx[-1]}"
                groups = collections.defaultdict(list)
                for k in rec.conditions:
                    if k.startswith(last + "_"):
                        g, j = k[len(last) + 1:].rsplit("_", 1)
                        groups[g].append((int(j), k))
                if groups:
                    s = np.ones(rec.T, bool)
                    for g, lst in groups.items():
                        s &= rec.conditions[max(lst)[1]][:rec.T].astype(bool)
                else:
                    s = rec.conditions[last][:rec.T].astype(bool)
                first = int(np.flatnonzero(s)[0]) + 1 if s.any() else None
                done = [e for e in events if e.get("name") == "SUBTASK_COMPLETED"]
                row["conditions"] = {"last": last, "first_true_step": first, "final": bool(s[-1]),
                                     "log_completed_step": done[-1]["step"] if done else None}
                tot["cond_eps"] += 1
                if first is not None and done:
                    cond_lag.append(first - done[-1]["step"])
                if (first is not None) == bool(done):
                    tot["cond_agree_with_log"] += 1
            if rec.body_forces:
                pads_any = np.zeros(rec.T, bool)
                for o in rec.objects:
                    pads_any |= rec.contact(o)
                labels = sorted({k.split("__")[0] for k in rec.body_forces})
                body_steps += rec.T
                for lab in labels:
                    any_obj = np.zeros(rec.T, bool)
                    for k, v in rec.body_forces.items():
                        if k.startswith(lab + "__") and not k.endswith("__table"):
                            any_obj |= v[:rec.T] > 0.1
                    body_touch[lab] += int(any_obj.sum())
                    body_only[lab] += int((any_obj & ~pads_any).sum())
            rows.append(row)
            print(f"{row['task']} run {run_index} env {env_id}: T={rec.T} state={row.get('state')} events={row['events']} cond={row.get('conditions')}")

    print("\n=== totals ===")
    if tot["state_steps"]:
        print(f"replay vs recorded tracker: grasped agrees on {tot['grasped_agree']}/{tot['state_steps']} steps "
              f"({100 * tot['grasped_agree'] / tot['state_steps']:.2f} %), attempt_closed {100 * tot['attempt_agree'] / tot['state_steps']:.2f} %; "
              f"(object, episode) pairs with a carry: {tot['carry_pairs']}, exact on {tot['carry_pairs_exact']}")
        if onset_g:
            print(f"  carry onset (replay - recorded) steps: median {np.median(onset_g):.1f}, |max| {max(abs(x) for x in onset_g)}; n={len(onset_g)}")
        if onset_a:
            print(f"  attempt onset diffs: median {np.median(onset_a):.1f}, |max| {max(abs(x) for x in onset_a)}; n={len(onset_a)}")
    ev = {k[3:]: v for k, v in tot.items() if k.startswith("ev_")}
    print(f"replayed lines vs log (detected step ±2): {ev}")
    if tcp_gap:
        print(f"finger-link midpoint vs base_link: median {np.median([g for g, _ in tcp_gap]) * 100:.1f} cm, max {max(m for _, m in tcp_gap) * 100:.1f} cm (link origins, not pads)")
    if carry_dist:
        print(f"base_link+TCP_OFFSET to object while carried: median {np.median(carry_dist) * 100:.1f} cm, p90 {np.percentile(carry_dist, 90) * 100:.1f} cm (n={len(carry_dist)} steps)")
    if tot["cond_eps"]:
        print(f"ladder-final condition vs log SUBTASK_COMPLETED: agree on {tot['cond_agree_with_log']}/{tot['cond_eps']} episodes; "
              f"first-true minus log step: {sorted(cond_lag)}")
    if body_steps:
        print(f"extra-body contact over {body_steps} steps:")
        for lab in sorted(body_touch):
            print(f"  {lab:22s} touching an object {body_touch[lab]:5d} steps, of which no pad touches anything: {body_only[lab]}")
    if args.json:
        json.dump({"rows": rows, "totals": dict(tot)}, open(args.json, "w"), indent=1)


if __name__ == "__main__":
    main()
