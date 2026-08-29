#!/usr/bin/env python3
"""Runtime verification for the fork's flag patches.

Every patch in ``docs/docs/verified/changes.md.md`` claims a change in what a *run* emits.
This turns each of those claims into a predicate over recorded episode logs so the
claim is checked, not asserted. Point it at a run directory:

    scripts/verify_patches.py output/rc4_BlackItemsInBinTask
    scripts/verify_patches.py --baseline output/rc3_X output/rc4_X   # before/after

A predicate returns PASS (the patched behaviour is present and correct), FAIL (the
old behaviour is still there) or N/A (this run cannot exercise the patch — e.g. no
episode lost a destination, so P76 has nothing to say). N/A is never a pass: the
ledger's "verified on hardware" column may only be set from a PASS.

Exit code is 1 if anything FAILs, so this can gate CI on a fetched run.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from collections import Counter
from dataclasses import dataclass, field

# Mirrors of the runtime constants. Duplicated on purpose: this script must be able
# to judge a recording made by *another* build, so it may not import the constants
# the recording was made with.
SETTLE_WARMUP_S = 1.0
GRASP_ATTEMPT_BURST_S = 2.0
DEST_HINTS = ("bin", "crate", "rack", "shelf", "pail", "box", "plate", "bowl", "table", "container")

# StatusCode values used below, by name so the predicates read like the ledger.
OBJECT_IN_CONTAINER_SUCCESS = 125
OBJECT_GRABBED_SUCCESS = 139
SUBTASK_COMPLETED = 190
GRASP_ATTEMPT_FAILED = 266
OBJECT_RELEASED = 267
PLACED_WITHOUT_LIFT = 274
TARGET_LOST = 273
OBJECT_CARRIED = 283
SUCCESS_CLASS = 200  # codes < 200 are the success class


@dataclass
class Episode:
    path: str
    task: str
    env_id: int
    run: int
    dt: float
    success: bool
    final_step: int
    events: list

    @property
    def label(self) -> str:
        return f"{self.task} env{self.env_id}"

    def t(self, step: int) -> float:
        return step * self.dt

    def of(self, *codes: int) -> list:
        return [e for e in self.events if e["code"] in codes]


@dataclass
class Result:
    patch: str
    title: str
    verdict: str                      # PASS | FAIL | N/A
    detail: str
    evidence: list = field(default_factory=list)
    opportunities: int = 0            # chances this run gave the patch to be caught out


def load(run_dir: str) -> list[Episode]:
    eps = []
    for f in sorted(glob.glob(os.path.join(run_dir, "*", "log_*.json"))):
        d = json.load(open(f))
        eps.append(Episode(f, d["task"], d["env_id"], d["run"], d["dt"],
                           bool(d["success"]), int(d["final_step"]), d["events"]))
    return eps


def object_of(event: dict) -> str | None:
    """The object an event is about, from the quoted name in its info line."""
    info = event.get("info", "")
    if "'" in info:
        return info.split("'")[1]
    for key in ("object=",):
        if key in info:
            return info.split(key, 1)[1].split(",")[0].split(")")[0].strip()
    return None


_CATALOG_CLASS: dict[str, str] | None = None


def _catalog_class(name: str) -> str | None:
    """Catalog class of a scene object name (``bowl_1`` -> ``bowl``), or None if unknown.
    Read once from assets/objects/object_catalog.json next to this script's repo."""
    global _CATALOG_CLASS
    if _CATALOG_CLASS is None:
        path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                            "assets", "objects", "object_catalog.json")
        try:
            _CATALOG_CLASS = {row["name"]: (row.get("class") or "") for row in json.load(open(path))}
        except (OSError, ValueError):
            _CATALOG_CLASS = {}
    for key in (name, name.rsplit("_", 1)[0] if "_" in name else name):
        if key in _CATALOG_CLASS:
            return _CATALOG_CLASS[key]
    return None


def looks_like_destination(name: str | None) -> bool:
    """Is this scene object a container / fixture rather than something to pick?

    The catalog decides when it knows the object: ``sugar_box`` and ``raisin_box`` are
    class ``food`` (products, not boxes to put things in) -- the substring "box" alone
    called them containers and marked rc6b's P72 a regression. Scene-authored names the
    catalog does not know (``grey_bin``, ``purple_crate``, ``franka_table``) fall back to
    the name hints.
    """
    if not name:
        return False
    cls = _catalog_class(name)
    if cls in ("container", "fixture"):
        return True
    if cls not in (None, "", "dishware", "kitchenware", "vase"):
        return False          # a product (food, condiment, tool, toy, ...) is never a destination
    return any(h in name.lower() for h in DEST_HINTS)   # plates and bowls: by name, as before


# --------------------------------------------------------------------------- #
# Predicates. Each takes the episode list and returns a Result.
# --------------------------------------------------------------------------- #

def p61_onset_stamping(eps):
    """Tracker events are stamped at grasp ONSET, with detection recorded separately."""
    stamped = [(e, ev) for e in eps for ev in e.events
               if "detected_step" in ev and ev["detected_step"] != ev["step"]]
    inverted = [(e, ev) for e, ev in stamped if ev["detected_step"] < ev["step"]]
    if inverted:
        return Result("P61", "onset stamping", "FAIL",
                      f"{len(inverted)} events stamped AFTER detection (step > detected_step)",
                      [f"{e.label} step={ev['step']} detected={ev['detected_step']}" for e, ev in inverted[:5]])
    if not stamped:
        return Result("P61", "onset stamping", "N/A", "no event in this run has an onset earlier than its detection")
    lead = [(ev["detected_step"] - ev["step"]) * e.dt for e, ev in stamped]
    return Result("P61", "onset stamping", "PASS",
                  f"{len(stamped)} events carry an onset {min(lead):.2f}-{max(lead):.2f}s before detection "
                  f"(mean {sum(lead)/len(lead):.2f}s)",
                  [f"{e.label} {ev['name']} onset={e.t(ev['step']):.2f}s detected={e.t(ev['detected_step']):.2f}s"
                   for e, ev in stamped[:3]], len(stamped))


def p71_no_green_on_tracker_grabs(eps):
    """The tracker's grab line is neutral OBJECT_CARRIED; only the ladder emits green."""
    # A ladder line always names its predicate ("success: object_grabbed(...)").
    tracker_green = [(e, ev) for e in eps for ev in e.of(OBJECT_GRABBED_SUCCESS)
                     if not ev.get("info", "").startswith("success:")]
    carried = [(e, ev) for e in eps for ev in e.of(OBJECT_CARRIED)]
    if tracker_green:
        return Result("P71", "tracker grab is neutral", "FAIL",
                      f"{len(tracker_green)} tracker grabs still emit the green OBJECT_GRABBED_SUCCESS",
                      [f"{e.label} @{e.t(ev['step']):.2f}s {ev['info'][:70]}" for e, ev in tracker_green[:5]])
    if not carried:
        return Result("P71", "tracker grab is neutral", "N/A", "no grasp was established in this run")
    return Result("P71", "tracker grab is neutral", "PASS",
                  f"{len(carried)} tracker grabs emitted as neutral OBJECT_CARRIED; 0 green tracker lines",
                  [f"{e.label} @{e.t(ev['step']):.2f}s {ev['info'][:70]}" for e, ev in carried[:3]], len(carried))


def p72_no_attempts_on_containers(eps):
    """Grasp-attempt tracking skips containers and fixtures."""
    attempts = [(e, ev) for e in eps for ev in e.of(GRASP_ATTEMPT_FAILED)]
    bad = [(e, ev) for e, ev in attempts if looks_like_destination(object_of(ev))]
    if bad:
        return Result("P72", "no attempts on containers", "FAIL",
                      f"{len(bad)} of {len(attempts)} attempt lines are on a container or fixture",
                      [f"{e.label} @{e.t(ev['step']):.2f}s {ev['info'][:70]}" for e, ev in bad[:5]])
    if not attempts:
        return Result("P72", "no attempts on containers", "N/A", "no grasp attempts recorded in this run")
    return Result("P72", "no attempts on containers", "PASS",
                  f"0 of {len(attempts)} attempt lines name a container or fixture", [], len(attempts))


def p73_no_attempt_after_release(eps):
    """Contact flickers as an object leaves the hand; that is not a new attempt."""
    offenders = []
    total = 0
    for e in eps:
        released = {}
        for ev in sorted(e.events, key=lambda x: x["step"]):
            obj = object_of(ev)
            if ev["code"] == OBJECT_RELEASED and obj:
                released[obj] = ev["step"]
            elif ev["code"] == GRASP_ATTEMPT_FAILED and obj:
                total += 1
                prev = released.get(obj)
                if prev is not None and (ev["step"] - prev) * e.dt <= GRASP_ATTEMPT_BURST_S:
                    offenders.append((e, ev, prev))
    if offenders:
        return Result("P73", "no attempt right after a release", "FAIL",
                      f"{len(offenders)} of {total} attempts open within {GRASP_ATTEMPT_BURST_S}s of releasing that object",
                      [f"{e.label} attempt@{e.t(ev['step']):.2f}s released@{e.t(prev):.2f}s {ev['info'][:50]}"
                       for e, ev, prev in offenders[:5]])
    if not total:
        return Result("P73", "no attempt right after a release", "N/A", "no grasp attempts recorded in this run")
    return Result("P73", "no attempt right after a release", "PASS",
                  f"0 of {total} attempts fall inside {GRASP_ATTEMPT_BURST_S}s of a release of the same object",
                  [], total)


def p74_nothing_credited_before_settle(eps):
    """No score is awarded until the scene is at rest and the spawn probe has run."""
    offenders = []
    for e in eps:
        cutoff = max(1, round(SETTLE_WARMUP_S / e.dt))
        for ev in e.events:
            if ev["code"] < SUCCESS_CLASS and ev["step"] < cutoff:
                offenders.append((e, ev))
    if offenders:
        return Result("P74", "nothing credited before settle", "FAIL",
                      f"{len(offenders)} success-class events land inside the {SETTLE_WARMUP_S}s settle warm-up",
                      [f"{e.label} @{e.t(ev['step']):.2f}s {ev['name']} {ev['info'][:60]}" for e, ev in offenders[:5]])
    firsts = [(e, min((ev["step"] for ev in e.events if ev["code"] < SUCCESS_CLASS), default=None)) for e in eps]
    scored = [(e, s) for e, s in firsts if s is not None]
    if not scored:
        return Result("P74", "nothing credited before settle", "N/A", "no episode scored anything")
    return Result("P74", "nothing credited before settle", "PASS",
                  f"earliest credit across {len(scored)} scoring episodes is {min(e.t(s) for e, s in scored):.2f}s "
                  f"(warm-up ends at {SETTLE_WARMUP_S:.2f}s)",
                  [f"{e.label} first credit @{e.t(s):.2f}s" for e, s in scored[:4]], len(scored))


def p75_placed_without_lift_retired(eps):
    hits = [(e, ev) for e in eps for ev in e.of(PLACED_WITHOUT_LIFT)]
    if hits:
        return Result("P75", "PLACED_WITHOUT_LIFT retired", "FAIL",
                      f"{len(hits)} PLACED_WITHOUT_LIFT events still emitted",
                      [f"{e.label} @{e.t(ev['step']):.2f}s {ev['info'][:70]}" for e, ev in hits[:5]])
    return Result("P75", "PLACED_WITHOUT_LIFT retired", "PASS",
                  f"0 PLACED_WITHOUT_LIFT events across {len(eps)} episodes", [], len(eps))


def p76_lost_destination_ends_episode(eps):
    """A destination leaving the table makes the task unachievable; stop the clock."""
    lost = [(e, ev) for e in eps for ev in e.of(TARGET_LOST)]
    if not lost:
        return Result("P76", "a lost destination ends the episode", "N/A",
                      "no episode in this run lost a destination")
    late = [(e, ev) for e, ev in lost if e.final_step > ev["step"]]
    if late:
        return Result("P76", "a lost destination ends the episode", "FAIL",
                      f"{len(late)} episodes kept running after TARGET_LOST",
                      [f"{e.label} lost@{e.t(ev['step']):.2f}s but ran to {e.t(e.final_step):.2f}s" for e, ev in late[:5]])
    return Result("P76", "a lost destination ends the episode", "PASS",
                  f"{len(lost)} episodes ended on the step the destination left the table",
                  [f"{e.label} lost@{e.t(ev['step']):.2f}s, episode ends {e.t(e.final_step):.2f}s "
                   f"({ev['info'][:44]})" for e, ev in lost[:4]], len(lost))


PREDICATES = [p61_onset_stamping, p71_no_green_on_tracker_grabs, p72_no_attempts_on_containers,
              p73_no_attempt_after_release, p74_nothing_credited_before_settle,
              p75_placed_without_lift_retired, p76_lost_destination_ends_episode]


def p77_contact_force_recorded(run_dir: str) -> Result:
    """P77 lives in the HDF5, not the event log: force per pad plus destination contact."""
    try:
        import h5py
    except ImportError:
        return Result("P77", "contact force recorded", "N/A", "h5py not installed; cannot read the HDF5")
    files = glob.glob(os.path.join(run_dir, "*", "run_*.hdf5"))
    if not files:
        return Result("P77", "contact force recorded", "N/A", "no HDF5 in this run directory")
    with h5py.File(files[0], "r") as f:
        demos = [k for k in f.get("data", {})]
        if not demos:
            return Result("P77", "contact force recorded", "N/A", "HDF5 has no demos")
        g = f["data"][demos[0]].get("contact")
        if g is None:
            return Result("P77", "contact force recorded", "FAIL", "no contact/ group in the HDF5")
        pads = {k: g[k] for k in g if g[k].ndim == 2 and g[k].shape[1] == 2}
        dests = [k for k in g if "__" in k]
        if not pads:
            return Result("P77", "contact force recorded", "FAIL", "contact/ holds no per-pad (T,2) columns")
        boolean = [k for k, v in pads.items() if v.dtype.kind in "bu"]
        if boolean:
            return Result("P77", "contact force recorded", "FAIL",
                          f"{len(boolean)} pad columns are still boolean, not force: {boolean[:4]}")
        return Result("P77", "contact force recorded", "PASS",
                      f"{len(pads)} objects carry (T,2) float pad FORCE; {len(dests)} object-destination contact columns",
                      [f"{k} {tuple(v.shape)} {v.dtype}" for k, v in list(pads.items())[:3]] +
                      [f"{d} (destination contact)" for d in dests[:2]], len(pads))


def _check_applied(applied: dict, tol: float = 1e-3) -> list[str]:
    """Requested-vs-PhysX comparison for one friction_applied.json (a copy of
    robolab.core.physics.friction.check_applied, kept here so this script can judge a
    recording without importing the build that made it)."""
    req = applied.get("requested") or {}
    problems = []
    for name, mat in (req.get("objects") or {}).items():
        rows = applied.get("objects", {}).get(name)
        if not rows:
            problems.append(f"{name}: no PhysX readback")
            continue
        for s, d, _r in rows:
            if abs(s - mat["static"]) > tol or abs(d - mat["dynamic"]) > tol:
                problems.append(f"{name}: requested {mat['static']}/{mat['dynamic']}, PhysX holds {s}/{d}")
                break
    pad = req.get("gripper")
    for body in req.get("gripper_bodies") or []:
        rows = applied.get("gripper", {}).get(body)
        if not isinstance(rows, list) or not rows:
            problems.append(f"pad {body}: no PhysX readback ({rows})")
            continue
        if pad:
            for s, d, _r in rows:
                if abs(s - pad["static"]) > tol or abs(d - pad["dynamic"]) > tol:
                    problems.append(f"pad {body}: requested {pad['static']}/{pad['dynamic']}, PhysX holds {s}/{d}")
                    break
    return problems


def _coeff_summary(applied: dict) -> str:
    objs = Counter(f"{row[0]:g}/{row[1]:g}" for rows in applied.get("objects", {}).values() for row in rows[:1])
    pads = Counter(f"{row[0]:g}/{row[1]:g}" for rows in applied.get("gripper", {}).values()
                   if isinstance(rows, list) for row in rows[:1])
    fmt = lambda c: ", ".join(f"{k} x{v}" for k, v in sorted(c.items())) or "none"
    return f"objects static/dynamic: {fmt(objs)}; pads: {fmt(pads)}"


def p79_friction_applied(run_dir: str) -> Result:
    """P79: what --friction asked for is what PhysX holds after start-up. Judged from
    friction_applied.json (a PhysX readback), never from env_cfg.json alone."""
    files = sorted(glob.glob(os.path.join(run_dir, "*", "friction_applied.json")))
    if not files:
        return Result("P79", "friction override applied", "N/A",
                      "no friction_applied.json (recording predates P79)")
    applied = json.load(open(files[0]))
    req = applied.get("requested") or {}
    if req.get("mode", "upstream") == "upstream":
        return Result("P79", "friction override applied", "N/A",
                      f"run used the upstream materials; PhysX readback: {_coeff_summary(applied)}")
    problems = _check_applied(applied)
    n_obj = len(req.get("objects") or {})
    n_pad = len(req.get("gripper_bodies") or [])
    if problems:
        return Result("P79", "friction override applied", "FAIL",
                      f"{len(problems)} of {n_obj + n_pad} targets differ from the request ({req.get('spec')})",
                      problems[:5], n_obj + n_pad)
    if not req.get("gripper_bodies"):
        return Result("P79", "friction override applied", "FAIL",
                      "objects overridden but the robot declared no friction_bodies -- pads still authored",
                      [], n_obj)
    return Result("P79", "friction override applied", "PASS",
                  f"--friction {req.get('spec')}: {n_obj} objects + {n_pad} pad bodies hold the requested "
                  f"coefficients ({_coeff_summary(applied)})",
                  [f"{n}: {m['static']}/{m['dynamic']} ({m['source']})" for n, m in list(req["objects"].items())[:3]],
                  n_obj + n_pad)


def report(run_dir: str) -> list[Result]:
    eps = load(run_dir)
    if not eps:
        return [Result("-", "load", "N/A", f"no episode logs under {run_dir} (run still in flight?)")]
    results = [f(eps) for f in PREDICATES]
    results.append(p77_contact_force_recorded(run_dir))
    results.append(p79_friction_applied(run_dir))
    return results


def summarise(run_dirs: list[str]) -> int:
    """One row per patch across every run: how many runs could test it, and how they went.

    A patch is only as verified as its widest PASS. A patch that is N/A everywhere is
    reported as untested, not as passing, because no run exercised it.
    """
    rows: dict[str, dict] = {}
    per_run = []
    for run_dir in run_dirs:
        name = os.path.basename(run_dir)
        results = report(run_dir)
        per_run.append((name, results))
        for r in results:
            row = rows.setdefault(r.patch, {"title": r.title, "PASS": [], "FAIL": [], "N/A": [], "n": 0})
            row[r.verdict].append(name)
            row["n"] += r.opportunities

    rows.pop("-", None)
    width = max((len(v["title"]) for v in rows.values()), default=10)
    print(f"{'patch':6} {'title':{width}}  {'pass':>4} {'fail':>4} {'n/a':>4}  verdict")
    print("-" * (6 + width + 40))
    failed = False
    for patch in sorted(rows):
        v = rows[patch]
        if v["FAIL"]:
            verdict, failed = "REGRESSED in " + ", ".join(v["FAIL"][:3]), True
        elif v["PASS"]:
            verdict = f"verified on {len(v['PASS'])} run(s), {v['n']} observations"
        else:
            verdict = "UNTESTED — no run exercised it"
        print(f"{patch:6} {v['title']:{width}}  {len(v['PASS']):>4} {len(v['FAIL']):>4} {len(v['N/A']):>4}  {verdict}")

    print("\nper run:")
    for name, results in per_run:
        marks = " ".join(f"{r.patch}:{ {'PASS':'ok','FAIL':'XX','N/A':'--'}[r.verdict] }" for r in results if r.patch != "-")
        marks = marks or "(no episodes yet)"
        print(f"  {name:34s} {marks}")
    return 1 if failed else 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run_dirs", nargs="+")
    ap.add_argument("--baseline", help="a pre-patch run of the same task, shown alongside for contrast")
    ap.add_argument("--summary", action="store_true",
                    help="one aggregate table across all runs instead of the per-run detail")
    args = ap.parse_args()

    if args.summary:
        return summarise(args.run_dirs)

    failed = False
    for run_dir in args.run_dirs:
        eps = load(run_dir)
        head = f"{os.path.basename(run_dir)}  —  {len(eps)} episodes"
        print(f"\n{head}\n{'=' * len(head)}")
        if eps:
            print(f"  {eps[0].task}   final steps: {[e.final_step for e in eps]}   success: {[e.success for e in eps]}")
        base = {r.patch: r for r in report(args.baseline)} if args.baseline else {}
        for r in report(run_dir):
            mark = {"PASS": "PASS", "FAIL": "FAIL", "N/A": " -- "}[r.verdict]
            print(f"  [{mark}] {r.patch}  {r.title}")
            print(f"          {r.detail}")
            for line in r.evidence:
                print(f"            · {line}")
            if r.patch in base and base[r.patch].verdict != r.verdict:
                print(f"          baseline({os.path.basename(args.baseline)}): {base[r.patch].verdict} — {base[r.patch].detail}")
            failed |= r.verdict == "FAIL"

        counts = Counter(ev["name"] for e in eps for ev in e.events)
        print(f"  events/episode: {sum(counts.values()) / max(1, len(eps)):.1f}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
