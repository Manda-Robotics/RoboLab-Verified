#!/usr/bin/env python3
"""Find scene objects that are not at rest when the episode starts (authored above, below, or into something).

Data-driven, no simulator: for every recorded episode under ``--output``, take
each rigid object's height at step 0 and at ``--after`` seconds, and report
objects whose *median* drop across episodes exceeds ``--threshold`` (the median
ignores episodes where the robot moved the object early). An object that falls
at reset settles under the policy's first actions, fires ``OBJECT_BUMPED`` /
``OBJECT_MOVED`` for nothing, and can bounce off the table (findings.md B12,
C3, H-R5-7, H-R8-8, H-R8-14).

    python scripts/check_rest_heights.py --output output --tasks robolab/tasks/benchmark

Exit code 1 if any object exceeds the threshold. Fix = lower the object's
``xformOp:translate`` z in the scene's .usda by the reported drop (P20).
"""
from __future__ import annotations

import argparse
import collections
import glob
import json
import re
import statistics
import sys
from pathlib import Path


def scene_of_task(tasks_dir: Path) -> dict[str, str]:
    """task class name → scene file, from the bundled task metadata (covers every
    task folder) with a regex over the task files as fallback."""
    out = {}
    meta = tasks_dir.parent / "_metadata" / "task_metadata.json"
    if meta.exists():
        try:
            for t in json.loads(meta.read_text()):
                if isinstance(t, dict) and t.get("task_name") and t.get("scene"):
                    out[t["task_name"]] = t["scene"]
        except (OSError, json.JSONDecodeError):
            pass
    for f in tasks_dir.glob("*.py"):
        src = f.read_text()
        m = re.search(r'import_scene\("([^"]+)"', src)
        for c in re.finditer(r"class (\w+)\(Task\)", src):
            out.setdefault(c.group(1), m.group(1) if m else "?")
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", default="output")
    ap.add_argument("--tasks", default="robolab/tasks/benchmark")
    ap.add_argument("--after", type=float, default=1.0, help="seconds after reset to compare against")
    ap.add_argument("--threshold", type=float, default=0.005, help="median drop in metres that counts")
    args = ap.parse_args()
    try:
        import h5py
    except ImportError:
        print("h5py is required", file=sys.stderr)
        return 2
    scenes = scene_of_task(Path(args.tasks))

    def scene_for(task: str) -> str:
        if task in scenes:
            return scenes[task]
        # embodiment-suffixed names (BananaInBowlTaskAloha) share the base task's scene
        best = max((t for t in scenes if task.startswith(t)), key=len, default=None)
        return scenes[best] if best else "?"
    drops: dict[tuple[str, str], list[float]] = collections.defaultdict(list)
    for h5 in glob.glob(f"{args.output}/*/*/run_*.hdf5"):
        task = Path(h5).parent.name
        scene = scene_for(task)
        try:
            with h5py.File(h5) as h:
                for demo in h["data"]:
                    g = h[f"data/{demo}/states/rigid_object"]
                    dt = None
                    log = Path(h5).parent / f"log_{Path(h5).stem.split('_')[-1]}_env{demo.split('_')[-1]}.json"
                    if log.exists():
                        try:
                            dt = json.loads(log.read_text()).get("dt")
                        except (OSError, json.JSONDecodeError):
                            dt = None
                    k = int(round(args.after / (dt or 1 / 15)))
                    for obj in g:
                        if obj == "table":
                            continue
                        pos = g[obj]["root_pose"][:, :3]
                        if len(pos) <= k:
                            continue
                        dz = float(pos[0, 2] - pos[k, 2])
                        dxy = float(((pos[k, :2] - pos[0, :2]) ** 2).sum() ** 0.5)
                        drops[(scene, obj)].append((dz, dxy))
        except (OSError, KeyError):
            continue
    rows = [(s, o, statistics.median([a for a, _ in v]), statistics.median([b for _, b in v]), len(v)) for (s, o), v in drops.items()]
    bad = sorted((r for r in rows if abs(r[2]) > args.threshold or r[3] > 2 * args.threshold), key=lambda r: -(abs(r[2]) + r[3]))
    print(f"{len(rows)} scene objects observed in {args.output}; {len(bad)} move more than {args.threshold*1000:.0f} mm vertically (or {2*args.threshold*1000:.0f} mm sideways) within {args.after:.1f} s of reset")
    for s, o, dz, dxy, n in bad:
        what = "drops" if dz > 0 else "RISES"
        print(f"  {s:40s} {o:22s} {what} {abs(dz)*100:5.1f} cm, rolls {dxy*100:4.1f} cm  (median over {n} episodes)")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
