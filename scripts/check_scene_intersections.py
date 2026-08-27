#!/usr/bin/env python3
"""Report objects whose authored bounding boxes overlap, per scene.

Two scene "fixes" of ours shipped without this check and both made their scene
worse, in the same way:

  P56b  lowered a lemon 1 cm -> it entered the basket mesh -> PhysX ejected 27
        unrelated objects across four episodes. Reverted.
  P63   placed two lemons "on the table" -> they were inside the clay_plates
        volume -> lemon_01 was thrown 740 mm and took four neighbours with it.
        Reverted.

A scene edit is not verifiable by re-measuring the object you edited. Overlap at
spawn is resolved by PhysX as an impulse, and the damage lands on objects you did
not touch. Run this before committing any scene change.

LIMITATION: this compares axis-aligned bounding boxes, not meshes. Fruit resting
inside a bowl legitimately overlaps that bowl's bbox, so the absolute count is an
upper bound and most pairs in a cluttered scene are benign. Use it differentially
-- run it before and after a scene edit, and treat any NEW pair as a red flag.

    python scripts/check_scene_intersections.py [--scenes-dir assets/scenes] [--scene NAME]
"""
from __future__ import annotations

import argparse
import glob
import os
import sys

MIN_OVERLAP_M = 0.0005   # ignore sliver contact; report real interpenetration


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenes-dir", default="assets/scenes")
    ap.add_argument("--scene", default=None, help="check a single scene file")
    ap.add_argument("--min-overlap", type=float, default=MIN_OVERLAP_M)
    args = ap.parse_args()

    try:
        from pxr import Usd, UsdGeom
    except ImportError:
        print("needs usd-core (pip install usd-core)", file=sys.stderr)
        return 2

    files = ([os.path.join(args.scenes_dir, args.scene)] if args.scene
             else sorted(glob.glob(os.path.join(args.scenes_dir, "*.usda"))))

    total = 0
    for path in files:
        try:
            stage = Usd.Stage.Open(path)
        except Exception:
            continue
        cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), ["default", "render"], useExtentsHint=False)
        prims = [p for p in stage.Traverse()
                 if p.GetParent() and p.GetParent().GetName() == "world"
                 and p.GetAttribute("xformOp:translate").Get() is not None]

        boxes = {}
        for p in prims:
            r = cache.ComputeWorldBound(p).ComputeAlignedRange()
            if not r.IsEmpty():
                boxes[p.GetName()] = (r.GetMin(), r.GetMax())

        hits = []
        names = sorted(boxes)
        for i, a in enumerate(names):
            if a in ("table", "ground", "floor"):
                continue
            alo, ahi = boxes[a]
            for b in names[i + 1:]:
                if b in ("table", "ground", "floor"):
                    continue
                blo, bhi = boxes[b]
                ov = [min(ahi[k], bhi[k]) - max(alo[k], blo[k]) for k in range(3)]
                if all(v > args.min_overlap for v in ov):
                    hits.append((min(ov), a, b))

        if hits:
            total += len(hits)
            print(f"\n{os.path.basename(path)} — {len(hits)} overlapping pair(s)")
            for depth, a, b in sorted(hits, reverse=True)[:12]:
                print(f"   {a:22s} <-> {b:22s} overlap {depth*1000:5.1f} mm")

    print(f"\ntotal overlapping pairs: {total}")
    return 1 if total else 0


if __name__ == "__main__":
    sys.exit(main())
