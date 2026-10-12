# SPDX-License-Identifier: Apache-2.0
"""Audit the compiled shelf's support planes, macro envelopes and openings.

This checks the deliberate macro approximation, not a sub-millimetre bound on
all original surface detail. Run GPU contact/scene tests after regeneration.
"""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np
from scipy.spatial import ConvexHull


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def audit(repo):
    build = load('build_shelf', repo / 'scripts/build_newton_shelf.py')
    shelf = load('shelf', repo / 'robolab/core/environments/newton_shelf.py')
    asset = repo / 'assets/objects/vomp/rack_l04/rack_l04_colliders.usda'
    manifest, parts = shelf.load_colliders(asset)
    source = build.source_components(asset.with_name('rack_l04.usd'))
    def group(suffix):
        name = 'M_Rack_L01_' + suffix
        return source[name][2], [parts[p][0].astype(float) for p in manifest['meshes'][name]['paths']]
    # Exact macro bounds are retained within float32 storage precision.
    tolerance = 1e-6
    decks, simplified_decks = group('Shelf')
    errors = []
    bounds = []
    for i, proxy in enumerate(simplified_decks):
        original = np.vstack([v for v, _ in decks[14*i:14*(i+1)]])
        error = max(abs(proxy.min(0) - original.min(0)).max(), abs(proxy.max(0) - original.max(0)).max())
        assert error <= tolerance, ('deck envelope', i, error)
        errors.append(error * .2)
        bounds.append([proxy.min(0)*.2, proxy.max(0)*.2])
    posts, simplified_posts = group('Frame')
    for (original, _), proxy in zip(posts, simplified_posts):
        a, b = ConvexHull(original), ConvexHull(proxy)
        np.testing.assert_allclose(a.volume, b.volume, rtol=1e-5)
        assert np.max(original @ b.equations[:, :3].T + b.equations[:, 3]) < tolerance
    joints, simplified_joints = group('FrameJoints')
    for i, ((original, _), proxy) in enumerate(zip(joints[:4], simplified_joints[:4])):
        assert abs(proxy[:, 2].max() - .002) <= tolerance
        assert abs(proxy[:, 2].min()) <= tolerance
        eq = ConvexHull(simplified_posts[i]).equations
        tabs = original[original[:, 2] > .002001]
        assert np.max(tabs @ eq[:, :3].T + eq[:, 3]) <= tolerance, ('uncovered foot tab', i)
    diagonal_ratios = []
    for i, ((original, _), proxy) in enumerate(zip(joints[4:12], simplified_joints[4:12])):
        h = ConvexHull(proxy)
        assert np.max(original @ h.equations[:, :3].T + h.equations[:, 3]) <= tolerance
        if i in (0, 4):
            ratio = h.volume / np.prod(np.ptp(original, axis=0))
            assert ratio < .15, ('diagonal brace filled opening', i, ratio)
            diagonal_ratios.append(float(ratio))
    rails, simplified_rails = group('ShelfElements')
    for (original, _), proxy in zip(rails, simplified_rails):
        np.testing.assert_allclose(proxy.min(0), original.min(0), atol=tolerance, rtol=0)
        np.testing.assert_allclose(proxy.max(0), original.max(0), atol=tolerance, rtol=0)
    # Prove a 20 mm diameter object can pass through sampled points across each
    # opening. A separating hull plane >radius away guarantees no intersection.
    equations = [ConvexHull(v.astype(float)*.2).equations for v, _ in parts.values()]
    clearance = []
    for i in range(3):
        gap = bounds[i+1][0][2] - bounds[i][1][2]
        clearance.append(float(gap))
        z = (bounds[i+1][0][2] + bounds[i][1][2])/2
        for x in np.linspace(-.09, .09, 19):
            for y in (-.12, 0., .12):
                p = np.array([x, y, z])
                assert all(np.max(eq[:, :3] @ p + eq[:, 3]) > .01 for eq in equations), ('blocked opening', p)
    return dict(collision_asset_sha256=hashlib.sha256(asset.read_bytes()).hexdigest(),
                part_count=len(parts), max_vertices=max(len(v) for v, _ in parts.values()),
                triangles=sum(len(f) for _, f in parts.values()),
                max_deck_envelope_error_m=float(max(errors)),
                support_heights_m=[float(hi[2]) for lo, hi in bounds],
                inter_deck_clearances_m=clearance, diagonal_to_aabb_volume_ratios=diagonal_ratios,
                opening_sphere_diameter_m=.02, opening_probes=171,
                approximations=manifest['approximations'])


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--repo', type=Path, default=Path(__file__).resolve().parents[1])
    p.add_argument('--output', type=Path)
    a = p.parse_args()
    report = audit(a.repo)
    result = json.dumps(report, indent=2) + '\n'
    if a.output:
        a.output.parent.mkdir(parents=True, exist_ok=True)
        a.output.write_text(result)
    print(result)


if __name__ == '__main__':
    main()
