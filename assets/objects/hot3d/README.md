# HOT3D keyboard collision asset

`keyboard_colliders.usda` is derived from `keyboard.usd` (HOT3D Model Data,
© Meta Platforms Technologies, LLC). It is covered by **CC BY-SA 4.0 and the
additional HOT3D Dataset License Agreement non-sale restriction**, as described
in [LICENSE](LICENSE). The original asset and visuals are unchanged.

The source has 110 connected components: base pieces and individual keycaps.
Native CoACD generates 5,503 pieces. The compiled library uses one convex hull
per component, without bridging gaps between keys. Only open keycap bottoms
are closed. Original body mass, COM, inertia, materials, transforms and explicit
collision filters are handled by Newton's native importer.

Build with the pinned Newton 1.5.2 environment, from the repository root:

```bash
.venv/bin/python scripts/build_newton_keyboard.py
.venv/bin/python -m pytest --confcutdir=tests/newton tests/newton/test_keyboard_colliders.py -q
```

The builder checks the full source file SHA-256, component count and contact
boundary geometry. Every original triangle centroid is on its component hull
within 0.2 micrometres (measured maximum 0.0981 micrometres). Every hull plane
matches an authored triangle within 0.5 micrometres, except the open keycap
bottom plane. This tolerance accounts for float32 triangulation slivers.
Component bounds are preserved, and no hull exceeds 44 vertices.

The runtime checks the source mesh geometry, compiled asset hash, expected
component and collider counts, and scale at most 1. Rebuild and revalidate if
the source changes. This is a collision geometry library in source mesh
coordinates; the original USD supplies all instance transforms and rendering.
The library itself is about 226 kB. The geometry checks do not establish
hardware-calibrated friction or inertia.
