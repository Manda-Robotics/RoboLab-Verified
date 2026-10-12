# Compiled rack collision geometry

`rack_l04_colliders.usda` is a reusable collision geometry library derived from
`rack_l04.usd`. The four original source meshes become 44 convex pieces:

| Structure | Pieces |
|---|---:|
| Upright post envelopes | 4 |
| Thin foot plates | 4 |
| Side braces and horizontal bars | 8 |
| Continuous shelf decks | 4 |
| Rails and corner brackets | 24 |

The 838 collision triangles replace 35,903 native CoACD hulls. Each piece has
at most 49 vertices. The authored units and coordinates are retained; benchmark
instances use scale 0.2. Bolt, nut, washer and thread collisions are removed.
Decks close the approximately 1 mm seams between slats at benchmark scale.
Posts use convex outer envelopes; braces use tight oriented boxes; rails and
brackets use solid boxes. This approximation is intended for manipulation of
objects on an assembled shelf, not disassembly or fastener contact.

The Newton adapter loads this USD and validates its companion manifest and the
four original source mesh fingerprints. Newton's native importer retains the
original visual meshes, transforms, materials, body mass/COM/inertia and
collision filters. The library itself does not author a rigid body, mass or
material: those belong to the source scene. No expensive rack decomposition is
performed at runtime. PhysX continues to use the original asset.

On the pinned Newton stack, from the repository root:

```bash
python scripts/build_newton_shelf.py
python scripts/audit_newton_shelf.py --output /tmp/shelf-geometry.json
python scripts/probe_newton_shelf.py --output /tmp/shelf-gpu-contacts.json
python -m pytest --confcutdir=tests/newton tests/newton/test_shelf_colliders.py -q
```

The probe defaults to CUDA MuJoCo-Warp. `--cpu` is a separate local diagnostic.
Regeneration needs NumPy, SciPy, USD and Newton; it does not need CoACD.
Source geometry changes require reviewing component selection and rerunning
geometry, contact and integrated scene checks. The adapter rejects changed
source geometry, a changed compiled file or scales larger than 0.2.
Set `cfg.newton_contact_settings = {"compact_shelf_colliders": False}` to restore
the expensive original Newton rack decomposition for a diagnostic comparison.

Asset license: **CC BY 4.0**, inherited from the VOMP asset in this repository,
© 2026 NVIDIA Corporation. This derivative simplifies collision geometry as
described above. See [repository asset notices](../../../../THIRD_PARTY_NOTICES.md)
and the [CC BY 4.0 license](https://creativecommons.org/licenses/by/4.0/).
The build and adapter Python code uses Apache-2.0.
