# Compiled shelf collision asset — October 13, 2026

The rack_l04 shelf no longer runs detailed CoACD decomposition in the default
Newton preset. A compiled USD contains 44 convex collision pieces, replacing
35,903 rack geoms. The whole TakeMugsOffOfShelf model falls from 36,723 to 864
geoms. The expensive startup described in the [previous mesh audit](newton-mesh-audit-2026-10-12.md)
is resolved for the tested shelf tasks.

## Geometry and intended approximation

The 50,990-byte USD contains 838 triangles and at most 49 vertices per piece:
four posts, four thin feet, eight side braces/bars, four decks and 24 rails or
corner brackets. It removes all fastener collisions and closes the approximately
1 mm seams between shelf slats at the benchmark scale of 0.2. Uprights use convex
outer envelopes, braces use tight oriented boxes, and rails/brackets use solid
boxes. This supports manipulation on an assembled shelf; it does not preserve
fastener/disassembly interactions or contact with the removed small recesses.

The original visual meshes remain in the scene. The asset is a collision library,
not a replacement render mesh or independently parameterized rigid body. The
adapter substitutes compiled pieces through Newton's native importer, preserving
transforms, materials, collision filters and explicit body mass properties. It
checks complete source geometry fingerprints, the compiled USD hash, component
counts and scale, and rejects mismatches instead of invoking expensive fallback
decomposition. Unrelated meshes retain their existing import path.

The rebuild is deterministic on the tested local stack (byte-identical USD and
JSON). It needs Newton, USD, NumPy and SciPy, but performs no CoACD computation.
See the [asset README](../../assets/objects/vomp/rack_l04/README.md) for commands,
configuration, provenance and inherited CC BY 4.0 licensing. The compiled USD is
stored as ordinary reviewable Git text, not an LFS pointer.

## Verification

Geometry checks preserve all four deck bounds and support heights exactly at
float32 source precision: 124.6468, 244.6857, 364.6857 and 484.6858 mm above the
rack origin. Inter-deck clearances are 115.59–115.63 mm. All 171 sampled opening
probes admit a 20 mm diameter sphere. The diagonal brace boxes occupy about 5%
of their axis-aligned bounding-box volume, so they do not fill the side openings.
The foot tabs omitted from the foot plates are contained in the post envelopes.

The actual integrated baseline and compiled MuJoCo models have bitwise-identical
mass, inertia, inertial frames and authored body transforms for all 25 bodies.
The dynamic shelf remains 20 kg. Its imported friction and contact-response rows
also match. Native importer tests separately cover explicit cross-body filter
propagation, original visual mesh identity, coefficients, scales, fingerprint
failure and hook restoration.

All **106 Newton regression tests passed** on the pinned node. Local checks
passed 99 cases with seven engine-dependent skips; the seven shelf-specific
cases were additionally run with the exact Newton source and passed. The prior
313-case Verified offline and 238-case simulator suites were not rerun for this
geometry-only follow-up.

Three ordinary managed-environment runs completed 20 simulated seconds each,
with RTX cameras, finite object/robot state and contact sensor values, no active
immovable/immovable response contacts, no capacity overflow and no passive
confirmed success:

| Task | Startup to first state | Peak host RSS | Total wall time | Maximum object XY span, final 3 s |
|---|---:|---:|---:|---:|
| TakeMugsOffOfShelf | 19.04 s | 7.84 GiB | 142.34 s | 0.01068 mm |
| PutMugsOnShelf | 18.03 s | 7.99 GiB | 140.35 s | 0.00491 mm |
| PutBowlOnShelfTop | 18.03 s | 7.98 GiB | 139.32 s | 0.00768 mm |

The comparable detailed TakeMugsOffOfShelf baseline needed approximately ten
minutes to initialize and peaked at 91.61 GiB RSS. RSS was sampled once per
second. In the simplified TakeMugsOffOfShelf run, rack displacement never
exceeded 1.16 micrometres. Final camera frames were reviewed; mugs remained
supported and all camera streams contained nonconstant images.

The isolated CUDA MuJoCo-Warp probe dropped a 100 g sphere and a 100 g box onto
each of the four decks from 30 mm above contact. Boxes also began with a 0.15 m/s
tangential speed. All eight probes settled within 2.11 micrometres of their
expected support height, with final linear speeds below 0.01 mm/s and no
observable late position drift at float32 sampling precision. Boxes travelled
about 11.9 mm before stopping. This probe uses a fixed rack and friction 0.5 to
isolate geometry; it does not replace the dynamic-rack scene checks or calibrate
benchmark material parameters. The corresponding CPU diagnostic also passed.

## Scope and evidence

Testing used the retained RTX PRO 5000 instance 55558470 and Isaac Sim 6.1.0.0,
Isaac Lab 4ed09d414e93f35b8573db8850f8b27572779176, Newton 1.5.2, MuJoCo/MJWarp
3.11.0 and Warp 1.16.0. The implementation applies on top of local revision
36fc9ec. The remote checkout's older Git HEAD does not identify its deployed
working tree; the candidate archive and source hashes identify the tested files.
The native CoACD cache was warm for other objects. The shelf uses only its compiled
USD. Startup and memory measurements include application startup and RTX cameras;
they are not a clean-machine install benchmark. A 32 GiB per-process memory guard
and ten-minute per-task deadline remained enabled throughout the scene queue.

There are three tested task configurations over two distinct scene layouts.
The unused shelf2_with_condiments and shelf2_without_condiments scene files also
reference rack_l04 but were not simulated in this follow-up. These checks do not
establish full benchmark coverage, fresh-policy success rates, calibrated inertia
or parity with PhysX. The original high-resolution visual asset remains unchanged.

Downloaded evidence: `macro-shelf-evidence.tar.gz`, 23,523,909 bytes, SHA256
`b5797203eb509a7eb6c8519f2b8e71ddabd7301e9f4d60e0524c88e247513bff`.
All 71 individual file hashes in `macro-evidence-files.json` were verified after
download. The bundle includes three-camera videos and start/end frames for each
task, object state histories, result JSON, RSS monitors, logs, the geometry/model
comparisons, the GPU probe, and the two exact deployed code archives. Full binary
simulator models are retained on the node and excluded from this download.

Compiled collision USD SHA256:
`97a3690f84fbb3588ec131cd71c174ee61a25507f82dc897c9727676dc048db3`.
Original source USD SHA256:
`94f332ada51a44466386d2d23e40b77d5ead54aa2c918aad32f9d76da71dff0b`.
The GPU node remains running for follow-up work.
