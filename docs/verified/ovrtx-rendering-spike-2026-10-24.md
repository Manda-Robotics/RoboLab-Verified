# Kit-less OVRTX rendering spike — 2026-10-24

Experimental branch: `codex/ovrtx-rendering-spike`, based on GA integration
`51437f3ce85503fefe5cf04da32bb196f423a479`. No production rendering default or
policy-scoring path has changed.

Follow-up: the [motion video and batched-capture speed comparison](ovrtx-video-2026-10-24.md)
is now complete. Its timing method supersedes the sequential diagnostic loop
below when comparing rendering speed.

The experiment compares Isaac Lab v3.0.0 with Newton 1.6.1 and MJWarp 3.12.0
in separate Python 3.12 environments. One uses Isaac Sim 6.2.0.0 / Isaac RTX;
the other uses OVRTX 0.5.1.385782 and OVStage 0.2.1.385922 without Isaac Sim.
Both retain the production Newton physics configuration.

## Scope and method

- Fresh runtime installations on one RTX PRO 4000 Blackwell (24 GB).
- Installed environment sizes; these are not compressed container image sizes.
- Process-start-to-environment, first Newton action, and camera capture timings.
- Identical frozen BagelsOnPlate states and three calibrated cameras, including
  the wrist sensor. The fixture uses initial state and step 120 of an existing
  GA recording. Images accumulate 32 render updates without advancing physics.
- Live pose and intrinsic readbacks must agree before images are compared.
  Both arms explicitly set `CameraCfg.update_latest_camera_pose=True`.
  Captures are saved without exposure/color adjustments or geometric alignment.
- This is a rendering feasibility experiment, not a policy-score equivalence
  test or a throughput benchmark.

## Functional changes

`install_newton61.py --renderer-runtime ovrtx` installs the upstream OVRTX extra
instead of Isaac Sim. `--renderer-runtime newton-warp` omits both renderer extras;
this installation mode alone does not constitute validation of Warp rendering.
The existing default remains `isaacrtx`.

The capture example explicitly selects all camera renderer configurations and
sets the experimental `experimental_kitless` environment flag. This lets Lab's
launcher own the Kit-less runtime and uses its USD stage API instead of
`omni.usd`. The normal evaluation path still requires Kit.

The world-state module imports visualization helpers eagerly. Those helpers now
use the existing version-compatible USD stage accessor, so merely importing
world state does not require `omni.usd`.

## Results

Both runtime installs completed. The OVRTX environment contains no Isaac Sim/Kit
distributions. The measured sizes are:

| Runtime | Installed | tar + zstd -3 |
| --- | ---: | ---: |
| OVRTX, Kit-less | 10,821,847,543 bytes (10.08 GiB) | 4,828,575,333 bytes (4.50 GiB) |
| Isaac RTX | 26,371,931,804 bytes (24.56 GiB) | 11,117,593,912 bytes (10.35 GiB) |

This is a 59% installed / 57% compressed environment reduction. It excludes
assets, upstream source checkout, OS/driver libraries and caches, and is not a
container image measurement. Both installs include the full upstream workspace
dependency set and pytest, so this is not a minimal deployment package.

CUDA 13.0 is selected through the same pinned
Torch build. The rental container initially lacked NVIDIA graphics userspace
libraries; adding the exact host-driver version inside the container restored
Vulkan. No host kernel driver was installed or changed.

The final `isaacrtx-v4` and `ovrtx-v4` runs both completed successfully. Each
initialized the production Newton configuration, advanced one action with finite
joint states, and captured two restored states through all three RGB cameras.
No Kit app was loaded by the OVRTX process. This is a short smoke check, not a
replacement for the previous GA physics/integration validation.

### Image quality and camera motion

The first OVRTX run was visibly noisy because the rental container could not
load the driver libraries through their expected `libnvoptix.so.1` and
`libnvidia-ngx.so.1` names. Its log explicitly reported that temporal denoising
was unavailable. Adding aliases to the already extracted, matching NVIDIA
595.58.03 libraries removed these initialization errors and the heavy noise.
No geometry, material, lighting or physics changes were made for this repair.

A second issue was exposed by moving the robot to the recorded step-120 pose:
OVRTX kept the initial wrist viewpoint when `update_latest_camera_pose=False`.
Isaac RTX followed the USD hierarchy, but both sensors' pose metadata remained
stale, so equal metadata alone initially hid the mismatch. Isaac Lab's camera
pose refresh also sends transforms to OVRTX. Enabling the public option in both
arms fixed the rendered viewpoint and metadata. No explicit per-frame wrist
pose workaround or installed-library patch was needed. The wrist position now
changes by approximately 0.316 m between captures, identically in both arms.
This verifies tracking across two restored poses, not a continuous policy run.

All six final image pairs have exactly equal recorded physical states, camera
positions, orientations and intrinsics. Visual inspection shows matching
geometry, framing, textures, background and reflections. OVRTX is slightly
darker in several regions; images are close, not pixel-identical.

Mean absolute RGB difference on the 0–255 channel scale, after 32 frozen render
updates (lower is closer; this is a whole-image statistic, not a perceptual or
policy-equivalence test):

| Camera | Initial state | Step 120 |
| --- | ---: | ---: |
| Egocentric | 2.61 | 2.68 |
| Over shoulder | 1.26 | 1.27 |
| Wrist | 3.14 | 2.50 |

The first saved frames have differences of 1.69–5.02/255. The first wrist image
after the pose change shows more denoising softness than the settled image;
neither has the heavy noise of the failed-denoiser attempt. This still test did
not cover continuous motion; the linked follow-up adds a moving-state replay.
Do not use the v2/v3 moved-wrist pair as a
valid image-quality comparison: it depicts different actual viewpoints.

### Size and startup

The corrected v4 runs used warm shader/kernel caches on the same RTX PRO 4000.
Times below start at Python process entry, before imports:

| Measurement | Isaac RTX | OVRTX |
| --- | ---: | ---: |
| Environment ready | 40.04 s | 48.92 s |
| First Newton action complete | 42.96 s | 51.21 s |
| Peak process host RSS | 8.72 GiB | 7.55 GiB |

The earlier warm v3 run measured 39.28/49.52 s to environment readiness and
42.24/51.75 s to the first action. Its camera-motion defect disqualifies its
moved-wrist image, but its startup timings are retained as a limited repeat.
The experiment demonstrates a smaller environment, **not faster startup**.
First cold attempts included graphics repair, compilation, cache warming and
some overlapping setup work; they are not a controlled cold-start comparison.

The v4 diagnostic capture loop took roughly 27–37 ms for Isaac RTX and
107–110 ms for OVRTX per update of the three cameras. It explicitly calls
`sim.render()` and then updates sensors sequentially, so this is not a tuned
batched-camera or policy-throughput benchmark. No performance port is justified
by these numbers alone.

### Validation and preserved evidence

- Existing installer tests: 6 passed. Requirement generation was also checked
  against the exact GA workspace for all three renderer options. Actual separate
  OVRTX and Isaac RTX installations and CUDA capture runs passed.
- Modified Python files compile; tracked changes pass `git diff --check`.
- Capture assertions check restored states, finite/nonconstant RGB, unchanged
  physical states during accumulation, and Kit absence for OVRTX.
- All completed attempts and the failed initial Kit launch are downloaded with
  SHA-256 archive and member verification. Small setup/provenance records and
  generated USD stage hashes are preserved; bulky debug stages remain on the
  stopped node. The failed public example's reporting wrapper is not counted
  as a passed benchmark.

The compact [verification summary](evidence/ovrtx-rendering-spike-2026-10-24.json)
records hashes, timings and audit metrics. Raw logs, fixture data, images,
installed-environment details and host metadata remain outside Git under the
workspace's `output/ovrtx-spike-20261024` directory. Final side-by-side images are
in `comparison-v4/`; source captures are in `downloaded/*-v4/`.

Instance `57554439` was stopped after evidence download and confirmed
`actual_status=exited`, `cur_state=stopped`, `intended_status=stopped`, within the
authorized two-hour rental window. It was not destroyed.

## Decision and remaining work

OVRTX is promising enough for an optional evaluation-path prototype after the
user reviews these images. Keep Isaac RTX for scored runs until reference-policy
scores and rendering behavior have been compared on a representative task
subset. One scene and two frozen
states do not establish equivalent policy observations across the benchmark.

The initial spike added the installer choice, opt-in runtime plumbing,
capture example and this documentation/evidence. Follow-ups add reusable
[video comparison](ovrtx-video-2026-10-24.md) and
[parallel rendering/physics throughput](ovrtx-scaling-2026-10-24.md) examples
and reports. The latter includes an unresolved non-finite-state replay failure. Normal renderer defaults are
unchanged. The fixture is intentionally external to Git. Reproduce with fresh,
separate Linux Python 3.12 environments:

```sh
<venv>/bin/python scripts/install_newton61.py --isaaclab-root <lab-checkout> --renderer-runtime ovrtx --install
<venv>/bin/python examples/run_renderer_spike.py --backend ovrtx --fixture <fixture.json> --output <fresh-output> --headless --device cuda:0
```

Use `isaacrtx` for both options in the baseline environment. The optional
`newton-warp` mode is available in the harness/installer but was not tested here.

## Sources

- [GA renderer documentation](https://isaac-sim.github.io/IsaacLab/v3.0.0/source/concepts/renderers.html)
- [Isaac Lab v3.0.0](https://github.com/isaac-sim/IsaacLab/releases/tag/v3.0.0)

GA documents OVRTX and Isaac RTX real-time path tracing; that establishes a
related rendering capability, not pixel parity. Default lighting/material
handling and renderer configuration still need direct comparison.
