# OVRTX motion video and capture speed — 2026-10-24

Follow-up to the [frozen-scene comparison](ovrtx-rendering-spike-2026-10-24.md),
on `codex/ovrtx-rendering-spike`. The same RTX PRO 4000 Blackwell instance was
successfully restarted; no additional instance was rented.

## Method

`examples/run_renderer_video.py` plays the same 345 recorded physical states
through Isaac RTX and Kit-less OVRTX, in separate processes and virtual
environments. The recording is a 23-second, 15 Hz GA bagel trajectory. The
initial scene and camera calibration come from the same recording. The mounted
wrist camera follows the robot using `update_latest_camera_pose=True`.

Each process warms the initial pose for 32 capture updates before each pass,
then runs three timing-only passes and one separate video pass. There is one
fresh three-camera capture for each recorded state, with no extra accumulation
at each moving pose. `scene.reset_to()` writes physical poses without resetting
sensors or temporal denoiser history. No physics steps occur during playback;
this is neither a fresh-policy episode nor a physics speed comparison.

Capture uses the public `SensorBase.update_batch()` API. The synchronized wall
clock timer includes scene/camera transform publication, `sim.render()`, batched
camera rendering and output delivery on the GPU. It excludes recorded-state
restoration/kinematics, CPU image readback, audit checks and video encoding. The
video pass runs after the timing-only passes so CPU encoders do not compete
with those measurements. OVRTX synchronous mode provides fresh frames.

The first 15 frames of each timed pass are excluded from the steady-state
summary, leaving 990 measured camera sets per renderer. Full per-frame timing
records and all-frame summaries are also retained. Rates refer to complete
sets of three cameras in **one environment**, not parallel environments.

The videos play at the original 15 Hz; playback speed does not indicate which
renderer generated frames faster. The comparison montage resizes the scene and
wrist views for display. Individual camera videos retain native resolution.
Video encoding uses H.264 CRF 16; unchanged PNG checkpoints preserve samples for
image inspection. Montage encoding is outside the benchmark.

## Validation

The harness checks physical state restoration, unchanged physics step count,
exactly one sensor-frame increment per replay frame, and nonconstant finite RGB.
The independent audit compares live camera poses and calibration between
renderers for every frame of every pass. It verifies decoded video counts,
frame rates and resolutions and screens cross-renderer temporal alignment.

Both processes completed all four 345-frame passes. Across all paired frames,
the recorded physical-state error and differences in live camera position,
orientation and calibration were exactly zero. Playback advanced zero physics
steps. All six native camera videos decode to 345 frames at 15 fps with the
expected dimensions. The two final case archives and every member were
SHA-256 verified after download.

Native resolutions are 864×480 for the egocentric camera and 1280×720 each for
the shoulder and wrist cameras, totaling 2,257,920 pixels per capture set.

## Rendering speed

Steady-state synchronized capture wall times, using all three cameras together:

| Renderer | Mean | Median | 95th percentile | Three-camera sets/s |
| --- | ---: | ---: | ---: | ---: |
| Isaac RTX | 35.39 ms | 34.13 ms | 45.26 ms | 28.26 |
| OVRTX | 39.13 ms | 39.64 ms | 41.96 ms | 25.56 |

OVRTX takes approximately **10.6% longer per capture set** in this scene. The
three pass means were 35.92/35.77/34.47 ms for Isaac RTX and
39.33/40.17/37.88 ms for OVRTX. These are measurements on one GPU and one
trajectory, not a general claim about renderer scaling or policy throughput.

Batching makes the OVRTX result substantially better than the earlier
107–110 ms sequential diagnostic loop. The scenes and motion also differ
between those measurements, so that comparison is not a controlled estimate
of the batching optimization alone. This motion test is the useful direct
comparison: both renderers receive the same recorded states and capture API.

## Video findings and artifacts

The 23-second comparison shows the external scene and moving wrist, with Isaac
RTX on the left and OVRTX on the right. Sampled frames through approach, grasp,
motion and withdrawal have matching framing and object placement. The original
heavy denoiser noise and frozen wrist viewpoint are absent from these samples.
Minor lighting/denoising differences remain. This sampled visual review is not
an exhaustive temporal-artifact or policy-equivalence assessment.

An independent screen decoded every frame and compared 256×144 thumbnails at
offsets of −2 through +2 frames. Zero offset gave the lowest error for all three
cameras. For the moving wrist, zero-offset mean absolute RGB error was
1.92/255 versus approximately 9.4/255 at either one-frame offset. This supports
frame alignment, rather than a delayed OVRTX stream that merely reports fresh
pose metadata. It is not a perceptual-quality score; compression, downsampling
and whole-image averaging affect it.

Raw videos, frame metadata and logs remain outside Git in
`output/ovrtx-spike-20261024`. The branch retains the reusable capture example
and [compact verification report](evidence/ovrtx-video-2026-10-24.json) only.
The evidence directory contains:

- `video-comparison/isaacrtx-vs-ovrtx.mp4`: scene and wrist montage.
- `video-comparison/wrist-native-side-by-side.mp4`: native-resolution wrist pair.
- `downloaded/{isaacrtx,ovrtx}-video-v1/video/*.mp4`: all six native camera videos.
- `video-comparison-audit.json`: count, calibration, alignment and timing checks.
- `downloaded/*-video-v1/timing_*/frames.json`: timing-only per-frame records.

Normal evaluation defaults and production contact settings remain unchanged.
The separate OVRTX environment retains the installation-size advantage measured
in the first spike. Reference-policy scores remain untested. The subsequent
[parallel rendering and full Newton-stepping study](ovrtx-scaling-2026-10-24.md)
measures multi-environment capacity and throughput on a replacement RTX PRO 4000.
It also records a non-finite-state replay failure that remains unresolved.

Instance `57554439` was stopped again after the follow-up evidence was downloaded
and verified. The service confirmed `actual_status=exited`, `cur_state=stopped`
and `intended_status=stopped`. It was not destroyed.
