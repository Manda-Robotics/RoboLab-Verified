# Parallel rendering and Newton stepping on RTX PRO 4000

On the same 24 GB RTX PRO 4000 Blackwell, the rendering-only BagelsOnPlate sweep passed
12 environments with Isaac RTX and 16 with Kit-less OVRTX. The immediately next
counts, 13 and 17, failed to allocate CUDA memory during the first Newton step.
These are measured limits of this complete runtime configuration, not theoretical
renderer limits or guarantees for other scenes. No policy model occupies the GPU.
The later physics-stepping test exposed a non-finite-state failure at eight
OVRTX environments. Memory fit must not be read as a stability guarantee.

## Recorded-state rendering throughput

Each environment produces three RGB images: 864×480 egocentric, 1280×720 shoulder
and 1280×720 wrist. One camera set means all three images (2,257,920 pixels).
The physics runtime is allocated and executes one initial action, then captures
play recorded poses without physics stepping. Distinct phase offsets give each
environment the same full 345-state trajectory in a different temporal phase.

| Renderer | Environments | Camera sets/s, aggregate | Frames/s per environment | Sampled peak GiB |
|---|---:|---:|---:|---:|
| Isaac RTX | 1 | 23.09 | 23.09 | 3.99 |
| Isaac RTX | 2 | 31.65 | 15.82 | 5.52 |
| Isaac RTX | 4 | 42.86 | 10.71 | 8.50 |
| Isaac RTX | 8 | 48.29 | 6.04 | 15.18 |
| Isaac RTX | 12 | 53.45 | 4.45 | 20.37 |
| OVRTX | 1 | 23.32 | 23.32 | 4.69 |
| OVRTX | 2 | 28.69 | 14.35 | 7.04 |
| OVRTX | 4 | 60.59 | 15.15 | 8.72 |
| OVRTX | 8 | 91.78 | 11.47 | 13.45 |
| OVRTX | 16 | 106.50 | 6.66 | 22.36 |

At eight environments OVRTX delivers about 1.90× the aggregate rendering
throughput. At each renderer’s largest passing batch, it delivers about 1.99×.
Per-environment latency rises with batching: higher aggregate throughput does
not mean that each episode runs faster. The largest batch is not necessarily
the fastest or most practical batch for a full evaluation service.

Capture timing includes transform publication, batched camera rendering and
synchronized GPU output delivery. Pose restoration/kinematics, image transfer
to CPU and encoding are excluded. Restore-plus-capture is also recorded in
the evidence. Each fresh process uses 32 moving warmup frames and one full
345-frame timed cycle. Shader/kernel caches have been populated by earlier
runs; these are warm-runtime measurements, not fresh-install startup timings.
Whole-device VRAM is sampled at 1 Hz, so brief peaks
can be missed. The failed allocations are independently confirmed in the logs.

## Independent rendering repeats and native images

The repeat harness keeps the same capture, restore and metric functions. It
also saves lossless native-resolution images after timing. No resolution,
physics or renderer-quality settings are reduced for larger batches.

| Renderer | Environments | Repeated aggregate camera sets/s |
|---|---:|---:|
| Isaac RTX | 2 | 31.31 |
| Isaac RTX | 4 | 42.59 |
| Isaac RTX | 9 | 51.26 |
| Isaac RTX | 12 | 53.39 |
| OVRTX | 2 | 28.80 |
| OVRTX | 4 | 60.43 |
| OVRTX | 9 | 97.64 |
| OVRTX | 16 | 105.34 |

Every camera instance is checked for advancing frame counters; sampled states
and camera calibration are compared to the source trajectory. Thumbnails from
all environments are screened against the previously validated video. Native
pixel crops receive visual review for texture detail and noise. These checks
do not establish equal internal ray/sample counts or reference-policy scores.

The requested Isaac RTX setting is `RaytracedLighting`; the effective runtime
readback is `RealTimePathTracing`. The OVRTX RGB path also authors that mode.
Isaac RTX reports AA mode 3 and DLSS execution mode 2 in the repeat runs;
those readbacks are retained without inferring identical internal algorithms.
OVRTX renders tiled products; installed Isaac Lab code authors full per-camera
output resolution multiplied by its tile layout. The cause of its nonlinear
two-to-four-environment timing transition has not been isolated.

## Physics stepping plus rendering

The user requested this additional variant after the rendering sweep. It uses
the normal RoboLab `env.step()` path: Newton physics, action processing,
termination checks, observations and default recorder terms. Production
simulation timing remains 120 Hz, four Newton substeps, eight simulation steps
per 15 Hz control action. The default lazy camera-observation path is preserved.
This differs from the explicit batched capture API in the rendering-only test.

All environments receive the same 345 recorded actions from the same initial
physical state. There is no pose teleportation between timed actions. A separate
32-action warmup precedes a fresh reset and verified initial-state restoration.
Premature frozen/terminated environments fail the run rather than being counted
as active throughput. These are unscored, open-loop diagnostic rollouts;
numerical trajectories can diverge between separately compiled GPU runs.
Native CoACD compilation also varies: the one-environment runs reported
637 versus 642 shapes and the four-environment runs 2581 versus 2565.
The eight-environment pair both reported 5129 shapes, but equal counts alone
do not prove identical geometry. These are fresh-start pipeline measurements,
not an isolated renderer-only causal ablation of one shared physics model.

Step timing ends after GPU synchronization. All three native RGB tensors are
then copied to CPU in a separately timed operation. No policy inference, network
transport, image preprocessing or final video/HDF5 export is included. Default
per-step recorder work is included. Thus this is simulator throughput, not a
complete hosted-policy evaluation throughput claim.

Presentation clips show environment 0 at 640×360 and 5 fps, sampling every
third control action. Playback follows simulation time, not measured wall time.
The timed cameras still produce native-resolution images every control action;
lossless native checkpoints are retained for the first and last environments.

| Renderer | Environments | Env steps/s | Including RGB CPU transfer | Hz per environment | Peak GiB |
|---|---:|---:|---:|---:|---:|
| Isaac RTX | 1 | 0.79 | 0.78 | 0.79 | 3.99 |
| Isaac RTX | 4 | 2.80 | 2.78 | 0.70 | 8.50 |
| Isaac RTX | 8 | 4.96 | 4.92 | 0.62 | 15.18 |
| Isaac RTX | 12 | 6.88 | 6.81 | 0.57 | 20.37 |
| OVRTX | 1 | 1.19 | 1.19 | 1.19 | 4.94 |
| OVRTX | 4 | 4.07 | 4.03 | 1.02 | 8.78 |
| OVRTX | 8 | 7.30 | 7.19 | 0.91 | 13.50 |
| OVRTX | 16 | 11.58 | 11.04 | 0.72 | 22.42 |

Including native RGB transfer to CPU, the four-environment pair delivers
4.03 versus 2.78 aggregate control steps/s, a 45% gain for OVRTX.
The largest fully completed batches deliver 11.04 steps/s with 16 OVRTX
environments versus 6.81 with 12 Isaac RTX environments, a 62% gain.
These rates remain below real time for each individual episode.
The OVRTX eight-environment timing is from a successful ordinary
repeat; the prior numerical failure described below remains unresolved.

The first Isaac RTX one-environment replay completed all 345 steps and its
HDF5 trajectory before a missing OpenCV video codec stopped final export.
Its timing rows, applied actions, finite states and native still images
were independently audited. Those measured timings are retained; no full
video is available for that attempt. Subsequent runs use verified system
FFmpeg after timing. The timed loop is AST-identical across the repair.

The original OVRTX eight-environment run failed a finite-state check at
control step 173 (11.53 simulated seconds). It used 13.50 GiB sampled GPU memory;
collision candidates remained below capacity with no buffer growth. This is
not classified as a memory-limit failure. Its timings are excluded from the
comparison. The original guard did not persist the offending state before
raising, and its HDF5 file has no exported demos. Complete failure evidence
and native images at the failure checkpoint are retained.

One repeat adds state checks after every action and saves offending components,
partial trajectories and video before aborting. That extra-checking diagnostic
is excluded from the comparable timing table. The independent 12/16-environment
cases preserve the original checkpoint frequency and measurement sections.
No physical coefficients, assets, solver settings or actions are changed.
The diagnostic completed all 345 steps with all eight trajectories finite,
correct actions and camera updates. It had the same 5129 native shape count
as the failed attempt, but full geometry identity was not established.
The failure did not reproduce; its cause and repeatability remain unresolved.

A final ordinary eight-environment repeat completed all 345 actions with all eight HDF5 trajectories finite, matching source actions and initial state/camera metadata. It delivered 7.19 aggregate control steps/s including native RGB transfer to CPU, versus 4.92 for Isaac RTX. All three environment-0 videos decoded completely; sampled videos and first/last-environment native images show normal camera/robot motion. This passing ordinary repeat and the passing extra-checking diagnostic do not explain or repair the original non-finite-state failure. No production physics settings were changed.


## Reproduction and scope

The reusable examples are `examples/run_renderer_scaling.py` and
`examples/run_renderer_physics_scaling.py`. Supply the recorded state fixture
and, for the latter, its matching action fixture. Use fresh output directories.
Physics-video export requires system FFmpeg with the libx264 encoder.
The compact evidence JSON records source hashes, package versions, configuration
checks, camera/state audits and failed allocation traces. Raw media and transport
archives remain outside Git in the study directory.

The study directory also contains `physics-rendering-throughput.png` and
`physics-renderer-comparison-n4-cpu.mp4`. The video compares equal simulation
times and labels throughput including RGB transfer to CPU. All 115 frames
were decoded successfully and representative frames were inspected visually.

The rented GPU is instance `57565659`, NVIDIA driver 595.58.03, with matching
Isaac Lab GA, Newton 1.6.1, MuJoCo-Warp 3.12.0, Warp 1.17.0 and Torch 2.11.0+cu130
in both environments. Isaac RTX uses Isaac Sim 6.2.0.0; OVRTX uses 0.5.1.385782
without Isaac Sim or Kit installed. Runs are serialized on this one GPU.

All eight main capacity failures occur in MuJoCo-Warp convex-collision scratch
allocation while renderer memory is resident. Tuning those allocations could
change capacity; this experiment does not tune them. A co-resident policy model,
different scene, camera resolution or renderer settings can change the limits.

The initial cold-start v1 attempt timed out before capture. Missing `libXt.so.6`
was independently repaired without claiming it caused that timeout. The v2
Isaac RTX capture completed, but Kit shutdown bypassed Python `finally` and lost
its summary; its frame data was audited and preserved. V3 persists summaries
before shutdown. All failed and successful attempts remain available.

Instance `57565659` was stopped after evidence download; provider state was `exited` with intended state `stopped` at 2026-10-25T00:05:14.637550+00:00. It was not destroyed.
