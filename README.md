<h1><picture><source media="(prefers-color-scheme: dark)" srcset="docs/images/ribble.gif"><img src="docs/images/ribble-dark.gif" alt="" height="42" align="absmiddle" /></picture> RoboLab Verified</h1>

[![offline tests](https://github.com/Manda-Robotics/RoboLab-Verified/actions/workflows/offline-tests.yml/badge.svg)](https://github.com/Manda-Robotics/RoboLab-Verified/actions/workflows/offline-tests.yml)

RoboLab Verified is a fork of [NVIDIA RoboLab](https://github.com/NVlabs/RoboLab) v0.3.1, a task-based simulation benchmark for robot manipulation policies built on [Isaac Lab](https://github.com/isaac-sim/IsaacLab). The fork keeps upstream's 120 tasks, scenes, success predicates and physics defaults. It changes how episodes are scored, how subtasks are credited, what the event log records, and what the sensors and recordings capture.

<h4 align="center"><a href="docs/verified/README.md">Why this fork</a> · <a href="docs/verified/changes.md">Changes</a> · <a href="docs/verified/findings.md">Findings</a> · <a href="docs/verified/migration.md">Migration</a> · <a href="docs/verified/verification.md">Verification</a></h4>

<h4 align="center"><a href="assets/objects/README.md">Objects</a> · <a href="assets/scenes/README.md">Scenes</a> · <a href="robolab/tasks/README.md">Tasks</a> · <a href="robolab/robots/README.md">Robots</a> · <a href="policies/README.md">Policy Clients</a></h4>

<div align="center">
  <img src="docs/images/robolab.png" alt="RoboLab Overview" width="800"/>
</div>

## What is different

Inspection of a 328-episode corpus produced with the upstream harness found the following. All 88 successes were declared while a target object was still moving. 64 % of logged "object grabbed" events occurred with the hand open. One task completed its subtask ladder at 0.07 s, before the arm had moved. In four of four successes of a stacking task, the wrong bowl was stacked. Each finding and its measurement is in [`docs/verified/findings.md`](docs/verified/findings.md).

The changes by area follow; [`docs/verified/changes.md`](docs/verified/changes.md) documents each one with its evidence.

- **Scoring**: a success is confirmed only after the targets are at rest. An episode ends as a failure once it can no longer succeed (a required object or the destination has left the table). Containment is capped at the rim.
- **Subtask crediting**: conditions already true at reset earn no credit. Nothing is credited before the scene settles or before a placed object comes to rest. `score` is judged on the final frame; `score_peak` keeps the live number.
- **Event log**: a grasp requires a carry. A pick is logged as `OBJECT_GRIPPED → OBJECT_CARRIED → OBJECT_GRABBED_SUCCESS`. Releases and drops are distinguished by the commanded gripper state. Wrong objects delivered into the goal are flagged. An object moving with an open hand marks the episode as a physics artifact. On the same tasks, π0.5's event count per episode fell from 45.5 to 31.0, and each remaining event corresponds to one physical transition.
- **Sensing and recording**: both finger pads carry a contact sensor (upstream read one). The HDF5 records per-pad contact force and object-to-destination contact, so flag rules can be re-evaluated on existing recordings.
- **Physics**: friction is a run parameter (`--friction`), read back from PhysX and written to every run directory. The default is upstream's. In a 32-episode-per-condition sweep, the success rate was insensitive to a 4× change in μ; the behaviour metrics were not ([`docs/physics.md`](docs/physics.md)).
- **Embodiments and backends**: a bimanual YAM rig (two I2RT arms, the side-by-side station most bimanual data is collected on) with a client for Ai2's released MolmoAct 2 bimanual checkpoint ([`docs/bimanual_yam.md`](docs/bimanual_yam.md)), a dual-Franka rig, a bimanual ViperX (ALOHA) config (its asset is not shipped), and a connector for running a pointing-capable VLM as a policy.
- **Tooling**: offline audits of task definitions and scenes that run without a simulator, and a verifier that evaluates each flag change as PASS / FAIL / N/A over recorded runs.

Status: the offline suite (222 tests) runs in CI. Each change is marked RUNTIME, OFFLINE or NONE according to how it was verified. About 25 of the 120 tasks have been run against the patched code, most with π0.5 only. [`docs/verified/verification.md`](docs/verified/verification.md) lists the verification status of each change. Numbers reported from this fork should include the tag.

## Key Features

- **RoboLab-120**: 120 benchmark [tasks](robolab/tasks/README.md) spanning pick-and-place, stacking, rearrangement, tool use, and more, each with language instructions and automated success/failure detection via composable predicates.
- **Bring your own robot**: tasks are not tied to a specific embodiment; any robot compatible with Isaac Lab can be used. Single-arm DROID, Franka and Kinova ship, along with two bimanual rigs ([`robolab/robots/README.md`](robolab/robots/README.md)).
- **Rich asset libraries**: [objects](assets/objects/README.md), [scenes](assets/scenes/README.md), and curated [backgrounds](assets/backgrounds/README.md) for creating new scenes and tasks.
- **AI-enabled workflows**: generate new scenes and tasks from natural language with the [/robolab-scenegen](skills/robolab-scenegen/) and [/robolab-taskgen](skills/robolab-taskgen/) Claude Code skills.
- **Multi-environment parallel evaluation**: parallel episodes with vectorized conditionals and per-environment termination.
- **Results dashboard**: a self-contained web [dashboard](docs/dashboard.md) for browsing scenes and tasks, replaying episodes (all cameras on one transport, the event timeline as the scrubber, `?t=` permalinks), and comparing experiments.

See the [Ecosystem](docs/ecosystem.md) page for projects built on RoboLab.

## Getting Started

Requires [uv](https://docs.astral.sh/uv/getting-started/installation/), [Git LFS](https://git-lfs.com) and a system `ffmpeg` (used for video recording). All `.usd`/`.usda` assets are LFS objects; without Git LFS, each scene file is a 130-byte pointer stub. The Isaac Sim / Isaac Lab stack is selected at install time with one of two mutually exclusive extras: `isaac51` (Isaac Sim 5.1 / Isaac Lab 2.3.2.post1, recommended; every result in this fork was produced on it) or `isaac50` (Isaac Sim 5.0 / Isaac Lab 2.2.0). Isaac Sim 5.0 segfaults at startup on NVIDIA drivers ≥ 580; use `isaac51` on current drivers. See [Requirements](#requirements) for hardware.

### Installation

```bash
sudo apt install ffmpeg git-lfs
# On a bare Linux image (a cloud GPU box without a desktop) Isaac Sim also needs the
# GL/Vulkan runtime; without libegl1/libvulkan1 it crashes at RTX renderer init:
# sudo apt install libegl1 libgl1 libglvnd0 libopengl0 libglx0 libgles2 libglu1-mesa libxt6 libvulkan1 vulkan-tools
git lfs install
git clone https://github.com/Manda-Robotics/RoboLab-Verified.git robolab
cd robolab
git lfs pull                     # fetch the USD assets (~6 GB); with git-lfs installed the clone above already did this (GIT_LFS_SKIP_SMUDGE=1 git clone defers it)
uv venv --python 3.11
source .venv/bin/activate
uv sync --extra isaac51 --extra test   # Isaac Sim 5.1 / Isaac Lab 2.3.2.post1 (recommended)
# uv sync --extra isaac50 --extra test # Isaac Sim 5.0 / Isaac Lab 2.2.0 (drivers < 580 only)
```

`--extra test` installs `pytest`, which the verification suite below requires.

The two stacks cannot coexist in one environment. To keep both available, install each into its own venv via `UV_PROJECT_ENVIRONMENT`:

```bash
UV_PROJECT_ENVIRONMENT=.venv    uv sync --extra isaac50
UV_PROJECT_ENVIRONMENT=.venv-51 uv sync --extra isaac51
```

Verify the installation:

```bash
python -m pytest offline_tests     # 222 tests, no simulator: the evaluation-correctness suite
uv run --no-sync pytest tests/     # boots Isaac Sim: isaaclab importable, all task definitions valid, one full episode runs
```

The Isaac suite auto-accepts the NVIDIA Omniverse EULA, so the run is headless with no prompts. More details at [Debugging → Diagnostic Scripts](docs/debug.md#diagnostic-scripts).

> **Running without activating the venv**: if you don't `source .venv/bin/activate`, prefix every `python` command with `uv run --no-sync` (e.g. `uv run --no-sync pytest tests/`). A bare `uv run` first re-syncs the environment to the default dependency set, which removes the Isaac extra you just installed.

> **Long runs**: launch with `python -u` (or `PYTHONUNBUFFERED=1`). With stdout redirected to a file, Python block-buffers, and Isaac Sim frequently dies at shutdown before the buffer is flushed. Per-episode results are always flushed to `episode_results.jsonl`. A finished run writes `run_complete.json` next to it; a run directory without that file is partial.

> **EULA outside the test suite**: when running other entry points (e.g. `policies/pi0_family/run.py`) for the first time, set `export OMNI_KIT_ACCEPT_EULA=Y` once. Cached after first acceptance.

### Run without a policy

```bash
# Run an empty episode with random actions
python examples/run_empty.py --headless

# Same, with a friction override, and read back what PhysX holds (docs/physics.md)
python examples/run_empty.py --task BananaInBowlTask --headless --friction 0.5

# Playback recorded demonstration data
python examples/run_recorded.py --headless

# Toggle the gripper open/closed while holding the arm fixed (sanity-check
# the gripper action path; saves sensor + viewport video to
# output/run_gripper_toggle/<task>/)
python examples/run_gripper_toggle.py --task BananaInBowlTask --headless

# Drive the dual-Franka rig with the scripted client
python policies/bimanual/run.py --task BimanualLiftToteTask --num-envs 2 --headless

# Move the bimanual YAM rig with the scripted client (no checkpoint needed)
python policies/bimanual/run.py --robot yam --task YamPutEverythingInBoxTask --headless

# Drive the bimanual YAM rig with MolmoAct 2 (serve the checkpoint first, see policies/molmoact2/README.md)
python policies/molmoact2/run.py --task YamPutEverythingInBoxTask --task-dirs bimanual --num-envs 4 --headless
```

> **Replay**: `run_recorded.py` restores the recorded initial state and replays the recorded actions open-loop, by default with the env configuration saved next to the recording (`env_cfg.json`). The recorded outcome is not invariant across simulator versions, since contact dynamics change between Isaac Sim / Isaac Lab releases (see [Requirements](#requirements)). Faithful reproduction requires recording and replaying with a single env. See [Replaying Recorded Episodes](docs/replay.md).

### Run with a policy

RoboLab uses a server-client architecture: the model runs as a standalone server, and RoboLab connects to it through an inference client. For a quick test, use [π0.5 via OpenPI](policies/pi0_family/README.md) and serve the joint-position checkpoint (`scripts/serve_pi05.sh`). The OpenPI `--env DROID` convenience flag serves delta actions, which this action space does not accept.

```bash
cd robolab
uv run --no-sync python policies/pi0_family/run.py --policy pi05 --task BananaInBowlTask --num-envs 10
```

Use the [dashboard](#dashboard) to view the output written to your local folder.

### Common CLI Options

```bash
# Run headlessly
python policies/pi0_family/run.py --policy pi05 --headless

# Run on specific tasks (these two are good for sanity checking)
python policies/pi0_family/run.py --policy pi05 --task BananaInBowlTask RubiksCubeAndBananaTask

# Run on a tag of tasks
python policies/pi0_family/run.py --policy pi05 --tag semantics

# Run 12 parallel episodes per task
python policies/pi0_family/run.py --policy pi05 --headless --num-envs 12

# Run at a stated friction (upstream | <mu> | realistic | table.json); recorded in the run directory
python policies/pi0_family/run.py --policy pi05 --headless --friction realistic

# Disable subtask progress tracking (on by default; drops score/reason from results)
python policies/pi0_family/run.py --policy pi05 --disable-subtask

# Resume a previous run (skips completed episodes)
python policies/pi0_family/run.py --policy pi05 --output-folder-name my_previous_run
```

Run one process per task. Passing several tasks to one invocation instantiates every task's environments up front and crashes (upstream issue, see [`docs/environment_run.md`](docs/environment_run.md)).

### Check a run

```bash
scripts/verify_patches.py output/<run>                # PASS / FAIL / N/A per change, from the recording
scripts/verify_patches.py --summary output/rc*        # one table across runs; exit 1 on any FAIL
scripts/flag_regression.py --output-dir output        # score the flags against the human verdicts
```

See [`docs/verified/verification.md`](docs/verified/verification.md).

## Example Tasks

See the full [Benchmark Task Library](robolab/tasks/README.md) for all 120 tasks.

<div align="center">
  <img src="docs/images/Make_sure_all_the_white_mugs_are_upright_so_that_the_opening_is_facing_upwards_0_hstack_3X_fps24_width800.gif" alt="Make sure all the white mugs are upright so that the opening is facing upwards" width="800"/>
  <br><em>"Make sure all the white mugs are upright so that the opening is facing upwards."</em>
</div>

<div align="center">
  <img src="docs/images/Put_all_plastic_bottles_away_in_the_bin_3_hstack_3X_fps24.gif" alt="Put all plastic bottles away in the bin" width="800"/>
  <br><em>"Put all plastic bottles away in the bin."</em>
</div>

<div align="center">
  <img src="docs/images/Put_the_orange_measuring_cup_and_the_blue_measuring_cup_outside_of_the_plate_0_hstack_3X_fps24_width800.gif" alt="Put the orange measuring cup and the blue measuring cup outside of the plate" width="800"/>
  <br><em>"Put the orange measuring cup and the blue measuring cup outside of the plate."</em>
</div>

## Dashboard

A self-contained web dashboard for browsing the benchmark (scenes and tasks) and analyzing experiment results.

```bash
uv run --no-sync robolab-dashboard --host 127.0.0.1
# open http://localhost:8080
```

The dashboard has no authentication; bind it to `127.0.0.1` on shared machines.

See [docs/dashboard.md](docs/dashboard.md) for the feature tour, CLI flags, and API endpoints.

## Documentation

Full documentation is at [docs/README.md](docs/README.md), covering:

- **RoboLab Verified**: [why](docs/verified/README.md), [changes](docs/verified/changes.md), [findings](docs/verified/findings.md), [migration](docs/verified/migration.md), [verification](docs/verified/verification.md), [physics](docs/physics.md)
- [Objects](docs/objects.md), [Scenes](docs/scene.md), [Tasks](docs/task.md): creating and managing assets and benchmark tasks
- [Robots](docs/robots.md), [Cameras](docs/camera.md), [Lighting](docs/lighting.md), [Backgrounds](docs/background.md): configuring simulation parameters
- [Environment Registration](docs/environment_registration.md): combining tasks with robot/observation/action configs
- [Inference Clients](policies/README.md): supported open-source models and clients
- [Replaying Recorded Episodes](docs/replay.md): playing back recorded HDF5 episodes faithfully
- [Analysis and Results](docs/analysis.md), [Data and Output](docs/data.md): summarizing, comparing, and auditing results; the output schema
- [Dashboard](docs/dashboard.md): interactive web viewer for benchmark, tasks, scenes, and eval results
- [Subtask Checking](docs/subtask.md), [Conditionals](docs/task_conditionals.md), [Event Tracking](docs/event_tracking.md)
- [Ecosystem](docs/ecosystem.md): task libraries and projects built on RoboLab

## Requirements

| Dependency | Version |
|---|---|
| Isaac Sim | 5.1 (recommended) or 5.0 |
| Isaac Lab | 2.3.2.post1 (recommended) or 2.2.0 |
| Python | 3.11 |
| Linux | Ubuntu 22.04+ |

> **Note on simulator versions**: Isaac Sim 5.0 and 5.1 ship different PhysX builds, so contact-rich dynamics (grasping, object settling) are not invariant across the two stacks. Results are best compared against runs on the same stack. Every recorded run in this fork's documentation is Isaac Sim 5.1. Isaac Sim 6.0 renders the Robotiq gripper incompletely in some scenes ([findings](docs/verified/findings.md#known-defects-not-changed)) and is not supported.

- **Disk space**: ~8 GB (assets account for ~7 GB)
- **GPU**: NVIDIA RTX GPU required. Recommend 48 GB+ VRAM. See [Isaac Lab's hardware requirements](https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/index.html#system-requirements) and the per-task [`num_envs` guide](docs/env_vram_size_guide.md).
- **Speed**: a 4-env run of one task takes 5–17 minutes on an L40 with ~200 ms inference; the full benchmark is ~30 GPU hours per policy.

## License

RoboLab Verified is released under the [Apache License 2.0](./LICENSE), as is upstream RoboLab; see [NOTICE](./NOTICE). Third-party dependency licenses are listed in [THIRD_PARTY_NOTICES.md](./THIRD_PARTY_NOTICES.md).

## Citation

RoboLab is the work of NVIDIA's Seattle Robotics Lab. If you use the benchmark, cite the paper:

```bibtex
@inproceedings{yang2026robolab,
    author    = {Xuning Yang and Rishit Dagli and Alex Zook and Hugo Hadfield and Ankit Goyal and Stan Birchfield and Fabio Ramos and Jonathan Tremblay},
    title     = {{RoboLab: A High-Fidelity Simulation Benchmark for Analysis of Task Generalist Policies}},
    booktitle = {Proceedings of Robotics: Science and Systems},
    year      = {2026},
    address   = {Sydney, Australia},
    month     = {July},
    url       = {https://arxiv.org/abs/2604.09860}
}
```

If your numbers were produced with this fork, say so and give the tag, e.g. "RoboLab Verified (`verified-rc7`, https://github.com/Manda-Robotics/RoboLab-Verified)". The differences from upstream are listed in [`docs/verified/changes.md`](docs/verified/changes.md).

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for how changes are documented and verified, and for the upstream acknowledgements.
