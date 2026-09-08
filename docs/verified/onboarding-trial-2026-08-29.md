# RoboLab Verified — hands-on trial log

Date: 2026-08-29
Repo: https://github.com/Manda-Robotics/RoboLab-Verified (main @ `4bf3f80`)
Goal: clone, set up on RunPod, run ~5–10 π0.5 episodes across several tasks, try the fork's features, note bugs.

## 1. Clone (local, macOS)

- `git clone` took >2 min and the shell timed out, but the clone completed. Reason: git-lfs is installed locally with smudge on, so the clone pulled all LFS objects (~6 GB, 1649 LFS files) instead of stubs. README says "git lfs pull (~6 GB)" as a separate step; worth noting that with git-lfs installed the clone itself does it. Final on-disk: `assets/` 3.9 GB + `.git/` 2.1 GB.
- Repo layout is clean: `robolab/` (core), `policies/` (clients), `scripts/`, `offline_tests/` (222 no-sim tests), `docs/verified/` (the fork's rationale/changes/findings/verification), `dashboard/`, `analysis/`, `docker/`.

## 2. Compute (RunPod)

- Resumed an existing exited pod (L40 48 GB, 120 GB persistent volume at `/workspace`) rather than build a new one, to save install time and cost.
- Resumed via the GraphQL API (`podResume`). SSH up ~1 min later. Driver 570.124.04, L40 46 GB.
- **Container disk is ephemeral**: after the restart, `git-lfs`, the GL/Vulkan libs and `~/.cache/openpi` (the 11.6 GB π0.5 checkpoint) were gone; only `/workspace` survived. Had to re-`apt install libegl1 libgl1 libglvnd0 libopengl0 libglx0 libgles2 libglu1-mesa libxt6 libvulkan1 vulkan-tools git-lfs`. The README's `sudo apt install ffmpeg git-lfs` is not enough on a bare RunPod image — without `libegl1`/`libvulkan1` Isaac Sim crashes at RTX init (the docker image has them; a bare-metal/pip install does not).
- Fresh clone of the public repo on the pod (`GIT_LFS_SKIP_SMUDGE=1 git clone` + `git lfs pull`): ~3 min, 9.5 GB on disk.
- Started the OpenPI π0.5 server from the sibling `openpi` checkout with `OPENPI_DATA_HOME=/workspace/.cache/openpi` so the checkpoint persists on the volume. `scripts/serve_pi05.sh` is a thin wrapper around `uv run scripts/serve_policy.py policy:checkpoint --policy.config=pi05_droid_jointpos --policy.dir=gs://openpi-assets-simeval/pi05_droid_jointpos`; checkpoint download is 11.6 GB (~4 min).
- `uv venv --python 3.11 && uv sync --extra isaac51 --extra test` in the fresh clone. Despite a 27 GB `UV_CACHE_DIR` from the previous install, it re-downloaded the Isaac Sim wheels (extscache-kit 2.8 GB, kit-sdk 1.3 GB, …) — the cache from the previous pod session did not help, probably because that env was built from a different lock (upstream RoboLab). Not a fork bug, just a time cost (~10 min).

## 3. Offline features (run locally on macOS, no GPU)

### BUG-1: `uv sync` / `uv pip install -e .` cannot resolve on macOS
`pyproject.toml` pins `torch = { index = "pytorch-cu128" }` under `[tool.uv.sources]` with no platform marker, so any resolution on macOS fails with *"torch (v2.7.0+cu128) only has wheels for manylinux/win"*. This blocks the documented `python -m pytest offline_tests` on a laptop, which is exactly the use-case the offline suite advertises ("no simulator, no GPU, ~5 s"). Fix: add `marker = "sys_platform == 'linux'"` to the torch/torchvision sources (uv supports this), or drop the explicit index for non-Linux.
Workaround used: create the venv, `uv pip install` the dependency list from *outside* the project dir (so `[tool.uv]` is not read), then `uv pip install --no-deps -e .`.

### Offline test suite
`python -m pytest offline_tests -q` → **222 passed in 12.95 s** (doc says ~5 s; fine). Clean, no warnings printed at `-q`.

### Offline audits
- `scripts/audit_task_definitions.py` (needs `usd-core`, installs fine on macOS): runs in 5 s, exit 0. `conflicts: 0  notes: 10  not asserted: 4`. Notes are sensible (e.g. D2: 9 tasks whose success predicate references a container the subtask ladder never mentions; F: `BananaInBowlTableTask` defined twice under `test_tasks/`). Matches what verification.md claims.
- `scripts/check_scene_intersections.py`: `total overlapping pairs: 0`.

## 4. On the pod: smoke test

- `examples/run_empty.py --task BananaInBowlTask --headless --friction 0.5`: exit 0 in 3.5 min (Isaac Sim boot ≈ 2.5 min of that, episode itself 50 steps at ~7 it/s). Output in `output/run_empty_env/BananaInBowlTask/` with `env_cfg.json`, `log_0_env0.json`, and `friction_applied.json`.
- `friction_applied.json` works as advertised: every shape of `table`, `bowl`, `banana` and both finger pads reads back `[0.5, 0.5, restitution]`. Nit: the file is extremely verbose (one `[static, dynamic, restitution]` triple per collision shape, dozens of identical rows per object); a per-object summary plus the raw list would be easier to read.
- Nit (display): at reset the subtask printout shows `❌ object_in_container(... ) (score=1.00)` for a condition that is not satisfied (banana is on the table), while the object's overall score is 0.00. Probably the per-condition number is something other than "satisfied", but next to a ❌ it reads as a contradiction.
- Benign noise in the log: `[Error] [carb.graphics-vulkan.plugin] Could not get NGX parameters block because NGX isn't enabled` at boot (no DLSS in the container). Would be worth a line in docs/debug.md so people don't chase it.

## 5. Incident: pod terminated mid-run (RunPod, not RoboLab)

About 4 min into the first π0.5 task (`t1_BananaInBowl`), SSH started refusing and the API returned `pod: null` for the pod: it and its 120 GB volume were **terminated**, not stopped. Nothing in my chain terminates pods (the old `rc7_friction_sweep.sh` on the volume only *stops*, and it was not running). Cause unknown (host reclaim?). Lost: the fresh install and the partial run output. Lesson recorded: rsync `output/` to local after every task.
The region holding the surviving network volume only had H100/RTX PRO 6000 in stock (≥ $2.09/h), so I deployed a fresh **A40 48 GB secure pod** ($0.44/h, 100 GB container disk, no volume) and redid the setup with a single script (`setup.sh`, ~15 min: apt → parallel clone+LFS / OpenPI sync+checkpoint → `uv sync --extra isaac51 --extra test`).

### Dashboard (local, no results)
- `robolab-dashboard --host 127.0.0.1 --port 8081` serves fine from the default dependency set. `/api/tasks/summary` → 120 tasks / 47 scenes / difficulty & attribute histograms; `/api/scenes/_stats` → 61 scenes, 79 unique objects; `/api/tasks/<name>/subtasks` gives the ladder (`pick_and_place`, conditions `object_grabbed`, `object_in_container`).
- Behaviour to know: with no `--output-dir` it silently loads whatever sources were persisted in `~/.config/robolab-dashboard/sources.json` by an earlier dashboard on the machine (documented in dashboard.md, but surprising the first time — it showed 169 runs from another checkout).
- **`--output-dir` is ignored when a persisted source list exists**: `robolab-dashboard --output-dir trial_output` started with only the old persisted source (`/api/sources` did not contain `trial_output`), so my runs were invisible until I added the directory with `POST /api/sources {"path": …}`. dashboard.md says the flag is an "initial results directory … still editable in the UI"; it should at least be merged into the persisted list.
- Local macOS quirk (not RoboLab's fault): Python 3.11 skipped every `.pth` in the uv venv ("Skipping hidden .pth file") because the files carried the macOS `hidden` flag, so the editable `robolab`/`dashboard`/`policies` packages were invisible outside the repo dir. `chflags nohidden site-packages/*.pth` fixed it. Recorded only because someone else on a Mac will hit `ModuleNotFoundError: No module named 'dashboard'` from the console script and blame the repo.

## 6. Policy runs (π0.5, `pi05_droid_jointpos`, 4 envs/task, headless, A40)

Run chain (`chain.sh`): one process per task as the README insists; `--output-folder-name` per task; `PYTHONUNBUFFERED=1`; rsync to local after each task.

### t1 — BananaInBowlTask (default instruction) — **4/4 success**, 4.1 min wall (incl. ~2.5 min boot)
- Throughput 1.28 it/s for 4 envs; policy 336 ms avg, env step 367 ms avg, video write 78 ms.
- Per-episode fields from the fork all present: `score`/`score_peak`, `success_first_hold_s`/`success_confirmed_s` (equal on all 4 — banana was at rest when released), `physics_artifact=False`, `towed_objects=[]`, `collateral_placed=0`, `pre_satisfied=False`, `early_resets=0`.
- `log_0_env0.json` (schema v2) shows the documented chain, with onset/detection stamping:
  `GRIPPER_HIT_TABLE@98 → OBJECT_CARRIED(step 99, detected 110) → OBJECT_GRIPPED(103, det 110) → OBJECT_GRABBED_SUCCESS@110 → OBJECT_RELEASED@158 → SUBTASK_COMPLETED@164`.
- HDF5 has the new `contact/` group per demo: `banana`, `bowl`, `table` as `(T,2)` float32 pad forces, plus `banana__bowl`, `banana__table`, `bowl__table`, `table__bowl` `(T,)` contact columns. Matches docs/data.md.
- `scripts/verify_patches.py` runs on the rsynced copy on my Mac with only `h5py` (no Isaac, no `robolab` import) — the "reads only the run directory" claim holds. Result: P61 PASS, P71 PASS, P74 PASS, P75 PASS, P77 PASS; P72/P73/P76 N/A (clean run, nothing to test); P79 N/A (no override requested). Output is readable and each PASS cites the episode/time it was judged on.
- Sanity check from verification.md: they say a wrong checkpoint shows as the arm never approaching; here 4/4 clean picks, so the jointpos checkpoint is the right one.

### BUG-2: `events` counts in `episode_results.jsonl` are incomplete
docs/data.md says `events` is "counts per event name". For every t1 episode it contains only `{GRIPPER_HIT_TABLE, OBJECT_GRIPPED, OBJECT_RELEASED}`, while the same episode's `log_*.json` also has `OBJECT_CARRIED`, `OBJECT_GRABBED_SUCCESS` and `SUBTASK_COMPLETED`. Anyone computing "grasps per episode" from the jsonl (as the README's 45.5 → 31.0 figure invites) gets a different number than from the logs. Probably the jsonl counter only sees events that go through one emitter (the detector) and not the tracker-derived / post-processed ones.

### Oddity: `OBJECT_CARRIED` onset precedes `OBJECT_GRIPPED` onset
In all four t1 episodes the backdated onset of `OBJECT_CARRIED` (e.g. step 99) is *earlier* than the onset of `OBJECT_GRIPPED` (step 103), although both are detected at the same step (110) and the README describes the order as GRIPPED → CARRIED → GRABBED_SUCCESS. Reading the log in step order therefore shows a carry starting before the grip. Either the carry onset should be clamped to ≥ the grip onset, or the doc should say the two onsets are independent estimates.

### BUG-3: `analysis/check_results.py` crashes (`NameError: name 'hdf5_path' is not defined`)
Both code paths are broken: `check_folder()` (line 208) and `diagnose_hdf5()` (line 139) reference `hdf5_path`, which is never assigned. `python analysis/check_results.py output/t1_BananaInBowl` → `NameError`; `--diagnose` too. Inherited from upstream v0.1.0 (`git log -S hdf5_path` → initial release), so the fork's "every source must parse" offline check can't catch it — a smoke-run of the CLI over a fixture run dir in `offline_tests/` would. Also, unlike `read_results.py`, it prepends `DEFAULT_OUTPUT_DIR` to a relative path that already contains the directory (`output/output/t1_…` → "Folder not found").

### Nit: websocket keepalive timeout on the first inference
On the first step the OpenPI server JIT-compiles (~60 s). `websockets` fires `ConnectionClosedError: sent 1011 (internal error) keepalive ping timeout` and the client prints a full traceback, then `[Pi0DroidJointposClient] Connection lost (...), reconnecting (attempt 1/3)` and recovers. It works, but a fresh user sees a red traceback in every first run. Either raise `ping_timeout` on the client, warm the server with one dummy request, or log the reconnect without the traceback.

### Nit: `timing` is a run-level block copied into every episode row
`timing.policy_inference_s`, `wall_total_s`, etc. are identical across the 4 episodes of a run — they are totals for the process, not per-episode. Fine, but the field lives in a per-episode record.

### t2 — BananaOnPlateTask (not among the fork's ~25 tested tasks) — **4/4 success**, 5.5 min
- Messier episodes than t1 (env1/2/3 re-grasped; `TARGET_OBJECT_BUMPED` with the nudge distance in `info`, `GRIPPER_HIT_TABLE` ×4 in env3). `score_peak == score == 1.0`, no physics artifacts, `success_first_hold_s == success_confirmed_s` on all 8 episodes so far.
- Event-vocabulary observation (env3, three grasps): the 2nd grasp is logged as `OBJECT_CARRIED(199) → OBJECT_GRABBED_SUCCESS(201)` with **no `OBJECT_GRIPPED`**, while grasps 1 and 3 have the full `CARRIED/GRIPPED/GRABBED_SUCCESS` triple. That's why the jsonl shows `OBJECT_GRIPPED: 2` vs `OBJECT_RELEASED: 3`. So "a pick is logged as GRIPPED → CARRIED → GRABBED_SUCCESS" in the README is the common case, not an invariant; the carry alone is sufficient for GRABBED_SUCCESS. Also each re-grasp re-emits `OBJECT_GRABBED_SUCCESS … advanced 1 step(s) to step 1` although the ladder step was already credited — harmless for `score_peak`, but three "success" lines for one credit is confusing when reading a log.

### Dashboard on the pod, with real results
`robolab-dashboard --output-dir output` picks up the runs immediately (`family: "π0.5"` inferred from `policy: pi05`). `/summary` gives success rate with Wilson bounds and score CI; `/episodes` returns the jsonl rows; `/events` returns the log with `time_s` and a `severity` (`failure`/`neutral`/`success`) per event — that's the scrubber timeline; `/timeseries` returns actions (8-dim joint-pos + gripper) per step; `/video` streams the episode mp4 (906 KB, `video/mp4`). `/thumb` → `404 {"detail":"no thumbnail"}` for every episode: nothing in the run directory produces a thumbnail and no tool generates one, so the endpoint is dead unless something else writes them. Minor.

### t3 — BowlStackingLeftOnRightTask (the fork ran the mirror task RightOnLeft) — **0/4**, 3.8 min
- All four time out at 300 steps (20 s cap; the task's `episode_s` is 20). π0.5 mostly bumps/knocks the bowls; env1 grips `bowl_2` but never places it. `score = score_peak = 0`, `success_*_s = null` on failures (good: no fake timestamps).
- The new failure vocabulary shows up: `GRASP_ATTEMPT_FAILED … (contact lost before a carry was established)`, `MULTIPLE_OBJECTS_GRABBED`, `GRIPPER_HIT_OBJECT`, `TARGET_OBJECT_BUMPED` with distances. Reasons are specific and readable.
- `verify_patches.py`: P61/P71/P73/P75/P77 PASS, **P72 FAIL** — see BUG-4.

### BUG-4: `verify_patches.py` P72 false-FAIL on stacking tasks (target is a "container-like" object)
`p72_no_attempts_on_containers` uses `looks_like_destination(name)`, which decides from the object catalog class / a name hint list (`bowl`, `plate`, …). In `BowlStackingLeftOnRightTask` the pick target *is* `bowl_2`, so the three legitimate `GRASP_ATTEMPT_FAILED` events on it are reported as "3 of 3 attempt lines are on a container or fixture" → FAIL, and `--summary` exits 1. The docstring already records the same class of false regression for `sugar_box`/`raisin_box` in rc6b. The role of an object is task-specific (`object=` vs `container=` in the ladder's `object_in_container` params) — the verifier (or the emitted event) should use that, not the catalog. Affects all 6 `stacking` tasks and anything with a bowl/plate/mug as the target (e.g. `BowlInBinTask`, `DishesInBinTask`, `TakeMugsOffOfShelfTask`).

### t4 — MarkerInMugTask (affordance; not in the fork's tested set) — **0/4**, 13.5 min
- Slowest run (≈1.0 it/s; 600-step episodes). π0.5 hovers and taps the table (`GRIPPER_HIT_TABLE` ×5–12 per episode) and never lifts the marker. env1: a first attempt at step 166 is logged as `GRASP_ATTEMPT_FAILED (contact lost before a carry was established)` with no grab credit (the "grasp requires a carry" rule working), then at step 475–485 `OBJECT_GRIPPED + OBJECT_CARRIED → OBJECT_GRABBED_SUCCESS: advanced 1 step(s) to step 1 for marker`, released at 598 (2 steps before the cap). `GRIPPER_FULLY_CLOSED` (closed on nothing) appears in env1/env3 — at **step 2**, i.e. during warm-up, which looks like a reset artifact rather than a policy action.

### Note (not a bug after reading the code): `score_peak` = 0.0 although the log says "advanced 1 step(s) to step 1"
t4 env1's log credits ladder condition 1 of 2 at step 485 (`OBJECT_GRABBED_SUCCESS … advanced 1 step(s) to step 1 for marker`) but `score_peak` is 0.0. Reading `robolab/eval/summarize.py` and `subtask_recorder.py`: the score is per *subtask* (1 subtask here, so 0 or 1), not per condition — t1's `OBJECT_GRABBED_SUCCESS` events also carry `score: 0.0` and only `SUBTASK_COMPLETED` carries 1.0. So the numbers are consistent; the confusing part is that the event text says "advanced … to step 1" while `reason` uses "(step 1/2)" for the *condition* index. Two different meanings of "step" in adjacent outputs. The `events` counter omitting `OBJECT_GRABBED_SUCCESS` (BUG-2) makes it harder to notice from the jsonl alone.

### t5 — LargerObjectRaisinBoxInBinTask with `--friction 0.5` (size reasoning; not in the fork's tested set) — **0/4**, 5.4 min
- Friction feature works end to end: `--friction` is validated at parse time before Isaac boots; `env_cfg.json → friction` records `{mode: uniform, spec: "0.5", objects: {butter/grey_bin/raisin_box: static 0.5, dynamic 0.5, restitution, source}}`; `friction_applied.json` reads back μ=0.5/0.5 on all 733 collision shapes of the 4 objects and on both finger pads (`requested` block included); `verify_patches.py` → **P79 PASS**, with per-object lines ("butter: 0.5/0.5 (uniform)").
- Behaviour: 7 `GRASP_ATTEMPT_FAILED` across the 4 episodes, `OBJECT_BUMPED`, `GRIPPER_FULLY_CLOSED` ×3 — π0.5 reaches the raisin box but never establishes a carry. P72 PASS here (raisin_box is class `food`, so the catalog correctly says "not a container" — the case the docstring mentions), P73 PASS.
- Note: friction_applied.json is 733 rows of identical triples for this scene. A `unique`/count summary per object would make the human check instant.

## 7. The fork's audit tools over the completed runs (local, no GPU; run while t6 was still going)

- `scripts/verify_patches.py --summary t1..t5`: one table (PASS/FAIL/N/A counts per predicate, then a per-run grid `P61:ok P72:XX …`). P76 correctly reported "UNTESTED — no run exercised it" (no destination was ever lost). P72 shows "REGRESSED in t3_BowlStackLeftOnRight" — the false positive of BUG-4. Exit code is 1 with that run in the set, as documented, so a CI gate on stacking tasks would go red for the wrong reason.
- `scripts/reflag.py t3`: 11 → 11 events (nothing to re-derive on a fresh run, as expected). **BUG-5 (stale message)**: it prints "P62+: tow / drag: needs contact force, which is not recorded", but every one of these runs has `contact/<object>` pad-force datasets in the HDF5 (P77 PASS on all 5). Either reflag.py predates P77 and should now read the HDF5, or the message should say it doesn't use it.
- `scripts/flag_regression.py --output-dir trial_output`: 23 labels → all `no-episode` ("episode not on disk") because the labelled corpus (rc-series runs) isn't here. Harmless, but the README's "score the flags against the human verdicts" only works if you have those exact run directories; a note on where to get them (or a bundled fixture) would help.
- `scripts/find_open_hand_carries.py t1..t5`: 0 open-hand carries — consistent with `physics_artifact: false` on all 20 episodes.
- `scripts/find_sinking_objects.py t1..t5`: 124 (episode, object) pairs scanned; **one hit: `MarkerInMugTask` `marker` sinks 7.1 mm in the first second** — i.e. the marker's authored rest height in `workdesk.usda` is off and it drops at reset. That's a real, reproducible scene finding on a task the fork hadn't run yet, produced purely by their tooling. Nice.
- `analysis/read_results.py "t*"` (the README's glob form) resolved the pattern against the *cwd* and picked up the repo's `tests/` folder ("Found 1 folder(s): tests … No episode results found"). Globs should be resolved under `DEFAULT_OUTPUT_DIR` like bare names are (`read_results.py t1_BananaInBowl` works). Minor.

### t6 — GrabAFruitTask, `--instruction-type vague` ("Grab a fruit") — **0/4**, 14 min
- `instruction`/`instruction_type` are recorded per row (`Grab a fruit`, `vague`). Slow again (~0.7 it/s on this 7-object breakfast scene).
- env1 is a good illustration of the new semantics: `OBJECT_CARRIED(92→101) → OBJECT_GRABBED_SUCCESS(101) → OBJECT_RELEASED(106)` — a real grasp of `orange2`, dropped 5 steps later, twice. Ladder step 1 is credited (`advanced … to step 1 for orange2`) but the second condition (hold/lift) never is, so 0/4 is right.
- Nit: `reason` says `Best progress: object_grabbed(object=banana) (step 1/2)` although the object actually grabbed was `orange2` — for `logical=any` ladders the reason string seems to name the first object in the list, not the one that made progress.
- BUG-2 again, in its worst form: the jsonl `events` for env1 is `{OBJECT_RELEASED: 2, MULTIPLE_OBJECTS_GRABBED: 1}` — two releases and *no* grasp events at all, because `OBJECT_CARRIED`/`OBJECT_GRABBED_SUCCESS` are not counted. Read from the jsonl alone, this episode looks like a physics glitch.
- `verify_patches.py`: P61/P71/P72/P73/P74/P75/P77 PASS.

### BUG-6 (docs): `read_results.py --by-instruction-type` is a documented example but unimplemented
docs/analysis.md lists `python analysis/read_results.py <run> --by-instruction-type` under "Compare instruction types". Running it prints `[RoboLab] summarize_experiments_by_instruction_type is not yet implemented.` and exits. (`--by-attributes` was not tried.)

## 8. Results

`analysis/read_results.py "output/t*"` on the pod (Isaac Sim 5.1 / Isaac Lab 2.3.2.post1, A40, π0.5 `pi05_droid_jointpos`, 4 envs per task, seed 0):

| task | instr. | friction | success | mean t (s) | wall |
|---|---|---|---|---|---|
| BananaInBowlTask | default | upstream | **4/4** | 13.1 | 4.1 min |
| BananaOnPlateTask | default | upstream | **4/4** | 17.7 | 5.5 min |
| BowlStackingLeftOnRightTask | default | upstream | 0/4 | – | 3.8 min |
| MarkerInMugTask | default | upstream | 0/4 | – | 13.5 min |
| LargerObjectRaisinBoxInBinTask | default | **0.5** | 0/4 | – | 5.4 min |
| GrabAFruitTask | **vague** | upstream | 0/4 | – | 14.2 min |
| **total** | | | **8/24 (33 %, CI 18–54)** | | 46 min of GPU |

- 0 physics artifacts, 0 towed objects, 0 collateral placements, 0 pre-satisfied resets, 0 open-hand carries across 24 episodes.
- Every success had `success_first_hold_s == success_confirmed_s` (the object was already at rest when the predicate first held), so the "confirm only after rest" rule never had to delay a success here.
- Full outputs (HDF5, event logs, videos, `friction_applied.json`, per-task stdout logs, `chain.log`, `setup.log`) are in `trial_output/` next to this file (250 MB).

## 9. Bug / issue index

| # | severity | where | summary |
|---|---|---|---|
| BUG-1 | medium | `pyproject.toml` | `torch = {index = "pytorch-cu128"}` has no Linux marker → `uv sync` / `uv pip install -e .` impossible on macOS; the "no simulator" offline suite can't be installed the documented way on a laptop |
| BUG-2 | medium | `episode_results.jsonl` → `events` | counts omit `OBJECT_CARRIED`, `OBJECT_GRABBED_SUCCESS`, `SUBTASK_COMPLETED` (present in `log_*.json`); an episode can show 2 releases and 0 grasps |
| BUG-3 | medium | `analysis/check_results.py` | `NameError: hdf5_path` in both code paths (inherited from upstream v0.1.0); also double-prefixes `output/` |
| BUG-4 | medium | `scripts/verify_patches.py` P72 | target objects that are bowls/plates/mugs are classed as containers by name/catalog → false FAIL on stacking-type tasks, `--summary` exits 1 |
| BUG-5 | low | `scripts/reflag.py` | prints "needs contact force, which is not recorded" although P77 records it in every HDF5 |
| BUG-6 | low | docs/analysis.md | `--by-instruction-type` documented but "not yet implemented" |
| nit | low | `analysis/read_results.py` | glob patterns resolve against cwd, not `output/` |
| nit | low | dashboard `/thumb` | always 404: nothing writes `*_viewport_last.png` |
| nit | low | client | first-inference JIT (~60 s) trips the websocket keepalive → traceback + reconnect on every fresh server |
| nit | low | event log | `OBJECT_CARRIED` onset can precede `OBJECT_GRIPPED` onset; `OBJECT_GRABBED_SUCCESS` re-emitted per re-grasp; "step" means condition index in `reason` but ladder step in event text; `GRIPPER_FULLY_CLOSED` fires at step 2 (warm-up) |
| nit | low | `reason` | for `logical=any` ladders names the first listed object, not the one that progressed |
| nit | low | `friction_applied.json` | hundreds of identical rows per object; no per-object summary |
| docs | low | README | RunPod/bare-metal pip install needs `libegl1 libvulkan1 …` in addition to `ffmpeg git-lfs`; NGX errors at boot are benign |
| finding | – | `workdesk.usda` | `find_sinking_objects.py`: marker drops 7.1 mm at reset in `MarkerInMugTask` |

## 10. Overall impression

- **Install**: straightforward on Linux if you know the extra GL/Vulkan packages; ~15 min on a fresh pod with parallel downloads (Isaac wheels ≈ 5 GB, checkpoint 11.6 GB, LFS assets 6 GB). `uv sync --extra isaac51` "just worked" on driver 570.
- **Running**: the README's warnings (one process per task, `python -u`, `OMNI_KIT_ACCEPT_EULA`, the jointpos-vs-delta checkpoint trap) all turned out to matter and were accurate. Throughput on an A40 with 4 envs was 0.7–1.3 it/s, i.e. 4–14 min per task; their "5–17 min on an L40" is in line.
- **What the fork adds is real and checkable**: contact forces in the HDF5, onset/detection stamping, the grasp-requires-carry rule (visible in t4/t6), friction as a recorded run parameter with a PhysX readback, and a verifier that runs on a laptop from the run directory alone. `verify_patches.py` is the standout tool; `find_sinking_objects.py` found a genuine scene defect in the first task it was pointed at.
- **Rough edges** are mostly in the secondary tooling and in consistency between the jsonl and the event log (BUG-2 is the one I'd fix first, then BUG-4 since it makes the CI-style gate lie on stacking tasks).
- **Not tried** (budget): `tests/` install-verification suite, `run_recorded.py` replay, the bimanual rigs, the VLM connector, the dashboard UI in a browser (only its API), and the Claude Code skills.

## 11. Cost / time

- RunPod: resumed L40 pod ($0.69/h) for ~55 min before it vanished; fresh A40 pod ($0.44/h) for ~75 min. Two other exited pods disappeared from the account during the session as well, with no terminate call issued against them.
- Wall clock: ~2 h 40 min from clone to this write-up.
