# Findings — what was wrong, with the evidence

The register of defects found in upstream RoboLab v0.3.1 by reviewing recorded episodes and
auditing the code. Identifiers (`A2`, `B1`, `H-B6`, `H-R7-5`, …) are the ones code comments
and [changes.md](changes.md) refer to: letters A–G are the areas below, `H-B` / `H-E` are
code-audit candidates (event tracking, dashboard), `H-R<n>` are items from review session
*n*. Each row says what addressed it, or that nothing did.

The corpus figures come from 328 recorded episodes of π0.5 and Cosmos3 across 41 tasks on
upstream v0.3.1, plus nine human review sessions whose episode-level verdicts are kept in
`analysis/flag_labels.jsonl`.

## A. Scoring semantics

| ID | Finding | Evidence | Addressed by |
|---|---|---|---|
| A1 | Success has no exclusivity requirement: "put X in Y" scores 1.0 if the whole table is swept into Y. | `BBQSauceInBin` env 3: success with the mug, mustard and ranch also in the bin. | `WRONG_OBJECT_PLACED` and `collateral_placed` per episode (P36). Success itself unchanged. |
| A2 | The episode ends on the first frame the predicate holds; a success cannot be verified as stable. | 88 of 88 successes end with an object still moving, 61 % above 10 cm/s; 104 successes with a target trajectory: moving at the final frame in 86 %. | P30 (confirmed success), P46 (placement credit after rest). |
| A3 | Ordinal subtask progress discards demonstrated capability. | `ToolsPickingHammer` 0/4 with mean subtask score 0.75 reads identically on a leaderboard to `PinkSpoonInPot` 0/4 with 0.00. | `score_peak` alongside `score` (P34); per-stage boxes in the dashboard (P24, P25). |
| A4 | `reason` reports terminal state, not peak. | 43 % of failures lost progress before the buzzer; 82 % of failures labelled "grasp-stage" have a grab success in their own log. | P34. |
| A5 | Containment is loose: a full container height of slack above the rim. | A banana leaning on a bowl's rim is "arguably in or not". | P12 (rim + 5 cm cap), P37 (above the base). |
| A6 | Counting-task semantics are undefined when one object is already inside. | "Put two bananas in the crate" with one already in. | Not changed; the audit (P11) reads the spawn state so such tasks are not misreported. |
| A7 | Success can be reached by pushing rather than picking. | `ToolsPickingHammer` env 3: hammer dragged, one tip on the table, never lifted, subtask completed. | A flag was tried (P41) and retired (P75): no true positive in three runs. Open. |

## B. Event tracking

| ID | Finding | Evidence | Addressed by |
|---|---|---|---|
| B1 | `object_grabbed` is contact. No closure, no lift, no coupling. | 1 990 grabs: hand open 64 %, object does not follow the hand 77 %; worst episode 139 / 139 / 139 identical success / drop / failure lines. | P31, P60, P71, P78. |
| B2 | `TARGET_OBJECT_DROPPED` co-fires with `OBJECT_GRABBED_FAILURE` at the same step. | 94 % of the time; three channels, zero independent information. | P31 (`OBJECT_RELEASED` / `OBJECT_DROPPED`), P45 (one line per transition). |
| B3 | No debounce. | 426 events in one 240 s episode; 299 same-name events within 5 steps of each other. | P47 (attempt bursts), P57, P45. |
| B4 | `WRONG_OBJECT_GRABBED` reads the keyword `"conditions"` as its target list. | 129 bogus events across ≥ 9 tasks, including the correct object. | P14. |
| B5 | `GRIPPER_FULLY_CLOSED` fires while holding an object. | 25 % of 1 359 events; the joint reads 1.00 of its range while holding a soup can. | P32, P66. |
| B6 | An accidental drop and a deliberate placement are the same event. | Reviewer: "it's not a failure, it's on purpose you drop it." | P31, P54 (commanded channel). |
| B7 | Deliberate wrong-object placement is logged as a bump. | Grab-carry-release of the mustard into the bin appears as `OBJECT_BUMPED`. | P36. |
| B8, B9, B10, B11, B16 | Failure on intentional release; drop attributed to the wrong object; hit vs failed grasp conflated; grasps with no event at all; wrong-object over-firing near containers. | Review sessions 01–02. | All consequences of B1; addressed by the grasp tracker (P31) and target resolution (P14). |
| B12 | Reset settling is attributed to the robot. | 44 bump/move events in the first second; `red_onion` moved 6.9 cm before the arm could reach it. | P33 (`SCENE_SETTLING`), P20 / P56 (heights). |
| B13 | No thrashing / stuck-loop flag. | Corpus median path efficiency 0.117; 81 of 188 episodes hard-thrash. | Not adopted (reviewer: "tending no"). Behavioural features remain in `scripts/` for offline analysis. |
| B14 | No shakiness / twitching flag. | Review session 03. | Not adopted. |
| B15 | No "catastrophic scene disruption" event. | A target and its container thrown off the table read the same as a nudge. | P38 (`OBJECT_FELL_OFF_TABLE`, `TARGET_LOST`), P76. |

## C. Physics, assets, scenes

| ID | Finding | Evidence | Addressed by |
|---|---|---|---|
| C1 | Friction is unrealistically high and static = dynamic everywhere. | Pads 2.0, 289 of 312 objects 2.0, nine fruit 5.0, seven Objaverse objects 10.0; pad–bagel effective μ 6.0. | P79: friction as a run parameter; default unchanged; the sensitivity sweep is in [physics.md](../physics.md). |
| C2 | Arm has `disable_gravity=True`, PD 400/80, EEF offset (0, 0, 0). | Code read. | Documented, not changed: it is Isaac Lab's reference high-PD configuration ([physics.md](../physics.md)). |
| C3 | Shipped scenes are not settled. | 62 (task, object) pairs sink at reset, five of them 675–817 mm; objects authored intersecting their container roll at every reset. | P20, P33, P56 for five objects; `SCENE_SETTLING` reports the rest; `scripts/check_scene_intersections.py` lists the authored overlaps. Re-authoring scenes is deliberately out of scope. |
| C4 | Ground-plane heights inconsistent (−0.697 canonical vs legacy −0.65). | Upstream CHANGELOG 0.3.1. | Not changed (upstream locks it per scene for replay compatibility). |

## D. Harness and reproducibility

| ID | Finding | Evidence | Addressed by |
|---|---|---|---|
| D1 | Multi-task invocation crashes (CUDA illegal memory access; all envs instantiated up front). | Reproduced at 4 and 10 envs. | Not changed; run one process per task (documented in [environment_run.md](../environment_run.md)). |
| D2 | Recorded video is below the policy's input resolution. | "I can see the pixels." | Parked (P27). |
| D3 | The review camera is mirrored: robot's right is the viewer's left. | Caused a reviewer to call `BowlStackingRightOnLeft` "broken". | P17 (explicit tag). A camera behind the robot was tried and rejected as occluded (P40). |
| D4 | PhysX silently falls back to CPU after a container resume while `torch.cuda` still reports the GPU. | ~50× slower. | Documented; not a benchmark change. |
| D5 | `nohup` + block-buffered stdout loses the run's result lines. | Observed on every long run that died at shutdown. | P10. |
| D6 | The VRAM-bound env count is undocumented. | Cosmos3-Nano: 4 envs max on 48 GB. | [env_vram_size_guide.md](../env_vram_size_guide.md). |
| D7 | A cut-short run leaves HDF5 trajectories with no results row. | — | P10's marker lets tooling tell a partial run from a finished one; rows are not fabricated. |

## E. Dashboard

| ID | Finding | Addressed by |
|---|---|---|
| E1 | Fresh install 500s (removed Starlette signature). | P01 |
| E2 | Episode grid blank without thumbnails the runner does not emit. | Not changed. |
| E3, E4 | Two per-episode videos play independently; autoplay on open. | P15 |
| E5 | The episode view does not show the instruction and subtask chain. | P24 |
| E6 | The results table shows SR and score but not which subgoals were reached. | P25, P70 |
| E7 | Events cannot be clicked to seek the video. | P15 (timeline as scrubber), P26 (`?t=`) |
| E8 | No indication the viewport is mirrored. | P17 |

## F. Docs, install, tests, packaging

| ID | Finding | Addressed by |
|---|---|---|
| F1 | `pytest` lives in an optional extra the documented install omits. | P08 (documented) |
| F2 | Bare `uv run` prunes the Isaac extra mid-session. | P08 |
| F3 | `conftest.py` destroys pytest's exit code. | P07 |
| F4 | Git LFS required, undocumented; every `.usda` a pointer stub without it. | P08 + `tests/test_lfs_pointers.py` |
| F5 | Isaac Sim 5.0 segfaults on drivers ≥ 580. | P08 (`isaac51` documented default) |
| F6 | README `git clone <repo_url>` placeholder. | P08 |

## G. Analysis tooling

| ID | Finding | Addressed by |
|---|---|---|
| G1–G3 | Behavioural feature extraction, anomaly triage and the human-review loop existed only as working scripts. | The review loop is the [method](README.md#method); the detectors that survived ship under `scripts/` ([changes.md](changes.md#tooling-shipped-with-the-fork)). |
| G4 | The metrics extractor produced `NaN` end-effector path lengths and silently returned `None` for demos without an `ee_pose` group. | P67. |

## Code-audit findings

From a line-by-line read of the event tracker, conditionals, state machine and dashboard.
Rows marked **open** are real and deliberately unchanged in this release; they are the
starting list for the next one.

### H-B. Event tracking, conditionals, scoring

| ID | Finding | Status |
|---|---|---|
| H-B1 | List-form `Subtask(conditions=[…])` builds parallel groups, not a sequence. | P28, P29 |
| H-B2 | `group1..N` names poison the intended-object set. | P14 |
| H-B3 | Completed stages latch forever — no regression. | P34 (final-frame judgement); per-step regression deliberately not done |
| H-B4 | `choose K` means `== K` for terminations and `>= K` for subtasks; docs say "exactly K". | **open** (P13 parked; five benchmark tasks use `choose`) |
| H-B5 | `pick_and_place` score is binary per object; docs describe a four-step 0.25 ladder. | **open** |
| H-B6 | Open-top containment is an infinite +z column. | P12 |
| H-B7 | `stacked` accepts leaning / inside via the `above_bottom` fallback. | **open** |
| H-B8 | Footprint checks use the world AABB of a rotated OBB (45° plate inflated up to √2). | **open** |
| H-B9 | Directional predicates have no minimum separation and use the root pivot, not the centroid. | **open** |
| H-B10 | `record_final_status` can overwrite a completed state machine's status with a stale error code. | **open** (P22 parked, needs evidence on real HDF5s) |
| H-B11 | `success` was the OR of all non-timeout termination terms; only a term literally named `success` was validated. | P38 (success term only), P65 (`success*` matching) |
| H-B12 | Episodes terminating within two steps are silently reset and re-run. | P09 |
| H-B13 | `WRONG_OBJECT_GRABBED` and `GRIPPER_FULLY_CLOSED` are dead on every robot except DROID (no per-robot closure config). | P48 (closure config per pad on DROID); other robots declare their own |
| H-B14 | `WRONG_OBJECT_DETACHED` emitted with code `OK` and picked as the success reason. | P05 |
| H-B15 | Six predicates map to `UNKNOWN_SUCCESS/FAILURE`; completion lines named after the first condition. | P45, P45b (codes 276–281, 190) |
| H-B16 | `OBJECT_TIPPED_OVER` and `OBJECT_STARTED_MOVING` are effectively dead but documented. | **open** (documented as such in [event_tracking.md](../event_tracking.md)) |
| H-B17 | Movement / out-of-scene events tracked only for non-target objects, "intended" = current stage only. | P38, P51 |
| H-B18 | Per-object events never re-arm although docs promise it. | **open** (documented) |
| H-B19 | `MULTIPLE_OBJECTS_GRABBED` counts any contact including the destination container. | **open** |
| H-B20 | Bump / move / workspace thresholds hard-coded to a Franka table. | P50 (bump 2 cm); workspace bounds **open** for bimanual / shelf scenes |
| H-B21 | Every tracker sub-check swallows all exceptions. | **open** |
| H-B22 | On DROID, `object_grabbed` used the *left finger only*. | P48 |
| H-B23 | `object_in_contact` validates `logical` / `K` then ignores them. | P06 |
| H-B24 | `object_at`'s release check contradicts its docs for list grippers. | **open** |
| H-B25 | Ordinal placement latches on the first frame (bounces count). | P46 |
| H-B26 | `tolerance` accepted but ignored by all hull containment predicates. | **open** |
| H-B27 | `get_subtask_state` counts stages while docs show condition counts. | **open** (doc) |

### H-E. Dashboard

Items H-E1–H-E36 from the same audit cover confidence-interval maths (Beta CI excludes the
point estimate at 0 % / 100 %, Wilson and Beta mixed, population vs sample std), double-counted
resumed runs, silent empty data on malformed inputs, path-traversal on the `task` parameter
with a `0.0.0.0` default bind, per-request globbing, and CDN-hosted Tailwind / Plotly. The
ones fixed are listed in [changes.md](changes.md#dashboard); the confidence-interval work
(P23) is parked. **Bind the dashboard to `127.0.0.1` on shared machines.**

## Known defects, not changed

Things a user of this fork should know that no change above removes.

- **`PutMugsOnShelf` is not a runnable task** under π0.5: the rack leaves the table in 7 of 8
  recorded envs within 30 s. P76 stops the clock; it does not make the task achievable.
  Every published number for this task is a number for an episode that was already over.
- **Objects authored interpenetrating their containers** roll at reset (`fruits_in_basket`,
  `fruits_out_of_basket`, others listed by `scripts/check_scene_intersections.py`). Reported
  per episode as `SCENE_SETTLING`; the scenes are as upstream authored them.
- **The "stuck to a finger" artifact.** An object occasionally moves with an open hand
  (object-to-hand distance variance 0.03 mm over 70 mm of motion). It is flagged as
  `TOWED_WITHOUT_GRASP` and the episode marked `physics_artifact`; it survives a 4× cut in
  friction, so it is a contact/solver effect, not a material one. Cause not established.
- **Rigid food.** A bagel is a rigid mesh: a finger through the hole never holds, and the
  policy's sound real-world strategy is penalised.
- **Isaac Sim 6.0 renders the Robotiq gripper as detached fragments in some scenes** (a
  policy input), observed in three of five sampled tasks and absent in two. Not present on
  Isaac Sim 5.1, which this fork targets.
- **The confidence intervals the dashboard shows** are not the Beta intervals the docs
  promise in every cell (H-E1, H-E2).
- The `H-B` rows marked **open** above.
