# Migration — what a downstream tool has to know

Everything upstream v0.3.1 wrote is still written. This page lists what is **added**, what
is **renamed or retired**, and the knobs that put the semantics back to upstream for an
A/B comparison. Field-level schema documentation lives in [data.md](../data.md); this is
the diff.

## Results row (`episode_results.jsonl`)

Added fields, one row per episode:

| field | type | meaning | change |
|---|---|---|---|
| `score_peak` | float | the live monotone subtask score; `score` is now judged on the final frame | P34 |
| `success_first_hold_s`, `success_confirmed_s` | float or null | when the success predicate first held, and when it was confirmed with the targets at rest; equal when the object was already still | P30 |
| `physics_artifact` | bool | an object moved with an open hand; the episode is not trustworthy and its grasp credit was withheld | P43 |
| `towed_objects` | list | the objects behind `physics_artifact` | P43 |
| `collateral_placed` | int | non-target objects released inside a goal container after reset | P36 |
| `early_resets`, `pre_satisfied` | int, bool | how often the episode was silently re-reset for terminating within two steps, and whether it still did after the cap | P09 |
| `events` | dict | counts per event name; the vocabulary is the new one below | — |

`success` is now read from the task's `success` term only. Upstream took the OR of every
non-timeout termination term, which would have scored a failure term as a win (P38).

## Episode event log (`log_<run>_env<env>.json`)

`schema_version` is 2. Each event carries `step` (the **onset**, P61), `detected_step` (when
the detector fired; absent on ladder lines, which have no detection lag), `code`, `name`,
`info`, `score`.

### Event codes added

| code | name | meaning | severity |
|---|---|---|---|
| 190 | `SUBTASK_COMPLETED` | a ladder stage finished | success |
| 266 | `GRASP_ATTEMPT_FAILED` | the hand touched / closed on the object but never established a carry; bursts within 2 s are one line with a count | failure |
| 267 | `OBJECT_RELEASED` | a grasped object left the hand while the hand was commanded open | neutral |
| 268 | `OBJECT_DROPPED` | a grasped object left the hand while it stayed closed (slip) | failure |
| 269 | `SCENE_SETTLING` | objects moved during the 1 s reset warm-up without hand contact; one per env | neutral |
| 270 | `WRONG_OBJECT_PLACED` | a non-target the hand had held was released inside a goal container | failure |
| 272 | `OBJECT_FELL_OFF_TABLE` | an object dropped 15 cm below its starting height | failure |
| 273 | `TARGET_LOST` | the success condition can no longer be met — terminal | failure |
| 275 | `TOWED_WITHOUT_GRASP` | the object moved with an open hand — physics artifact | failure |
| 276–281 | `OBJECT_ON_TOP_FAILURE` … `OBJECT_GROUPS_IN_CONTAINERS_FAILURE` | ladder lines named after their predicate instead of `UNKNOWN_FAILURE` | failure |
| 282 | `TARGET_OBJECT_BUMPED` | the policy nudged an object the task is about | neutral |
| 283 | `OBJECT_CARRIED` | the grasp detector saw a carry established | neutral |
| 284 | `OBJECT_GRIPPED` | the jaws closed on this object, before the carry | neutral |

Neutral codes are listed in `robolab.core.task.status.NEUTRAL_STATUS_CODES`; the dashboard
colours by that set first and by name as a fallback.

### Retired or changed

| name | status |
|---|---|
| `TARGET_OBJECT_DROPPED` (263) | no longer emitted; replaced by `OBJECT_RELEASED` / `OBJECT_DROPPED` |
| `OBJECT_GRABBED_FAILURE` (248) | no longer emitted as a tracker line; failed attempts are 266 |
| `WRONG_OBJECT_PUSHED_IN` (271) | merged into `WRONG_OBJECT_PLACED` (270) |
| `PLACED_WITHOUT_LIFT` (274) | off by default (`EMIT_PLACED_WITHOUT_LIFT`); no true positive was ever observed |
| `WRONG_OBJECT_DETACHED` (257) | emitted with its own code (was `OK`), and folded into the release that follows it within 0.5 s |
| `GRIPPER_FULLY_CLOSED` (256) | only when closed on nothing, at 98 % of the closure span; neutral |
| `OBJECT_GRABBED_SUCCESS` (139) | the ladder's progress line only; the detector's observation is `OBJECT_CARRIED` |
| `OBJECT_BUMPED` (258) | threshold 2 cm (was 5 cm); targets included; not emitted while the policy holds the object or within 1 s of releasing it |
| `OBJECT_STARTED_MOVING` (261), `OBJECT_TIPPED_OVER` (262) | unchanged and effectively dead, as upstream |

A pick-and-place therefore reads
`OBJECT_GRIPPED → OBJECT_CARRIED → OBJECT_GRABBED_SUCCESS → OBJECT_RELEASED → OBJECT_IN_CONTAINER_SUCCESS → SUBTASK_COMPLETED`.

## HDF5 (`run_<i>.hdf5`, per `demo_<env>`)

Added group `contact/`:

| dataset | shape | meaning |
|---|---|---|
| `contact/<object>` | `(T, 2)` float32 | contact **force** on the left and right finger pad, N | P77 (P62 recorded booleans) |
| `contact/<object>__<destination>` | `(T,)` uint8 | the object touches that container / surface | P77 |

The rest of the layout is upstream's. Bimanual rigs record `left_ee_pose` / `right_ee_pose`
instead of `ee_pose` (upstream's `ee_recorder_bodies` label); `compute_metrics` reads either
(P67).

## Run directory

| file | meaning |
|---|---|
| `run_complete.json` | written once every task/run finished; a directory without it is partial (P10) |
| `env_cfg.json` → `friction` | the requested friction per object, with the catalog class it was resolved from, and the pad material (P79) |
| `friction_applied.json` | the PhysX readback after start-up — per object shape and per pad body — written on every run, upstream materials included (P79) |
| `env_cfg.json` → `renderer`, `policy` | run provenance (upstream 0.3.x) |

## Task metadata

`task_metadata.json` `num_subtasks` changes for the 11 list-form tasks (a sequence is one
group, P28) and for the five rewritten Rubik's-cube / mug ladders (P29). Regenerate with
`python robolab/tasks/_utils/generate_task_metadata.py` (boots Isaac).

## CLI and configuration

- `--friction upstream | <μ> | realistic | <table.json>` on every runner (P79).
- Robot configs may declare `friction_bodies` (the pad bodies a friction override targets);
  `ee_recorder_bodies` remains mandatory as in upstream 0.3.1.
- `contact_gripper` on DROID declares two concrete sensors and the `gripper` alias group
  (P48); custom robots that copied the one-pad form keep working.
- Dashboard URLs are permalinks: `#/results/run/<run>/task/<task>/ep/<env>/<runIndex>?t=<s>`.
- `pytest tests/` returns a real exit code (P07).

## Compatibility knobs

Constants in `robolab/constants.py`. Setting each to the value in the last column restores
the upstream behaviour of that one change, for a controlled comparison. They are not a
supported configuration space — the defaults are the benchmark.

| constant | default | change | upstream value |
|---|---|---|---|
| `SUCCESS_REST_S` | 0.2 | P30 confirmed success | 0 |
| `PLACEMENT_REST_S` | 0.2 | P46 placement after rest | 0 |
| `SUCCESS_MAX_SPEED` | 0.02 m/s | P30 / P46 rest threshold | — |
| `GRASP_HOLD_S`, `GRASP_COUPLING_M`, `GRASP_HAND_MOVE_M` | 0.2 s, 5 mm, 1 cm | P31 grasp = carry | (contact only: no equivalent) |
| `GRASP_ATTEMPT_CLOSURE`, `GRASP_RELEASE_CLOSURE` | 0.3, 0.1 | P31 attempt / release | — |
| `GRASP_ATTEMPT_BURST_S` | 2.0 | P47 attempt bursts | — |
| `GRASP_TOW_CLOSURE`, `GRASP_TOW_OFFSET_M`, `GRASP_TOW_LIFT_M` | 0.1, 3 cm, 2 cm | P43 / P53 tow rule | — |
| `SETTLE_WARMUP_S` | 1.0 | P33 / P74 settle window | 0 |
| `SUBTASK_EXCLUDE_SPAWN_TRUE_RUNGS` | True | P64 | False |
| `GRIPPER_CLOSED_EVENT_THRESHOLD` | 0.98 | P66 | 0.75 |
| `DETACH_FOLD_S` | 0.5 | P57 | 0 |
| `EMIT_PLACED_WITHOUT_LIFT` | False | P75 | — (did not exist) |
| `OFF_TABLE_DROP_M` | 0.15 | P38 / P76 | 0 disables |
| `FRICTION` | `"upstream"` | P79 | `"upstream"` |
| `open_top_cap_margin` (argument of `build_local_hull`) | 0.05 m | P12 | `None` |

Changes with no knob — the rewritten ladders (P19, P29, P65), the scene height edits (P20,
P33, P56), the second contact sensor (P48), the target resolution (P14) and the event
renames — are what upstream should have done; comparing against them means checking out
upstream.
