# Changes

One row per change against upstream RoboLab v0.3.1. The `P<nn>` identifiers are used in
commit titles, code comments and test names; `git log --grep P46` or `grep -rn P46 robolab`
finds the implementation of P46. Numbering has gaps where a candidate was withdrawn.
Withdrawn candidates are listed at the end with the reason.

Columns: Scores states whether re-running an existing recording could change a number in
`episode_results.jsonl` (`SR` = success rate, `sub` = subtask score, `ev` = event counts only).
Level is the verification level defined in [verification.md](verification.md): RUNTIME
(a recorded GPU run demonstrates it: verifier PASS with a matching FAIL on the pre-patch
baseline, or confirmed by a human on a linked episode), OFFLINE (unit-tested or replayed
over recorded logs, never executed on a GPU), NONE (written, not executed). Finding
refers to [findings.md](findings.md).

The knobs that restore upstream behaviour for each semantic change are listed in
[migration.md](migration.md#compatibility-knobs).

## Scoring and termination

| ID | Change | Scores | Level | Finding | Evidence |
|---|---|---|---|---|---|
| P30 | **Confirmed success.** The success predicate must hold with every target object at rest (< 2 cm/s for 0.2 s). An object already still ends the episode on that frame; a moving one makes the episode wait; the policy stays active. `success_first_hold_s` / `success_confirmed_s` recorded per episode. | SR ↓ only | RUNTIME | A2 | 104 corpus successes: target still moving at the final frame in 86 %, above 10 cm/s in 44 %, worst 101 cm/s. Reviewer on four separate episodes: "unclear if it would have dropped out again." |
| P12 | **Open-top containment capped** at the container's rim + 5 cm (`open_top_cap_margin`; `None` restores the unbounded column). | SR (rare) | OFFLINE | H-B6, A5 | Upstream's open-top hull was an infinite +z column: an object on a shelf above a bin, or still falling into it, was "in the container". `YellowAndWhiteObjectsInBin`: all three successes ended with a required object 27–95 mm above the rim, still moving. |
| P37 | **Nested containment needs the object above the container's base.** `object_in_container` also requires the object's root at or above the container's root. | SR | OFFLINE | H-R7-5 | `BowlStackingRightOnLeft`, 4 of 4 "successes": the robot stacked the left bowl onto the right one, the opposite of the instruction, and the symmetric hull test passed (final z 0.05 vs 0.07). |
| P38 | **An unwinnable episode ends.** `OBJECT_FELL_OFF_TABLE` for any object 15 cm below its start height; `TARGET_LOST` ends the episode as a failure only when the success condition can no longer be met (`all` → any required object gone; `any` → all gone; `choose K` → fewer than K remain). `success` is read from the `success` term only; other non-timeout termination terms do not set it. | none (those episodes were failures) | OFFLINE (fired on rc3, not judged) | H-B17, H-B11, B15 | `ToolOrganization` ejected a target hammer 576 mm at 2.3 m/s with zero events and ran to the buzzer. Before this, any failure termination term would have scored as a success (`env.py` took the OR of all non-timeout terms). |
| P76 | **A lost destination ends the episode too.** The success term's container or surface leaving the table is `TARGET_LOST`. Scoped to single-group success terms. | none | RUNTIME | H-R7-2 | `PutMugsOnShelf`, 4 of 4 envs: the rack is tipped off the table at 8.9–15.5 s and the episode ran its full 180 s with nothing left to achieve. Now ends at the step the rack leaves. |
| P09 | **The silent re-reset is counted and capped.** An episode that terminates within two steps was reset and re-run with no record; now at most three re-resets, logged, and `early_resets` / `pre_satisfied` per episode. | none (adds fields) | NONE | H-B12 | A predicate already true at reset restarted the episode forever with nothing in the results. |

## Subtask crediting

| ID | Change | Scores | Level | Finding | Evidence |
|---|---|---|---|---|---|
| P28 | **List-form ladders are sequences.** `Subtask(conditions=[a, b, c])` with `logical="all"` is one ordered group; `any` / `choose` stay alternatives. No task file changes. | sub (6 tasks) | RUNTIME | H-B1 | Upstream built parallel one-condition groups (`group1..N`), so `object_dropped` (= not in contact) completed at reset and `object_left_of` was credited while the cube was still in the hand. Corpus: 16 of 40 episodes credited a condition within 0.5 s of reset; estimated subtask score 0.70 → 0.58, four failures scored 1.0 drop. |
| P29 | **Five ladders with a spawn-true step rewritten**: `RubiksCube{LeftOf,RightOf,Behind,InFrontOf}Bowl`, `WhiteMugInCenter`: grab, then place with `require_gripper_detached=True`; the standalone `object_dropped` step removed. | sub (5 tasks) | RUNTIME | H-B1, H-R6-11 | The 0.07 s `OBJECT_DROPPED_SUCCESS` subgoal reviewers asked about. |
| P64 | **A rung already true at reset earns no credit.** The state machine probes every rung once; true ones stay in the ladder but carry no score, the rest are renormalised. General form of P29's hand fixes. `SUBTASK_EXCLUDE_SPAWN_TRUE_RUNGS`. | sub (11 tasks, 60 corpus episodes) | RUNTIME | H-R9-9 | `BananasOutOfBin` logged "Completed subtask 1/1" at 0.07 s, then flagged its own bananas against `(target objects: [])`; `BlackItemsInBin`'s keyboard spawns in the bin. Root cause was the state machine taking the highest satisfied rung ("if the later condition holds, the earlier one must have been"), false for spawn-true rungs. |
| P74 | **Nothing is credited before the scene has settled.** P64's probe runs at the end of the 1 s warm-up. | sub | RUNTIME | — | `BlackItemsInBin` still credited its subtask at 0.27 s in all four envs: `object_in_container` also needs the object at rest, so at step 0 the rung read False and escaped exclusion. Earliest credit now 10.07 s. |
| P46 | **A placement is credited only after the object settles** (0.2 s under 2 cm/s): `object_in_container`, `object_on_top`, `object_on_center`. `PLACEMENT_REST_S`. | sub | OFFLINE | A2, H-B25 | `BananaInBowl` env 3: released at 20.87 s, "Completed subtask" at 20.93 s, then the banana bounced onto the rim and ended 8 cm off centre, mostly outside. The ladder had already scored 1.0. |
| P34 | **`score` is judged on the final frame; the live monotone number is `score_peak`.** | sub | OFFLINE | H-B3, A3, A4 | `PutTwoMugsOnShelf`: mug credited at 70.2 s (score 1.0), the other mug `OBJECT_OUT_OF_SCENE` at 70.3 s, episode fails at 180 s with score 1.0 and one mug on the shelf. Corpus: 29 of 62 credited placements moved > 5 cm afterwards. |
| P65 | **A ladder's last rung is the success predicate** for `GrabABagel` / `GrabAFruit` (`object_picked_up`, 50 mm, instead of contact); the task validator matches `success*` termination names. | sub (2 tasks) | RUNTIME | H-R9-T4, T5 | `GrabAFruit` env 1: "Completed subtask" at 6.20 s with `success: false`; `GrabABagel` credited its subtask with the bagel at exactly its start height. Under exact-name matching neither task had a validated success term at all. |
| P19 | `FoodPackingByColor`'s ladder matches its success placement (the two were inverted). | sub (1 task) | OFFLINE | H-R8-2 | Found by the task-definition lint (P11). |
| P45b | Completion lines use `SUBTASK_COMPLETED` (190) instead of the stage's first condition's code. | ev | RUNTIME | H-B15 | "Is this an object-grab success, or is this just the success?" |

## Event vocabulary

| ID | Change | Scores | Level | Finding | Evidence |
|---|---|---|---|---|---|
| P31 | **A grasp is a carry.** `object_grabbed` = contact for 0.2 s with the object's offset to the hand changing < 0.5 cm while the hand moves ≥ 1 cm (`GraspTracker`). Contact that ends earlier with the hand ≥ 30 % closed → `GRASP_ATTEMPT_FAILED`. After a grasp, contact lost with the hand < 10 % closed → `OBJECT_RELEASED`, otherwise `OBJECT_DROPPED`. `TARGET_OBJECT_DROPPED` no longer emitted. | ev; grab-stage credit; SR of the 4 pick-up tasks | RUNTIME | B1, B2, B6 | 1 990 corpus grabs vs the finger joint and trajectories: hand open at the "grab" 64 %, object neither lifts nor follows the hand within 1 s 77 %, followed by a drop within 2 ticks 33 %. `TARGET_OBJECT_DROPPED` co-fired with `OBJECT_GRABBED_FAILURE` 93 % of the time. |
| P60 | The grasp line is emitted by the tracker, paired structurally with the ladder's grab. | ev | OFFLINE | — | Five tasks whose ladder is a single placement condition (the stacking family) logged releases and drops but never a pick. |
| P61 | **Every event is stamped at its onset** (`step`), with `detected_step` kept alongside. | ev | RUNTIME (457 obs.) | — | Onsets 0.2–4.7 s ahead of detection, mean 1.8 s; the dashboard seeks to the onset. |
| P71 | The detector's grab line is the neutral `OBJECT_CARRIED` (283); the ladder keeps `OBJECT_GRABBED_SUCCESS` as the progress line. | ev | RUNTIME | — | Every grasp printed twice with the same name, and 28 grasps of objects the task then flagged WRONG carried a green success line. |
| P78 | **A pick reads `OBJECT_GRIPPED → OBJECT_CARRIED → OBJECT_GRABBED_SUCCESS`.** The grip (284) is emitted only when it becomes a carry, stamped when the jaws closed; a fumble is `GRASP_ATTEMPT_FAILED` alone. | ev | RUNTIME | — | Reviewer: "it needs to be grip, carry, and then grab success." A stale grip stamp that attached to the next carry was caught on the first GPU run and fixed. |
| P47 | **Consecutive failed attempts on one object within 2 s are one line with a count** (`Grasp attempts on 'X' failed ×9 over 5.1 s`). | ev | OFFLINE | B3 | A fumble at a banana produced 9 lines in 5 s. |
| P72 | Containers and fixtures are excluded from grasp-attempt tracking. | ev | RUNTIME | — | 14 of 81 attempt lines were on a bin or shelf. |
| P73 | No attempt opens within 2 s of releasing the same object. | ev | RUNTIME | — | 10 of 81; contact flickers as an object leaves the hand. |
| P32 | `GRIPPER_FULLY_CLOSED` only when closed on nothing, message "Gripper closed on nothing"; re-armed per open. | ev | OFFLINE | B5 | 1 359 corpus events, 343 (25 %) while holding an object. The joint cannot tell: at those flags `finger_joint` read 1.00 of its range while holding a soup can; the drive reaches its target and the linkage gives. |
| P66 | The `GRIPPER_FULLY_CLOSED` event threshold is 0.98 of the closure span (was 0.75). | ev | RUNTIME | H-R9-13 | `BlackItemsInBin` env 0: a smartphone wedged between the fingers held the joint at 0.83 and the event fired 69 times, 93 % of that episode's log. At 0.98: none. The joint reaches 100 % in 68 of 80 episodes, so this is a narrow fix. |
| P14 | `WRONG_OBJECT_GRABBED` resolves its target list from the condition's bound kwargs (`object` / `objects`), across all stages, with containers excluded. | ev | OFFLINE (corpus replay) | B4, H-B2 | Tasks that build `Subtask(conditions=partial(...))` had the literal keyword `"conditions"` as their target list; list-form tasks had `group1..N`. 129 bogus events across ≥ 9 tasks; `GrabABagel` flagged the fruit that had just completed its subtask 0.27 s later. |
| P44 | `WRONG_OBJECT_GRABBED` only for a carried non-target; a touch does not qualify. | ev | OFFLINE | B16 | A lime episode: 6 lines → 1. |
| P36 | **`WRONG_OBJECT_PLACED`** (270): a non-target the hand had held is released inside a goal container. Objects already in the container at reset never count. (`WRONG_OBJECT_PUSHED_IN` was merged into it, P55.) | ev; `collateral_placed` per episode | RUNTIME | A1, B7 | `BBQSauceInBin` env 3 succeeded with the mug, mustard and ranch also in the bin; `AnimalsInBin` env 0 delivered four wrong objects into the goal bin with no placement event at all. |
| P43 | **`TOWED_WITHOUT_GRASP`** (275): an object moving with the hand while the hand is < 10 % closed, off-centre along the jaw axis by ≥ 3 cm, and lifted ≥ 2 cm clear. The episode is marked `physics_artifact` and its grasp credit withheld. | ev; SR of affected episodes | RUNTIME (rc7) | H-R6-2, H-R7-1 | Calibrated on the reviewer's verdicts: 5 of 5 known tows caught, both known wide-object grips (closure 0.23 and 0.00 centred) rejected. Reviewer on one instance: "a crazy bug, 100 %." First runtime firings in the friction sweep at μ 0.5; the artifact is independent of friction. |
| P52 | A bumped target is `TARGET_OBJECT_BUMPED` (282), neutral. | ev | RUNTIME | — | "It sometimes flags the core item as bumped — it either shouldn't, or show it in green." |
| P50 | Bump threshold 5 cm → 2 cm. | ev | RUNTIME | H-B20 | A 2.8 cm bowl nudge at 44 s produced nothing. |
| P51 | Movement events for every object, targets included, suppressed while the policy holds it or within 1 s of releasing it. | ev | RUNTIME | H-B17 | The onion nudge at 57 s produced nothing because targets were excluded from tracking. |
| P33 | **`SCENE_SETTLING`** (269): motion during the first 1 s without hand contact is logged as one note per env and is not attributed to the robot. | ev | OFFLINE (observed on rc3, not judged) | B12, C3 | 47 of 631 bump/move events fired in the first second, before the arm could reach anything. |
| P49 | `GRIPPER_FULLY_CLOSED`, `OBJECT_RELEASED`, `SCENE_SETTLING` (and later `TARGET_OBJECT_BUMPED`, `OBJECT_CARRIED`, `OBJECT_GRIPPED`) are classed as neutral in the dashboard. | none | RUNTIME | — | "It should be a grey flag, it's not particularly good or bad." |
| P45 | **One line per physical transition.** A tracker drop/release/tow absorbs the ladder's same-tick regression line; ladder lines are named after their predicate (six new codes 276–281). | ev | RUNTIME | B2, H-B14, H-B15 | Simulated on all 328 corpus logs: 1 780 twin lines (17 % of all events) collapse; p90 events per episode 82 → 64. |
| P57 | A `WRONG_OBJECT_DETACHED` line is folded into the release/drop that follows it within 0.5 s. | ev | RUNTIME | — | 43 of 43 corpus detach lines were followed by a release for the same object within 0.5 s (median 0.07 s). 493 → 450 lines on the rc2 corpus. |
| P05 | `WRONG_OBJECT_DETACHED` is emitted with its own code (257) instead of `OK`. | `reason` text | NONE | H-B14 | Six successful corpus episodes carried the detach message as their official `reason`, because the summary picked "last event with code < 200". |
| P54 | Release vs drop is decided by the commanded gripper channel instead of the measured joint. | ev | RUNTIME | B6 | The measured closure was rising in both a deliberate release and a slip; the command channel separates them: 1.00 → 0.00 → 1.00 vs pinned at 1.00. |
| P53 | A tow must lift the object ≥ 2 cm clear of where the contact began. | ev | RUNTIME | — | Both earlier "tows" were drags along the table (`cheez_it` 7 cm below its start). |
| P75 | `PLACED_WITHOUT_LIFT` (P41) retired; `EMIT_PLACED_WITHOUT_LIFT=False`. | ev | RUNTIME | A7 | Three runs, four firings, all the same keyboard that starts in the bin (the P74 bug). No true positive was ever observed; sliding a cube behind a bowl, the most pushable placement in the benchmark, never fired it. |
| P06 | `object_in_contact` honours its `logical` / `K` arguments (they were validated, then ignored). | none (no task calls it) | OFFLINE | H-B23 | Library correctness. |

## Contact sensing and recording

| ID | Change | Scores | Level | Finding | Evidence |
|---|---|---|---|---|---|
| P48 | **Contact sensors on both Robotiq pads**; `contact_gripper` declares `gripper_left` / `gripper_right` and the `gripper` alias group (either pad). Closure config per pad. | contact detection everywhere (more sensitive) | RUNTIME | H-B22, H-B13 | `BowlStacking` env 0 @ 11.87 s: the bowl 2 cm from the fingertips at closure 0.79, and the log said "Gripper closed on nothing"; the left-pad-only sensor did not detect it. |
| P62 | Per-pad gripper contact recorded in the HDF5 (`contact/<object>` as `(T, 2)`). | none | OFFLINE | — | The reviewer's six labelled open-hand carries: the three real tows were held by one pad, the three non-tows by both. Jaw offset did not separate them; pad count did. |
| P77 | **Contact force per pad** (`(T, 2)` float32) and boolean object-to-destination contact (`contact/<object>__<container>`). | none | RUNTIME (102 obs.) | — | Booleans could not separate a labelled drag from a "magnetic" mug; both read both-pads at closure ~1.00, like an ordinary grasp. Over 77 confirmed carries vs 19 failed attempts, peak force does not separate them (AUC 0.39); whether both pads are ever loaded does (AUC 0.95). |

## Physics and scenes

| ID | Change | Scores | Level | Finding | Evidence |
|---|---|---|---|---|---|
| P79 | **Friction is a run parameter** (`--friction upstream / <μ> / realistic / table.json`), applied at start-up through Isaac Lab's material event term on every rigid object and the robot's `friction_bodies`; request in `env_cfg.json`, PhysX readback in `friction_applied.json`. Default unchanged. | none by default | RUNTIME (24 runs) | C1 | Pads and 289 of 312 objects are authored at μ 2.0 (fruit 5.0, bagels 10.0). 32-episode sweep per condition: success 16 / 19 / 25 / 22 % at μ 2.0 / 1.0 / 0.5 / realistic (indistinguishable), failed grasp attempts ×4.5. [physics.md](../physics.md). |
| P20 | **Five objects placed where they settle**: `lemon_02` (−14.2 cm), `lemon_01`, `measuring_cup`, `plasticpail_a02`, and `orange_02` moved off the plate it pre-satisfied (`FruitsOrangesOnPlate` credited a subgoal at 0.07 s in all eight corpus episodes). Rest-height lint `scripts/check_rest_heights.py`. | fewer reset artifacts; one task no longer pre-credited | RUNTIME | B12, C3, H-R5-7, H-R6-17 | 62 (task, object) pairs sink at reset; five by 675–817 mm. |
| P33 | `red_onion` authored at z = 0, inside the table, moved to 0.038; `measuring_cup` to its settled xy. | fewer bogus bumps | RUNTIME | H-R8-8 | It popped up 3.8 cm and 3.4 cm sideways at every reset in 56 episodes across three tasks. |
| P56 | `red_onion` in `clutter_fruit_bottle_bluebin` corrected a second time from a measured run: the scene's authored frame is offset −2.5 cm from world, so "set authored = settled" is wrong by that much. | fewer bogus bumps | RUNTIME | — | Scene height edits must be re-measured after a run. |
| C2 | **Not changed, documented.** `disable_gravity=True`, PD 400/80, EEF offset (Isaac Lab's `FRANKA_PANDA_HIGH_PD_CFG`); gravity-off stands in for the real controller's gravity compensation. | — | — | C2 | [physics.md](../physics.md#the-arm-controller-gravity-off-pd-40080-eef-offset). |

## Harness, reproducibility, packaging

| ID | Change | Scores | Level | Finding | Evidence |
|---|---|---|---|---|---|
| P10 | `run_complete.json` written when every task/run finished; stdout line-buffered in `run_evaluation`. | none | RUNTIME | D5 | With stdout redirected, Isaac often died at shutdown before Python flushed; the run "succeeded" with an empty log. |
| P07 | `pytest`'s exit code survives Isaac's shutdown (`conftest.py` reports the status before Kit exits the process). | none | OFFLINE | F3 | CI saw 0 on every failure. |
| P08 | README install fixes: Git LFS first, `isaac51` as the documented default (5.0 segfaults on drivers ≥ 580), `--extra test`, `uv run --no-sync` everywhere, real clone URL; `tests/test_lfs_pointers.py` scans every asset for pointer stubs. | none | OFFLINE (pointer scan) | F1, F2, F4, F5, F6 | Without LFS every `.usda` is a 130-byte stub and every scene fails to load. |
| P11 | Task-definition lint as a test: each task's success placement AST-diffed against its subtask ladder. | none | OFFLINE | H-R8-3 | Found `FoodPackingByColor` (P19). |
| P67 | `compute_metrics` reads per-arm end-effector channels and fails loudly when there are none. | none (adds metrics for bimanual) | OFFLINE | G4 | Bimanual demos have `left_ee_pose` / `right_ee_pose`, no `ee_pose`; upstream's bare `except` turned that into `None`, so every bimanual run produced no metrics. |
| P68 | `object_picked_up` takes `gripper_name` like every other predicate; a list means every listed gripper must hold the object. | none | OFFLINE | — | The hook that lets a user-authored task express a two-handed lift. |
| — | Every Python source must parse (`offline_tests/test_all_sources_parse.py`); every option a runner reads must be declared (`test_runner_args.py`); every robot label must be shadowed on the scene cfg (`test_p79_friction.py`). | none | OFFLINE | — | Each guards a class of bug that cost a GPU launch: a stray paren, an `AttributeError` 40 s into a boot, an `Unknown asset config type` 2 min into one. |

## Dashboard

| ID | Change | Checked | Finding |
|---|---|---|---|
| P01 | Fresh install no longer 500s on `/` (request-first `TemplateResponse`). | unit test | E1 |
| P02 | Event severity by whole name token, then `StatusCode` range (`"HIT" in name` matched `WHITE`). | unit test | H-E25 |
| P03 | Stable string sort (`localeCompare`). | `node --check` | H-E22 |
| P04 | The video endpoint honours HTTP `Range` (206 / `Content-Range` / 416), so seeking works in Chromium and Safari. | curl, browser | H-E30 |
| P15 | One transport under all camera tiles: play/pause, time, speed, the event timeline as the scrubber; per-tile link toggle; native controls mirrored across tiles. | browser | E3, E4, H-E29 |
| P16 | Runs carry a recorded date and sort newest-first. | browser | H-E31 |
| P17 | `mirrored · robot R = your L` tag on the review viewport (the review camera looks back at the robot). A camera behind the robot was tried and rejected as occluded (P40). | browser | D3, E8 |
| P24 | Subgoal checklist under the instruction; boxes fill with the playhead. | browser | E5 |
| P25 | Task page: per-episode stage boxes; outcome buckets as filter chips. | Playwright | E6, H-E17 |
| P26 | Permalinks `#/results/run/<run>/task/<task>/ep/<env>/<runIndex>?t=<s>`; deep links, reload and Back work; `?t=` opens the episode paused there. | browser | H-E15 |
| P35 | Sidebar groups runs by policy family with pooled success rate, collapsible. | browser | — |
| P39 | Head camera upright (it was rolled 90°); camera names burned into the tiled recording. | GPU run | H-R5-1, H-R5-3 |
| P69 | "verified" mark under the wordmark. | — | — |
| P70 | Click a task on the results overview to see it across every experiment, grouped by policy. | unit test | E6, H-E20, H-E21 |

## Embodiments and policy backends

| ID | Change | Level | Notes |
|---|---|---|---|
| A2 | **Dual-Franka rig** (`robolab/robots/bimanual_franka.py`, asset `assets/robots/bimanual_franka_robotiq_2f85/`): one articulation, 16-dim joint-position action, per-arm `left_ee_pose` / `right_ee_pose`, tracking wrist cameras, unchanged success predicates (`gripper_name="gripper"` = either hand). | RUNTIME (scripted lift 6/6 clean) | The default two-arm choice. |
| A2 | **Bimanual ViperX (ALOHA) rig** (`robolab/robots/aloha.py`): opposing / station / mobile variants, 14-dim action. | RUNTIME (rig) | No released policy works on it. π0.5 base scored 0/6: coherent reach, no grasp. The limitation is stated in the code and the docs. The rig runs; an ALOHA success rate is a property of the checkpoint. |
| A1 | **VLM pointing connector** (`policies/vlm_pinpoint/`): a pointing-capable VLM returns an image point and a phase, a geometric controller does the metric work. The controller lives in its own package (`vlm-pinpoint`); the connector is ~90 lines and imports it lazily. | RUNTIME (fork) | Same Gemini model: 0/4 driven as Cartesian deltas, 6/6 through this path on `BananaInBowl`; 5/6 on `BananasInCrate`. Top-down, single-object, uncluttered picks only. |
| — | **Bimanual runner** (`policies/bimanual/run.py`): registers the two-arm envs and drives them with a scripted client, so the rigs are verifiable at runtime. Three smoke-test tasks under `robolab/tasks/bimanual/` (outside the benchmark set, `--task-dirs bimanual`). | RUNTIME | Its results show only that the stack runs end to end, and it prints that on every run. |

## Tooling shipped with the fork

| Tool | What it does | Needs |
|---|---|---|
| `scripts/verify_patches.py` | Per recorded run, PASS / FAIL / N/A for each flag change above; `--summary` across runs; exit 1 on FAIL. | h5py for P77 |
| `scripts/reflag.py`, `scripts/flag_regression.py`, `analysis/flag_labels.jsonl` | Replay the post-processable rules over an episode on disk; score against human verdicts. | — |
| `scripts/audit_task_definitions.py` | Every task definition against its scene: objects that do not exist, dead contact sensors, ladders that contradict the success term, checked against the spawn state before asserting a conflict. Over 120 tasks: no real success-vs-ladder conflict (two candidates were withdrawn once the spawn state was read). | usd-core |
| `scripts/check_scene_intersections.py`, `scripts/check_rest_heights.py`, `scripts/find_sinking_objects.py` | Objects authored interpenetrating, objects that drop/rise/roll at reset. | usd-core, h5py |
| `scripts/find_open_hand_carries.py`, `scripts/contact_force_profile.py` | Candidate open-hand carries for the tow analysis, and the pad-force statistics behind P77. | h5py |
| `scripts/friction_sweep_report.py` | The per-condition table of a `--friction` sweep. | h5py (optional) |
| `scripts/find_task_definition_conflicts.py` | The P11 lint, standalone. | — |

## Fixes from the first external trial (2026-08-29)

A reader with no prior context cloned the published repository, ran six tasks (four outside
the set the fork had been verified on) and read the outputs the way a new user would. What
that found, and what changed:

| ID | Change | Level | Evidence |
|---|---|---|---|
| P80 | `episode_results.jsonl` → `events` counts every line of the episode log, ladder lines included. | OFFLINE | Upstream's tally kept a fixed subset of codes: a `BananaInBowl` row showed `GRIPPER_HIT_TABLE`, `OBJECT_GRIPPED`, `OBJECT_RELEASED` while its log held the carry and the credit; a `GrabAFruit` episode read as two releases and no grasp. |
| P81 | `verify_patches.py` P72 decides "container" from the task's own roles (`object=` / `container=` in the run's `env_cfg.json`), falling back to the catalog and name hints only for objects the task does not name. | OFFLINE | `BowlStackingLeftOnRight`: three attempts on the pick target `bowl_2` were reported as attempts on a container, and `--summary` exited 1. |
| P82 | `analysis/check_results.py` runs: `hdf5_path` was never assigned in either code path (upstream since v0.1.0); a cwd-relative path is no longer prefixed with `output/` a second time. | OFFLINE (subprocess test) | `NameError` on every invocation. The "every source must parse" guard cannot catch a name that is only undefined at run time; the script now has a smoke test. |
| P83 | `read_results.py --by-instruction-type` is implemented: success per task and instruction type, pooled per type, `--csv`. | OFFLINE | Documented in `analysis.md`; printed "not yet implemented". |
| P84 | A carry's onset is never stamped earlier than the grip's. | OFFLINE | `OBJECT_CARRIED` at step 99 preceded `OBJECT_GRIPPED` at step 103 in every `BananaInBowl` episode, against the documented order. |
| P85 | `GRIPPER_FULLY_CLOSED` is not reported during the reset warm-up; `reason` for an `any` ladder names the object that progressed furthest. | OFFLINE | "Gripper closed on nothing" at step 2 on `MarkerInMug`; `GrabAFruit`'s reason named the banana while the orange was the object grabbed. |
| P86 | `friction_applied.json` carries a per-object `summary` (coefficients, shape count, whether every shape agrees) above the per-shape rows. | OFFLINE | 733 identical rows for one scene. |
| — | `pyproject.toml`: the CUDA torch index is Linux-only, so `uv sync` resolves on macOS and the no-simulator suite installs on a laptop the documented way. A bare glob (`t*`) in `read_results.py` resolves under `output/` before the cwd. `reflag.py` no longer says contact force is not recorded. README lists the GL/Vulkan packages a bare Linux image needs; `debug.md` lists the benign boot and first-inference messages. | OFFLINE | Reported against the published tree. |

Reported and left as is: the `timing` block is a run-level total repeated per row (upstream
schema); the dashboard's `/thumb` endpoint has no producer (E2); a ladder line is re-emitted
when a regressed rung is credited again (documented in [migration.md](migration.md)).

## Withdrawn, reverted, rejected

Kept with the reason for each.

| ID | What | Why |
|---|---|---|
| P13 | One semantic for `choose K` (terminations use `== K`, subtasks `>= K`). | No corpus episode exhibits the mismatch; parked until one does. The inconsistency stands and is listed in [findings.md](findings.md#known-defects-not-changed). |
| P18, P22, P23, P27 | Camera clean-ups, an HDF5 status overwrite, dashboard CI math, recording at policy resolution. | Parked: no evidence yet (P22), or a display-only change (P23), or a cost decision (P27). |
| P40 | Review viewport behind the robot. | Tried on hardware and rejected: the arm occludes the scene. The mirrored view with the P17 tag is the fix. |
| P41 | `PLACED_WITHOUT_LIFT`. | Retired by P75: four firings, all one artifact, no true positive. |
| P56b | A second 1 cm nudge of two lemons. | Reverted: it ejected 27 unrelated objects at reset. The criterion for a scene edit is the whole scene's event count, including objects other than the one edited. |
| P58, P59 | Two "ladder omits a required object" conflicts from the static audit. | Withdrawn: in both scenes the object already satisfies the relation at spawn; the audit now reads the spawn state before asserting. |
| P63 | Place two loose lemons flat on the table. | Reverted; the objects that keep moving at reset are authored intersecting their basket, and nudging height never converges. Reported as `SCENE_SETTLING` instead. |
| — | Thrashing / stuck-loop flag (B13), a `TARGET_NEVER_CONTACTED` flag, an empty-log summary line, hit-object re-arming. | Discussed and not adopted, or awaiting evidence. |
