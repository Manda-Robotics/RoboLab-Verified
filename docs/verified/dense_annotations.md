# Dense annotations for RoboLab Verified: feasibility, design, and test plan

Status: plan written 2026-09-01; phases A and C implemented the same night, R1/R2/R4 recorded on a pod 2026-09-02 (§9.11); see §9 for what exists and what was measured. The basis is **this fork**, RoboLab Verified
(`Manda-Robotics/RoboLab-Verified`, main `ae33894`, tag `v0.3.1-verified.1`), not upstream RoboLab:
every path below is relative to this repository's root, every rule builds on the fork's event
tracker, recorder and verifier (P-rows in [changes.md](changes.md)), and the work lands as new rows
in that register with a verification level per [verification.md](verification.md). A throwaway
pilot (`analysis/pilots/dense_phases_pilot.py`) ran over the 24 episodes of the first external
trial of this fork (2026-08-29; the run directories are not in the repo, 250 MB) and produced the
numbers quoted below. Every number is from that run unless a file path says otherwise.

The question: can the fork emit a second timeline underneath the flag bar that states, for every
step of every episode, what the robot is doing, using a fixed schema, derived from the simulator
state we already record, without a VLM? Short answer: yes for the manipulation layer, with three
gaps that need a runtime change and one class of things (where the policy is "looking", what it
"intends") that state cannot give and pixels only partly can.

---

## 1. What we have today (measured facts)

### 1.1 The recording

Every episode writes `run_<i>.hdf5` at the control rate, 15 Hz (`sim.dt` 1/120 × `decimation` 8,
`env_cfg.json`). The two mp4s are 15 fps with one frame per control step (t1 env0: 164 steps,
163 frames, 10.87 s), so a step index is a video frame index. Per step, per episode
(`data/demo_<env>/`):

| dataset | shape | what it is |
|---|---|---|
| `actions` | (T, 8) | 7 commanded joint positions + gripper command in [0, 1] (`BinaryJointPositionZeroToOneAction`, 0 = open, 1 = close) |
| `states/articulation/robot/joint_position`, `joint_velocity` | (T, 13) | `panda_joint1..7`, `finger_joint` (column 7, span 0 to π/4), then 4 mimic joints (`robolab/robots/droid.py:83-96`). Joint names are not stored in the file; `scripts/find_open_hand_carries.py` hard-codes column 7 |
| `ee_pose/position`, `orientation`, `linear_velocity`, `angular_velocity` | (T, 3/4/3/3) | pose of `base_link` of the Robotiq (`ee_recorder_bodies: {ee_pose: base_link}`, robot-root frame, which equals env-local for the Franka family), **not the fingertips**. Measured on 4 clean grasps: the held banana sits at `[+0.15, +0.03, 0.00] m` in this frame, so the TCP is 15 cm ahead of the recorded point |
| `states/rigid_object/<obj>/root_pose`, `root_velocity` | (T, 7), (T, 6) | **every** rigid object in the scene, not only the targets (t4 records 14 objects, t6 20) |
| `bbox/bbox_mm/<obj>`, `bbox/centroid/<obj>` | (T, 8, 3) int16, (T, 3) float16 | world-space oriented bounding box corners and centroid, all objects |
| `contact/<obj>` | (T, 2) float32 | contact force of the **left and right inner finger pad** against that object (P77), all objects in `contact_object_list` |
| `contact/<a>__<b>` | (T,) uint8 | object-to-object contact (target↔destination, target↔table) |
| `contact/table` | (T, 2) | pad force against the table |
| `initial_state/cameras/<cam>/position`, `orientation` | (2, 3/4) | the three camera poses at reset (wrist cam moves with the arm; its offset from `base_link` is fixed) |
| `subtask/completed`, `score`, `status` | (E,) | event-indexed, not per step |

`subtask/*` is written only on steps where some env produced a non-empty info string (`subtask_recorder.py:267-268`), so it is **not step-aligned** with `actions` despite `results.py:2875` indexing it as if it were; do not use it as a dense channel.

Not recorded: joint efforts/torques, camera images (`obs/` exists only with `RECORD_IMAGE_DATA=True`; by default the mp4s are the only imagery), per-object pad force **positions**, contact of anything other
than the two inner finger pads (outer fingers, knuckles, wrist, forearm), the object's contact
normal, the per-step boolean of each ladder condition (evaluated every step by
`robolab/eval/gt_state.py::_check_all_subtask_conditions`, never persisted).

Object roles are resolvable offline: `env_cfg.json → terminations.<name>.func/params` carries the
success predicate and its arguments (`object`, `container`, `surface`, `distance`, tolerances);
`subtasks[].conditions` carries the ladder. Multi-target tasks appear as one termination per
candidate (`GrabAFruit`: `success_banana`, `success_banana_01`, `success_apple`, `success_orange`,
each `object_picked_up(object, surface=table, distance=0.05)`). P81 already reads roles this way in
`verify_patches.py`.

### 1.2 The event layer

`log_<run>_env<env>.json` (schema v2) is sparse: 1 to 18 events per trial episode (median 7), each
`{step, detected_step, code, name, info, score}`; `step` is the back-dated onset, `detected_step`
when the rule fired. The vocabulary is transitions (`OBJECT_GRIPPED`, `OBJECT_CARRIED`,
`OBJECT_GRABBED_SUCCESS`, `OBJECT_RELEASED`, `GRASP_ATTEMPT_FAILED`, `GRIPPER_HIT_TABLE`,
`TARGET_OBJECT_BUMPED`, `MULTIPLE_OBJECTS_GRABBED`, `GRIPPER_FULLY_CLOSED`, `SUBTASK_COMPLETED`, …).
The rules live in `robolab/core/task/event_tracker.py` and `robolab/core/task/grasp.py::GraspTracker`
with every threshold in `robolab/constants.py:84-164`. The ones that matter here: a contact is a pad
force > 0.1 N (`world_state.in_contact`); a **carry** is contact for ≥ 3 steps with the object-to-hand
offset drifting < 5 mm while the hand moved ≥ 1 cm (`GRASP_HOLD_S` 0.2, `GRASP_COUPLING_M` 0.005,
`GRASP_HAND_MOVE_M` 0.01); an attempt needs ≥ 30 % closure and a 2 s debounce; release vs drop is
decided by the commanded channel with 0.3 s memory; a bump is a velocity latch at 5 cm/s followed
by ≥ 2 cm displacement; the scene warm-up is 1.0 s (`SETTLE_WARMUP_S`). `GraspTracker` keeps a
per-(object, hand) state (`_PairState`: contact streak, grasped, attempt_closed, contact_start_z,
grip_step, …) that is exactly a per-step in-hand state; it is just never written out.

Between events nothing is said. In t3 env1 the log has two events in 20 s; the robot spent 13 of
those seconds alternating approach and retreat between the two bowls, which is the whole story of
the episode and is invisible in the log.

`scripts/contact_force_profile.py::windows()` already turns events into `(object, kind, start,
end)` spans (carry = `OBJECT_CARRIED` onset → `OBJECT_RELEASED`/`OBJECT_DROPPED`, 2 s windows
around `GRASP_ATTEMPT_FAILED`). That is the only interval logic in the repo and it is not exposed
anywhere.

One more fact with leverage: `offline_tests/test_grasp_tracker.py:32-37` monkeypatches the five
module-level accessors `GraspTracker` reads the world through (`hand_contact`, `hand_position`,
`object_position`, `jaw_offset`, `hand_closed`). Point those at the HDF5 arrays and **the real
tracker, with the real thresholds, runs over a recording with no simulator**. The in-hand layer of
the dense track does not need to be re-implemented; it needs to be replayed and written out per step.

### 1.3 The human layer

- In the private working clone, not in this fork: `analysis/reviews/episode_notes.jsonl`, 840
  reviewed episodes across four policies (runs made on upstream v0.3.1 scoring, Isaac Sim 6.0),
  each with a narrated `quote` and a free-form `modes` list (141 distinct modes; the top 30 are in
  the table in §3). Episode level, no timestamps. 123 quotes mention a time in passing. The
  reviewer is not named in this tree and the notes stay out of it.
- In this fork: `analysis/flag_labels.jsonl`, 27 time-stamped verdicts on individual flags (`t`,
  `flag`, `object`, `verdict`), consumed by `scripts/flag_regression.py`. This is the only
  interval-adjacent gold that exists.
- There is **no** human-labelled dense segmentation of any RoboLab episode. A gold set has to be
  made (§6, phase B).

### 1.4 The dashboard

One hardcoded strip (`dashboard/static/app.js::buildTransport`, 2665-2801; `.events-strip` in
`app.css:813`), children are point markers positioned by `left: time_s / maxTime %`, tooltips are
the native `title` attribute, colour is three-valued severity. No lane abstraction, no range
element. `transport.onTime` is a single assignable callback (claimed by `loadAndRenderEvents` at
2541); a second track must be chained through it, not assigned over it. `buildSubgoals` (2596) is
the established pattern for "consumes events, animates with the playhead". The backend has no
write path of any kind; the private clone's dashboard reads episode-level review annotations from
a read-only JSON keyed `run::task::env`, and that feature was not carried into this fork.

### 1.5 What the two references decided, and what transfers

`robot-episode-labeler` (ours) and the Macrodata post converge on the same contract, arrived at
by measurement rather than taste:

1. The unit is a **completed world-state change** (object becomes held, is released at a new
   place, a lid/door changes state). Approach, retreat, regrasp, hesitation are inside the segment
   they serve, never their own segment. Macrodata: models that split approach/grasp-adjust/retreat
   were the main over-segmentation failure. Ours (ADR-014): merging, not misplacement, was the
   dominant error once that rule was enforced.
2. **Pick and place are two segments.** Every object picked, every placement, its own segment.
3. Segments are ordered, non-overlapping, internally contiguous, start at 0.00, and **stop at the
   last completed manipulation** (ADR-008; a demanded trailing segment made the model invent
   `retract_arm`).
4. A reserved `no_completed_subtask` label, `result = fail`, for stretches in which nothing
   completes. Without it a closed vocabulary fabricates events on exactly the failure-heavy
   episodes an eval customer cares about.
5. Invariants live in code, not in the prompt; confidence is derived from disagreement between
   independent derivations, never self-reported; every automatic correction becomes a visible
   warning.
6. Scoring splits segmentation (greedy IoU ≥ 0.5 F1, boundary tolerance curve at ±0.25/0.5/1/2 s,
   internal boundaries only) from labelling (accuracy on matched pairs), and reports recall by
   segment duration.

What does **not** transfer: the whole boundary-finding problem. Both references spend most of their
effort estimating when the world changed from pixels (F1 0.31 for Macrodata, 0.70 for ours on
WGO-Bench). We have the world state at 15 Hz. Boundaries are a threshold crossing on a recorded
signal, exact to one step, and the open question moves from "where is the boundary" to "which
definition of the boundary do we want", which is a design decision to make once and test against
humans, not a per-episode inference.

The other difference: the user's ask is finer than the references' unit. "What is going on at each
point in time" is the approach/grasp/lift/transport layer the references deliberately fold away.
So we want both layers, and the sim makes the fine layer cheap.

---

## 2. Design

### 2.1 Three tracks, one file

```
L0  flags        existing point events (unchanged)
L1  phases       dense, contiguous, exhaustive: one label per step, 15 Hz, from state
L2  attempts     macrodata-style segments: one per completed (or failed) world-state change,
                 built from L1, carries result/attributes/description
```

L1 is the bar the user asked for. L2 is what makes L1 readable at a glance and what feeds the
per-policy statistics ("mean time from first contact to lift", "attempts per success", "fraction
of episode with the target in hand"). L0 stays as is; every L0 event should fall on an L1 boundary
or inside the L1 phase it names, and where it does not, that is a finding about one of the two.

### 2.2 Per-step channels (the state vector L1 is computed from)

All from the HDF5 plus `env_cfg.json`; no simulator, no `robolab` import. `T` steps, `O` objects.

| channel | formula | notes |
|---|---|---|
| `tcp_pos` (T,3) | `ee.position + R(ee.orientation) @ [0.15, 0.03, 0]` | offset measured on t1; to be replaced by a recorded TCP frame (§5, R1) |
| `tcp_speed` (T,) | `‖∇_t tcp_pos‖`, 3-step central difference | |
| `grip_cmd` (T,) | `actions[:, 7] > 0.5` | commanded, binary |
| `closure` (T,) | `joint_position[:, 7] / (π/4)` | measured closure fraction, as `gripper_fully_closed` and `find_open_hand_carries.py` compute it. The pilot's attempt to find the column by correlation with the command picked mimic joints (8, 10, 11) on different runs; read the layout from the robot cfg, or write `joint_names` as an HDF5 attribute (R1) |
| `pad_f[o]` (T,2) | `contact/<o>` | per pad |
| `touch[o]` (T,) | `(pad_f[o] > f_touch).any()` | one pad; `f_touch` = 0.1 N, the runtime `in_contact` default |
| `pinch[o]` (T,) | `(pad_f[o] > f_touch).all()` | both pads: the only pixel-free definition of "in the jaws" |
| `dist[o]` (T,) | `‖tcp_pos − obj_pos[o]‖` | to centroid; bbox-surface distance is available from `bbox_mm` if centroid distance proves too coarse for large objects (bins) |
| `closing_rate[o]` (T,) | `∇_t dist[o]` | negative = approaching |
| `lift[o]` (T,) | `obj_z[o] − rest_z[o]` | `rest_z` = median z over steps 3..15 (after settle), not step 0 |
| `disp_w[o]` (T,) | `‖obj_pos[o][t] − obj_pos[o][t−8]‖` | 0.53 s windowed displacement. **Use this, not `root_velocity`** (H3) |
| `on_table[o]`, `in_dest[o]` (T,) | `contact/<o>__table`, `contact/<o>__<dest>` | as recorded |
| `held_tracker` (T,) | `GraspTracker.grasped(o)` replayed offline over the recording | the runtime carry definition (coupling + hand motion), with its onset back-dating and tow logic |
| `held_geom` (T,) | `argmax_o pinch[o] ∧ lift[o] > 1 cm`, else none | an independent, purely geometric "what is in hand". Two derivations of the same fact give the disagreement signal the references use for confidence (§2.4) |
| `role[o]` | target / destination / distractor from `env_cfg.terminations` | fixed per episode |
| `in_wrist_view[o]` (T,) | project `bbox_mm[o]` corners into the wrist camera frustum (camera pose = `base_link` pose ∘ recorded wrist offset, intrinsics from the camera cfg) | pixel-free "the target is in the wrist camera's field of view". Not what the policy attends to, but the geometric precondition for it (H9) |

### 2.3 L1 phase vocabulary

Evaluated top to bottom per step; the first matching rule wins. Each phase carries the object it
refers to and that object's role, so the rendered label reads `approach(banana:target)`, and the
colour is by phase family with the role as a hatch/outline.

| family | phase | rule (channels above) |
|---|---|---|
| settle | `settle` | the 1.0 s warm-up the tracker itself observes (`SETTLE_WARMUP_S`), extended while any untouched object still moves (`disp_w` > 2 mm) |
| in-hand | `lift(o)` | `held = o` (where `held` = `held_tracker`, falling back to `held_geom` when the tracker replay is unavailable) and `∇_t obj_z[o] > +3 cm/s` |
| | `lower(o)` | `held = o` and `∇_t obj_z[o] < −3 cm/s` |
| | `transport(o)` | `held = o`, `tcp_speed > 2 cm/s` |
| | `hold(o)` | `held = o`, `tcp_speed ≤ 2 cm/s` |
| | `slip(o)` | `held = o` and `‖v_obj − v_tcp‖ > 5 cm/s` (object moving relative to the hand) |
| grasp | `close_on(o)` | `grip_cmd` closed, `touch[o]`, not `held` |
| | `pinch_no_lift(o)` | `grip_cmd` closed, `pinch[o]`, `lift[o] ≤ 1 cm` (gripped, dragging or pressing) |
| | `open_contact(o)` | `grip_cmd` open, `touch[o]` (nudging, pre-grasp contact, or an open-hand carry = physics artifact if `lift[o] > 1 cm`) |
| | `close_empty` | `grip_cmd` closed, no `touch` on anything, `finger_q` at the closed stop |
| | `release(o)` | first 0.5 s after `grip_cmd` opens while `pinch[o]` was true in the previous step |
| table | `press_table` | `contact/table` pad force > `f_touch`, no object touched |
| free motion | `approach(o)` | `tcp_speed > 2 cm/s`, `closing_rate[o] < −1 cm/s` for the nearest object `o`, `dist[o] < 25 cm` |
| | `reach(o)` | same with `dist[o] ≥ 25 cm` |
| | `retreat` | `tcp_speed > 2 cm/s`, `closing_rate` of the nearest object > +1 cm/s |
| | `reposition` | `tcp_speed > 2 cm/s`, neither approaching nor retreating |
| still | `hover(o)` | `tcp_speed ≤ 2 cm/s`, nearest object within 8 cm |
| | `idle` | `tcp_speed ≤ 2 cm/s`, nothing within 8 cm |
| scene | `disturb(o)` | overrides the free-motion/still families when `disp_w[o] > 1 cm` for an object that is not `held` and not `touch`ed: the arm body pushed it, or it is falling/rolling after a release |

Post-processing, in code, with each correction logged: mode filter over 5 steps (0.33 s), then
absorb runs shorter than 4 steps (0.27 s) into the predecessor, then merge equal neighbours. The
`release` phase is exempt from the minimum length. Nothing is inferred across a gap: every step has
exactly one L1 label, so L1 is exhaustive by construction.

Thresholds (2 cm/s, 1 cm, 8 cm, 25 cm, 0.53 s) are the pilot's first guesses and are the subject of
H4. `f_touch`, the carry constants and the warm-up come from `robolab/constants.py` and are not
re-declared: the pilot used 0.05 N and its own "both pads + lifted" carry, and read `open_contact`
at t2 env3 step 76 where the tracker had already declared `OBJECT_CARRIED` (contact streak plus
coupling, no lift required). Two definitions of "touching" and two of "carrying" is the first thing
the real implementation removes; the geometric derivation stays only as the independent check.

### 2.4 L2 attempt segments

Built from L1 by a small state machine, no new thresholds:

| L2 label | opens when | closes when | `result` |
|---|---|---|---|
| `pick(o)` | first `close_on(o)` or `pinch_no_lift(o)` after a non-in-hand phase | `held = o` for ≥ 0.5 s (`pass`) or `touch[o]` lost with no `held` (`fail`) | |
| `place(o → d)` | `held = o` ends with `release(o)` | object at rest (`disp_w[o] < 2 mm` for 0.5 s) | `pass` if `in_dest[o]` (or the task's own predicate on the final rest pose), `fail` otherwise; `d` is the nearest destination-role object, or `table` |
| `drop(o)` | `held = o` ends without `release(o)` (`grip_cmd` still closed) | object at rest | `fail` |
| `no_completed_subtask` | any stretch ≥ 1 s not inside a `pick`/`place`/`drop` and after the first `approach` | | `fail` |
| (unlabelled tail) | after the last `pick`/`place`/`drop` closes | end of episode | per ADR-008 the tail is not a segment; it is rendered as empty |

Per-segment `attributes` are derived from the L1 phases inside the segment, not judged:
`regrasp` (≥ 2 `close_on` runs inside one `pick`), `wrong_object` (role ≠ target), `press_table`
(any inside), `pinch_no_lift ≥ 1 s` (drag), `slip`, `multi_object` (two objects `pinch`ed),
`disturbed(<other>)`, `open_hand_carry`. `description` is a template over the same facts
("pick banana: 2 closing attempts, 1.1 s pressing the table, lifted at 7.1 s"), not free text.

This is the schema the references arrived at (`start, end, label, result, attributes,
description`). `confidence` is derived, as they do it, from disagreement between independent
derivations: `held_tracker` vs `held_geom` inside a `pick`, the L2 `place` result vs the task's own
predicate on the final frame (or the per-step predicate once R4 records it), and a boundary that
moves by more than 0.5 s when a threshold is halved or doubled (H4). Each disagreement is a flag on
the segment (`held_disagree`, `predicate_disagree`, `boundary_unstable_<x>s`); `provenance` records
which rule fired and with which constants. When the L0 event
log and L2 disagree (an `OBJECT_GRABBED_SUCCESS` with no `pick … pass`, a `pick … pass` with no
grab event) the segment carries `flags: ["disagrees_with_event_log"]`. That disagreement rate is a
metric (H7), and it is where the next batch of tracker patches will come from.

### 2.5 File format

One file per episode next to the log, `phases_<run>_env<env>.json`, schema v1:

```json
{
  "schema_version": 1,
  "dt": 0.0667,
  "task": "BananaInBowlTask", "env_id": 0, "run": 0,
  "annotator": {"name": "robolab.eval.phases", "version": "0.1", "thresholds": {"f_touch": 0.05, "v_move": 0.02, "...": "..."}},
  "roles": {"banana": "target", "bowl": "destination"},
  "phases":   [{"start": 3, "end": 30, "label": "reach", "object": "banana", "role": "target"}, "..."],
  "attempts": [{"start": 98, "end": 125, "label": "pick", "object": "banana", "result": "pass",
                "attributes": ["press_table"], "description": "pick banana: 1 closing attempt, 0.5 s on the table, lifted at 7.1 s",
                "flags": []}, "..."],
  "warnings": ["absorbed 3 runs < 4 steps", "..."]
}
```

Steps, not seconds, are canonical (`time_s = step × dt` is derived, as `/events` does today), so
the file lines up with the HDF5 and the video frame index without a rounding convention. The
dashboard route returns both.

---

## 3. Feasibility, phenomenon by phenomenon

What a reviewer says about an episode, and whether state can say it. Status: **have** (recorded
today), **derive** (computable from what is recorded), **runtime** (needs a change to what is
recorded, no pixels), **pixels** (needs the images), **no** (not observable from either).

| what the reviewer says | reviewer mode (count of 840) | signal | status |
|---|---|---|---|
| goes for the right / wrong object | `correct_object_chosen` 411, `wrong_object_first` 161, `wrong_object_lock_in` 61 | `approach(o)` with `role[o]`; first `close_on(o)` | **derive** |
| never gets a grip / gripper never closes | `grip_never_established` 86, `gripper_never_closes` 45 | no `pinch` on target; `grip_cmd` never closes | **have** |
| drops during transport / recovers after drop | `drops_during_transport` 87, `recovers_after_drop` 28 | `held` ends without `release`; a later `pick … pass` | **derive** |
| test gripping / closes on nothing | `test_gripping` 71 | `close_empty` runs, `grip_cmd` toggles with no `touch` | **have** |
| hovers, no commit / hover equilibrium | `hover_no_commit` 67, `hover_equilibrium` 31 | long `hover(target)` / `approach`↔`retreat` alternation | **derive** (t3 env1: 13 s of it) |
| stuck on / grinding on the table | `stuck_on_table` 45, `grinding_on_table` 24 | `press_table` (t4: 55 to 82 % of the episode) | **have** |
| tips object over / pushes off table | `tips_object_over` 54, `object_pushed_off_table` 19 | `disturb(o)` + object orientation from `root_pose` quaternion; off-table from position vs table bbox | **derive** |
| towed, not grasped / carries with empty gripper | `towed_not_grasped` 20, `carries_with_empty_gripper` 16 | `open_contact(o)` with `lift[o] > 1 cm`, one-pad force profile (P77) | **have** |
| poor grip quality / shaky arm | `poor_grip_quality` 41, `shaky_arm` 13 | `slip`, pad force asymmetry, `‖v_obj − v_tcp‖`; joint velocity spectral energy | **derive** |
| placement incomplete / object lands inaccessible | `placement_incomplete` 34, `object_lands_inaccessible` 21 | `place … fail` with final rest pose vs destination; inaccessible = outside the reachable envelope or under/behind a fixture: needs a reach model | **derive** (partly) |
| regrasp from a new angle / no regrasp | `regrasp_from_new_angle` 12, `no_regrasp_from_new_angle` 54 | `regrasp` attribute + TCP orientation delta between attempts | **derive** |
| ran out of time / slow success | `ran_out_of_time` 59, `slow_success` 24 | L2 timeline vs `episode_length_s` | **have** |
| container edge blocks reach | `container_edge_blocks_reach` 17 | `open_contact(dest)` or `disturb(dest)` while approaching the target inside it | **derive** |
| arm body pushes an object (no pad involved) | (folded into `tips_object_over` etc.) | today only `disp_w` on an untouched object (t5 env2: bin moved 6.5 cm, pad force 0.0); attribution needs contact sensors on the outer fingers, knuckles and wrist | **runtime** (R2) |
| grasp quality: where on the object the pads landed | `hard_to_grasp_object` 16, `unnatural_gripper_pose` 15 | pad contact position on the object surface; needs the contact sensor's contact point, not only its force | **runtime** (R3) |
| object stuck to the gripper | `object_stuck_to_gripper` 12 | `held` with `grip_cmd` open for > 0.5 s after release | **have** |
| target not visible from the wrist cam | (P92: a tow invisible in the wrist view) | `in_wrist_view[o]` from bbox corners and camera pose | **derive** (H9) |
| occluded by another object | `occluded_by_object` 12 | ray from camera to target bbox intersects another bbox | **derive** (coarse; bboxes are boxes) |
| scans / commits without scanning / attention exhausted / incoherent looking | `commits_without_scanning` 30, `scans_before_committing` 12, `attention_exhausted` 46, `incoherent_looking` 35, `insufficient_exploration` 41 | the wrist camera sweep is a proxy (`in_wrist_view` over time, yaw rate of the wrist); what the policy attends to is not observable | **derive** a proxy; the claim itself is **no** |
| task / category not understood, believes task done, success looks accidental | `task_not_understood` 29, `category_not_understood` 30, `believes_task_done` 15, `success_looks_accidental` 17 | inferences over the whole L2 timeline (no attempt on any target-role object; idle after a `place … fail`; a `pass` place with `slip`/`drop` immediately before) | **derive** as named patterns over L2, presented as such, not as phases |
| object class confusion | `object_class_confusion` 42 | first `pick` on a distractor of a related category: needs a category map over the object catalog | **derive** with a small table |
| pomegranate / red onion / red mug attractor | 43, 39, 29 | same as wrong-object with the object named | **derive** |

Reading the table: the top 30 reviewer modes are all in **have** or **derive**. The three
**runtime** items are cheap Isaac Lab config changes (§5). The **pixels** column is empty because
nothing the reviewer said about the *robot* needed pixels; what needed pixels was the reviewer's
own confidence ("it definitely sees it"), and no amount of imagery gives us that either. Pixels
become necessary only for a visual-QA layer (is the rendering broken, P98's unrendered table, the
Robotiq rendering defect on Isaac Sim 6.0), which is a different tool.

---

## 4. Pilot results (24 episodes, first external trial, π0.5)

Script: `analysis/pilots/dense_phases_pilot.py`. Rules as in §2.3 minus `slip`, `release`,
`in_wrist_view`, with `root_velocity` still used for `disturb`. Findings:

1. **Clean episodes segment into the textbook chain with no tuning.** t1 env0 (4/4 success task):
   `settle → reach → approach → hover → close_on → lift → transport → lower → transport → retreat`,
   10 segments, every one of the 6 log events falls in the phase that names it
   (`GRIPPER_HIT_TABLE`, `OBJECT_CARRIED`, `OBJECT_GRIPPED` inside `close_on`; `OBJECT_GRABBED_SUCCESS`
   inside `lift`; `OBJECT_RELEASED` at the end of the last `transport`).
2. **Failure episodes are where the track earns its place.** t3 env1 (0/4): 2 events, 28 phases;
   0 to 13.3 s is `approach(bowl_2) / retreat / approach(bowl_1) / idle` alternating, then one
   `close_on → pinch_no_lift → transport → lift → lower → hold` chain that ends with the bowl held
   1.8 s above the other bowl at the time cap. That is the reviewer's "hover, no commit" then
   "almost finished", stated by the machine. t4 env1: `press_table` covers 55 % of 40 s (82 % in
   env0); the reviewer's "ramming its finger into the table" (P93) is a number now.
3. **`root_velocity` is not a motion signal.** On 149 (object, episode) pairs that no pad ever
   touched, 28 % exceed 2 cm/s at some point after the first second, with zero net displacement:
   PhysX reports 2 to 4 cm/s jitter on resting bagels, forks and plates (`bagel_01` p50 3.8 cm/s,
   max displacement 0.000 m). Windowed displacement (0.53 s, > 1 cm) fires on 5 of 149, and all
   five are real: a bin pushed 6.5 cm by the arm body, knocked glasses, a nudged plate twice, and
   the P82-finding marker sinking 1 cm. Both signals recover 10 of 10 logged `*_BUMPED` events.
4. **The recorded EE frame is not the TCP.** Minimum `‖ee − target‖` at a successful grasp is
   14 to 17 cm. With a fixed `[0.15, 0.03, 0]` offset the pilot's `hover`/`approach` split becomes
   sensible; the offset is a stand-in for a recorded TCP frame (R1).
5. **Fragmentation is real behaviour, not noise, but needs the second layer.** Segment density is
   9 to 14 per 10 s on most episodes and 4 to 7 on the table-pressing ones. Nobody reads 28 segments
   in a 20 s bar; they read "one failed pick after 13 s of indecision" (L2) and open L1 when they
   want to know what the indecision looked like.
6. **The pilot's own thresholds already disagree with the tracker's.** 0.05 N vs 0.1 N for a touch,
   "both pads and lifted" vs "coupled and the hand moved" for a carry: at t2 env3 step 76 the tracker
   declared `OBJECT_CARRIED` where the pilot read `open_contact`. The fix is not a better threshold; it
   is replaying `GraspTracker` offline (§1.2) so there is one definition, and keeping the geometric
   one only as the disagreement check.
7. **20-object scenes break naive "nearest object" attribution.** In t6 (`GrabAFruit`, 20 objects)
   the pilot spent half the episode in `disturb(...)` because of the velocity jitter (fixed by
   finding 3), and its `approach(o)` picked whichever object the TCP happened to be nearest, which
   in a crowded scene is often not where the hand is going. Attribution needs the closing rate
   over the next ~0.5 s toward each candidate (which object the TCP is heading to), not the current
   nearest.

The whole pilot runs in ~3 s per episode on a laptop with `h5py` and `numpy` only.

---

## 5. Runtime changes worth making (none needed for phase A)

| id | change | why | cost |
|---|---|---|---|
| R1 | record a TCP frame (`ee_recorder_bodies: {tcp: <fingertip midpoint>}`; the Robotiq USD has a `tool0`/`tcp` xform, or add one at `base_link + 0.15 m`) and write `joint_names` as an attribute of `states/articulation/robot` | removes the measured offset and its ±1 cm error; makes `dist`, `hover`, `approach` honest; removes the hard-coded column 7 | one cfg line plus one attribute, verify on one run |
| R2 | contact sensors on outer fingers, knuckles, `wrist_3` / `panda_hand` body, with the same per-object filter as the pads | attributes pushes, tips and knock-overs to the robot instead of inferring them from displacement; today a bin pushed 6.5 cm by the forearm has no robot contact on record | new `ContactSensorCfg` entries; the `contact/` group grows by (bodies × objects) columns; VRAM and step-time cost to measure (H10) |
| R3 | record pad contact **points** (Isaac Lab `ContactSensor` exposes `force_matrix_w`; positions need `track_pose` or a PhysX contact report) | grasp location on the object, pad asymmetry, "grasped by the tail" | investigate; possibly a PhysX contact-report subscription rather than the sensor |
| R4 | per-step boolean of every ladder condition (`subtask/conditions` (T, C) uint8, plus the (T,) `grasped`/`attempt_closed` state of `GraspTracker` per object) | L2 `result` from the task's own predicate at every step instead of re-implementing `object_in_container`; also answers P88/P89 (the blinking checkmark, the airborne success) with data | `gt_state.py::_check_all_subtask_conditions` already evaluates every condition every step under `--enable-gt-state` and never persists it; a recorder term around it is ~30 lines |
| R5 | joint efforts (`applied_torque`) | contact with the environment through the arm without a sensor everywhere; stall detection | one recorder term |
| R6 | wrist camera intrinsics and the `base_link → wrist_cam` offset in `env_cfg.json` | `in_wrist_view` without hard-coding | write two numbers |

R1 and R4 first; R2 after H10 says what it costs.

---

## 6. Plan, with hypotheses

Each hypothesis has a metric and a pass line. A failed hypothesis is a result, and its row stays.
Every shipped change gets a P-row in [changes.md](changes.md) marked OFFLINE (phases A to C) or
RUNTIME (phase D) by the rules of [verification.md](verification.md); N/A is not a pass here either.

### Phase A: offline annotator over existing recordings (no GPU)

Deliverable, in order: (1) `robolab/eval/tracker_replay.py`: run `GraspTracker` over one HDF5
demo by binding its five accessors to the recorded arrays (the pattern of
`offline_tests/test_grasp_tracker.py`), emitting the per-step `grasped`/`attempt_closed`/`towed`
state per object and the event list; (2) `robolab/eval/phases.py` (library,
`annotate(run_dir, env) → PhasesFile`, channels of §2.2, rules of §2.3 and §2.4); (3)
`scripts/annotate_phases.py` (CLI over run dirs, writes `phases_*.json`, `--summary` table).
Reads the run directory only, like `verify_patches.py`. Roles via `verify_patches.task_roles()`
(P81), intervals via `contact_force_profile.windows()` for the L0 cross-check. Offline tests over a committed fixture: pick 6 of the 24 trial episodes (HDF5 + log + `env_cfg.json`,
no video; a few MB) and add them under `offline_tests/fixtures/`, the way the fork's other offline
checks avoid a GPU launch.

- **H1 (coverage, by construction).** For every episode, L1 labels cover [0, T) with no gap and
  no overlap; L2 segments are ordered and non-overlapping. Metric: invariant checks in
  `offline_tests/`; pass = 0 violations over the 24 trial episodes and the rc7 sweep (32 × 8 runs).
- **H2 (event consistency).** Every L0 event lands in an L1 phase that names it, per a fixed
  table (`OBJECT_GRIPPED → close_on|pinch_no_lift`, `OBJECT_CARRIED → lift|transport|hold`,
  `OBJECT_RELEASED → release`, `GRIPPER_HIT_TABLE → press_table`, `GRASP_ATTEMPT_FAILED → pick fail`,
  `SUBTASK_COMPLETED → place pass`). Metric: fraction of events consistent. Pass ≥ 0.95 on the trial
  set; every miss is listed and each is either a phase-rule bug or a tracker bug, and the ledger
  gets a row either way.
- **H3 (motion signal).** Windowed displacement (0.53 s, 1 cm) has ≤ 5 % false "moving" on
  never-touched objects across the rc7 sweep (256 episodes, many objects) while recovering ≥ 95 %
  of `*_BUMPED` events; `root_velocity` at any threshold cannot do both. Pilot: 3 % / 100 % vs
  28 % / 100 % on 24 episodes.
- **H4 (threshold stability).** Halving or doubling each threshold (`f_touch`, `v_move`, `lift`,
  `near`, window) changes the L2 segment set (labels + boundaries within ±0.5 s) on ≤ 10 % of
  episodes. Where it changes more, the threshold is doing real work and needs the gold set (H6) to
  fix it. Report as a sensitivity table, one row per threshold.
- **H5 (the replay is faithful).** The offline `GraspTracker` replay reproduces the recorded
  event log: same event names at the same onset steps (±1 step for the pre-step/post-step
  sampling offset) on ≥ 98 % of tracker lines over the trial set and rc7. Where it does not, the
  cause is a quantity the tracker reads live that the recording lacks (the docstring of
  `scripts/reflag.py:8-21` lists the known ones), and that quantity goes on the R-list. This is the
  precondition for everything above: if the replay is not faithful, L1's in-hand family is a
  re-implementation after all.
- **H5b (two derivations agree).** `held_tracker` and `held_geom` agree on ≥ 95 % of in-hand
  steps; the disagreements cluster in tows, one-pad carries and drags (`pinch_no_lift`), which is
  what the flag is for.
- **H7 (disagreement is informative).** Across the five `cli_*_robolab120` corpora (6,000
  episodes, upstream scoring), the rate of `disagrees_with_event_log` flags is reported per task;
  the top-10 tasks by rate overlap with the P89/P90/P96/P97 tasks (stacking family, spoons-in-pot)
  in ≥ 6 of 10. If it does not, either the flag or the ledger is looking at the wrong thing.

### Phase B: a gold set

There is none. 50 episodes, stratified: 10 clean successes, 10 slow successes, 10
"drop / regrasp", 10 "never grips", 10 crowded scenes (≥ 12 objects); at least one per policy;
none from the tuning fixture. Two annotators (the maintainer and the reviewer), the dashboard's episode view
with the L1 track **hidden**, keyboard marks for L2 boundaries and labels only (the references'
schema: `pick/place/drop/no_completed_subtask` + object + result + the attribute rubric in §2.4).
L1 is not human-annotated: nobody can place `approach → hover` to one step from video, and the
references say the fine layer is exactly what humans and models disagree on. L1 is validated
through L2 and H2 instead. Budget: the references measured > 10 min per minute of video for
dense human labels; for L2-only on 20 to 60 s episodes expect 3 to 5 min each, so ~4 h per
annotator.

- **H6 (agreement).** Inter-annotator boundary agreement at ±0.5 s ≥ 0.8 on internal boundaries,
  label agreement ≥ 0.9. If humans are below that, the schema is ambiguous and gets fixed before
  the machine is scored against it.
- **H8 (machine vs gold).** L2 segmentation F1 at IoU 0.5 ≥ 0.85 and boundary recall at ±0.5 s
  ≥ 0.8 against the consensus gold; label accuracy on matched segments ≥ 0.95; recall on segments
  shorter than 2 s ≥ 0.7 (the bucket where the pixel pipelines collapse to 7 %). Reuse the
  labeler's scorers (`src/rel/eval/metrics.py`: greedy IoU matching, boundary tolerance curve,
  recall by duration) by emitting its run-record format. The bar is far above the pixel numbers
  because the boundaries are on recorded state; if we do not clear it, the fault is in the
  definitions, and H4's sensitivity table says which one.

### Phase C: dashboard track

Deliverable: `GET .../phases` (sibling of `/events`, `app.py:468`, same `dt` resolution), a second
lane inside `buildTransport` (the strip becomes a two-row column; playhead moves to the wrapper),
`transport.onTime` converted to a subscriber list, L2 as coloured range blocks with the L1 phases
as a thinner sub-lane that expands on hover or click, `title` tooltips carrying the description and
the rule that fired, click seeks. Cached per file mtime like everything else in the loader.
Optional: a hotkey to write an L2 boundary correction to `analysis/phase_labels.jsonl` (the
first write path in the dashboard; keyed by the 4-tuple `(run_id, task, env_id, run_index)`, `t`
in seconds, mirroring `flag_labels.jsonl`). That is also how Phase B's annotation happens.

- **H11 (readability).** With the L2 lane on, the reviewer's time to answer "what happened in this
  episode" on 10 unseen episodes is ≤ half of the time without it, and the answers agree with the
  narrated notes. Small, subjective, and the point of the whole thing.

### Phase D: runtime additions (GPU)

R1 and R4 on one pod session, verified on `BananaInBowl` and `BowlStackingRightOnLeft` (P89's task).

- **H9 (wrist-view proxy).** `in_wrist_view[target]` computed from bboxes and camera pose agrees
  with a human "is the target in the wrist video" on ≥ 90 % of 100 sampled frames. If it does,
  P92's "tow invisible from the wrist camera" and the `commits_without_scanning` family get a
  measurable proxy; if not, it is dropped and the row says so.
- **H10 (R2 cost).** Adding contact sensors on 4 more robot bodies with per-object filtering
  costs ≤ 10 % step time and ≤ 1 GB VRAM at `num_envs = 4` on `GrabAFruit` (20 objects). If more,
  restrict the filter to target and destination roles.
- **H12 (attribution).** With R2, ≥ 90 % of `disturb(o)` phases longer than 0.3 s get a named
  robot body as cause; the remainder are falls, rolls and settling and are relabelled `settling(o)`.

### Phase E: corpus statistics for the blog and the ledger

Once H8 passes, run `annotate_phases.py` over the five `cli_*_robolab120` corpora and report per
policy: time budget per L1 family (fraction of the episode reaching, hovering, pressing the table,
in-hand), attempts per success, first-contact time, wrong-object-first rate, regrasp rate,
drop-during-transport rate. These are the reviewer's `modes` as numbers over 6,000 episodes instead
of 840 watched ones, and each one is checked against the watched subset first (does
`press_table > 30 %` pick out the episodes tagged `grinding_on_table`, does `drop` pick out
`drops_during_transport`; report precision/recall per mode for the 20 most frequent modes). Those
recordings are Isaac Sim 6.0 + upstream scoring; the annotator does not care, but the physics
findings should be stated against that stack.

---

## 7. Open decisions (the maintainer's call)

1. **Two layers or one.** The plan says L1 dense + L2 attempts. If only one can ship first, ship
   L2 (it is what the references converged on and what the gold set can score); L1 stays as the
   computed substrate and the expandable sub-lane.
2. **Where the phase definitions live.** Proposed: `robolab/eval/phases.py`, sharing constants
   with `robolab/core/events/grasp_tracker` through one `robolab/core/events/constants.py`. This
   touches the tracker; the alternative (a standalone script with its own thresholds) is faster
   and repeats the two-definitions problem of §4 finding 6.
3. **Whether the annotator runs at runtime too.** It can (it is a function of the same tensors
   the tracker sees), and then `episode_results.jsonl` can carry the L2 summary fields
   (`n_attempts`, `first_contact_s`, `time_in_hand_s`, `time_press_table_s`). Offline-first is the
   safer order; runtime is a second step once the definitions hold on the gold set.
4. **Gold-set labour.** ~4 h each for two people. Without it H8 cannot be stated and the track is
   "plausible", which is the status the fork has been avoiding for everything else.
5. **The "attention" family** (`scans`, `attention_exhausted`, `incoherent_looking`) is reported
   as a wrist-view proxy or not at all. Proposed: report the proxy, name it as one, never use the
   reviewer's words for it.

---

## 8. Assumptions to verify before Phase A starts

- The `contact/<obj>` columns are `[gripper_left, gripper_right]` = `[left_inner_finger,
  right_inner_finger]` (`grasp.py::pad_contact_columns`, `PAD_LABELS`); the `contact/<a>__<b>`
  destination columns are chosen by the name heuristic `is_destination()` (`grasp.py:380-384`),
  not by task role, so a bowl that is the pick target still gets a `__bowl` column and a task whose
  destination is not name-matched gets none. L2 `place` must not depend on that column existing.
- `GraspTracker` reads nothing the recording lacks. Known: it reads `env.action_manager.action`
  (recorded as `actions`, pre-step), the pad forces (recorded post-step), `base_link` pose and the
  object poses (recorded). The one-step phase difference between pre-step actions and post-step
  states is the ±1 step in H5.
- `actions[:, 7]` is the gripper channel on every single-arm embodiment (DROID/Franka/Kinova); the
  bimanual rigs have two, and `phases.py` should read the action layout from `env_cfg.actions`.
- Column 7 is `finger_joint` on every single-arm embodiment (true for DROID/Franka per
  `droid.py:83-96`; check Kinova and both bimanual rigs, whose `left_ee_pose`/`right_ee_pose`
  channels `compute_metrics.read_ee_channels` already handles).
- The wrist camera's fixed offset from `base_link` is the same on every task (it is part of the
  robot cfg, not the scene).
- `initial_state/cameras/*` orientation quaternions are `(w, x, y, z)` like the rest of Isaac Lab.
- `bbox_mm` corners are world-frame and updated per step (they are, per shape (T, 8, 3); confirm
  they follow the object and are not the rest-pose box).

---

## 9. Status after the first implementation pass (2026-09-01)

Phase A and phase C are implemented (register rows P100 to P103 in [changes.md](changes.md)).
Phase B is prepared and waits for the two annotators. Phases D and E are not started.

### 9.1 What exists

| piece | where | notes |
|---|---|---|
| tracker replay | `robolab/eval/tracker_replay.py` | the fork's `GraspTracker`, unmodified, over a recording; deviation and step convention documented in the module |
| annotator | `robolab/eval/phases.py` | channels of §2.2, the L1 rules of §2.3, the L2 machine of §2.4, the file of §2.5; tracker constants imported from `robolab/constants.py` |
| CLI | `scripts/annotate_phases.py` | write files, `--summary`, `--check-events`, `--check-replay`, `--show` |
| tests | `offline_tests/test_phases.py`, fixture `offline_tests/fixtures/dense` (t1 and t3 of the trial, 2 MB, exempt from the `*.hdf5` ignore) | 19 tests, 251 in the suite |
| dashboard | `dashboard/app.py` (`/phases`, `/phase_labels`), `dashboard/static/app.js` (`loadAndRenderPhases`, `buildLabelPanel`), `app.css` | three lanes under the strip; labels to `analysis/phase_labels.jsonl` |
| gold set | `scripts/select_gold_episodes.py`, `analysis/gold_set.jsonl` | 50 episodes, deep links to `http://127.0.0.1:8080` |

### 9.2 Hypotheses, measured

Corpus: the 24 trial episodes plus rc3 to rc7 (216 episodes), all π0.5, all with the
`contact/` group; 240 episodes, 2.1 s each on a laptop (the torch replay is the cost; proxy
mode without it is 0.1 s).

| id | result | reading |
|---|---|---|
| H1 | **0** coverage or order violations in 240 episodes | by construction, after two guards the corpus forced: a drop's search for the object coming to rest stops at the next attempt (an onion re-grasped before it settled), and a single step of contact at the last row is not a pick |
| H2 | **98.3 %** (4,285 of 4,361 log events) | remaining misses: pre-P84 carry onsets stamped before the grip (27), attempt lines stamped at a first contact that the phases read as approach or table (27), a drop that the phases read as the object still in the jaws (2) |
| H3 | 3 % vs 28 % false "moving" on 149 untouched (object, episode) pairs; 10 of 10 bumps recovered either way | adopted: `disp_w` for `disturb`; P30's own velocity criterion kept for "at rest" so a place ends where the fork's success confirmation ends |
| H4 | run over 106 episodes (trial, rc7 upstream, rc3), each threshold halved and doubled; table in §9.6 | the physical thresholds are not doing work (`LIFT_M`, `V_LIFT`, `SLIP_V`, `RELEASE_S`, `DISP_M`: 0 to 5 % of episodes change their L2 set). The judgement calls are: the `no_completed_subtask` rule (`NCS_BREAKER_SHARE` 28 to 43 %, `GAP_NCS_S` 20 to 27 %), the approach-break run (18 to 30 %), the TCP "moving" speed (19 to 31 %), the closing-rate sign (12 to 25 %), `PICK_MIN_HOLD_S` (10 to 16 %). Those are what the gold set must pin, and the first narrated episode already voted on the first: the excursions between three failed attempts belong to the attempts, not to a separate segment |
| H5 | **93.5 %** of logged tracker lines at the same detection step ±2; 89.6 % at the same onset ±1 | below the 98 % pass line. The shortfall is one cause: `hand_position` is `base_link` in the replay and the left inner finger live, so an object that shifts while the jaws close keeps a small `rel_dev` live and a larger one here (fewer carries, more failed attempts in the replay: 40 replay-only attempts against 34 log-only carries and 21 log-only drops). Fix is R1 (record the finger body), not a rule change |
| H5b | `held_disagree` flag on 501 of ~1,800 L2 segments (7.6 per episode) | expected in drags and one-pad carries; the flag is doing its job. Not yet audited against video |
| H7 | not run | needs the five `cli_*` corpora annotated in proxy mode (0.1 s each) |

Two findings for the ledger came out of H2 and H3 rather than from the track itself:

- `GRIPPER_FULLY_CLOSED` fires on steps where the phases read `pinch_no_lift` or `close_on`
  (93 cases before the checker was allowed to consult the channel). The event rule requires
  no object in contact via `get_objects_in_contact_with("gripper", contact_object_list)` while
  the recorded pad forces show force on an object at that step. Either the two contact readings
  differ (net force vs the filtered matrix, or an object outside `contact_object_list`) or the
  timing does; one episode should settle it (P66 follow-up).
- P30 confirms success on the recorded `root_velocity` being below 2 cm/s for 0.2 s, and that
  velocity jitters at 2 to 5 cm/s on some resting objects (bagels, forks, plates, `banana_01`).
  A target with that jitter would never confirm. No corpus success is known to have been held
  up by it, but the criterion is one the sweep in H4 should include.

### 9.3 L2 rules that were settled on the trial episodes

Recorded here because they are judgement calls the gold set will test:

- A pick opens on the tracker's `attempt_closed` (jaws at least 30 % closed while touching) or
  on a carry; an open-hand brush is not an attempt. Without this, 771 picks appeared that the
  tracker never counted.
- A pick passes when the tracker's carry lasts `PICK_MIN_HOLD_S` (0.5 s), lifted or not; a drag
  is a pass with the attribute `drag (carried, never lifted)`. The fork's definition of a grasp
  is coupled motion (P31), and `BananaOnPlate` is solved by dragging.
- Contact flicker within `GRASP_ATTEMPT_BURST_S` (2 s) of a parting is not a new attempt
  (P73's rule applied to the segments); failed picks on one object inside a burst fold into one
  with `regrasp xN` (P47's rule).
- The approach before a pick belongs to it, back to a retreat / idle / table / repositioning run
  of at least 0.5 s or an approach toward another object. The stretch before that becomes
  `no_completed_subtask` only when it is at least 5 s long and at least 40 % of it is not
  approaching; a slow reach stays inside the pick (the references' convention), indecision
  does not (t3 env 1: 10.5 s of approach / retreat alternation).
- A place passes when the object is inside the destination (the recorded pair-contact column,
  else the destination's box footprint) once at rest by P30's criterion; a release that has not
  come to rest by the end of the episode is `unknown` (t4 env 1: released 2 steps before the cap).
- The destination of a place is the nearest destination-role object at rest, where roles come
  from `object=` / `container=` / `surface=` / `reference_object=` in the run's `env_cfg.json`.

### 9.4 Labelling protocol for the gold set (phase B, ready)

1. `robolab-dashboard --host 127.0.0.1` with the run directories that hold rc3 to rc7 and the
   `cli_*` corpora as sources; open each link in `analysis/gold_set.jsonl` (`G01` to `G50`,
   already in a random order; 38 π0.5 episodes with pad forces, 12 from the other four policies).
2. Watch the episode once at 2× with the lanes visible; the machine's segments are a prior,
   not the answer. Then label L2 only: scrub to the start of each `pick`, `place`, `drop` or
   `no_completed_subtask`, press `i`, scrub to its end, press `o`, choose the kind, the object,
   the result, type what you saw, `Enter`. The next start is pre-filled with the previous end.
3. Conventions (the references' contract, §1.5): a pick ends when the object is securely held off
   its support (a drag that reaches the goal is a `place` after a `pick` with the note "drag");
   a place ends when the hand lets go and the object has landed (a dropped object: when it first
   rests on something without still falling, not when it stops rolling); approach, hesitation and re-seating belong to the
   segment they serve; failed attempts that end in a success belong to that success; a stretch
   in which nothing completes is `no_completed_subtask`; the tail after the last completed
   manipulation is not labelled; the first segment starts at 0.
4. Write the `annotator` field once; it persists in the browser. Both annotators label all 50
   without seeing each other's marks (the panel shows every mark on the episode, so the second
   annotator uses `ROBOLAB_PHASE_LABELS=analysis/phase_labels_<name>.jsonl`).
5. Scoring (H6, H8): `scripts/score_gold.py --gold analysis/gold_set.jsonl --labels <file 1>
   --labels2 <file 2> --sources <run dirs>` reads the label files and the machine's
   `phases_*.json` (annotating on the fly where none was written), matches segments by IoU ≥ 0.5
   and internal boundaries at ±0.25 / 0.5 / 1 / 2 s, and reports the numbers of §6 phase B: the
   second annotator against the first, then the machine against their consensus (segments both
   drew with the same label). Checked on synthetic labels over the fixture.

### 9.5 First narrated episodes (2026-09-01 evening) and what they changed

Three episodes narrated from the video with exact times (`analysis/phase_labels.jsonl`,
annotator `finn`): `cli_g05 MustardInRightBin` env 0 (three failed attempts, then a pick held to
the end), `cli_cosmos3 BananaOnPlate` env 1, `cli_cosmos3 BananasInCrate` env 8. All three are
recordings without the contact group. Against them the machine scored recall 1.0, label and
result accuracy 1.0, every boundary within ±0.5 s; F1 0.80, then 0.92 after two fixes the
episodes forced:

- the distance-only proxy touch read "touch" for 6.2 s of hovering next to the mustard; with
  evidence of interaction required (jaws part-closed, a close command, or the object moving) it
  reads 0.6 s outside the narrated contact windows and misses 0.3 of 7.9 s inside;
- the in-hand channel on such recordings came from geometry and a distance blink ended a carry
  (a spurious `place` where the reviewer saw the object held to the end); `proxy_pads()` now
  builds synthetic pad columns and the real tracker replays on them, so carries are coupled
  motion there too. A table proxy (fingertips within 1.5 cm above the top) gives `press_table`
  on those recordings as well.

The remaining machine-only segments are `place` after a lift the reviewer stopped narrating at;
in one of them the gripper opens on the final step. Narrating to the end of the episode is now
part of the protocol.

The gold set is split 30 tune / 20 test (seeded, `split` field): rules and thresholds change
against tune-split disagreements only; the 20 test episodes are labelled once and H8 is
reported on them. `analysis/gold_set.jsonl` also carries the machine's flag count per episode,
which orders the tune split by expected information for the next labelling batch.

### 9.6 H4 sensitivity table

106 episodes (trial, rc7 upstream, rc3). A changed L2 means a label, a result, or a boundary
moved by more than 0.5 s. `CONTACT_PROXY_M` applies only to recordings without the contact group
(none in this set).

| threshold | default | factor | value | episodes with a changed L2 | L1 steps changed | mean boundary shift (s) | n |
|---|---|---|---|---|---|---|---|
| `V_MOVE` | 0.02 | 0.5 | 0.01 | 18.9% | 7.6% | 0.08 | 106 |
| `V_MOVE` | 0.02 | 2 | 0.04 | 31.1% | 13.4% | 0.21 | 106 |
| `V_LIFT` | 0.03 | 0.5 | 0.015 | 0.0% | 3.7% | 0.00 | 106 |
| `V_LIFT` | 0.03 | 2 | 0.06 | 0.0% | 3.7% | 0.00 | 106 |
| `LIFT_M` | 0.01 | 0.5 | 0.005 | 0.0% | 0.0% | 0.00 | 106 |
| `LIFT_M` | 0.01 | 2 | 0.02 | 0.0% | 0.0% | 0.00 | 106 |
| `NEAR_M` | 0.08 | 0.5 | 0.04 | 11.3% | 2.2% | 0.03 | 106 |
| `NEAR_M` | 0.08 | 2 | 0.16 | 2.8% | 2.3% | 0.00 | 106 |
| `APPROACH_M` | 0.25 | 0.5 | 0.125 | 2.8% | 9.9% | 0.00 | 106 |
| `APPROACH_M` | 0.25 | 2 | 0.5 | 0.0% | 3.4% | 0.00 | 106 |
| `CLOSING_RATE` | 0.01 | 0.5 | 0.005 | 12.3% | 2.6% | 0.03 | 106 |
| `CLOSING_RATE` | 0.01 | 2 | 0.02 | 24.5% | 5.7% | 0.02 | 106 |
| `DISP_M` | 0.01 | 0.5 | 0.005 | 3.8% | 0.9% | 0.00 | 106 |
| `DISP_M` | 0.01 | 2 | 0.02 | 4.7% | 0.7% | 0.00 | 106 |
| `SLIP_V` | 0.05 | 0.5 | 0.025 | 0.0% | 0.3% | 0.00 | 106 |
| `SLIP_V` | 0.05 | 2 | 0.1 | 0.0% | 0.1% | 0.00 | 106 |
| `SMOOTH_W` | 5 | 0.5 | 2 | 3.8% | 2.3% | 0.00 | 106 |
| `SMOOTH_W` | 5 | 2 | 10 | 14.2% | 8.9% | 0.01 | 106 |
| `MIN_RUN` | 4 | 0.5 | 2 | 10.4% | 0.0% | 0.01 | 106 |
| `MIN_RUN` | 4 | 2 | 8 | 17.0% | 0.0% | 0.02 | 106 |
| `RELEASE_S` | 0.5 | 0.5 | 0.25 | 0.9% | 0.7% | 0.00 | 106 |
| `RELEASE_S` | 0.5 | 2 | 1.0 | 0.9% | 1.0% | 0.00 | 106 |
| `GAP_NCS_S` | 5.0 | 0.5 | 2.5 | 19.8% | 0.0% | 0.00 | 106 |
| `GAP_NCS_S` | 5.0 | 2 | 10.0 | 27.4% | 0.0% | 0.00 | 106 |
| `NCS_BREAKER_SHARE` | 0.4 | 0.5 | 0.2 | 28.3% | 0.0% | 0.00 | 106 |
| `NCS_BREAKER_SHARE` | 0.4 | 2 | 0.8 | 43.4% | 0.0% | 0.00 | 106 |
| `PICK_MIN_HOLD_S` | 0.5 | 0.5 | 0.25 | 10.4% | 0.0% | 0.07 | 106 |
| `PICK_MIN_HOLD_S` | 0.5 | 2 | 1.0 | 16.0% | 0.0% | 0.12 | 106 |
| `BREAK_RUN_S` | 0.5 | 0.5 | 0.25 | 17.9% | 0.0% | 0.09 | 106 |
| `BREAK_RUN_S` | 0.5 | 2 | 1.0 | 30.2% | 0.0% | 0.06 | 106 |
| `CONTACT_PROXY_M` | 0.02 | 0.5 | 0.01 | 0.0% | 0.0% | 0.00 | 1 |
| `CONTACT_PROXY_M` | 0.02 | 2 | 0.04 | 0.0% | 0.0% | 0.00 | 1 |

### 9.7 Reviewer modes against the phase statistics (phase E, first pass)

1,153 reviewed episodes (the private notes; four policies plus π0.5, all recordings without the
contact group, so proxy mode throughout), each annotated and reduced to a few numbers; one
fixed rule per reviewer mode (`scripts/phase_mode_check.py`). Precision is "the reviewer
tagged it when the rule fired", recall "the rule fired when the reviewer tagged it".

| reviewer mode | n | machine rule | precision | recall | rule fires |
|---|---|---|---|---|---|
| `grinding_on_table` | 46 | press_table_share >= 0.25 | 0.32 | 0.59 | 84 |
| `stuck_on_table` | 60 | press_table_share >= 0.15 | 0.30 | 0.65 | 130 |
| `grip_never_established` | 105 | no pick passed | 0.19 | 0.43 | 231 |
| `gripper_never_closes` | 51 | no closing on anything | 0.10 | 0.16 | 79 |
| `drops_during_transport` | 153 | a drop segment | 0.38 | 0.55 | 221 |
| `recovers_after_drop` | 66 | drop then a later pick pass | 0.17 | 0.38 | 151 |
| `wrong_object_first` | 217 | first pick on a non-target | 0.48 | 0.77 | 345 |
| `wrong_object_lock_in` | 91 | >= 2 picks on non-targets | 0.15 | 0.78 | 467 |
| `test_gripping` | 92 | close_empty >= 2 runs | 0.05 | 0.20 | 367 |
| `hover_no_commit` | 98 | still share >= 0.15 and no pick pass | 0.27 | 0.33 | 118 |
| `no_carry_attempt` | 91 | no pick at all | 0.25 | 0.22 | 80 |
| `target_never_attempted` | 85 | no pick on a target | 0.26 | 0.80 | 260 |
| `clean_first_try` | 108 | one pick, passed, no fail, no drop | 0.41 | 0.42 | 110 |
| `clean_transport` | 62 | pick pass and place pass, no drop | 0.05 | 0.19 | 247 |
| `towed_not_grasped` | 27 | towed / open-hand carry attribute | 0.03 | 0.70 | 577 |
| `carries_with_empty_gripper` | 16 | close_empty >= 1.0 s | 0.03 | 0.75 | 460 |
| `multi_object_progress` | 183 | >= 2 place passes | 0.58 | 0.43 | 135 |
| `no_regrasp_from_new_angle` | 64 | >= 3 failed picks on one object | 0.08 | 0.50 | 389 |
| `regrasp_from_new_angle` | 19 | pick pass after >= 1 failed pick | 0.02 | 0.74 | 606 |
| `ran_out_of_time` | 223 | still in hand / no place pass at the cap | 0.16 | 0.42 | 596 |
| `tips_object_over` | 67 | a disturb phase >= 0.5 s | 0.09 | 0.73 | 549 |
| `placement_incomplete` | 38 | a place that failed | 0.05 | 0.63 | 518 |

Reading it honestly:

- Recall is usable on the manipulation modes (wrong object first 0.77, target never attempted
  0.80, stuck or grinding on the table 0.6, drops 0.55); precision is low everywhere. Two
  reasons, and they are separable. The reviewer's tags are not exhaustive: he names what was
  salient in an episode, so "the rule fired and he did not say it" is not always a false
  positive. And proxy mode over-fires: the rules that fire on half the corpus (`towed`,
  `carries_with_empty_gripper`, `regrasp`, `tips_object_over`, `placement_incomplete`) are the
  ones that need pad forces or a settled place result, which these recordings cannot give.
  The two artifact attributes (`open_hand_carry`, `towed`) are now emitted only with real pads.
- What this says for the blogpost: per-policy rates of "first pick on a non-target",
  "drop after a pick", "time pressing the table" and "attempts per success" can be reported at
  corpus scale with the caveat stated; "clean transport", "towing" and "placement incomplete"
  cannot, until the corpus is re-run under this fork's recorder (pad forces, pair contact) or
  the tune loop has pinned the place rule.
- The per-episode features are regenerated by the script (`--features`) for a per-policy table
  once the rules are trusted.

### 9.8 Iteration 1 on the tune split (2026-09-02)

Eleven episodes narrated by the maintainer (G17, G03, G37, G46, G42, G12, G13, G29 and the
three of the evening before; 50 L2 segments; `analysis/phase_labels.jsonl`, annotator `finn`,
`gold_id` on each row). Every rule change below is traceable to one narrated moment, is in one
commit on the `dense-annotations` branch, and was checked against the 240-episode tripwire
(H1 0 violations, H2 97.1 % after; 98.3 % before, the difference being attempt lines that now
fall one segment earlier because of the two folding rules).

| narrated moment | machine before | rule now |
|---|---|---|
| can bumps the bin, gripper forced open, can lands in the bin: "place pass" (G17) | `drop fail` | no `drop` label: `place` with attribute "dropped (gripper opened with the close command still on)", result = outcome |
| 7 s hovering and circling over the target before the grip (G17) | 18 s `no_completed_subtask` | hover and `reposition` over the object then picked belong to its pick; only retreat / idle / press_table / close_empty / disturb break the approach |
| off the ground, dropped at once: "pick pass" (G17, G03) | `pick fail` | `PICK_MIN_HOLD_S` 0.5 → 0.25 s |
| fingers on the bin floor while lining up (G03) | `pick pass grey_bin` | a container is picked only when lifted; touching it opens no attempt |
| banana out of the bin onto the table: "place pass" (G03) | `place fail` | `object_outside_of` terminations: the container is the origin, the table the destination (`task_origins`) |
| final tuna pick 90.7 to 93.6 s (G17) | missed | recording gap: rc3's P62-era boolean pads read `[0, 0]` with the can 20 cm up between jaws closed at 0.6; the proxy fills in only while an object hangs lifted next to closed jaws with the close command on |
| brief close that ends in the successful pick 2 s later (G46) | two picks | a failed close within `GRASP_ATTEMPT_BURST_S` of a passing pick folds into it (the references' rule) |
| bottle still held at the cap, already inside the pail (G46) | `carry unknown` | `place unknown` with "still held at the end, inside the destination" |
| 30 s idle and pressing the table after the only attempt (G12) | nothing (tail) | a trailing stretch ≥ `GAP_NCS_S` that is mostly retreat / idle / table is `no_completed_subtask` |

H8 on the eleven, before and after: F1 0.71 → 0.81, label accuracy 0.79 → 0.95, result
accuracy 0.71 → 0.88, boundaries within ±0.5 s 55 → 65 %, within ±1 s 62 → 75 %. Recall by the
reviewer's segment length: 4 to 8 s 0.96, over 8 s 0.89, 2 to 4 s 0.69, 1 to 2 s 0.0. The
remaining error class is one thing: how fast a carry is seen to begin and end (quick
drop-then-regrasp), which on pads recordings is the finger-position gap (R1) and on proxy
recordings the composite touch. Rule tuning on it should stop here; the pod fixes it.

Two findings for the ledger from this iteration: the P62-era boolean contact missed a whole
grip (so the live tracker missed that carry too), and `cli_pi05_robolab120` has video for 240
of its 1,200 episodes (the rerun has all; the gold set now points at the rerun).

### 9.11 Pod session (2026-09-02): R1, R2 and R4 recorded, the replay is exact

One A40 pod session (about 3 h, $0.44/h), π0.5 jointpos, 4 envs × 2 runs per task, six tasks
(BananaInBowl, BowlStackingRightOnLeft, GrabAFruit twice, MustardInRightBin, FoodPacking2Cans,
BananasInCrate), recorded with the fork's recorder plus three new terms
(`robolab/core/events/dense_recorders.py`, register rows P104 to P106). Runs and logs in
`../TESTING/dense_output/d*`. Checked with `scripts/check_dense_channels.py <run dir>`.

| channel | what it settles |
|---|---|
| `finger_left` / `finger_right` (R1) | the replayed `GraspTracker` reads the same finger body as the live one. Replay vs the recorded live state (`tracker/<obj>`): **100 %** of object-steps agree in all three checked tasks (3,782 / 4,534 / 68,400), carry and attempt onsets exact. H5 is no longer a number to chase: the in-hand layer of the phases is the live tracker's own state, whether replayed or read from the file |
| `tracker/<obj>` (R4) | the per-step `grasped` / `attempt_closed` the plan asked for in §1.2; with it on disk the replay is a check, not a dependency |
| `conditions/*` + `conditions_legend.json` (R4) | every ladder predicate every step. The final rung first reads true on the step the log stamps `SUBTASK_COMPLETED` (16 / 16 with an outcome, lag 0). L2 `place` results can now be checked against the task's own predicate per step, and the `predicate_disagree` flag of §2.4 has its source |
| `contact_body/<label>__<obj>` (R2) | seven robot bodies beyond the pads. No measurable step cost on GrabAFruit (0.75 vs 0.73 it/s). In BananaInBowl the inner knuckles touch an object on 78 of 1,891 steps, 10 of them with no pad on anything: pushes the pads could not attribute |
| `robot_joint_names` in `env_cfg.json` (R1) | the closure column found by name; the hard-coded column 7 stays as the fallback |

Per task (4 envs × 2 runs each; "state" = replayed tracker vs the recorded live state over
every object-step; "lines" = replayed tracker lines vs the log at the same detection step ±2;
"ladder" = episodes where the final rung's first true step equals the log's
`SUBTASK_COMPLETED` step, or both are absent; "pad-free" = steps on which an extra robot body
touches an object while neither pad touches anything):

| task | success | steps | state | lines | ladder | H2 | pad-free body contact |
|---|---|---|---|---|---|---|---|
| BananaInBowl | 8/8 | 1,891 | 3,782 / 3,782 | 32 / 33 | 8 / 8 | 59 / 59 | 10 (inner knuckle) |
| BowlStackingRightOnLeft | 2/8 | 2,267 | 4,534 / 4,534 | 23 / 24 | 8 / 8 | 26 / 27 | 0 |
| GrabAFruit (20 objects) | 0/8 | 3,600 | 68,400 / 68,400 | 5 / 5 | 8 / 8 | 14 / 15 | 0 |
| MustardInRightBin | 3/8 | 2,692 | 8,076 / 8,076 | 32 / 32 | 8 / 8 | 55 / 56 | 5 (outer finger) |
| FoodPacking2Cans (180 s) | 2/8 | 18,217 | 109,300 / 109,302 | 157 / 160 | 8 / 8 | 326 / 334 | 403 (347 on the left outer finger) |
| BananasInCrate | 3/8 | 5,793 | 34,758 / 34,758 | 127 / 128 | 8 / 8 | 194 / 199 | 169 (knuckles and outer finger) |

Every carry onset agrees to the step (0 of 88 carries differ); attempt onsets differ by at
most one step. The eleven unmatched lines in 382 are release-vs-drop readings and attempt
lines one debounce apart, not state. The remaining H2 misses are the known class:
`OBJECT_CARRIED` back-dated into an `open_contact` phase (the tracker's onset precedes the
closure the phase reads).

Findings that change the plan:

- **The finger bodies are not a TCP.** Both inner finger link origins coincide with `base_link`
  when open and part by ~4.6 cm as the jaws close (the pad mesh is 9 to 12 cm out along the
  link's x). A least-squares fit over 429 carried steps places the held banana at
  `base_link + (0.151, 0.030, -0.013)` with a 1.5 cm median residual, in the finger frames no
  better. So the measured `TCP_OFFSET` (0.15, 0.03, 0) was right, it stays, and the "record a
  TCP frame" half of R1 is closed as unnecessary. The finger channels exist for the replay.
- **The P62 pad-fill was extending carries past real drops on force recordings.** The two
  `GRIPPER_FULLY_CLOSED in transport` misses of §9.2 are, on the new recordings, steps where the
  pads read 0 N and the live tracker had already dropped the object; the annotator's
  "lifted next to closed jaws with no pad contact is in the hand" fill (for rc3's boolean
  pads) kept the carry alive for 3 rows. Restricted to boolean-pad recordings: H2 on the
  240-episode tripwire 97.1 → **98.1 %** (4,276 / 4,361), H1 still 0.
- The remaining replay-vs-log difference is one line in 66: a `RELEASED` where the log says
  `DROPPED` at the same step (d1 run 1 env 1, step 426). The recorded tracker state agrees
  with the replay there; the difference is in the event tracker's release-vs-drop reading
  of the command history, not in the state. Open, small.
- **Arm pushes are now on record.** On FoodPacking2Cans the left outer finger touches an object
  on 412 steps, 347 of them with no pad on anything; on BananasInCrate the knuckles and outer
  finger add 169 such steps. Those are the `disturb(o)` phases the plan could only attribute by
  displacement (H12 has its data; the attribution rule is not written yet).
- What did not happen: the aggregate `conditions/s<i>` of the first two pod tasks used "all
  rungs at once" and is always 0 (the per-condition columns are complete and the check
  script recomputes the aggregate from the terminal rungs; fixed for later tasks).

What this means for the loop in §9.8: the error class that iteration 1 stopped at (how fast
a carry is seen to begin and end) was the replay's finger deviation, and on recordings made
with these terms it is gone by construction. Rule tuning against the gold set resumes on
episodes recorded this way; the rc3 to rc7 gold episodes keep the deviation and their H8 is
a lower bound.

### 9.8b Iteration 2: first review-mode episode (2026-09-02, d5 FoodPacking2Cans run 0 env 0)

Eight machine segments reviewed (P108): label right 7, result right 8, bounds right 5. What the
three bounds verdicts said, against the recording:

| reviewed moment | machine | data | rule now |
|---|---|---|---|
| second pick of the soup can: "a clear pick at 18.3, 20.3 is mid carry" | pick closes 20.3 s | both pads on the can from 17.4 s, lifted 6 cm at 18.2, 20 cm at 19.4; the live tracker's `grasped` only from 20.0 s. Its carry needs the object-to-hand offset to drift < 5 mm per 0.2 s in world coordinates, and the offset moved 6 to 12 mm per 0.2 s while the wrist turned with the can held | `HELD_LIFT_M`: pinched, commanded closed and more than 3 cm above rest is in hand whatever the coupling says (the tracker keeps its `held_disagree` flag). Pick now closes at 18.3 s |
| drop of the soup can: "the drop already happened at 6.4; after that it is still in motion" | place/drop closes at rest, 7.7 s | leaves the jaws 6.0 s, lands 6.3 s, rolls until 7.2 s, at rest by P30 at 7.7 s | **decided (delegated): a place or drop ends when the object lands**, the first support contact after leaving the hand while no longer falling faster than 5 cm/s (`LANDING_VZ`; a bounce off a bin wall on the way down is not a landing). The result is still read at rest. Now 6.1 s |
| 8.5 s between the place and the tuna pick: "it is definitely travelling to the tuna can, a bit distracted once there" (marked unsure) | `no_completed_subtask` (approach 4.1 s, retreat 3.1 s) | one object approached throughout, with retreats | **tried and reverted**: with retreats neutral, rc3 FoodPacking env 2's 28 s of ramming the bin (narrated `no_completed_subtask`) folds into one 33 s pick and H8 over the 12 labelled episodes falls (F1 0.883 → 0.864; counting open-hand contact with the bin as a breaker made it 0.860). The old rule stays, behind `RETREAT_BREAKS_APPROACH`; the vote is recorded and waits for more verdicts |

H8 over the 12 labelled episodes, before and after the `HELD_LIFT_M` rule: F1 0.833 → 0.883,
boundaries within ±0.5 s 59 → 67 %, label accuracy 0.96, result accuracy 0.94. With the landing
rule on top: F1 0.850, ±0.5 s 63 %, label 0.96, result 0.94; the difference is two narrated drops
in rc5 BananasOutOfBin env 3 (10.2 to 11.3 s and 36.0 to 38.0 s) whose ends were narrated at rest
under the old convention and now fall below IoU 0.5 against a 0.5 s landing segment: convention
lag in the gold, to be re-judged in review mode, not a rule error. Tripwire after `HELD_LIFT_M`:
H1 0, H2 98.1 → 97.8 % (13 carry lines now stamped after the phase already reads in-hand, which
is the finding below, not a regression); with the landing rule: H1 0, H2 97.8 %, unchanged.

Finding for the ledger: the tracker's carry test compares the object-to-hand offset in world
coordinates, so a held object whose wrist rotates drifts by the lever arm and is "not carried".
On the 240-episode corpus 168 passed picks (was 110) now carry `no OBJECT_CARRIED in log`: the
phases see a pinched, lifted object where the live tracker never issued the carry line. The fix
is the offset in the hand frame (a tracker change, next pod session); until then the phases'
override and the flag carry it.

### 9.9 Handoff

- Branch `dense-annotations` in this clone, never pushed; `main` equals `origin/main`. The
  maintainer does not want any of this on the public fork yet.
- Run things with `/usr/local/bin/python3.12` (torch, h5py, fastapi); the repo venv lacks torch.
  `python3.12 -m pytest offline_tests` (251 tests). Dashboard for labelling:
  `python3.12 -m dashboard.cli --host 127.0.0.1 --port 1880`.
- Labels: `analysis/phase_labels.jsonl` (transcribed from the maintainer's audio; `gold_id`
  per row). Score: `scripts/score_gold.py --gold analysis/gold_set.jsonl --labels
  analysis/phase_labels.jsonl --sources ../RoboLab/output`. Tripwire after any rule change:
  `scripts/annotate_phases.py --summary --check-events` over the trial and rc3 to rc7 runs
  (`../TESTING/trial_output/t?_*`, `../RoboLab/output/rc*`, ~9 min).
- Gold set: `analysis/gold_set.jsonl`, `split` tune/test; labelled so far: G17 G03 G37 G46
  G42 G12 G13 G29 (all tune). Do not tune on the test split. Next to label, other policies
  first: G41, G50, G10, G36; then G15, G21, G20, G07, G06, G01.
- Convention decisions taken so far are in §9.3 and the table above; the protocol for narrating
  is §9.4 (narrate through to where the object ends up).
- Pod recordings with the R1/R2/R4 channels: `../TESTING/dense_output/d1..d6` (48 π0.5 episodes,
  six tasks); `scripts/check_dense_channels.py` for the replay / ladder / body-contact checks.
  Pod recipe (fresh A40, ~15 min to first run) in the session memory; the pod is terminated.
- The private notes corpus and its mode check: `scripts/phase_mode_check.py --notes
  ../RoboLab/analysis/reviews/episode_notes.jsonl --sources ../RoboLab/output` (about an hour).

### 9.10 Not done, in order

1. Continue labelling the tune split (order in §9.9), score, triage; the second annotator into a separate file for H6; then one read of the test split for the reported H8.
2. Add the two findings above to the ledger (P62 booleans missing a grip; pi05 run without videos).
2. ~~R1 on a pod~~ done (§9.11): R1, R2, R4 recorded, replay exact. Next runtime step: the
   annotator on the run path so `phases_*.json` is written next to each log by default, and the
   L2 summary fields into `episode_results.jsonl` (open decision 3).
3. Annotate the five `cli_*` corpora in proxy mode and report H7 and the phase-E statistics
   against the reviewer's modes.
4. The `GRIPPER_FULLY_CLOSED` versus pad-force discrepancy on one episode.
