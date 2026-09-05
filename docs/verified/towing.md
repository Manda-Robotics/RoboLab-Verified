# The towing artifact

The object that rides on one finger with the hand open. Reviewers called it "magnetic", "stuck to a
finger", "physically impossible", "the gripper opens but it stays". This page records what it is,
how often it happened in the recorded corpora, how RoboLab Verified flags it, and the fix that
removed it in a controlled replay. Register rows: P43 (the first runtime rule), P124 (this page).

## 1. What it is

The Robotiq 2F-85 pad collider (`Defeatured_2F_85_PAD_OPEN_fingertipsstep_01` in
`assets/robots/franka_robotiq_2f_85_flattened.usd`) is a convex decomposition of a mesh that is not
watertight, with `shrinkWrap` off. At the fingertip it is a **6 mm slab** (42.5 to 48.5 mm off the
jaw axis when open) with air behind it; the finger body begins 4.5 mm further out and only over the
rear third of the pad. An object feature (a rim, a wall, a box edge, a stem, a tail) pressed more
than about 3 mm into that slab is depenetrated out of its *back* face, the nearest exit, and is
then hooked behind the pad. The pad itself blocks the way back. Opening the hand does not release
it; the object slips off when the pad edge passes it.

This needs a finger that cannot yield, and the recorded data say so: in every reviewed tow the
`finger_joint` sits at a hard limit, 0.000 (open, ~90 %) or 0.785 (closed on nothing). Mid-range,
the drive is compliant and the penetration stays below 1 mm. Friction plays no part, which is why
the P79 sweep found tows at mu 0.5 as at 2.0.

Evidence, all from recorded state (`states/rigid_object/*/root_pose`, `ee_pose`, the 13 joint
angles) plus the object collision meshes and a forward kinematics of the gripper validated to
0.00 mm against recorded finger poses:

| observation | value |
|---|---|
| rigid coupling of object to hand during a tow | rel-pose std 0.02 to 0.06 mm |
| `finger_joint` during a tow | exactly 0.000 or 0.785 in every reviewed case |
| object depth inside the pad's collision box during a tow | 3 to 9 mm (through the slab) |
| the same depth in genuine carries (25 sampled) | at most 0.7 mm, both pads |
| two-pad open-limit pinch (tuna can, 85.5 mm = the aperture) | 0.0 to 0.3 mm: not a tow |
| pad force in an instrumented tow (rc7 marker) | one pad, steady 6.8 N, object centre at the pad plane |
| onset | one 3 Hz sample: 0 to 6.6 mm |

Which objects hook (tier-A tows per episode in which a pad touched the object, all six corpora):
mugs 10 %, bowls 9 %, plates 11 %, glasses 9 %, remote 8 %, mouse 8 %, pudding box 16 %; blocks 0 %,
apple 0 %, bottles 0 %, bananas 1 %. Thin features tunnel, fat convex bodies do not.

**Replay test.** `examples/run_recorded.py --env-config current --num_envs 4` replays the recorded
initial state and actions of gr00t `GrabAFruitTask` env 1 (the reviewer's "the gripper is towing a
banana"). With the shipped asset the banana hooks in 4 of 4 envs at the recording's time (right pad
depth 8 to 9.6 mm from t = 13 s to the end, banana carried at 8 to 10 cm). With
`ROBOLAB_ROBOT_USD=assets/robots/franka_robotiq_2f_85_solidpad.usd`, the same asset with the pad
collider replaced by its bounding box and the finger body by its convex hull, the hook does not form
in any env: the contact reaches 1.6 to 2.4 mm at t = 12 to 13 s, separates by t = 14 to 19 s, and
the banana stays on the table. One episode, open-loop; the solid pad is not the default until a
normal run set confirms grasping is unchanged (the contact face is the same, the box is 19 mm deep
instead of 6).

## 2. How common it was

All six `cli_*_robolab120` corpora (120 tasks, 10 envs each, 7 186 episodes), scored offline with
the detector below. Tier A is the flag (section 3); A or B adds shorter or shallower hooks.

| policy | episodes | tier A | tier A or B |
|---|---|---|---|
| gr00t | 1 200 | 125 (10 %) | 245 (20 %) |
| pi05_rerun | 1 200 | 118 (10 %) | 188 (16 %) |
| pi05 | 1 200 | 99 (8 %) | 162 (14 %) |
| g05 | 1 198 | 83 (7 %) | 171 (14 %) |
| cosmos3 | 1 200 | 89 (7 %) | 151 (13 %) |
| molmoact2 | 1 188 | 49 (4 %) | 112 (9 %) |
| **all** | **7 186** | **563 (7.8 %)** | **1 029 (14.3 %)** |

- Tier-A episodes succeed 7.6 % of the time; the others 21.7 %. A tow usually ends the episode's
  useful life: median hook 11 s, mean 30 s, 34 % last to the end.
- Tasks with thin-walled objects run at 20 to 33 %: SpoonsInPot 33 %, ReorientAllMugs 28 %,
  PutTwoMugsOnShelf 28 %, FruitsOnPlate 27 %, CleanUpToys 27 %, the dishes-in-bin tasks 25 %.
- Open limit 560 of the 656 tier-A segments, closed limit 64, mixed 32. Right pad 362, left 294.
- 26 tier-A hooks begin within 3 s of reset (cordless_drill in the Tool tasks, red_mug in
  ReorientWhiteMugs): spawn poses in contact with the open gripper, a scene defect to fix separately.
- Beyond the tows, the same defect shows as 4 043 *drag* segments (hooked and pushed along the
  table) and 10 060 *press* segments (the arm pushing a pad 2 mm or more into a table-pinned object).

Against the reviewer's labels: 31 of the 32 labelled tow episodes in these corpora are recovered
(the miss is a hooked mug pushed along the table, classed drag); tier A alone recovers 26. Of 13
unlabelled flagged episodes the reviewer checked (5 tier A, 4 tier B, 4 tier C), all tier A and B
were confirmed; tier C held one proper tow and three marginal ones. The previous runtime rule (P43,
`TOWED_WITHOUT_GRASP`) fired in 2 of the 4 baseline replays above and on none of the 32 labelled
episodes: its `never grasped` term is blind to hooks after a real grasp, its jaw offset is the root
pivot, and its lift is measured from a stale contact start.

Per-policy comparisons of grasp counts, drops, attempts and success on these corpora are
contaminated unevenly by this artifact; the episodes are identifiable, see below.

## 3. The flag

`robolab/eval/tows.py`, run by default after every episode (`ROBOLAB_FLAG_TOWS=0` disables) and by
`scripts/flag_tows.py` over existing runs. Per object near a pad, every third control step: the
object's collision points (sampled from the scene USD, in the body frame) are moved to the world by
the recorded root pose, into the finger frame by the forward kinematics, and their signed distance
to the pad's bounding box is taken (negative inside).

- **hook**: one pad at least 2 mm inside the object for at least 0.6 s; the far pad is recorded too.
- **class** by what the object does meanwhile: `tow` rises 2 cm or more; `drag` moves 2 cm or more
  along its support; `press` neither. A deep one-sided contact with the finger mid-range *and* the
  far pad touching is a `squeeze` (a real pinch drawn with interpenetration: the marker, the orange)
  and is not a hook.
- **tier** on tows: **A** finger at a limit for at least half the segment, median depth 3 mm or more,
  2 s or more, rise 3 cm or more; **B** at a limit, 2.5 mm, 1 s; **C** the rest.

Output: `tows_<run>_env<env>.json` next to the log (every segment with object, pad, times, depth,
finger range, rise, path, class, tier, plus a summary), and in `episode_results.jsonl` a `tows`
summary; a tier-A tow sets `physics_artifact` and adds the object to `towed_objects`, the fields
P43 introduced. Thresholds are in `robolab/constants.py` (`TOW_*`). Limits: DROID rig only; objects
larger than 35 cm radius (racks) are skipped; needs `usd-core` for the meshes.

```
python scripts/flag_tows.py output/cli_pi05_robolab120 --workers 8      # writes tows_*.json, prints the table
python scripts/flag_tows.py output/cli_* --summary --csv tows.csv        # table + every tow segment for review
```

## 4. What to do with a flagged episode

Report it, do not score it as policy behaviour. The object never left the hand by the policy's
doing, so grasp, carry, release and drop counts after the onset are physics, and the task outcome
is usually a timeout with the object welded to a finger. For a leaderboard either exclude tier-A
episodes and publish the exclusion rate per policy (it differs, 4 to 10 %), or rerun with the
solid-pad asset once it is validated.
