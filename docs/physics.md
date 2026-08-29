# Physics: friction and the arm controller

Two things about the simulated contact physics that RoboLab Verified looked at, what
was found, and what was (and was not) changed. Both were raised by reviewers watching
episodes ("it's grabbing right at the edge of the banana — I'm not sure this would
work in the real world") before anyone read the numbers.

- [Friction](#friction) — a run parameter now (`--friction`, P79); **the default is
  unchanged**.
- [The arm controller](#the-arm-controller-gravity-off-pd-40080-eef-offset) — documented,
  deliberately unchanged.

## Friction

### What the assets carry

Friction is baked into the USD files. Every object carries its own `PhysicsMaterial`
(catalogued in [`assets/objects/object_catalog.json`](../assets/objects/object_catalog.json),
fields `static_friction` / `dynamic_friction`), the robot USD carries one for the finger
pads, and every scene defines a `/world/PhysicsMaterial` that nothing binds to. Read back
from the files (2026-08-28, upstream v0.3.1):

| surface | static = dynamic μ | count |
|---|---|---|
| Robotiq 2F-85 finger pads (`left_inner_finger`, `right_inner_finger`) | **2.0** | — |
| objects | 2.0 | 289 of 312 |
| fruit (`fruits_veggies` dataset: lemons, oranges, onion, avocado…) | 5.0 | 9 |
| Objaverse bagels, apple, bell pepper, lunch bag | **10.0** | 7 |
| basic blocks | 0.8 / 1.0 | 4 |
| no material (`gregorys_coffee_cup`, `large_storage_rack`) | PhysX default 0.5 | 2 |

PhysX combines the two materials of a contact pair with the **average** rule
(`physics:frictionCombineMode` is unset everywhere, and *average* is the default). So the
coefficient that decides whether a grasp holds is

| contact pair | effective μ |
|---|---|
| pad – typical object | (2.0 + 2.0) / 2 = **2.0** |
| pad – fruit | (2.0 + 5.0) / 2 = 3.5 |
| pad – bagel | (2.0 + 10.0) / 2 = **6.0** |
| object – tabletop (no material → PhysX default 0.5; confirmed by readback) | (2.0 + 0.5) / 2 = 1.25 |

Static equals dynamic everywhere, which real materials do not do (dynamic is lower).
Dry rubber on plastic, cardboard or glazed ceramic is roughly 0.5–1.0; on fruit skin
lower. Simulators without pad compliance typically sit at 0.5–1.0 to compensate for
rigid contacts (Isaac Lab's own default material is 0.5). Upstream's asset guide asks
for "2.0–5.0 for reliable grasping" ([objects.md](objects.md)) — i.e. the values were
chosen so that grasps hold, not measured.

The consequence reviewers saw: edge pinches, one-finger holds and towed objects that a
real pad would let slip (see `TOWED_WITHOUT_GRASP` in [event_tracking.md](event_tracking.md)
and the tow analysis in `docs/VERIFIED_PATCHES.md`).

### What Verified does: `--friction` (P79)

Friction is a **run parameter**. The benchmark default is still the authored material —
a run without the flag is bit-identical to upstream. Every eval runner accepts

| `--friction` | effect |
|---|---|
| `upstream` (default) | authored USD materials, untouched |
| `<number>`, e.g. `0.5` | that coefficient (static = dynamic) on **every rigid object and on the finger pads**; the effective pad–object μ is the same number |
| `realistic` | the bundled per-class table [`robolab/core/physics/friction_realistic.json`](../robolab/core/physics/friction_realistic.json): pads 0.8/0.7, plastics/cardboard 0.4/0.35, ceramics/glass and metal utensils 0.3/0.25, wood blocks 0.5/0.45; effective pad–object μ 0.55–0.65 |
| `path/to/table.json` | a user table in the same format |

A table maps the catalog's `class` field to `{static, dynamic}` (or a single number for
both); `default` covers unlisted classes and objects the catalog does not know, `gripper`
is applied to the robot's pads. Keys starting with `_` are comments. Dynamic must not
exceed static.

**How it is applied.** At env-cfg build time (`GeneratedTaskEnvCfg.__post_init__`)
`robolab.core.physics.friction.install` adds one Isaac Lab
`randomize_rigid_body_material` event term per rigid scene object, plus one for the
robot's `friction_bodies`, in `startup` mode with a degenerate range `(μ, μ)` and a
single bucket — i.e. it *sets* the PhysX shape materials once, at start-up. Restitution
is carried over from the catalog so only friction changes. No USD is edited; nothing
runs per step.

**What is and is not touched** (recorded so it is not mistaken for an oversight). The
tabletop (`/world/table`, `../fixtures/table_oak.usd`) is a *rigid object* in the scene —
PhysX-default 0.5/0.5 upstream, no catalog row — so an override reaches it; a table
resolves as class `fixture` (`realistic`: 0.5/0.45). The pedestal fixture under it and any
kinematic/static scene body are `AssetBaseCfg`, which the event term cannot address, and
keep their authored material. The ALOHA/Kinova/dual-Franka pad bodies are *declared* from
their asset link names but have not yet been read back on a GPU (DROID has).

**Provenance.** Every run records both what was asked for and what PhysX holds:

- `env_cfg.json` → `friction`: mode, the flag text, per-object requested
  `{static, dynamic, restitution, source}` where `source` names the catalog class the
  value was resolved from, the pad material and the pad body names.
- `friction_applied.json` (next to it): the PhysX readback after start-up, per object
  shape and per pad body — **written under `upstream` too**, so the baseline's 2.0 is a
  measurement rather than a claim.

`scripts/verify_patches.py` compares the two (predicate **P79**): PASS when every
target holds the requested coefficient, FAIL on any mismatch, N/A for an `upstream` run
or a recording that predates P79 (an N/A is never a pass).

**Probe without a policy.** `examples/run_empty.py --task BananaInBowlTask --headless
--friction 0.5` boots one env, steps it with zero actions and leaves the two files under
`output/run_empty_env/BananaInBowlTask*/`.

**Robot label.** A robot cfg names its pad bodies with the `friction_bodies` label
(assigned after the class, like `ee_recorder_bodies`; see
[robots.md](robots.md#label-assignment-rules)):

```python
DroidCfg.friction_bodies = ["left_inner_finger", "right_inner_finger"]
```

Without the label an override applies to the objects only and a warning is printed;
`verify_patches.py` then reports P79 as FAIL for that run, because half the contact pair
was left at 2.0.

### Why the default did not change

Changing the benchmark's physics changes every published number, and there is no
labelled ground truth for "the right μ". What there *is* is the sensitivity experiment
FINDINGS §5 called load-bearing: the same policy, the same tasks, at several
coefficients. RoboLab Verified ships the knob and publishes the measurement; the
default moves only if the measurement says the ranking of policies depends on it.

**Protocol (rc7 friction sweep).** π0.5 (`pi05_droid_jointpos`), 4 envs per task, the
eight tasks below, at `--friction upstream`, `1.0`, `0.5`, `realistic`; outputs
`output/rc7_<mode>_<Task>`. Reported per condition: success rate, `OBJECT_CARRIED`
count, `GRASP_ATTEMPT_FAILED` count, `TOWED_WITHOUT_GRASP` count, and the fraction of
carries that ever load both pads (`scripts/contact_force_profile.py`, P77).

| task | why |
|---|---|
| BananaInBowl | the reference pick-and-place; π0.5 succeeds often, so a drop is visible |
| FoodPacking2Cans | the spam-heaviest task; cans and boxes, many attempts |
| BlackItemsInBin | wedged smartphone, marker — the stuck-to-finger cases |
| FruitsOnion | fruit at μ 5.0 today; the biggest change under any override |
| GrabAFruit | a pure lift, scored on 50 mm of rise |
| StackWhiteMugs | the labelled tow ("a crazy bug, 100 %") |
| ClutterPumpkin | clutter, bumps, `SCENE_SETTLING` |
| BowlStackingRightOnLeft | rim grasps of nested bowls |

`scripts/friction_sweep_report.py output/rc7_*` renders the per-condition table (success
rate, carries / failed attempts / drops per episode, tows, one-pad carries, and the P79
verdict of every run, so a condition whose materials did not land is marked rather than
compared). Results are recorded in `docs/VERIFIED_PATCHES.md` under P79 as they come in.

### Results (rc7, 2026-08-29)

Pod `robolab-verified6` (L40), π0.5 jointpos, 4 envs per task, one process per task,
30 of 32 runs complete (the last two `realistic` runs were still on the pod when it
self-stopped; see the ledger). Every override run's `friction_applied.json` PASSes the
P79 predicate — the coefficients below are what PhysX held, not what was asked for.

| task | upstream (μ 2.0) | 1.0 | 0.5 | realistic (≈0.6) |
|---|---|---|---|---|
| BananaInBowl | 3/4 | 3/4 | 3/4 | 3/4 |
| BlackItemsInBin | 0/4 | 0/4 | 0/4 | 0/4 |
| BowlStackingRightOnLeft | 0/4 | 0/4 | 1/4 | — |
| ClutterPumpkin | 0/4 | 0/4 | 0/4 | — |
| FoodPacking2Cans | 0/4 | 0/4 | 0/4 | 0/4 |
| FruitsOnion | 1/4 | 2/4 | 3/4 | 2/4 |
| GrabAFruit | 0/4 | 0/4 | 0/4 | 0/4 |
| StackWhiteMugs | 1/4 | 1/4 | 1/4 | 2/4 |
| **success rate** | **5/32 (16 %)** | 6/32 (19 %) | 8/32 (25 %) | 7/24 (29 %) |
| carries / episode | 2.2 | 1.8 | 1.5 | 1.6 |
| failed grasp attempts / episode | **1.6** | 5.8 | **7.3** | 7.2 |
| drops (closed hand) / episode | 0.4 | 0.6 | 0.8 | 1.0 |
| events / episode | 20.0 | 27.5 | 30.0 | 30.6 |
| tows / physics-artifact episodes | 0 | 0 | 2 | 0 |
| objects off the table | 4 | 2 | 2 | 1 |
| one-pad carries | 2/69 (3 %) | 3/59 (5 %) | 2/47 (4 %) | 4/39 (10 %) |

**What it says.**

1. **The success rate does not fall when friction drops from 2.0 to 0.5.** If anything it
   rises (16 → 25 %), but 5/32 against 8/32 is not distinguishable (Fisher p = 0.54; the
   95 % Beta intervals are 0.07–0.32 and 0.13–0.42). FINDINGS §5's worry — that the
   published leaderboard is "substantially an artefact of contact tuning" — is **not
   supported** on this slice: π0.5's outcomes on these eight tasks are robust to a 4×
   change in μ.
2. **The behaviour behind the number is not robust.** Failed grasp attempts go 1.6 →
   7.3 per episode (×4.5), carries 2.2 → 1.5, closed-hand drops 0.4 → 1.0. At realistic
   friction the policy slips and re-grasps far more before reaching the same outcome —
   FoodPacking2Cans alone goes from 26 to 131–149 failed attempts across four episodes,
   ClutterPumpkin from 6 to 57. Every event-based metric this fork reports (attempt
   counts, drop counts, carries) is therefore friction-dependent, and a comparison of
   those metrics across runs is only meaningful at one stated `--friction`.
3. **Fruit at μ 5.0 was hurting, not helping.** FruitsOnion is the one task that moves
   (1 → 2 → 3 of 4 as μ drops); its onion and lemons are the objects authored at 5.0.
4. **The "magnetic object" artifact is not friction.** The only two `TOWED_WITHOUT_GRASP`
   flags in the sweep fired at μ 0.5 (`BlackItemsInBin` env 2 `marker` @ 108.7 s,
   `ClutterPumpkin` env 1 `orange_01` @ 54.3 s) — the same stuck-to-finger behaviour
   reviewers labelled at 2.0. It survives a 4× friction cut, so it is a contact/solver
   problem, not a material one. These are also the first runtime firings of P43.

**Decision.** The default stays `upstream`. The measurement is published with the
benchmark; anyone comparing *behaviour* (not just success) must state the friction they
ran at, and `friction_applied.json` in every run directory makes that checkable.
Statistical power is the caveat: 32 episodes per condition resolves a ~25-point
difference in success rate, not a 10-point one.

## The arm controller: gravity off, PD 400/80, EEF offset

`robolab/robots/droid.py` spawns the Franka with `disable_gravity=True`, implicit PD
actuators at stiffness 400 / damping 80 on all seven arm joints, and records the
end-effector at the Robotiq `base_link` with `EEF_OFFSET_POS = (0, 0, 0)`.
This was flagged during the code audit (plan item C2) as a possible source of
unrealistic behaviour. It is not one, and it is **left unchanged**:

- **It is Isaac Lab's reference configuration.** `FRANKA_PANDA_HIGH_PD_CFG` in
  `isaaclab_assets/robots/franka.py` is exactly `disable_gravity = True`, stiffness 400,
  damping 80, with the note "useful for task-space control using differential IK". The
  plain `FRANKA_PANDA_CFG` (gravity on, 80/4) is the compliant variant meant for
  torque-level control.
- **Gravity-off is the stand-in for gravity compensation.** Isaac Lab's implicit actuators
  are pure PD; they have no gravity-compensation term. A real Franka running the DROID
  stack compensates gravity in its controller, so its joints do not sag toward their
  targets. With gravity on and no compensation the arm would hang below every commanded
  pose by `τ_gravity / stiffness` (several degrees at the shoulder), which no real DROID
  arm does. Turning gravity on would therefore make the simulation *less* like the
  robot, not more, unless a compensation term were added at the same time — and that is
  a controller change, not a physics fix.
- **The EEF offset is a reporting convention.** `ee_pose` is the pose of the gripper
  base, not the fingertips ([frames.md](frames.md)); every analysis script in this fork
  reads it that way (`GRASP_JAW_BODY = "base_link"`). Moving the frame to the fingertips
  would change every recorded trajectory without changing any behaviour.

What was checked: the config matches the Isaac Lab source line for line (2026-08-28), and
the recorded action/joint traces on the rc-series runs show absolute joint targets
tracked without drift (`docs/VERIFIED_PATCHES.md`, "sanity gate": median hand→target
distance 1.3 cm on the corpus).

## See Also

- [Objects](objects.md) — how object materials are authored and catalogued
- [Robots](robots.md) — robot cfg labels (`friction_bodies`, `ee_recorder_bodies`)
- [Running Environments](environment_run.md#run_evalpy-cli-reference) — the `--friction` flag
- `docs/VERIFIED_PATCHES.md` — P79's row, with the measurements
