# Bimanual rigs in RoboLab Verified

Three rigs are adopted. They are not equally usable, and the difference is about the
*checkpoints available*, not about the rigs.

| rig | file | state |
|---|---|---|
| **bimanual YAM** | `bimanual_yam.py` | **verified with a released policy**: Ai2's MolmoAct 2 bimanual checkpoint, 5/8 on the parity task (Ai2's own simulator: 4/8); 0/4 on each of three benchmark tasks outside its training distribution. The rig to reach for: it is what the labs collect bimanual data on. |
| **dual-Franka** | `bimanual_franka.py` | **verified 6/6 clean**, max 0.18 rad/step. The default choice. |
| **bimanual ViperX (ALOHA)** | `aloha.py` | **config only in this repository**: the ALOHA 2 asset (`assets/robots/aloha2/`) and its MJCF import scripts were not shipped, so this rig does not load from a clean clone. Rig verified in the development fork (arms build, wrist cams track, per-arm metrics record). **No working policy**: the released pi05 checkpoint scores **0/6** — coherent reach, twitchy, no grasp. Out of distribution for this embodiment; fine-tuning is the only path we found. |

**Reporting rule.** An ALOHA score is a statement about the checkpoint, not about the
rig or about RoboLab. Do not place ALOHA numbers beside Franka numbers without saying
so — the reader will otherwise conclude the benchmark is harder than it is, or that
the rig is broken, and neither is what the evidence shows.

**Shared dependency.** Both rigs declare per-arm `ee_recorder_bodies`
(`left_ee_pose` / `right_ee_pose`), so a bimanual demo has no `ee_pose` group at all.
Upstream `compute_metrics` reads `demo["ee_pose"]` inside a bare `except`, which turns
that into a silent `None` — every bimanual run yields no metrics. P67 fixes it and is
a hard dependency of both rigs, not an optimisation.

The ALOHA transfer-cube task is included so the rig is runnable end to end. It is a
smoke test, not a benchmark result.
