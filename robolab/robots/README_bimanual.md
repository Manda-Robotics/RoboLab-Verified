# Bimanual rigs in RoboLab Verified

Two rigs are adopted. They are not equally usable, and the difference is about the
*checkpoints available*, not about the rigs.

| rig | file | state |
|---|---|---|
| **dual-Franka** | `bimanual_franka.py` | **verified 6/6 clean**, max 0.18 rad/step. The default choice. |
| **bimanual ViperX (ALOHA)** | `aloha.py` | rig verified (arms build, wrist cams track, per-arm metrics record). **No working policy**: the released pi05 checkpoint scores **0/6** — coherent reach, twitchy, no grasp. Out of distribution for this embodiment; fine-tuning is the only path we found. |

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
