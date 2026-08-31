# Scripts

Offline tools shipped with RoboLab Verified. None of them needs Isaac Sim; the ones marked
`h5py` read recorded HDF5 trajectories, the ones marked `usd-core` read USD files.

| script | what it does | needs |
|---|---|---|
| `verify_patches.py` | Per recorded run, PASS / FAIL / N/A for each evaluation change; `--baseline` for before/after, `--summary` across runs, exit 1 on FAIL. The evidence behind [changes.md](../docs/verified/changes.md). | h5py (P77) |
| `reflag.py` | Replay the post-processable event rules over an episode already on disk. | — |
| `flag_regression.py` | Score the flags of recorded runs against the human verdicts in `analysis/flag_labels.jsonl`. | — |
| `friction_sweep_report.py` | The per-condition table of a `--friction` sweep ([physics.md](../docs/physics.md)). | h5py (optional) |
| `contact_force_profile.py` | Pad-force statistics over confirmed carries vs failed attempts (the tow question). | h5py |
| `audit_task_definitions.py` | Every task definition against its scene and spawn state; exit 1 on a finding. | usd-core |
| `find_task_definition_conflicts.py` | AST diff of each task's success placement against its subtask ladder. | — |
| `check_scene_intersections.py` | Objects authored overlapping each other; differential by design. | usd-core |
| `check_rest_heights.py` | Objects that drop, rise or roll at reset, from recordings. | h5py |
| `find_sinking_objects.py` | Objects that sink into their support at reset. | h5py |
| `find_open_hand_carries.py` | Candidates for the "stuck to a finger" artifact. | h5py |
| `grip_before_carry.py` | Checks the `OBJECT_GRIPPED → OBJECT_CARRIED` ordering in recorded logs. | — |
| `read_subtask_status_from_hdf5.py` | Dump the per-step subtask status from a recording. | h5py |
| `convert_to_lerobot.py` | Export recordings to the LeRobot format (upstream). | — |
| `serve_pi05.sh`, `install_openpi.sh` | Serve π0.5 from the joint-position checkpoint; install the OpenPI server and client. | uv |

Usage lines: `python scripts/<name>.py --help`. See [verification.md](../docs/verified/verification.md)
for how these fit together.
