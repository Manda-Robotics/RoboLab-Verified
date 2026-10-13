# Newton scene placement corrections — October 13, 2026

The exhaustive task sweep exposed two startup impulses caused by authored
object intersections. Moving the desk cube 15 mm and the red hammer 32 mm
removes the cube flip and the black hammer's upward kick. Both corrected USD
scenes pass fresh 20-second physics, contact, camera, recording and reset checks.

These are placement corrections, not changes to contact surfaces or solver
parameters. Every object retains its geometry, orientation, mass, inertia and
material. The edits affect both physics backends; the measurements here use
Newton/MuJoCo Warp on the RTX PRO 5000. PhysX behavior after these translations
has not been measured.

| Scene | Authored intersection | Exact change | Measured effect |
|---|---|---|---|
| `workdesk.usda` | Cube/marker: 6.485 mm | `/world/rubiks_cube` Y −15 mm | Cube maximum displacement 60.558 → 0.512 mm; maximum rotation 179.972° → 0.381°. |
| `tools_container.usda` | Black/red hammer: 6.545 mm | `/world/red_hammer` Y +32 mm | Black hammer maximum upward motion 88.692 → 0 mm; maximum displacement 142.449 → 23.261 mm. It now settles downward by 22.766 mm. |

Initial intersections were measured on the saved, fully integrated native
collision models at the authored poses. Margins were set to zero only in the
CPU geometry diagnostic; the GPU runs use the production preset. A translation
scan checked the red hammer against all eligible neighboring bodies. A separate
3 mm black-hammer height experiment reduced but did not remove the kick, so it
was not applied.

![Measured startup motion before and after placement correction](evidence/newton-task-placement-comparison.png)

The corrected source-based runs are `physics_MouseOnKeyboardTask_deskfix` and
`physics_ToolOrganizationTask_toolsfix`. Each records 300 steps, all native free
bodies, three cameras, pad and extra-body contact channels, and two resets.
Finite-state, contact-capacity, recording-length and reset checks pass. Neither
scene passively triggers success or target loss. No native body exceeds the
startup review threshold of 30 mm displacement or 15° rotation after correction.
Those thresholds are triage flags, not definitions of physical validity.

Across the entire corrected desk scene, the largest late XY span is 1.225 mm;
the tools scene's is 0.128 mm. Thus the correction does not imply perfectly
motionless clutter. All bodies were included in the trajectory review, and
five frames from each camera were inspected. The comparison figure uses the
actual corrected scene runs, not the earlier configuration-offset controls.

The fresh `MouseOnKeyboardTask` policy episode succeeds at step 201 (13.4 s),
with a grasp, carry and release. This is one example, not evidence that the
placement change improves policy success rates. The fresh
`ToolOrganizationTask` episode also succeeds at step 2468 (164.53 s), after
several drops and retries; both hammer placements complete. Its finite-state,
recording and success-label checks pass. The full 120-task physics/policy
sweep continues separately.

The desk edit affects `MarkerInMugTask`, `MouseOnKeyboardTask` and
`PickGlassesTask`. The tools edit affects `ClampInRightBinTask`,
`NonHammerToolsInRightBinTask`, `ToolOrganizationBothTask`, `HammersInLeftBinTask`
and `ToolOrganizationTask`. The previously run desk/tools cases are invalidated
in the exhaustive queue and retained as baseline evidence. Remaining affected
tasks use the corrected scene assets.

Scene metadata positions are synchronized with the USD edits. No evaluation
predicate or episode time limit changes. The separately committed P126 fixes
HDF5 success labeling; its evidence is in the changes ledger.

[Detailed trajectories, checks, source labels and archive hashes](evidence/newton-task-placement-verification.json)
identify all four baseline/corrected recordings. The archives and extracted
files are retained under
`/Users/metz/Documents/manda/state-of-robot-aug-2026/output/newton-all-tasks-20261013/`.
Every archived evidence file was SHA-256 verified after download. The GPU source
label `778afa7+validated-corrections-ac8d6f18d6ff` identifies the deployment;
its manifest verifies 296 runtime Python files and 47 scene files. The node
remains running.
