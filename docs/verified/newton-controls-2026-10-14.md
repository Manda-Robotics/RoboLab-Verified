# Newton friction and physics timing controls (P127)

Validated on 2026-10-14 (Europe/Zurich) using the existing RTX PRO 5000 Blackwell
48 GB node. These controls are opt-in. The default friction and timing remain
unchanged. Changing the requested material or physics resolution can change scores.

## Finding and implementation

At baseline `b58499a`, Newton explicitly rejected numeric and table `--friction`
requests because Verified's existing material events were implemented through
PhysX. A retained `ReorientAllMugsTask` recording contains
`friction_applied.json` with `status: unsupported`, not measured coefficients.
The normal evaluation CLI also lacked simulation-rate and Newton-substep flags.
The baseline guard outputs and recorded-file hash are retained in the
[machine-readable evidence](evidence/newton-controls-p127-verification.json).

Newton now accepts `upstream`, a number, `realistic`, and a JSON material table.
The table's dynamic value maps to MuJoCo's single sliding coefficient. Its static
value stays in provenance; it is not independently implemented. The adapter
resolves compiled geoms in every world, updates Newton and CPU/GPU MuJoCo material
arrays before stepping, and verifies actual readback. Native priority/max material
combination remains in effect, including explicit contact pairs. Torsional and
rolling friction, contact response, and unrelated static extras are preserved.
The explicit-pair priority path has offline coverage; the GPU scene checks do not
establish exhaustive coverage of every possible pair configuration.

`--sim-rate HZ` works on both backends. It changes decimation and render interval
to preserve authored policy and camera rates. Incompatible rates fail before
simulation. `--newton-substeps N` changes internal solver resolution and is
rejected on PhysX. DROID defaults remain 120 Hz / 4 Newton substeps / 15 Hz policy.
Coarser internal timesteps increase the default contact time constant only as
required by the existing `timeconst >= 2 * solver_dt` stability constraint.

Requested/resolved timing is recorded in `env_cfg.json`; the output marker saves
requested timing and the resolved friction table. Parameter-stamped output folders
cannot be resumed with different controls. Legacy backend-only folders retain
default-run compatibility; nondefault controls need a fresh folder when parameter
provenance is absent.

```bash
python policies/pi0_family/run.py --policy pi05 --headless \
  --physics newton_mjwarp --friction 0.5 --sim-rate 240 --newton-substeps 2 \
  --task RubiksCubeTask --output-folder-name newton_mu05_240x2
```

## Verification

- Offline suite: **345 passed**.
- Newton CPU suite: **123 passed, 2 GPU-only tests skipped**.
- Seven recorded GPU integration conditions passed; table below.
- The public policy runner completed a fresh episode with all three controls.

Each passive GPU condition uses `BananaInBowlTask`, 30 control steps (2 simulated
seconds), startup/step/reset readback, finite robot/object states, three camera
streams, and an exported HDF5 trajectory. The two-environment condition has two
distinct 30-step trajectories. All cases retain 15 Hz control and rendering;
Newton's measured native timestep matches the resolved simulation dt / substeps.

| Backend | Friction | Simulation Hz | Substeps | Environments |
|---|---|---:|---:|---:|
| Newton | 0.5 | 240 | 2 | 2 |
| Newton | realistic | 120 | 2 | 1 |
| Newton | upstream | 60 | 4 | 1 |
| Newton | upstream (defaults) | 120 | 4 | 1 |
| Newton | custom JSON | 240 | 4 | 1 |
| PhysX | 0.5 | 240 | — | 1 |
| PhysX | upstream | 60 | — | 1 |

The custom table uses default static/dynamic 0.4/0.2, gripper 0.9/0.6, and fruit
0.7/0.3. The readback verifies the distinct dynamic values, not the static values.
P79 verifies the recorded overrides; upstream cases remain N/A for the override
check, even though their coefficients are measured.

An independent native-contact sliding test starts a box at 0.5 m/s and advances
240 steps at 1/240 s. With coefficient 0.1 it travels **0.126434 m**; at 0.9 it
travels **0.013880 m**. CPU and GPU results agree to less than 0.1 micrometre in
final X. This verifies an effect on dynamics beyond material-array bookkeeping.

The fresh pi0.5 `RubiksCubeTask` episode uses friction 0.5, 240 Hz and two substeps.
The public runner exited 0 and produced its completion marker, a 231-step HDF5,
and a 230-frame 15 fps video. The policy **failed** after 15.4 simulated seconds:
two failed grasp attempts, then `TARGET_LOST` / `OBJECT_FELL_OFF_TABLE`. JSON and
HDF5 both report false. The recording audit passed finite data, step lengths,
quaternions, both pad channels, 21 extra contact channels, and video lengths;
there were no audit review flags or `physics_artifact` flag. Five video frames
were sampled. This is integration evidence, not a successful policy result or
evidence of improved benchmark performance.

## Fixes found during validation

The initial solver wrapper collided with a constructor keyword named `solver`;
the corrected wrapper uses `self`. It also now preserves the constructor's
signature with `functools.wraps`, which Isaac Lab needs to pass solver settings.
Without the signature, the prototype lost its contact capacity setting and hit
a CCD buffer error. A regression checks signature preservation.

The coarser-timestep case exposed float32 rounding at the contact time-constant
boundary. Validation now permits only conversion roundoff; the regression still
rejects a time constant 0.1% below the minimum. An isolated-checkout packaging
omission of the bundled realistic table was corrected. Finally, the diagnostic
probe's first N2 recording reused an episode ID. The corrected probe assigns one
ID per environment, as the production runner already did, and N2 `r4` supersedes
`r3` for recorded parallel coverage. Failed and superseded attempts remain in the
evidence archive rather than being counted as passes.

## Evidence and scope

The [verification JSON](evidence/newton-controls-p127-verification.json) contains
baseline evidence, source hashes, all selected cases, recording audits and test
counts. The full archive contains logs, layered source snapshots, complete
readbacks, HDF5 trajectories, images and the policy video:

- Remote: `/workspace/newton-controls-20261014/controls-evidence.tar.gz`
- SHA-256: `5ca41ef325b5d7309cb86f0cf04f7a73ad26f590776e2339a8715f435696ea8f`
- Size: 32,548,840 bytes; 156 hashed evidence files.

The archive and every file were downloaded and hash-verified locally under
`output/newton-controls-20261014/` in the `state-of-robot-aug-2026` workspace.
Reproduce passive checks with `scripts/probe_physics_controls.py --task
BananaInBowlTask --headless --probe-output <fresh-directory>` plus the shared
physics flags. This uses the pinned Isaac Lab/Newton native-contact DROID setup.
Other robots, non-native contact modes, large environment counts, and all 120 tasks
under changed friction/timing remain unvalidated.

Tests ran in an isolated source checkout. The ongoing all-task sweep keeps its
original physics configuration and code; its results are not mixed with these
controls experiments. A controls manifest was briefly uploaded over sweep
metadata during setup, then restored byte-for-byte. All 343 runtime source hashes
were unchanged and no monitor recorded the temporary manifest; the incident
record is included in the evidence. The node remains running.
