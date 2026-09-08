# Dashboard

A self-contained web dashboard for browsing the RoboLab benchmark and inspecting eval results. Boots in seconds, binds `0.0.0.0` so anyone on your LAN can reach it via your IP.

<video src="https://github.com/user-attachments/assets/5992e61b-9043-4602-8402-04459da38421" autoplay controls muted loop playsinline width="800">
  Your viewer doesn't render inline video — see <a href="https://github.com/user-attachments/assets/5992e61b-9043-4602-8402-04459da38421">robolab_dashboard.mp4</a>.
</video>

## Quickstart

```bash
# Already installed if you ran `uv sync` / `uv pip install -e .` — the
# dashboard ships in the default RoboLab dependency set.

# Bare minimum: starts empty, add output directories from the sidebar.
uv run robolab-dashboard

# Or seed it with one up-front (still editable in the UI):
uv run robolab-dashboard --output-dir /path/to/output
# → Serving /path/to/output on http://0.0.0.0:8080
```

Then open `http://localhost:8080` locally or `http://<your-lan-ip>:8080` from another machine.

### CLI flags

| Flag | Default | Purpose |
|---|---|---|
| `--output-dir <path>` | *(none — optional)* | Optional initial results directory. If omitted, the dashboard starts with whatever sources you previously added from the sidebar (persisted to `~/.config/robolab-dashboard/sources.json`). Add or remove sources at runtime from the sidebar. |
| `--scenes-metadata-dir <path>` | auto-detected (see resolution order below) | Where to find `scene_metadata.json` + `_images/`. Only needed if running from a worktree that has `assets/` sparse-excluded. Also reads `$ROBOLAB_SCENES_METADATA_DIR`. Resolution order: CLI flag → env var → `<PACKAGE_DIR>/assets/scenes/_metadata` → sibling `robolab/` checkout. |
| `--port <int>` | `8080` | Bind port. |
| `--host <ip>` | `0.0.0.0` | Bind host. `127.0.0.1` to restrict to localhost. |
| `--reload` | off | Auto-reload on Python edits (dev). |

## What it shows

### Scenes

Browse every USD scene in the asset library — a card grid with a preview thumbnail per scene, the object count, and the number of tasks that reference each one. Click into any scene to see its full prim list (name, payload USD path, description, static-body flag) and a back-link to every task that uses it.

Data sources:
- `assets/scenes/_metadata/scene_metadata.json` — per-scene prim manifests
- `assets/scenes/_metadata/scene_statistics.json` — aggregate stats
- `assets/scenes/_images/<scene>.png` — preview images served via `/api/scenes/<file>/image`

### Tasks

Browse benchmark tasks defined under `robolab/tasks/<folder>/`. The folder dropdown defaults to `constants.DEFAULT_TASK_SUBFOLDERS[0]` (i.e. `benchmark`); any other subfolder with `.py` files (e.g. `test_tasks`, `randomize_initial_pose`) can be selected.

Above the task list, a summary card shows aggregate counts (Tasks · Unique scenes · Avg variants/task · Avg episode length) plus stackable **mini-card filters**: click a difficulty (simple / moderate / complex) or attribute (semantics, spatial, color, …) to filter the list. `Ctrl/Cmd-click` stacks multiple selections (OR within a kind, AND across kinds).

Per-task detail shows all instruction variants, a clickable scene preview, contact objects, terminations, subtasks, and subtask-stage counts. Click the scene to jump to its detail page.

Data sources:
- `robolab/tasks/_metadata/task_metadata.json` — pre-generated task manifest (run the standard `task_metadata` regen step to update)
- `robolab/tasks/_metadata/task_timing.json` — optional per-task wall-clock timing
- These JSONs are read directly; the dashboard never imports IsaacLab.

### Results

Add your experiment output directory to the left hand side bar. Each experiment folder must contain Task output folders, and an `episode_results.jsonl`. The dashboard parses each `hdf5` and compiles a report based on the data. If videos are available, they are shown.

All SR / Score cells carry **95% confidence intervals** with the half-width annotation: `29.7% [24.5–35.4] ±5.4`. SR uses an exact Beta credible interval (see `robolab.core.logging.results.beta_ci_bounds`); Score uses Student-t.

#### Dense annotation lanes

Review mode: the label panel lists every machine segment with three verdicts (label, result, bounds within 0.5 s). `j` / `k` select and seek, `1` / `2` / `3` toggle, `Enter` saves a `kind: review` row to the labels file; a ✗ opens the correction fields (label `none` removes the segment). `scripts/score_gold.py` consumes the rows.

Under the event strip of an episode, three lanes share the strip's time axis
([dense_annotations.md](verified/dense_annotations.md)):

- **Attempts**: one block per `pick` / `place` / `drop` / `no completed subtask`, coloured by
  result (green pass, red fail, grey unknown, hatched for a stretch in which nothing completed).
  Hover for the description and any flags; click to seek. The same segments are listed under
  the lanes and highlight with the playhead.
- **Phases**: one block per contiguous phase (reach, approach, hover, close on, lift, transport,
  lower, release, press table, ...), coloured by family. Hover for the object and its role.
- **Labels**: your own marks (see below). Hidden until the Labels panel is opened, unless the
  episode already carries marks.

The lanes are served by `GET .../episodes/{env}/run/{run}/phases`. When
`scripts/annotate_phases.py` has written `phases_<run>_env<env>.json` next to the log, that
file is served; otherwise the episode is annotated on the fly from the recording (a few seconds
the first time, cached afterwards, nothing written). Without `torch` the in-hand state comes from
geometry alone and the legend says so.

#### Labelling an episode

The **Labels** panel under the lanes is a review tool and is closed by default: click the
**Labels** heading to open it (the choice is remembered per browser). While it is closed its
keys are inert. It writes human marks to `analysis/phase_labels.jsonl`
(override with `ROBOLAB_PHASE_LABELS=<path>`), one JSON line per mark, keyed by
`(run, task, env, run_index)`; a delete appends a tombstone rather than rewriting the file.
Keys while the episode view is open: `i` sets the start to the playhead, `o` the end, `Enter`
inside the panel saves; `space` and the arrow keys still drive the transport. Pick the kind
(`pick`, `place`, `drop`, `no_completed_subtask`, or a bare `boundary`), the object (the
scene's objects are offered), the result, and a note in your own words. After a save the next
mark's start is pre-filled with the previous end, so a whole episode can be labelled by
scrubbing forward. `GET .../phase_labels` returns the marks of one episode;
`DELETE /api/phase_labels/{id}` removes one.

## Hosting on the LAN

The default `--host 0.0.0.0` binding makes the dashboard reachable on your machine's LAN IP. Share the URL with a colleague (e.g. `http://<your-lan-ip>:8080`) and they can browse without setting anything up locally. Tighten to `--host 127.0.0.1` if you'd rather keep it loopback-only.
