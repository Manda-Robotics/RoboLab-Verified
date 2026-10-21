"""An episode gets the videos of its own run only (dashboard/loaders/local.py, 2026-09-02)."""
import importlib
import sys
import types
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def _loader():
    return importlib.import_module("dashboard.loaders.local")


def test_two_runs_in_one_folder_do_not_share_videos(tmp_path):
    L = _loader()
    names = [f"Pack_{r}_env{e}{sfx}.mp4" for r in (0, 1) for e in (0, 1) for sfx in ("", "_viewport")]
    for n in names:
        (tmp_path / n).write_bytes(b"")
    row_cls = L.EpisodeRow
    src = L.LocalLoader.__new__(L.LocalLoader)
    for run_index in (0, 1):
        e = row_cls.__new__(row_cls)
        e.env_id, e.run_index, e.last_frame_path, e.videos = 1, run_index, None, []
        src._attach_media(tmp_path, e)
        got = sorted(Path(v.path).name for v in e.videos)
        assert got == [f"Pack_{run_index}_env1.mp4", f"Pack_{run_index}_env1_viewport.mp4"], got


def test_legacy_names_without_a_run_token_still_attach(tmp_path):
    L = _loader()
    for n in ("Pick_env0.mp4", "Pick_env0_viewport.mp4"):
        (tmp_path / n).write_bytes(b"")
    e = L.EpisodeRow.__new__(L.EpisodeRow)
    e.env_id, e.run_index, e.last_frame_path, e.videos = 0, 0, None, []
    L.LocalLoader.__new__(L.LocalLoader)._attach_media(tmp_path, e)
    assert sorted(v.name for v in e.videos) == ["recording", "viewport"]


def test_outcome_audit_and_reason_provenance_reach_episode_api(tmp_path):
    import json
    from dataclasses import asdict

    run = tmp_path / 'run'
    (run / 'Task').mkdir(parents=True)
    integrity = {'status': 'conflict', 'eligible_for_comparison': False,
                 'observed_success': {'verified': False, 'hdf5': True}}
    (run / 'episode_results.jsonl').write_text(json.dumps({
        'env_name': 'Task', 'run': 0, 'env_id': 0, 'success': False,
        'artifact_integrity': integrity, 'reason_source': 'subtask_final_state',
        'reason': 'object_grabbed is false on final frame',
    }) + '\n')
    rows = _loader().LocalLoader(tmp_path).list_episodes('run', 'Task')
    assert len(rows) == 1
    value = asdict(rows[0])
    assert value['artifact_integrity'] == integrity
    assert value['reason_source'] == 'subtask_final_state'
    assert value['success'] is False


def test_raw_totals_disclose_invalid_and_unaudited_outcomes(tmp_path):
    import json
    run = tmp_path / 'run'
    (run / 'Task').mkdir(parents=True)
    rows = [
        {'env_name': 'Task', 'run': 0, 'env_id': 0, 'success': True,
         'artifact_integrity': {'eligible_for_comparison': False}},
        {'env_name': 'Task', 'run': 1, 'env_id': 0, 'success': False},
    ]
    (run / 'episode_results.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in rows))
    loader = _loader().LocalLoader(tmp_path)
    meta = loader.list_runs()[0]
    task = loader.list_tasks('run')[0]
    for summary in [meta, task]:
        assert summary.num_integrity_invalid == 1
        assert summary.num_unaudited == 1
        assert summary.num_episodes == 2  # raw historical totals are preserved, labelled raw
    assert len(loader.list_episodes('run', 'Task')) == 2
