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
