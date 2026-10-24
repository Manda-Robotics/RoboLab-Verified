"""The source installer must retain upstream overrides and editable members."""

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("installer_under_test", ROOT / "scripts/install_newton61.py")
installer = importlib.util.module_from_spec(spec)
spec.loader.exec_module(installer)


def test_requirements_use_workspace_and_newton_overrides(tmp_path):
    (tmp_path / "pyproject.toml").write_text('''
[project]
dependencies = ["isaaclab", "newton[sim]>=1.2.0", "warp-lang==1.17.0"]
[project.optional-dependencies]
isaacsim = ["isaacsim[all,extscache]==6.2.0.0"]
[tool.uv]
override-dependencies = ["newton[sim]==1.6.1", "torch==2.11.0", "mujoco~=3.12.0", "mujoco-warp~=3.12.0", "torchaudio==2.11.0", "torchvision==0.26.0"]
[tool.uv.sources]
isaaclab = {path="source/isaaclab", editable=true}
''')
    requirements, overrides = installer.prepare_requirements(tmp_path, ROOT)
    assert requirements == [f"-e {tmp_path}/source/isaaclab", "newton[sim]>=1.2.0",
                            "warp-lang==1.17.0", "isaacsim[all,extscache]==6.2.0.0", f"-e {ROOT}", "pytest"]
    assert overrides == ["newton[sim]==1.6.1", "torch==2.11.0+cu130", "mujoco==3.12.0",
                         "mujoco-warp==3.12.0", "torchaudio==2.11.0+cu130", "torchvision==0.26.0+cu130"]


def test_checkout_clones_release_tag_and_verifies_commit(tmp_path, monkeypatch):
    calls = []

    def fake_run(*args, **kwargs):
        calls.append(args)
        return SimpleNamespace(stdout=installer.ISAACLAB_COMMIT if "rev-parse" in args else "")

    monkeypatch.setattr(installer, "run", fake_run)
    assert installer.prepare_checkout(tmp_path / "new-lab") == installer.ISAACLAB_COMMIT
    assert calls[0][0:6] == ("git", "clone", "--depth", "1", "--branch", "v3.0.0")
    assert any("rev-parse" in call for call in calls)
    assert any("status" in call for call in calls)


@pytest.mark.parametrize("head,dirty,error", [
    ("4ed09d414e93f35b8573db8850f8b27572779176", "", "Expected Isaac Lab v3.0.0"),
    (installer.ISAACLAB_COMMIT, " M source/isaaclab/pyproject.toml", "must be clean"),
])
def test_checkout_rejects_old_revision_or_local_edits(tmp_path, monkeypatch, head, dirty, error):
    def fake_run(*args, **kwargs):
        assert "clone" not in args and "checkout" not in args  # never replace an existing checkout
        return SimpleNamespace(stdout=head if "rev-parse" in args else dirty)

    monkeypatch.setattr(installer, "run", fake_run)
    with pytest.raises(ValueError, match=error):
        installer.prepare_checkout(tmp_path)


def test_install_rejects_conflicting_usd_provider(monkeypatch):
    monkeypatch.setattr(installer.metadata, "version", lambda name: "26.8")
    with pytest.raises(ValueError, match="fresh Python 3.12 venv"):
        installer.check_usd_provider()


def test_install_accepts_environment_without_usd_core(monkeypatch):
    def missing(name):
        assert name == "usd-core"
        raise installer.metadata.PackageNotFoundError(name)

    monkeypatch.setattr(installer.metadata, "version", missing)
    installer.check_usd_provider()
