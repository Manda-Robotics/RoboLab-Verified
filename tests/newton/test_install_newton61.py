"""The source installer must retain upstream overrides and editable members."""

import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("installer_under_test", ROOT / "scripts/install_newton61.py")
installer = importlib.util.module_from_spec(spec)
spec.loader.exec_module(installer)


def test_requirements_use_workspace_and_newton_overrides(tmp_path):
    (tmp_path / "pyproject.toml").write_text('''
[project]
dependencies = ["isaaclab", "newton[sim]>=1.2.0", "warp-lang==1.16.0"]
[project.optional-dependencies]
isaacsim = ["isaacsim[all,extscache]==6.1.0.0"]
[tool.uv]
override-dependencies = ["newton[sim]==1.5.2", "torch==2.11.0", "mujoco~=3.11.0"]
[tool.uv.sources]
isaaclab = {path="source/isaaclab", editable=true}
''')
    requirements, overrides = installer.prepare_requirements(tmp_path, ROOT)
    assert requirements == [f"-e {tmp_path}/source/isaaclab", "newton[sim]>=1.2.0",
                            "warp-lang==1.16.0", "isaacsim[all,extscache]==6.1.0.0", f"-e {ROOT}[test]"]
    assert overrides == ["newton[sim]==1.5.2", "torch==2.11.0+cu128", "mujoco~=3.11.0"]
