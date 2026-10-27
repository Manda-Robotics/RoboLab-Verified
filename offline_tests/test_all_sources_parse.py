"""Every Python file in the package and the dashboard must parse — the offline
suite does not import most of them, and a stray paren once reached a commit."""
import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_all_python_sources_parse():
    bad = []
    for d in ("robolab", "dashboard", "scripts", "policies"):
        for f in (ROOT / d).rglob("*.py"):
            try:
                ast.parse(f.read_text(), filename=str(f))
            except SyntaxError as e:
                bad.append(f"{f.relative_to(ROOT)}:{e.lineno}: {e.msg}")
    assert not bad, "\n".join(bad)


def _top_level_names(tree):
    names = set()
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names.add(node.name)
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            names.update((a.asname or a.name).split(".")[0] for a in node.names)
        elif isinstance(node, (ast.Assign, ast.AnnAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            for t in targets:
                names.update(n.id for n in ast.walk(t) if isinstance(n, ast.Name))
        elif isinstance(node, (ast.If, ast.Try)):
            blocks = [node.body, node.orelse, getattr(node, "finalbody", [])]
            blocks += [h.body for h in getattr(node, "handlers", [])]
            for block in blocks:
                names |= _top_level_names(ast.Module(body=block, type_ignores=[]))
    return names


# Broken before this check existed; each needs its own fix, and fixing one should delete its line.
KNOWN_BROKEN_IMPORTS = {
    # These names live in robolab.robots.franka_definitions.
    "robolab/registrations/example/auto_env_registration.py: robolab.robots.franka.FrankaJointPositionActionCfg",
    "robolab/registrations/example/auto_env_registration.py: robolab.robots.franka.contact_gripper",
    # Never committed to robolab.variations (8a5da4d).
    "robolab/registrations/aloha/auto_env_registrations.py: robolab.variations.backgrounds.DarkVoidBackgroundCfg",
    "robolab/registrations/aloha/auto_env_registrations.py: robolab.variations.lighting.GymMatchLightCfg",
}


def test_robolab_imports_name_something_that_exists():
    """A rename must update its importers: cf17e6e renamed droid._to_torch to
    isaaclab_compat.as_torch and the three bimanual rigs stopped importing."""
    bad = []
    for f in (ROOT / "robolab").rglob("*.py"):
        for node in ast.walk(ast.parse(f.read_text())):
            if not (isinstance(node, ast.ImportFrom) and node.level == 0
                    and (node.module or "").startswith("robolab.")):
                continue
            path = ROOT / node.module.replace(".", "/")
            src = path.with_suffix(".py") if path.with_suffix(".py").exists() else path / "__init__.py"
            if not src.exists():
                continue
            defined = _top_level_names(ast.parse(src.read_text()))
            submodules = {p.stem for p in path.glob("*.py")} | {p.name for p in path.glob("*/")}
            for alias in node.names:
                key = f"{f.relative_to(ROOT)}: {node.module}.{alias.name}"
                if alias.name != "*" and alias.name not in defined | submodules and key not in KNOWN_BROKEN_IMPORTS:
                    bad.append(f"{f.relative_to(ROOT)}:{node.lineno}: {node.module}.{alias.name}")
    assert not bad, "\n".join(bad)
