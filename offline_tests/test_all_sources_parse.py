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
