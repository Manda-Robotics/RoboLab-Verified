"""Every USD asset must be a real file, not an unresolved Git LFS pointer (P08 / F4)."""
from pathlib import Path

ASSETS = Path(__file__).resolve().parents[1] / "assets"


def test_no_unresolved_lfs_pointers():
    stubs = [
        p for p in ASSETS.rglob("*.usd*")
        if p.is_file() and p.open("rb").read(40).startswith(b"version https://git-lfs")
    ]
    assert not stubs, (
        f"{len(stubs)} USD files are Git LFS pointer stubs — run `git lfs install && git lfs pull`. "
        f"First: {stubs[:3]}"
    )
