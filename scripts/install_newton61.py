#!/usr/bin/env python3
"""Install the pinned Isaac Lab 3.0 GA source stack into a RoboLab Python 3.12 venv.

Run with the target venv's Python. Default is resolution only; --install applies
the plan. --replace-isaacsim first uninstalls the old Sim wheels to save space.
Assets, output directories, and other virtual environments are never touched.
"""

import argparse
import importlib.metadata as metadata
import json
import os
import platform
from pathlib import Path
import subprocess
import sys
import tomllib

ISAACLAB_REF = "v3.0.0"
ISAACLAB_COMMIT = "e3f7d9ff1f5d7f8280de3ece9000b7117f66bcf4"


def run(*args, **kwargs):
    return subprocess.run(args, check=True, text=True, **kwargs)


def prepare_requirements(lab, robolab):
    """Use upstream's workspace and overrides, not the obsolete beta2 wheel."""
    cfg = tomllib.loads((lab / "pyproject.toml").read_text())
    sources = cfg["tool"]["uv"]["sources"]
    requirements = []
    for requirement in cfg["project"]["dependencies"]:
        source = sources.get(requirement)
        if isinstance(source, dict) and "path" in source:
            requirements.append(f'-e {lab / source["path"]}')
        else:
            requirements.append(requirement)
    requirements.extend(cfg["project"]["optional-dependencies"]["isaacsim"])
    # usd-exchange already supplies pxr. RoboLab's offline [test] extra also
    # installs usd-core, which would overwrite the same USD Python files.
    requirements.extend([f"-e {robolab}", "pytest"])
    overrides = cfg["tool"]["uv"]["override-dependencies"]
    # uv pip does not inherit the workspace's torch source routing. Keep the
    # exact MuJoCo versions audited by our private solver adapters; upstream's
    # compatible-release ranges would otherwise admit unvalidated patch builds.
    replacements = {
        "torch==2.11.0": "torch==2.11.0+cu130",
        "torchaudio==2.11.0": "torchaudio==2.11.0+cu130",
        "torchvision==0.26.0": "torchvision==0.26.0+cu130",
        "mujoco~=3.12.0": "mujoco==3.12.0",
        "mujoco-warp~=3.12.0": "mujoco-warp==3.12.0",
    }
    overrides = [replacements.get(value, value) for value in overrides]
    return requirements, overrides


def prepare_checkout(lab):
    """Clone the release tag, then verify its immutable identity and clean tree."""
    if not lab.exists():
        run("git", "clone", "--depth", "1", "--branch", ISAACLAB_REF,
            "https://github.com/isaac-sim/IsaacLab.git", str(lab),
            env={**os.environ, "GIT_LFS_SKIP_SMUDGE": "1"})
    head = run("git", "-C", str(lab), "rev-parse", "HEAD", capture_output=True).stdout.strip()
    if head != ISAACLAB_COMMIT:
        raise ValueError(f"Expected Isaac Lab {ISAACLAB_REF} ({ISAACLAB_COMMIT}), found {head}; "
                         "use a new --isaaclab-root or a checkout of the pinned commit")
    if run("git", "-C", str(lab), "status", "--porcelain", capture_output=True).stdout.strip():
        raise ValueError("Isaac Lab checkout must be clean for a reproducible installation")
    return head


def check_usd_provider():
    """Refuse an in-place install that would mix two owners of the pxr package."""
    try:
        metadata.version("usd-core")
    except metadata.PackageNotFoundError:
        return
    raise ValueError("The target venv contains usd-core, but Isaac Lab 3.0 uses usd-exchange. "
                     "Use a fresh Python 3.12 venv; do not mix these USD providers.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--isaaclab-root", type=Path, required=True)
    parser.add_argument("--install", action="store_true")
    parser.add_argument("--replace-isaacsim", action="store_true")
    args = parser.parse_args()
    if (sys.version_info[:2] != (3, 12) or sys.prefix == sys.base_prefix
            or sys.platform != "linux" or platform.machine() != "x86_64"):
        parser.error("Run using the target Linux x86_64 Python 3.12 virtual environment's Python")
    if args.replace_isaacsim and not args.install:
        parser.error("--replace-isaacsim requires --install")
    lab = args.isaaclab_root.resolve()
    robolab = Path(__file__).resolve().parents[1]
    try:
        if args.install:
            check_usd_provider()
        head = prepare_checkout(lab)
    except ValueError as exc:
        parser.error(str(exc))
    # Fresh, durable logs beside the environment, never in episode result folders.
    import tempfile
    record = Path(tempfile.mkdtemp(prefix="newton61-install-", dir=robolab))
    requirements, overrides = prepare_requirements(lab, robolab)
    (record / "requirements.in").write_text("\n".join(requirements) + "\n")
    (record / "overrides.txt").write_text("\n".join(overrides) + "\n")
    (record / "stack.json").write_text(json.dumps({"isaaclab_ref": ISAACLAB_REF, "isaaclab_commit": head, "python": sys.executable}, indent=2))
    (record / "before.txt").write_text(run("uv", "pip", "freeze", "--python", sys.executable,
                                           capture_output=True).stdout)
    common = ["--python", sys.executable, "--index-strategy", "unsafe-best-match", "--prerelease", "allow",
              "--index", "https://pypi.nvidia.com", "--index", "https://download.pytorch.org/whl/cu130",
              "--override", str(record / "overrides.txt")]
    run("uv", "pip", "compile", *common, str(record / "requirements.in"),
        "--output-file", str(record / "resolved.txt"))
    print(f"Installation record: {record}", flush=True)
    if not args.install:
        return
    if args.replace_isaacsim:
        # Resolve exact installed distribution names before removing anything.
        old_sim = []
        for dist in metadata.distributions():
            name = dist.metadata["Name"].lower().replace("_", "-")
            if name == "isaacsim" or name.startswith("isaacsim-"):
                old_sim.append(dist.metadata["Name"])
        old_sim = sorted(set(old_sim))
        if old_sim:
            run("uv", "pip", "uninstall", "--python", sys.executable, *old_sim)
    # Vast base images may export UV_LINK_MODE=copy, doubling a ~20 GB Sim
    # installation during staging. Use a temporary cache beside this venv and
    # explicit hardlinks; deleting the cache afterward leaves the files intact.
    with tempfile.TemporaryDirectory(prefix=".newton61-cache-", dir=Path(sys.prefix).parent) as cache:
        run("uv", "pip", "install", *common, "--no-cache", "--link-mode", "hardlink",
            "--requirements", str(record / "resolved.txt"), env={**os.environ, "TMPDIR": cache})
    (record / "after.txt").write_text(run("uv", "pip", "freeze", "--python", sys.executable,
                                          capture_output=True).stdout)
    print("ROBOLAB_NEWTON61_INSTALL_COMPLETE", flush=True)


if __name__ == "__main__":
    main()
