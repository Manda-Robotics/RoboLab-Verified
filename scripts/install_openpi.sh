#!/bin/bash
# Install the OpenPI policy server (pi0, pi0-FAST, pi05, PaliGemma) next to RoboLab, and
# the openpi-client package into RoboLab's Isaac venv.
#
# Two things that are easy to get wrong:
#   - GIT_LFS_SKIP_SMUDGE=1: a dependency carries an LFS object that 404s, and without
#     this the whole sync dies on `git clone` of that package.
#   - The client must be installed from git, not `-e ../openpi/...`: a later `uv sync`
#     on the RoboLab project prunes editable installs it does not know about, silently
#     removing openpi_client mid-session.
#
# Usage, from the RoboLab checkout:
#   OPENPI_DIR=../openpi VENV=.venv-51 scripts/install_openpi.sh
set -euxo pipefail
ROBOLAB_DIR="$(cd "$(dirname "$0")/.." && pwd)"
OPENPI_DIR="${OPENPI_DIR:-$ROBOLAB_DIR/../openpi}"
VENV="${VENV:-.venv}"
export GIT_LFS_SKIP_SMUDGE=1

[ -d "$OPENPI_DIR" ] || git clone --depth 1 https://github.com/xuningy/openpi.git "$OPENPI_DIR"
(cd "$OPENPI_DIR" && uv sync)

cd "$ROBOLAB_DIR"
uv pip install --python "$VENV/bin/python" \
  "openpi-client @ git+https://github.com/xuningy/openpi.git#subdirectory=packages/openpi-client"
"$VENV/bin/python" -c "import openpi_client; print('openpi_client ok')"
echo OPENPI_DONE
