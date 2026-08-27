#!/bin/bash
# Install the OpenPI policy server (serves pi0, pi0-fast, pi05, paligemma) plus
# the openpi-client package into RoboLab's eval venv.
#
# Two things that are easy to get wrong and cost an hour each:
#   - GIT_LFS_SKIP_SMUDGE=1: a dependency carries an LFS object that 404s, and
#     without this the whole sync dies on `git clone` of that package.
#   - The client must be installed from git, not `-e ../openpi/...`: any later
#     `uv sync` on the RoboLab project prunes editable installs it doesn't know
#     about, silently removing openpi_client mid-session.
set -euxo pipefail
export HOME=/root
export PATH="$HOME/.local/bin:$PATH"
export UV_CACHE_DIR=/workspace/.uv-cache
export GIT_LFS_SKIP_SMUDGE=1

cd /workspace
[ -d openpi ] || git clone --depth 1 https://github.com/xuningy/openpi.git
cd openpi
uv sync

cd /workspace/robolab
uv pip install --python .venv-51/bin/python \
  "openpi-client @ git+https://github.com/xuningy/openpi.git#subdirectory=packages/openpi-client"
.venv-51/bin/python -c "import openpi_client; print('openpi_client ok')"
echo OPENPI_DONE
