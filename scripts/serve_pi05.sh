#!/bin/bash
# Serve pi0.5 for RoboLab: the *joint-position* checkpoint.
#
# RoboLab's pi0 client (policies/pi0_family/client.py, Pi0DroidJointposClient) and the
# DROID joint-position action space (DroidJointPositionActionCfg, use_default_offset=False)
# expect ABSOLUTE joint targets. OpenPI's convenience flag `serve_policy.py --env DROID`
# serves pi05_droid, which emits DELTAS -- the arm then wanders and never approaches the
# object, and the run looks like a harness regression (35 cm minimum hand-to-target
# distance against 1.3 cm for the correct checkpoint).
#
# Diagnostic if a run looks wrong: recorded actions with j3 ~ -1.9 and j5 ~ 2.3 are absolute
# and correct; values near zero mean the wrong checkpoint is being served.
#
# Usage: OPENPI_DIR=/path/to/openpi scripts/serve_pi05.sh
set -euo pipefail
OPENPI_DIR="${OPENPI_DIR:-$(dirname "$0")/../../openpi}"
cd "$OPENPI_DIR"
export XLA_PYTHON_CLIENT_MEM_FRACTION="${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.5}"
uv run scripts/serve_policy.py policy:checkpoint \
  --policy.config=pi05_droid_jointpos \
  --policy.dir=gs://openpi-assets-simeval/pi05_droid_jointpos
