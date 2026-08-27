#!/bin/bash
# pi0.5 for RoboLab: the *jointpos* checkpoint.
#
# Pi0DroidJointposClient + DroidJointPositionActionCfg(use_default_offset=False) expect
# ABSOLUTE joint targets. openpi's default `--env DROID` serves pi05_droid, which emits
# DELTAS -- the arm then flails and never approaches the object (pod 2026-08-26: 35 cm
# minimum hand-target distance against 1.3 cm in the corpus).
#
# Diagnostic if a run looks wrong: recorded actions with j3 ~ -1.9 and j5 ~ 2.3 are
# absolute and correct; near-zero values mean the wrong checkpoint is being served.
cd /workspace/openpi
export HOME=/root PATH="/root/.local/bin:$PATH" UV_CACHE_DIR=/workspace/.uv-cache XLA_PYTHON_CLIENT_MEM_FRACTION=0.5
uv run scripts/serve_policy.py policy:checkpoint \
  --policy.config=pi05_droid_jointpos \
  --policy.dir=gs://openpi-assets-simeval/pi05_droid_jointpos
