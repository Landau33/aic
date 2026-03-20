#!/usr/bin/bash
set -e

export RMW_IMPLEMENTATION=rmw_zenoh_cpp
export ZENOH_CONFIG_OVERRIDE="transport/shared_memory/enabled=false"

# Prefer workspace sources for local Python packages so edits take effect
# immediately without rebuilding the pixi environment.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PYTHONPATH="${SCRIPT_DIR}/aic_example_policies${PYTHONPATH:+:${PYTHONPATH}}"
