#!/usr/bin/bash

export RMW_IMPLEMENTATION=rmw_zenoh_cpp
export ZENOH_CONFIG_OVERRIDE="transport/shared_memory/enabled=false"

# Prefer workspace sources for local Python packages so edits take effect
# immediately without rebuilding the pixi environment.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
HIL_SERL_DIR="$(cd "${SCRIPT_DIR}/../hil-serl_aic" && pwd)"
PIXI_ENV_DIR="${CONDA_PREFIX:-${SCRIPT_DIR}/.pixi/envs/default}"
ORIGINAL_PYTHONPATH="${PYTHONPATH:-}"

if [ -f "${PIXI_ENV_DIR}/setup.sh" ]; then
  source "${PIXI_ENV_DIR}/setup.sh"
fi

export PYTHONPATH="${SCRIPT_DIR}/aic_example_policies:${HIL_SERL_DIR}/serl_launcher:${HIL_SERL_DIR}/serl_robot_infra${ORIGINAL_PYTHONPATH:+:${ORIGINAL_PYTHONPATH}}"
export PATH="${PIXI_ENV_DIR}/bin:${SCRIPT_DIR}/toolkit${PATH:+:${PATH}}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-${SCRIPT_DIR}/.pixi/matplotlib}"
