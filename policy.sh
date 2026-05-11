#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/pixi_env_setup.sh"
cd "${SCRIPT_DIR}"

# ros2 run aic_model aic_model \
#   --ros-args \
#   -p use_sim_time:=true \
#   -p policy:=aic_example_policies.ros.HilSerlPolicy

ros2 run aic_model aic_model \
  --ros-args \
  -p use_sim_time:=true \
  -p policy:=aic_example_policies.ros.TestPolicy
