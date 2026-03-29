#!/bin/bash

export RMW_IMPLEMENTATION=rmw_zenoh_cpp
export ZENOH_CONFIG_OVERRIDE="transport/shared_memory/enabled=false"

/home/young/.pixi/bin/pixi run -m /home/young/ws_aic/aic_hilserl_env/pixi.toml \
  ros2 run aic_model aic_model \
  --ros-args \
  -p use_sim_time:=true \
  -p policy:=aic_example_policies.ros.HilSerlPolicy

# /home/young/.pixi/bin/pixi run -m /home/young/ws_aic/aic_hilserl_env/pixi.toml \
#   ros2 run aic_model aic_model \
#   --ros-args \
#   -p use_sim_time:=true \
#   -p policy:=aic_example_policies.ros.TestPolicy
