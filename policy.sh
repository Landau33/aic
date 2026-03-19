#!/bin/bash

pixi run ros2 run aic_model aic_model \
  --ros-args \
  -p use_sim_time:=true \
  -p policy:=aic_example_policies.ros.TestPolicy
