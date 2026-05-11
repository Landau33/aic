#!/bin/bash

CURRENT_CONFIG="${CURRENT_CONFIG:-/home/yuang/ws_aic/aic/aic_engine/config/sample_config_task_3.yaml}"

/entrypoint.sh \
  ground_truth:=true \
  start_aic_engine:=true \
  aic_engine_config_file:="${CURRENT_CONFIG}"
