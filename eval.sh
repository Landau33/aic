#!/bin/bash

# 指定 GPU UUID
GPU_UUID="GPU-45a924c1-5d32-a7cc-ed8d-674d0179dcee"
CURRENT_CONFIG="/home/young/ws_aic/src/aic/aic_engine/config/sample_config.yaml"

# 将所有变量直接写在 bash -c 的引号内部
# 用分号 ; 将 export 语句和脚本路径连接起来
export __NV_PRIME_RENDER_OFFLOAD=1
export __GLX_VENDOR_LIBRARY_NAME=nvidia
export CUDA_VISIBLE_DEVICES=$GPU_UUID
export CUDA_DEVICE_ORDER=PCI_BUS_ID
/entrypoint.sh \
  ground_truth:=true \
  start_aic_engine:=true \
  aic_engine_config_file:=${CURRENT_CONFIG}
 