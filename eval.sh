#!/bin/bash

# 指定 GPU UUID
GPU_UUID="GPU-45a924c1-5d32-a7cc-ed8d-674d0179dcee"

# 将所有变量直接写在 bash -c 的引号内部
# 用分号 ; 将 export 语句和脚本路径连接起来
distrobox enter -r aic_eval -- bash -c "
export __NV_PRIME_RENDER_OFFLOAD=1
export __GLX_VENDOR_LIBRARY_NAME=nvidia
export CUDA_VISIBLE_DEVICES=$GPU_UUID
export CUDA_DEVICE_ORDER=PCI_BUS_ID
/entrypoint.sh ground_truth:=false start_aic_engine:=true
"