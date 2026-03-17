#!/bin/bash

echo "=== 强制使用 GPU 1 (NVIDIA RTX 4090) ==="
echo ""

# 设置 GPU 1 的环境变量
export CUDA_VISIBLE_DEVICES=1
export NVIDIA_VISIBLE_DEVICES=1
export CUDA_DEVICE_ORDER=PCI_BUS_ID

# 强制 NVIDIA GPU 渲染（避免使用 Intel 核显）
export __NV_PRIME_RENDER_OFFLOAD=1
export __GLX_VENDOR_LIBRARY_NAME=nvidia
export __VK_LAYER_NV_optimus=NVIDIA_only

echo "主机环境变量："
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
echo "__NV_PRIME_RENDER_OFFLOAD=$__NV_PRIME_RENDER_OFFLOAD"
echo "__GLX_VENDOR_LIBRARY_NAME=$__GLX_VENDOR_LIBRARY_NAME"
echo ""

# 记录开始前的 GPU 状态
echo "开始前的 GPU 状态："
nvidia-smi --query-gpu=name,index,utilization.gpu,memory.used --format=csv
echo ""

distrobox enter -r aic_dev_2404 -- bash -lc '
# 在容器内设置相同的环境变量
export CUDA_VISIBLE_DEVICES=1
export NVIDIA_VISIBLE_DEVICES=1
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export __NV_PRIME_RENDER_OFFLOAD=1
export __GLX_VENDOR_LIBRARY_NAME=nvidia

echo "容器内 GPU 信息："
nvidia-smi --query-gpu=name,index,uuid --format=csv
echo ""
echo "容器内环境变量："
env | grep -E "CUDA|NVIDIA|GLX" | sort
echo ""

cd ~/ws_aic/src/aic

echo "执行策略模型..."
echo "使用 GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader | sed -n 2p)"
echo ""

pixi run ros2 run aic_model aic_model \
  --ros-args \
  -p use_sim_time:=true \
  -p policy:=aic_example_policies.ros.CheatCode
'