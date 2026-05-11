#!/usr/bin/env bash

set -euo pipefail

if [[ $# -lt 5 || $# -gt 6 ]]; then
  echo "Usage: $0 <left|center|right> <width> <height> <offset_x> <offset_y> [node_name]" >&2
  echo "Example: $0 right 400 300 -60 -20 /aic_model" >&2
  exit 1
fi

camera="$1"
width="$2"
height="$3"
offset_x="$4"
offset_y="$5"
node_name="${6:-/aic_model}"

case "$camera" in
  left|center|right)
    ;;
  *)
    echo "Invalid camera: $camera (expected left, center, or right)" >&2
    exit 1
    ;;
esac

if ! [[ "$width" =~ ^[0-9]+$ ]]; then
  echo "width must be a positive integer" >&2
  exit 1
fi

if ! [[ "$height" =~ ^[0-9]+$ ]]; then
  echo "height must be a positive integer" >&2
  exit 1
fi

ros2 service call "$node_name/set_parameters" rcl_interfaces/srv/SetParameters "{
parameters: [
  {name: 'hil_serl.roi.$camera.width', value: {type: 2, integer_value: $width}},
  {name: 'hil_serl.roi.$camera.height', value: {type: 2, integer_value: $height}},
  {name: 'hil_serl.roi.$camera.offset_x', value: {type: 3, double_value: $offset_x}},
  {name: 'hil_serl.roi.$camera.offset_y', value: {type: 3, double_value: $offset_y}}
]}"
