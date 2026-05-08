#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Disaggregated serving on Intel XPU: prefill on tile 0, decode on tile 1.
# XPU Tiles: 2 (uses ZE_AFFINITY_MASK for device selection)

set -e
trap 'echo Cleaning up...; kill 0' EXIT

SCRIPT_DIR="$(dirname "$(readlink -f "$0")")"
source "$SCRIPT_DIR/../../../../common/gpu_utils.sh"   # build_sglang_gpu_mem_args
source "$SCRIPT_DIR/../../../../common/launch_utils.sh" # print_launch_banner, wait_any_exit

MODEL="${MODEL:-Qwen/Qwen3-0.6B}"

DISAGG_BOOTSTRAP_PORT="${DYN_DISAGG_BOOTSTRAP_PORT:-12345}"

# XPU device assignments (tile IDs)
PREFILL_DEVICE="${DYN_PREFILL_DEVICE:-0}"
DECODE_DEVICE="${DYN_DECODE_DEVICE:-1}"

HTTP_PORT="${DYN_HTTP_PORT:-8000}"
print_launch_banner "Launching Disaggregated Serving XPU (2 tiles)" "$MODEL" "$HTTP_PORT"

# run ingress
OTEL_SERVICE_NAME=dynamo-frontend \
python3 -m dynamo.frontend &

# run prefill worker
echo "Starting prefill worker on XPU tile $PREFILL_DEVICE..."
OTEL_SERVICE_NAME=dynamo-worker-prefill DYN_SYSTEM_PORT=${DYN_SYSTEM_PORT1:-8081} \
ZE_AFFINITY_MASK=$PREFILL_DEVICE \
python3 -m dynamo.sglang \
  --model-path "$MODEL" \
  --served-model-name "$MODEL" \
  --page-size 16 \
  --tp 1 \
  --trust-remote-code \
  --device xpu \
  --disaggregation-mode prefill \
  --disaggregation-bootstrap-port "$DISAGG_BOOTSTRAP_PORT" \
  --host 0.0.0.0 \
  --port 40000 \
  --disaggregation-transfer-backend nixl \
  --enable-metrics \
  --disable-cuda-graph \
  --grammar-backend none &

# run decode worker
echo "Starting decode worker on XPU tile $DECODE_DEVICE..."
OTEL_SERVICE_NAME=dynamo-worker-decode DYN_SYSTEM_PORT=${DYN_SYSTEM_PORT2:-8082} \
ZE_AFFINITY_MASK=$DECODE_DEVICE \
python3 -m dynamo.sglang \
  --model-path "$MODEL" \
  --served-model-name "$MODEL" \
  --page-size 16 \
  --tp 1 \
  --trust-remote-code \
  --device xpu \
  --disaggregation-mode decode \
  --disaggregation-bootstrap-port "$DISAGG_BOOTSTRAP_PORT" \
  --host 0.0.0.0 \
  --disaggregation-transfer-backend nixl \
  --enable-metrics \
  --disable-cuda-graph \
  --grammar-backend none &

# Exit on first worker failure; kill 0 in the EXIT trap tears down the rest
wait_any_exit
