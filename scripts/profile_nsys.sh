#!/usr/bin/env bash
# Capture a Nsight Systems timeline of the streamed lookup pipeline. The trace
# shows H2D copies, kernels, and D2H copies overlapping across CUDA streams.
# Nsight Systems does not require special profiling permissions.
set -euo pipefail
cd "$(dirname "$0")/.."

BIN=build/streams_benchmark
OUT=results/nsys_streams
[ -x "$BIN" ] || { echo "build first: cmake --build build -j"; exit 1; }

nsys profile \
    --trace=cuda,nvtx,osrt \
    --force-overwrite=true \
    -o "$OUT" \
    "$BIN"

echo "Wrote ${OUT}.nsys-rep"
echo "Open with:  nsys-ui ${OUT}.nsys-rep"
echo
echo "Transfer/kernel overlap summary:"
nsys stats --report cuda_gpu_trace "${OUT}.nsys-rep" 2>/dev/null | head -30 || true
