#!/usr/bin/env bash
# Profile the lookup kernels with Nsight Compute (kernel-level metrics:
# memory throughput, sectors/request for coalescing, achieved occupancy).
#
# Nsight Compute needs access to GPU performance counters. If you hit
# "ERR_NVGPUCTRPERM", either run with sudo, or enable counter access for all
# users (persists across reboot):
#   sudo sh -c 'echo "options nvidia NVreg_RestrictProfilingToAdminUsers=0" \
#       > /etc/modprobe.d/nvidia-profiler.conf' && sudo reboot
set -euo pipefail
cd "$(dirname "$0")/.."

BIN=build/profile_kernels
OUT=results/ncu_lookup
[ -x "$BIN" ] || { echo "build first: cmake --build build -j"; exit 1; }

# --set full captures all sections. Restrict to the two lookup kernels; one
# launch each is enough since profile_kernels issues each exactly once.
ncu --set full \
    --kernel-name "regex:lp_find_kernel|bk_find_kernel" \
    --launch-count 2 \
    --force-overwrite \
    -o "$OUT" \
    "$BIN"

echo "Wrote ${OUT}.ncu-rep"
echo "Open with:  ncu-ui ${OUT}.ncu-rep    (or: ncu --import ${OUT}.ncu-rep --page details)"
