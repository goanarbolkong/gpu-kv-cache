#!/usr/bin/env bash
# Build (if needed) and run the full benchmark + load-factor study + streams
# overlap test, then render the plots.
set -euo pipefail
cd "$(dirname "$0")/.."

if [ ! -x build/benchmark ]; then
  cmake -B build -DCMAKE_BUILD_TYPE=Release
  cmake --build build -j
fi

mkdir -p results
echo "== correctness =="
ctest --test-dir build --output-on-failure

echo "== throughput sweep + load-factor study =="
./build/benchmark results/benchmark.csv

echo "== CUDA streams overlap =="
./build/streams_benchmark

echo "== Unified Memory vs explicit pinned =="
./build/unified_memory_benchmark results/unified_memory.csv

echo "== plots =="
python3 scripts/plot_results.py || echo "(install matplotlib to render plots)"
