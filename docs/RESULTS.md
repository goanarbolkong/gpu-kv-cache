# Results

All numbers below were measured on the machine described under **Setup**. Raw
data is in [`results/benchmark.csv`](../results/benchmark.csv) and
[`results/loadfactor.csv`](../results/loadfactor.csv); regenerate everything with
`./scripts/run_benchmark.sh`.

## Setup

| | |
|---|---|
| GPU | NVIDIA GeForce RTX 4060 Ti (Ada Lovelace, **sm_89**), 8 GB, 288 GB/s peak DRAM |
| Host | x86-64, Ubuntu 24.04, g++ 13.3 |
| Toolkit | CUDA 13.3, compiled `-O3 -lineinfo` for `sm_89` |
| Key/Value | 64-bit / 64-bit, empty sentinel `0xFFFF…FF` |
| Timing | CUDA events (device), `std::chrono` (host); 5 timed reps after 1 warmup |

## 1. CPU vs GPU lookups — the parallelism crossover

End-to-end GPU time **includes** H2D query upload and D2H result download.
Throughput in Mkeys/s (higher is better).

| batch (keys) | CPU `unordered_map` | GPU linear (e2e) | GPU bucketed (e2e) | GPU vs CPU |
|---:|---:|---:|---:|---:|
| 1,000 | 177.6 | 62.0 | 57.3 | **CPU wins** |
| 10,000 | 135.6 | 272.3 | 159.8 | GPU 2.0× |
| 100,000 | 86.8 | 428.3 | 382.1 | GPU 4.9× |
| 1,000,000 | 41.5 | 532.2 | 484.4 | GPU 12.8× |
| 8,000,000 | 32.0 | 432.7 | 397.1 | GPU 13.5× |

**Crossover point ≈ 3–4K keys.** Below it, the fixed cost of launching a kernel
and copying the batch over PCIe outweighs the parallelism; above it the GPU pulls
away to a >13× end-to-end win at large batches. See
[`results/crossover.png`](../results/crossover.png).

## 2. Host–device transfer overhead

For large batches the lookup *kernel* is no longer the bottleneck — PCIe
transfers are. Breakdown of one 8M-key GPU linear lookup (ms):

| H2D upload | kernel | D2H download | total | transfer share |
|---:|---:|---:|---:|---:|
| 5.59 | 7.21 | 5.69 | 18.49 | **61%** |

This is the entire motivation for the streamed pipeline in §5: if 61% of the
wall-clock is copies, overlapping them with compute is where the time is.

## 3. Inserts (concurrent `atomicCAS`)

GPU insert throughput vs single-threaded `unordered_map` build (kernel-only for GPU):

| batch (keys) | CPU build | GPU linear | GPU bucketed | speedup |
|---:|---:|---:|---:|---:|
| 1,000,000 | 13.6 | 1307.8 | 1818.4 | ~96–134× |
| 8,000,000 | 10.5 | 484.8 | 458.3 | ~44–46× |

Thousands of threads claim slots concurrently with `atomicCAS`; correctness under
this contention is verified by the cross-check against `unordered_map` in the
test suite.

## 4. Load factor — where warp-cooperative probing wins

Fixed capacity (2²³ = 8,388,608 slots), varying key count, **kernel-only** uniform
lookups. This isolates probing efficiency from transfer cost.

| load factor | linear (Mkeys/s) | bucketed (Mkeys/s) | winner |
|---:|---:|---:|---:|
| 0.50 | 1433 | 1202 | linear |
| 0.70 | 1085 | 1045 | linear |
| 0.85 | 835 | 952 | **bucketed +14%** |
| 0.95 | 606 | 897 | **bucketed +48%** |

**This is the key design result.** At low load factor, probe chains are short and
the simple thread-per-key table is fastest — it does the least work per key. As
the table fills, linear probing's chains grow and its throughput collapses (606
Mkeys/s at 0.95). The bucketed table degrades far more gracefully: each probe
loads a whole 32-slot bucket in one coalesced transaction and 32 lanes inspect it
in parallel, so worst-case probe length is bounded by buckets visited, not slots.
The crossover is around load factor **0.8**. See
[`results/loadfactor.png`](../results/loadfactor.png).

Takeaway for sizing a real cache: provision for load factor ≤ 0.7 and use the
thread-per-key table; if memory forces a fuller table, switch to the bucketed
design.

## 5. CUDA streams — overlapping transfers with compute

16M-key lookup, batch split into 32 chunks over 4 streams with pinned host memory
(`./build/streams_benchmark`):

| path | wall-clock | throughput |
|---|---:|---:|
| synchronous (1 copy → kernel → 1 copy) | 40.75 ms | 392.7 Mkeys/s |
| streamed (overlapped) | 23.11 ms | 692.3 Mkeys/s |
| **speedup** | **1.76×** | |

Both paths return identical, fully-correct results (0 missed keys). The speedup
comes from hiding H2D/D2H behind kernel execution — consistent with the 61%
transfer share measured in §2.

## 6. Unified Memory — the cost of convenience

Everything above uses explicit `cudaMalloc` + `cudaMemcpy` from **pinned** host
memory. Unified Memory (`cudaMallocManaged`) is the convenient alternative: one
pointer valid on host and device, with the runtime migrating pages on demand.
This measures what that convenience costs for the same lookup round trip
(query up → kernel → results touched on host), `./build/unified_memory_benchmark`:

| batch (keys) | explicit pinned | unified (naive) | unified (prefetch) |
|---:|---:|---:|---:|
| 1,000,000 | 1.90 ms / 527.8 | 3.97 ms / 252.2 | 3.09 ms / 323.6 |
| 4,000,000 | 9.67 ms / 413.6 | 17.10 ms / 234.0 | 13.34 ms / 299.8 |
| 8,000,000 | 19.82 ms / 403.7 | 32.88 ms / 243.3 | 25.69 ms / 311.5 |

(ms = mean total round-trip; second number is Mkeys/s, higher is better.)

**Explicit pinned wins at every size.** Naive Unified Memory — relying on
on-demand page-fault migration — is **~1.7× slower** (32.9 ms vs 19.8 ms at 8M)
because pages fault in one at a time during the kernel and fault back out during
the host checksum, serialising migration with compute. Adding
`cudaMemPrefetchAsync` hints to bulk-migrate the query to the GPU before the
kernel and the results back to the host afterwards recovers most of that gap
(25.7 ms), but still trails explicit pinned transfers by **~1.3×**.

The takeaway matches the thesis of §2/§5: when transfers are a measurable share
of wall-clock, controlling them explicitly — pinned staging buffers, and
overlapping copies with compute via streams — beats letting the runtime page
memory for you. Unified Memory's real wins are elsewhere (programmer time,
memory oversubscription beyond VRAM via `cudaMemAdvise`), not raw throughput on a
known, batchable access pattern.

## 7. Nsight findings

### Nsight Systems (`results/nsys_streams.nsys-rep`)

The timeline confirms the overlap: in the synchronous phase H2D, `bk_find_kernel`,
and D2H run strictly back-to-back on the default stream; in the streamed phase the
chunks' copies and kernels interleave across 4 streams. Measured from the trace:

- `bk_find_kernel` over 16M keys: **20.3 ms** (≈ 789 Mkeys/s, kernel-only).
- `bk_insert_kernel` over 16M keys: **37.8 ms** (≈ 423 Mkeys/s).
- **Pinned** H2D of 128 MB: ~12.4 GB/s vs **pageable** ~8.2 GB/s — pinned memory
  alone is ~1.5× faster on the wire (RTX 4060 Ti is PCIe 4.0 ×8).
- Register usage: `bk_insert` 28 reg/thread, `bk_find` 19 reg/thread — low enough
  not to limit occupancy on Ada.

### Nsight Compute (`scripts/profile_ncu.sh`)

Kernel-level counter metrics (achieved occupancy, sectors/request for coalescing,
DRAM throughput vs the 288 GB/s roofline) require GPU performance-counter access.
On a fresh machine this needs a one-time, admin-only enable (documented at the top
of `scripts/profile_ncu.sh`); the script is ready to capture a full `--set full`
report once that is done. Expected story to confirm there: the bucketed kernel
should show markedly fewer sectors-per-request (better coalescing) than the linear
kernel at high load factor, explaining the §4 crossover.
