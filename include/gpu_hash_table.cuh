// gpu_hash_table.cuh — GPU-resident open-addressing key/value tables.
//
// Two implementations share one header so the benchmark can swap them in place:
//
//   LinearProbeTable  one thread per key, linear probing over a flat slot array.
//                     Simple and readable; the baseline GPU design.
//
//   BucketedTable     one warp per key, cooperative probing over 32-slot buckets.
//                     Each probe loads a full cache-line-friendly bucket so the
//                     32 lanes of a warp inspect it in a single coalesced access
//                     (the design used by NVIDIA's cuCollections / cuco).
//
// Keys and values are 64-bit. 0xFFF...F is reserved as the empty sentinel and is
// therefore not a storable key. Semantics are cache-like: re-inserting a key
// overwrites its value (last writer wins).
#pragma once

#include <cstddef>
#include <cstdint>
#include <cuda_runtime.h>

namespace gpukv {

using Key = unsigned long long;    // 64-bit; matches atomicCAS overloads
using Value = unsigned long long;

// Reserved sentinels. kEmptyKey is the byte pattern 0xFF*8, so a free table can
// be created with a single cudaMemset(..., 0xFF, ...).
constexpr Key kEmptyKey = 0xFFFFFFFFFFFFFFFFULL;
constexpr Value kNotFound = 0xFFFFFFFFFFFFFFFFULL;

// 64-bit MurmurHash3 finalizer: cheap, no multiplies on the critical path beyond
// two, and mixes well enough to keep probe chains short at load factor <= 0.7.
__host__ __device__ inline uint64_t hash64(uint64_t k) {
  k ^= k >> 33;
  k *= 0xff51afd7ed558ccdULL;
  k ^= k >> 33;
  k *= 0xc4ceb9fe1a85ec53ULL;
  k ^= k >> 33;
  return k;
}

// Round up to the next power of two (>= 1). Capacities are powers of two so the
// modulo reduction is a single bitwise AND with (capacity - 1).
inline size_t next_pow2(size_t n) {
  size_t p = 1;
  while (p < n) p <<= 1;
  return p;
}

// --------------------------------------------------------------------------
// Thread-per-key linear-probing table.
// --------------------------------------------------------------------------
class LinearProbeTable {
 public:
  // capacity is rounded up to a power of two. Pick capacity >= 2 * num_keys
  // (load factor <= 0.5) to keep probe chains short.
  explicit LinearProbeTable(size_t capacity);
  ~LinearProbeTable();

  LinearProbeTable(const LinearProbeTable&) = delete;
  LinearProbeTable& operator=(const LinearProbeTable&) = delete;

  // Batched ops over device pointers. Launch is asynchronous on `stream`.
  void insert(const Key* d_keys, const Value* d_values, size_t n,
              cudaStream_t stream = 0);
  void find(const Key* d_keys, Value* d_values_out, size_t n,
            cudaStream_t stream = 0);

  // Reset every slot to empty.
  void clear(cudaStream_t stream = 0);

  size_t capacity() const { return capacity_; }

 private:
  Key* d_keys_ = nullptr;
  Value* d_values_ = nullptr;
  size_t capacity_ = 0;
};

// --------------------------------------------------------------------------
// Warp-cooperative bucketed table. Slots are grouped into buckets of
// kBucketSize (== warp size). A warp handles one key: all 32 lanes probe one
// bucket together, vote with __ballot_sync, and elect a single lane to claim a
// slot with atomicCAS. Buckets are probed linearly on overflow.
// --------------------------------------------------------------------------
class BucketedTable {
 public:
  static constexpr int kBucketSize = 32;  // == warpSize on all current GPUs

  // num_buckets is rounded so that num_buckets * kBucketSize is a power of two.
  // Choose total capacity >= 2 * num_keys, same as the linear table.
  explicit BucketedTable(size_t capacity);
  ~BucketedTable();

  BucketedTable(const BucketedTable&) = delete;
  BucketedTable& operator=(const BucketedTable&) = delete;

  void insert(const Key* d_keys, const Value* d_values, size_t n,
              cudaStream_t stream = 0);
  void find(const Key* d_keys, Value* d_values_out, size_t n,
            cudaStream_t stream = 0);
  void clear(cudaStream_t stream = 0);

  size_t capacity() const { return num_buckets_ * kBucketSize; }
  size_t num_buckets() const { return num_buckets_; }

 private:
  Key* d_keys_ = nullptr;
  Value* d_values_ = nullptr;
  size_t num_buckets_ = 0;
};

}  // namespace gpukv
