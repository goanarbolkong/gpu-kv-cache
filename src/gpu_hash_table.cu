// gpu_hash_table.cu — kernels and host wrappers for both table designs.
#include "gpu_hash_table.cuh"
#include "cuda_utils.cuh"

namespace gpukv {

namespace {
constexpr unsigned kFullMask = 0xFFFFFFFFu;  // all 32 lanes active
constexpr int kLinearBlock = 256;
constexpr int kBucketBlock = 128;  // 4 warps / block
}  // namespace

// ===========================================================================
// LinearProbeTable kernels (one thread per key)
// ===========================================================================

__global__ void lp_insert_kernel(Key* keys, Value* values, size_t mask,
                                  const Key* in_keys, const Value* in_vals,
                                  size_t n) {
  const size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
  if (i >= n) return;

  const Key key = in_keys[i];
  const Value val = in_vals[i];
  size_t slot = hash64(key) & mask;

  // Probe at most `capacity` slots; bail if the table is full (cache: drop).
  for (size_t probe = 0; probe <= mask; ++probe) {
    const Key prev = atomicCAS(&keys[slot], kEmptyKey, key);
    if (prev == kEmptyKey || prev == key) {
      values[slot] = val;  // claimed an empty slot, or updating an existing key
      return;
    }
    slot = (slot + 1) & mask;  // collision -> linear probe
  }
}

__global__ void lp_find_kernel(const Key* keys, const Value* values, size_t mask,
                               const Key* in_keys, Value* out_vals, size_t n) {
  const size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
  if (i >= n) return;

  const Key key = in_keys[i];
  size_t slot = hash64(key) & mask;

  for (size_t probe = 0; probe <= mask; ++probe) {
    const Key cur = keys[slot];
    if (cur == key) {
      out_vals[i] = values[slot];
      return;
    }
    if (cur == kEmptyKey) break;  // an empty slot means the key was never inserted
    slot = (slot + 1) & mask;
  }
  out_vals[i] = kNotFound;
}

// ===========================================================================
// BucketedTable kernels (one warp per key, cooperative probing)
// ===========================================================================

__global__ void bk_insert_kernel(Key* keys, Value* values, size_t num_buckets,
                                  const Key* in_keys, const Value* in_vals,
                                  size_t n) {
  const unsigned lane = threadIdx.x & 31u;
  const size_t warp_id = (blockIdx.x * (size_t)blockDim.x + threadIdx.x) >> 5;
  if (warp_id >= n) return;  // warp-uniform: all 32 lanes share warp_id

  const Key key = in_keys[warp_id];
  const Value val = in_vals[warp_id];
  const size_t bmask = num_buckets - 1;
  size_t bucket = hash64(key) & bmask;

  const size_t kMaxAttempts = num_buckets * BucketedTable::kBucketSize;
  size_t advanced = 0;  // number of distinct buckets visited

  for (size_t attempt = 0; attempt < kMaxAttempts && advanced < num_buckets;
       ++attempt) {
    const size_t base = bucket * BucketedTable::kBucketSize;
    const Key slot_key = keys[base + lane];  // one coalesced 32-wide load

    // (1) Key already present in this bucket? Update its value, done.
    const unsigned match = __ballot_sync(kFullMask, slot_key == key);
    if (match) {
      const int leader = __ffs(match) - 1;
      if (lane == leader) values[base + leader] = val;
      return;
    }

    // (2) No empty slot here -> bucket is full of other keys, advance.
    const unsigned empties = __ballot_sync(kFullMask, slot_key == kEmptyKey);
    if (empties == 0) {
      bucket = (bucket + 1) & bmask;
      ++advanced;
      continue;
    }

    // (3) Try to claim the lowest-indexed empty slot with one CAS.
    const int leader = __ffs(empties) - 1;
    Key prev = kEmptyKey;
    if (lane == leader) prev = atomicCAS(&keys[base + leader], kEmptyKey, key);
    prev = __shfl_sync(kFullMask, prev, leader);

    if (prev == kEmptyKey || prev == key) {
      if (lane == leader) values[base + leader] = val;  // won the slot / same key
      return;
    }
    // CAS lost the race to another warp: re-read this bucket and retry. This is
    // what prevents duplicate keys -- a concurrent same-key insert is caught by
    // the match test on the next iteration.
  }
}

__global__ void bk_find_kernel(const Key* keys, const Value* values,
                               size_t num_buckets, const Key* in_keys,
                               Value* out_vals, size_t n) {
  const unsigned lane = threadIdx.x & 31u;
  const size_t warp_id = (blockIdx.x * (size_t)blockDim.x + threadIdx.x) >> 5;
  if (warp_id >= n) return;

  const Key key = in_keys[warp_id];
  const size_t bmask = num_buckets - 1;
  size_t bucket = hash64(key) & bmask;

  for (size_t step = 0; step < num_buckets; ++step) {
    const size_t base = bucket * BucketedTable::kBucketSize;
    const Key slot_key = keys[base + lane];

    const unsigned match = __ballot_sync(kFullMask, slot_key == key);
    if (match) {
      const int leader = __ffs(match) - 1;
      Value v = values[base + leader];
      v = __shfl_sync(kFullMask, v, leader);
      if (lane == 0) out_vals[warp_id] = v;
      return;
    }
    // An empty slot in the bucket means the key cannot be anywhere further.
    if (__ballot_sync(kFullMask, slot_key == kEmptyKey)) break;
    bucket = (bucket + 1) & bmask;
  }
  if (lane == 0) out_vals[warp_id] = kNotFound;
}

// ===========================================================================
// LinearProbeTable host wrappers
// ===========================================================================

LinearProbeTable::LinearProbeTable(size_t capacity) {
  capacity_ = next_pow2(capacity);
  CUDA_CHECK(cudaMalloc(&d_keys_, capacity_ * sizeof(Key)));
  CUDA_CHECK(cudaMalloc(&d_values_, capacity_ * sizeof(Value)));
  clear();
  CUDA_CHECK(cudaDeviceSynchronize());
}

LinearProbeTable::~LinearProbeTable() {
  cudaFree(d_keys_);
  cudaFree(d_values_);
}

void LinearProbeTable::clear(cudaStream_t stream) {
  // 0xFF bytes == kEmptyKey. Values need no init (only read after a key match).
  CUDA_CHECK(cudaMemsetAsync(d_keys_, 0xFF, capacity_ * sizeof(Key), stream));
}

void LinearProbeTable::insert(const Key* d_keys, const Value* d_values, size_t n,
                              cudaStream_t stream) {
  if (n == 0) return;
  const unsigned grid = (n + kLinearBlock - 1) / kLinearBlock;
  lp_insert_kernel<<<grid, kLinearBlock, 0, stream>>>(
      d_keys_, d_values_, capacity_ - 1, d_keys, d_values, n);
  check_last_kernel("lp_insert");
}

void LinearProbeTable::find(const Key* d_keys, Value* d_values_out, size_t n,
                            cudaStream_t stream) {
  if (n == 0) return;
  const unsigned grid = (n + kLinearBlock - 1) / kLinearBlock;
  lp_find_kernel<<<grid, kLinearBlock, 0, stream>>>(
      d_keys_, d_values_, capacity_ - 1, d_keys, d_values_out, n);
  check_last_kernel("lp_find");
}

// ===========================================================================
// BucketedTable host wrappers
// ===========================================================================

BucketedTable::BucketedTable(size_t capacity) {
  const size_t total = next_pow2(capacity);
  num_buckets_ = total / kBucketSize;
  if (num_buckets_ == 0) num_buckets_ = 1;  // at least one bucket
  const size_t slots = num_buckets_ * kBucketSize;
  CUDA_CHECK(cudaMalloc(&d_keys_, slots * sizeof(Key)));
  CUDA_CHECK(cudaMalloc(&d_values_, slots * sizeof(Value)));
  clear();
  CUDA_CHECK(cudaDeviceSynchronize());
}

BucketedTable::~BucketedTable() {
  cudaFree(d_keys_);
  cudaFree(d_values_);
}

void BucketedTable::clear(cudaStream_t stream) {
  const size_t slots = num_buckets_ * kBucketSize;
  CUDA_CHECK(cudaMemsetAsync(d_keys_, 0xFF, slots * sizeof(Key), stream));
}

void BucketedTable::insert(const Key* d_keys, const Value* d_values, size_t n,
                           cudaStream_t stream) {
  if (n == 0) return;
  const unsigned warps_per_block = kBucketBlock / 32;
  const unsigned grid = (n + warps_per_block - 1) / warps_per_block;
  bk_insert_kernel<<<grid, kBucketBlock, 0, stream>>>(
      d_keys_, d_values_, num_buckets_, d_keys, d_values, n);
  check_last_kernel("bk_insert");
}

void BucketedTable::find(const Key* d_keys, Value* d_values_out, size_t n,
                         cudaStream_t stream) {
  if (n == 0) return;
  const unsigned warps_per_block = kBucketBlock / 32;
  const unsigned grid = (n + warps_per_block - 1) / warps_per_block;
  bk_find_kernel<<<grid, kBucketBlock, 0, stream>>>(
      d_keys_, d_values_, num_buckets_, d_keys, d_values_out, n);
  check_last_kernel("bk_find");
}

}  // namespace gpukv
