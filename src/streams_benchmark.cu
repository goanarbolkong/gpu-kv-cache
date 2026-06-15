// streams_benchmark.cu — overlap host<->device transfers with kernel execution.
//
// A single large batch of lookups is processed two ways:
//   * synchronous : one H2D copy, one kernel, one D2H copy, in order.
//   * streamed    : the batch is split into chunks spread round-robin over
//                   several CUDA streams using *pinned* host memory, so the H2D
//                   of one chunk overlaps the kernel of another and the D2H of a
//                   third. Pinned memory is required for async copies to overlap.
//
// Reports wall-clock time for each and the resulting speedup. Run under Nsight
// Systems (scripts/profile_nsys.sh) to see the overlapped timeline.
//
// Usage: ./streams_benchmark [batch] [num_streams] [num_chunks]
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <vector>

#include "cuda_utils.cuh"
#include "gpu_hash_table.cuh"
#include "workload.cuh"

using namespace gpukv;
using Clock = std::chrono::high_resolution_clock;

static double ms_since(Clock::time_point t0) {
  return std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
}

int main(int argc, char** argv) {
  const size_t batch = argc > 1 ? std::strtoull(argv[1], nullptr, 10) : 16000000;
  const int num_streams = argc > 2 ? std::atoi(argv[2]) : 4;
  const int num_chunks = argc > 3 ? std::atoi(argv[3]) : 32;
  const int reps = 5;

  // Table holds `batch` distinct keys; queries are all hits (uniform).
  auto keys = make_keys(batch, 17);
  auto vals = make_values(keys, 19);
  auto query = lookup_uniform(keys, batch, 23);

  // Build the (persistent) table.
  Key* d_keys = nullptr;
  Value* d_vals = nullptr;
  CUDA_CHECK(cudaMalloc(&d_keys, batch * sizeof(Key)));
  CUDA_CHECK(cudaMalloc(&d_vals, batch * sizeof(Value)));
  CUDA_CHECK(cudaMemcpy(d_keys, keys.data(), batch * sizeof(Key), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_vals, vals.data(), batch * sizeof(Value), cudaMemcpyHostToDevice));
  BucketedTable table(2 * batch);
  table.insert(d_keys, d_vals, batch);
  CUDA_CHECK(cudaDeviceSynchronize());

  // Pinned host buffers (required for true async overlap).
  Key* h_query = nullptr;
  Value* h_out = nullptr;
  CUDA_CHECK(cudaHostAlloc(&h_query, batch * sizeof(Key), cudaHostAllocDefault));
  CUDA_CHECK(cudaHostAlloc(&h_out, batch * sizeof(Value), cudaHostAllocDefault));
  for (size_t i = 0; i < batch; ++i) h_query[i] = query[i];

  // Device query/result buffers (chunks use disjoint sub-ranges).
  Key* d_query = nullptr;
  Value* d_out = nullptr;
  CUDA_CHECK(cudaMalloc(&d_query, batch * sizeof(Key)));
  CUDA_CHECK(cudaMalloc(&d_out, batch * sizeof(Value)));

  std::vector<cudaStream_t> streams(num_streams);
  for (auto& s : streams) CUDA_CHECK(cudaStreamCreate(&s));

  auto verify = [&]() {
    size_t misses = 0;
    for (size_t i = 0; i < batch; ++i)
      if (h_out[i] == kNotFound) ++misses;
    return misses;
  };

  // ---- synchronous path --------------------------------------------------
  auto run_sync = [&]() {
    auto t0 = Clock::now();
    CUDA_CHECK(cudaMemcpy(d_query, h_query, batch * sizeof(Key), cudaMemcpyHostToDevice));
    table.find(d_query, d_out, batch);
    CUDA_CHECK(cudaMemcpy(h_out, d_out, batch * sizeof(Value), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaDeviceSynchronize());
    return ms_since(t0);
  };

  // ---- streamed path -----------------------------------------------------
  const size_t chunk = (batch + num_chunks - 1) / num_chunks;
  auto run_streamed = [&]() {
    auto t0 = Clock::now();
    for (int c = 0; c < num_chunks; ++c) {
      const size_t off = (size_t)c * chunk;
      if (off >= batch) break;
      const size_t len = std::min(chunk, batch - off);
      cudaStream_t s = streams[c % num_streams];
      CUDA_CHECK(cudaMemcpyAsync(d_query + off, h_query + off, len * sizeof(Key),
                                 cudaMemcpyHostToDevice, s));
      table.find(d_query + off, d_out + off, len, s);
      CUDA_CHECK(cudaMemcpyAsync(h_out + off, d_out + off, len * sizeof(Value),
                                 cudaMemcpyDeviceToHost, s));
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    return ms_since(t0);
  };

  cudaDeviceProp prop{};
  CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
  std::printf("Device: %s | batch=%zu | streams=%d | chunks=%d\n\n", prop.name,
              batch, num_streams, num_chunks);

  run_sync();      // warmup
  run_streamed();  // warmup

  double sync_ms = 0, stream_ms = 0;
  for (int r = 0; r < reps; ++r) sync_ms += run_sync();
  size_t sync_misses = verify();
  for (int r = 0; r < reps; ++r) stream_ms += run_streamed();
  size_t stream_misses = verify();
  sync_ms /= reps;
  stream_ms /= reps;

  std::printf("synchronous : %8.3f ms  (%.1f Mkeys/s)\n", sync_ms,
              (batch / 1e6) / (sync_ms / 1e3));
  std::printf("streamed    : %8.3f ms  (%.1f Mkeys/s)\n", stream_ms,
              (batch / 1e6) / (stream_ms / 1e3));
  std::printf("speedup     : %8.2fx\n", sync_ms / stream_ms);
  std::printf("correctness : sync misses=%zu, streamed misses=%zu (expect 0)\n",
              sync_misses, stream_misses);

  for (auto& s : streams) cudaStreamDestroy(s);
  cudaFreeHost(h_query);
  cudaFreeHost(h_out);
  cudaFree(d_query);
  cudaFree(d_out);
  cudaFree(d_keys);
  cudaFree(d_vals);
  return 0;
}
