// unified_memory_benchmark.cu — the cost of convenience.
//
// The rest of the project uses explicit device allocations + cudaMemcpy from
// pinned host memory. This benchmark measures what Unified Memory
// (cudaMallocManaged) costs for the same batched-lookup round trip, comparing:
//
//   explicit_pinned   cudaMalloc + cudaMemcpy from pinned host memory (baseline)
//   unified_naive     cudaMallocManaged, rely on on-demand page-fault migration
//   unified_prefetch  cudaMallocManaged + cudaMemPrefetchAsync hints
//
// Each timed region covers the realistic cycle: get the query batch onto the
// GPU, run the lookup kernel, and make the results readable on the host again
// (a host-side checksum touches every result, forcing it back).
//
// Usage: ./unified_memory_benchmark [out.csv]
#include <chrono>
#include <cstdio>
#include <string>
#include <vector>

#include "cuda_utils.cuh"
#include "gpu_hash_table.cuh"
#include "workload.cuh"

using namespace gpukv;
using Clock = std::chrono::high_resolution_clock;

static double ms_since(Clock::time_point t0) {
  return std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
}

// Touch every result on the host so its page must be resident there; returns a
// checksum to keep the compiler honest.
static Value host_checksum(const Value* out, size_t n) {
  Value sum = 0;
  for (size_t i = 0; i < n; ++i) sum += out[i];
  return sum;
}

// cudaMemPrefetchAsync's signature changed in CUDA 13: the old
// (ptr, bytes, int device) overload was replaced by one taking a
// cudaMemLocation. Wrap it so this benchmark builds on both toolkits.
// `device == cudaCpuDeviceId` requests migration to host memory.
static void prefetch(const void* p, size_t bytes, int device) {
#if CUDART_VERSION >= 13000
  cudaMemLocation loc{};
  loc.type = (device == cudaCpuDeviceId) ? cudaMemLocationTypeHost
                                         : cudaMemLocationTypeDevice;
  loc.id = (device == cudaCpuDeviceId) ? 0 : device;
  CUDA_CHECK(cudaMemPrefetchAsync(p, bytes, loc, 0));
#else
  CUDA_CHECK(cudaMemPrefetchAsync(p, bytes, device));
#endif
}

int main(int argc, char** argv) {
  const std::string out_path = argc > 1 ? argv[1] : "results/unified_memory.csv";
  const std::vector<size_t> sweep = {1000000, 4000000, 8000000};
  const int reps = 5;

  int dev = 0;
  CUDA_CHECK(cudaGetDevice(&dev));
  cudaDeviceProp prop{};
  CUDA_CHECK(cudaGetDeviceProperties(&prop, dev));

  std::FILE* csv = std::fopen(out_path.c_str(), "w");
  std::fprintf(csv, "n,variant,total_ms,mkeys_per_s\n");
  std::printf("Device: %s | concurrentManagedAccess=%d\n\n", prop.name,
              prop.concurrentManagedAccess);
  std::printf("%10s %-18s %10s %10s\n", "n", "variant", "total_ms", "Mkeys/s");

  auto report = [&](size_t n, const char* variant, double ms) {
    const double mk = (n / 1e6) / (ms / 1e3);
    std::printf("%10zu %-18s %10.3f %10.1f\n", n, variant, ms, mk);
    std::fprintf(csv, "%zu,%s,%.5f,%.3f\n", n, variant, ms, mk);
  };

  for (size_t n : sweep) {
    auto keys = make_keys(n, 201);
    auto vals = make_values(keys, 203);
    auto query = lookup_uniform(keys, n, 207);

    // Persistent table in ordinary device memory.
    Key* d_keys = nullptr;
    Value* d_vals = nullptr;
    CUDA_CHECK(cudaMalloc(&d_keys, n * sizeof(Key)));
    CUDA_CHECK(cudaMalloc(&d_vals, n * sizeof(Value)));
    CUDA_CHECK(cudaMemcpy(d_keys, keys.data(), n * sizeof(Key), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_vals, vals.data(), n * sizeof(Value), cudaMemcpyHostToDevice));
    LinearProbeTable table(2 * n);
    table.insert(d_keys, d_vals, n);
    CUDA_CHECK(cudaDeviceSynchronize());

    // ---- explicit pinned baseline -----------------------------------------
    {
      Key* h_query = nullptr;
      Value* h_out = nullptr;
      Key* dq = nullptr;
      Value* dout = nullptr;
      CUDA_CHECK(cudaHostAlloc(&h_query, n * sizeof(Key), cudaHostAllocDefault));
      CUDA_CHECK(cudaHostAlloc(&h_out, n * sizeof(Value), cudaHostAllocDefault));
      CUDA_CHECK(cudaMalloc(&dq, n * sizeof(Key)));
      CUDA_CHECK(cudaMalloc(&dout, n * sizeof(Value)));
      for (size_t i = 0; i < n; ++i) h_query[i] = query[i];

      auto once = [&]() {
        auto t0 = Clock::now();
        CUDA_CHECK(cudaMemcpy(dq, h_query, n * sizeof(Key), cudaMemcpyHostToDevice));
        table.find(dq, dout, n);
        CUDA_CHECK(cudaMemcpy(h_out, dout, n * sizeof(Value), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaDeviceSynchronize());
        volatile Value s = host_checksum(h_out, n);
        (void)s;
        return ms_since(t0);
      };
      once();  // warmup
      double ms = 0;
      for (int r = 0; r < reps; ++r) ms += once();
      report(n, "explicit_pinned", ms / reps);

      cudaFreeHost(h_query);
      cudaFreeHost(h_out);
      cudaFree(dq);
      cudaFree(dout);
    }

    // ---- unified memory ----------------------------------------------------
    Key* m_query = nullptr;
    Value* m_out = nullptr;
    CUDA_CHECK(cudaMallocManaged(&m_query, n * sizeof(Key)));
    CUDA_CHECK(cudaMallocManaged(&m_out, n * sizeof(Value)));
    for (size_t i = 0; i < n; ++i) m_query[i] = query[i];

    // naive: no hints. Before each rep, park both buffers on the host so the
    // kernel must fault-migrate the query in, and the checksum faults results
    // back -- the worst-case demand-paging path.
    {
      auto once = [&]() {
        prefetch(m_query, n * sizeof(Key), cudaCpuDeviceId);
        prefetch(m_out, n * sizeof(Value), cudaCpuDeviceId);
        CUDA_CHECK(cudaDeviceSynchronize());
        auto t0 = Clock::now();
        table.find(m_query, m_out, n);
        CUDA_CHECK(cudaDeviceSynchronize());
        volatile Value s = host_checksum(m_out, n);
        (void)s;
        return ms_since(t0);
      };
      once();
      double ms = 0;
      for (int r = 0; r < reps; ++r) ms += once();
      report(n, "unified_naive", ms / reps);
    }

    // prefetch: explicitly migrate query to GPU before the kernel and results
    // back to the CPU after -- managed memory used the way it should be.
    {
      auto once = [&]() {
        prefetch(m_query, n * sizeof(Key), cudaCpuDeviceId);
        prefetch(m_out, n * sizeof(Value), cudaCpuDeviceId);
        CUDA_CHECK(cudaDeviceSynchronize());
        auto t0 = Clock::now();
        prefetch(m_query, n * sizeof(Key), dev);
        table.find(m_query, m_out, n);
        prefetch(m_out, n * sizeof(Value), cudaCpuDeviceId);
        CUDA_CHECK(cudaDeviceSynchronize());
        volatile Value s = host_checksum(m_out, n);
        (void)s;
        return ms_since(t0);
      };
      once();
      double ms = 0;
      for (int r = 0; r < reps; ++r) ms += once();
      report(n, "unified_prefetch", ms / reps);
    }

    cudaFree(m_query);
    cudaFree(m_out);
    cudaFree(d_keys);
    cudaFree(d_vals);
    std::printf("\n");
  }

  std::fclose(csv);
  std::printf("Wrote %s\n", out_path.c_str());
  return 0;
}
