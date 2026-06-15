// profile_kernels.cu — a deliberately minimal driver for Nsight Compute.
//
// It performs exactly one insert and one lookup for each table at a fixed,
// representative size (load factor ~0.7), so `ncu` has a small, clean set of
// kernel launches to attribute metrics to (memory throughput, sectors/request,
// achieved occupancy) without the noise of the full benchmark sweep.
#include <cstdio>
#include <vector>

#include "cuda_utils.cuh"
#include "gpu_hash_table.cuh"
#include "workload.cuh"

using namespace gpukv;

int main() {
  const size_t cap = size_t(1) << 23;  // 8,388,608 slots
  const size_t n = 6000000;            // load factor ~0.715

  auto keys = make_keys(n, 101);
  auto vals = make_values(keys, 103);
  auto query = lookup_uniform(keys, n, 107);

  Key *d_keys, *d_query;
  Value *d_vals, *d_out;
  CUDA_CHECK(cudaMalloc(&d_keys, n * sizeof(Key)));
  CUDA_CHECK(cudaMalloc(&d_vals, n * sizeof(Value)));
  CUDA_CHECK(cudaMalloc(&d_query, n * sizeof(Key)));
  CUDA_CHECK(cudaMalloc(&d_out, n * sizeof(Value)));
  CUDA_CHECK(cudaMemcpy(d_keys, keys.data(), n * sizeof(Key), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_vals, vals.data(), n * sizeof(Value), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_query, query.data(), n * sizeof(Key), cudaMemcpyHostToDevice));

  {
    LinearProbeTable lin(cap);
    lin.insert(d_keys, d_vals, n);   // lp_insert_kernel
    CUDA_CHECK(cudaDeviceSynchronize());
    lin.find(d_query, d_out, n);     // lp_find_kernel
    CUDA_CHECK(cudaDeviceSynchronize());
  }
  {
    BucketedTable buck(cap);
    buck.insert(d_keys, d_vals, n);  // bk_insert_kernel
    CUDA_CHECK(cudaDeviceSynchronize());
    buck.find(d_query, d_out, n);    // bk_find_kernel
    CUDA_CHECK(cudaDeviceSynchronize());
  }

  cudaFree(d_keys); cudaFree(d_vals); cudaFree(d_query); cudaFree(d_out);
  std::printf("profile_kernels done (n=%zu, capacity=%zu, lf=%.3f)\n", n, cap,
              (double)n / cap);
  return 0;
}
