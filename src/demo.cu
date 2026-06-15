// demo.cu — minimal end-to-end usage of both tables.
//
//   ./demo              uses the bucketed (warp-cooperative) table
//   ./demo --linear     uses the thread-per-key linear-probing table
//
// Inserts a handful of key/value pairs, looks them up (including a key that was
// never inserted), and prints the results.
#include <cstdio>
#include <cstring>
#include <vector>

#include "cuda_utils.cuh"
#include "gpu_hash_table.cuh"

using namespace gpukv;

template <typename Table>
static void run(const char* name) {
  std::printf("=== %s ===\n", name);

  const std::vector<Key> h_keys = {10, 20, 30, 40, 50};
  const std::vector<Value> h_vals = {1000, 2000, 3000, 4000, 5000};
  // Look up three present keys and one absent key (999).
  const std::vector<Key> h_query = {30, 10, 50, 999};

  const size_t n = h_keys.size();
  const size_t q = h_query.size();

  Key *d_keys, *d_query;
  Value *d_vals, *d_out;
  CUDA_CHECK(cudaMalloc(&d_keys, n * sizeof(Key)));
  CUDA_CHECK(cudaMalloc(&d_vals, n * sizeof(Value)));
  CUDA_CHECK(cudaMalloc(&d_query, q * sizeof(Key)));
  CUDA_CHECK(cudaMalloc(&d_out, q * sizeof(Value)));
  CUDA_CHECK(cudaMemcpy(d_keys, h_keys.data(), n * sizeof(Key), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_vals, h_vals.data(), n * sizeof(Value), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_query, h_query.data(), q * sizeof(Key), cudaMemcpyHostToDevice));

  Table table(/*capacity=*/4 * n);
  table.insert(d_keys, d_vals, n);
  table.find(d_query, d_out, q);
  CUDA_CHECK(cudaDeviceSynchronize());

  std::vector<Value> h_out(q);
  CUDA_CHECK(cudaMemcpy(h_out.data(), d_out, q * sizeof(Value), cudaMemcpyDeviceToHost));

  for (size_t i = 0; i < q; ++i) {
    if (h_out[i] == kNotFound)
      std::printf("  GET %3llu -> (not found)\n", h_query[i]);
    else
      std::printf("  GET %3llu -> %llu\n", h_query[i], h_out[i]);
  }
  std::printf("\n");

  cudaFree(d_keys);
  cudaFree(d_vals);
  cudaFree(d_query);
  cudaFree(d_out);
}

int main(int argc, char** argv) {
  const bool linear = (argc > 1 && std::strcmp(argv[1], "--linear") == 0);
  if (linear)
    run<LinearProbeTable>("LinearProbeTable (one thread per key)");
  else
    run<BucketedTable>("BucketedTable (one warp per key)");
  return 0;
}
