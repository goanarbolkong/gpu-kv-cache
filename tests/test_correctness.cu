// test_correctness.cu — dependency-free correctness tests for both tables,
// cross-checked against std::unordered_map as the source of truth.
//
// Exits non-zero on any failure so it plugs straight into CTest.
#include <cstdio>
#include <unordered_map>
#include <vector>

#include "cuda_utils.cuh"
#include "gpu_hash_table.cuh"
#include "workload.cuh"

using namespace gpukv;

static int g_failures = 0;
#define CHECK(cond, msg)                                       \
  do {                                                         \
    if (!(cond)) {                                             \
      std::printf("  [FAIL] %s\n", msg);                       \
      ++g_failures;                                            \
    }                                                          \
  } while (0)

// Small device-pointer helpers to keep the test body readable.
template <typename T>
static T* to_device(const std::vector<T>& v) {
  T* d = nullptr;
  CUDA_CHECK(cudaMalloc(&d, v.size() * sizeof(T)));
  CUDA_CHECK(cudaMemcpy(d, v.data(), v.size() * sizeof(T), cudaMemcpyHostToDevice));
  return d;
}

template <typename Table>
static void test_table(const char* name) {
  std::printf("Testing %s\n", name);

  const size_t n = 200000;        // inserted keys
  const size_t q = 50000;         // absent-key queries
  auto keys = make_keys(n, /*seed=*/42);
  auto vals = make_values(keys, /*seed=*/7);

  // Reference map.
  std::unordered_map<Key, Value> ref;
  ref.reserve(n * 2);
  for (size_t i = 0; i < n; ++i) ref[keys[i]] = vals[i];

  Table table(/*capacity=*/2 * n);  // load factor 0.5
  Key* d_keys = to_device(keys);
  Value* d_vals = to_device(vals);
  table.insert(d_keys, d_vals, n);
  CUDA_CHECK(cudaDeviceSynchronize());

  // (1) Every inserted key resolves to its reference value.
  Value* d_out = nullptr;
  CUDA_CHECK(cudaMalloc(&d_out, n * sizeof(Value)));
  table.find(d_keys, d_out, n);
  CUDA_CHECK(cudaDeviceSynchronize());
  std::vector<Value> out(n);
  CUDA_CHECK(cudaMemcpy(out.data(), d_out, n * sizeof(Value), cudaMemcpyDeviceToHost));
  size_t hit_mismatch = 0;
  for (size_t i = 0; i < n; ++i)
    if (out[i] != ref[keys[i]]) ++hit_mismatch;
  CHECK(hit_mismatch == 0, "all inserted keys return their value");

  // (2) Keys never inserted must report kNotFound. Build absent keys by
  //     flipping a high bit so they cannot collide with the inserted set.
  std::vector<Key> absent(q);
  for (size_t i = 0; i < q; ++i) absent[i] = keys[i] ^ 0x8000000000000000ULL;
  // Drop any that happen to exist in the reference set.
  std::vector<Key> absent_clean;
  for (Key k : absent)
    if (ref.find(k) == ref.end() && k != kEmptyKey) absent_clean.push_back(k);
  Key* d_absent = to_device(absent_clean);
  Value* d_absent_out = nullptr;
  CUDA_CHECK(cudaMalloc(&d_absent_out, absent_clean.size() * sizeof(Value)));
  table.find(d_absent, d_absent_out, absent_clean.size());
  CUDA_CHECK(cudaDeviceSynchronize());
  std::vector<Value> absent_out(absent_clean.size());
  CUDA_CHECK(cudaMemcpy(absent_out.data(), d_absent_out,
                        absent_clean.size() * sizeof(Value), cudaMemcpyDeviceToHost));
  size_t false_hits = 0;
  for (Value v : absent_out)
    if (v != kNotFound) ++false_hits;
  CHECK(false_hits == 0, "absent keys report not-found");

  // (3) Update semantics: re-insert the same keys with value+1 -> last wins.
  std::vector<Value> vals2(n);
  for (size_t i = 0; i < n; ++i) vals2[i] = (vals[i] + 1) & 0x00FFFFFFFFFFFFFFULL;
  Value* d_vals2 = to_device(vals2);
  table.insert(d_keys, d_vals2, n);
  table.find(d_keys, d_out, n);
  CUDA_CHECK(cudaDeviceSynchronize());
  CUDA_CHECK(cudaMemcpy(out.data(), d_out, n * sizeof(Value), cudaMemcpyDeviceToHost));
  size_t update_mismatch = 0;
  for (size_t i = 0; i < n; ++i)
    if (out[i] != vals2[i]) ++update_mismatch;
  CHECK(update_mismatch == 0, "re-insert overwrites value (last writer wins)");

  cudaFree(d_keys);
  cudaFree(d_vals);
  cudaFree(d_vals2);
  cudaFree(d_out);
  cudaFree(d_absent);
  cudaFree(d_absent_out);
}

int main() {
  test_table<LinearProbeTable>("LinearProbeTable");
  test_table<BucketedTable>("BucketedTable");

  if (g_failures == 0) {
    std::printf("\nALL TESTS PASSED\n");
    return 0;
  }
  std::printf("\n%d CHECK(S) FAILED\n", g_failures);
  return 1;
}
