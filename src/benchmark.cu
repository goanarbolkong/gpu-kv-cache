// benchmark.cu — GPU vs CPU throughput across batch sizes.
//
// Builds a table of N keys, then times batched lookups of N keys on:
//   * CPU   std::unordered_map
//   * GPU   LinearProbeTable   (one thread per key)
//   * GPU   BucketedTable      (one warp per key)
//
// For the GPU it reports H2D, kernel, and D2H separately so the host-device
// transfer overhead -- the thing that decides the parallelism crossover point
// -- is visible. Inserts are timed too. Results go to a CSV and a stdout table.
//
// Usage: ./benchmark [out.csv]
#include <chrono>
#include <cstdio>
#include <string>
#include <unordered_map>
#include <vector>

#include "cuda_utils.cuh"
#include "gpu_hash_table.cuh"
#include "workload.cuh"

using namespace gpukv;
using Clock = std::chrono::high_resolution_clock;

namespace {

constexpr int kReps = 5;  // timed repetitions; we report the average

double ms_since(Clock::time_point t0) {
  return std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
}

// One row of the CSV / stdout table.
struct Row {
  const char* distribution;
  size_t n;
  const char* impl;
  const char* op;
  double h2d_ms;
  double kernel_ms;
  double d2h_ms;
  double total_ms;
};

double mkeys_per_s(size_t n, double total_ms) {
  return total_ms > 0 ? (n / 1e6) / (total_ms / 1e3) : 0.0;
}

void print_header() {
  std::printf("%-7s %10s %-13s %-7s %9s %9s %9s %10s %10s\n", "dist", "n",
              "impl", "op", "h2d_ms", "kern_ms", "d2h_ms", "total_ms",
              "Mkeys/s");
}

void print_row(const Row& r) {
  std::printf("%-7s %10zu %-13s %-7s %9.3f %9.3f %9.3f %10.3f %10.1f\n",
              r.distribution, r.n, r.impl, r.op, r.h2d_ms, r.kernel_ms,
              r.d2h_ms, r.total_ms, mkeys_per_s(r.n, r.total_ms));
}

void write_row(std::FILE* f, const Row& r) {
  std::fprintf(f, "%s,%zu,%s,%s,%.5f,%.5f,%.5f,%.5f,%.3f\n", r.distribution,
               r.n, r.impl, r.op, r.h2d_ms, r.kernel_ms, r.d2h_ms, r.total_ms,
               mkeys_per_s(r.n, r.total_ms));
}

// Time a GPU lookup pass, broken into H2D / kernel / D2H. The table is already
// populated. `d_query` already holds the query keys on device.
template <typename Table>
Row time_gpu_lookup(Table& table, const char* impl, const char* dist, size_t n,
                    const std::vector<Key>& h_query) {
  Key* d_query = nullptr;
  Value* d_out = nullptr;
  CUDA_CHECK(cudaMalloc(&d_query, n * sizeof(Key)));
  CUDA_CHECK(cudaMalloc(&d_out, n * sizeof(Value)));

  GpuTimer timer;
  double h2d = 0, kern = 0, d2h = 0;

  for (int rep = 0; rep < kReps + 1; ++rep) {
    timer.start();
    CUDA_CHECK(cudaMemcpy(d_query, h_query.data(), n * sizeof(Key),
                          cudaMemcpyHostToDevice));
    float t_h2d = timer.stop();

    timer.start();
    table.find(d_query, d_out, n);
    float t_kern = timer.stop();

    std::vector<Value> h_out(n);
    timer.start();
    CUDA_CHECK(cudaMemcpy(h_out.data(), d_out, n * sizeof(Value),
                          cudaMemcpyDeviceToHost));
    float t_d2h = timer.stop();

    if (rep > 0) {  // rep 0 is warmup
      h2d += t_h2d;
      kern += t_kern;
      d2h += t_d2h;
    }
  }
  cudaFree(d_query);
  cudaFree(d_out);

  h2d /= kReps;
  kern /= kReps;
  d2h /= kReps;
  return Row{dist, n, impl, "lookup", h2d, kern, d2h, h2d + kern + d2h};
}

// Time a GPU insert pass (clears the table each rep so we measure cold inserts).
template <typename Table>
Row time_gpu_insert(Table& table, const char* impl, size_t n,
                    const std::vector<Key>& h_keys,
                    const std::vector<Value>& h_vals) {
  Key* d_keys = nullptr;
  Value* d_vals = nullptr;
  CUDA_CHECK(cudaMalloc(&d_keys, n * sizeof(Key)));
  CUDA_CHECK(cudaMalloc(&d_vals, n * sizeof(Value)));
  CUDA_CHECK(cudaMemcpy(d_keys, h_keys.data(), n * sizeof(Key), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_vals, h_vals.data(), n * sizeof(Value), cudaMemcpyHostToDevice));

  GpuTimer timer;
  double kern = 0;
  for (int rep = 0; rep < kReps + 1; ++rep) {
    table.clear();
    CUDA_CHECK(cudaDeviceSynchronize());
    timer.start();
    table.insert(d_keys, d_vals, n);
    float t = timer.stop();
    if (rep > 0) kern += t;
  }
  cudaFree(d_keys);
  cudaFree(d_vals);
  kern /= kReps;
  return Row{"uniform", n, impl, "insert", 0, kern, 0, kern};
}

// Lookup kernel-only time for an already-populated table (no transfers), used
// by the load-factor study to isolate probing efficiency.
template <typename Table>
double time_lookup_kernel(Table& table, Key* d_query, Value* d_out, size_t n) {
  GpuTimer timer;
  double kern = 0;
  for (int rep = 0; rep < kReps + 1; ++rep) {
    timer.start();
    table.find(d_query, d_out, n);
    float t = timer.stop();
    if (rep > 0) kern += t;
  }
  return kern / kReps;
}

// Study: at fixed key count, sweep the load factor and compare lookup kernel
// time for the two designs. This isolates the regime where the warp-cooperative
// bucketed table's coalesced full-bucket loads beat thread-per-key probing.
void load_factor_study(const std::string& out_path) {
  // Capacity is fixed at a power of two; we vary the key count to hit each
  // target load factor (varying capacity instead would just round back to the
  // same power of two and collapse the sweep).
  const size_t cap = size_t(1) << 23;  // 8,388,608 slots
  const size_t n_max = (size_t)(0.95 * cap);
  auto keys = make_keys(n_max, 31);
  auto vals = make_values(keys, 37);
  auto q = lookup_uniform(keys, n_max, 41);

  Key *d_keys, *d_query;
  Value *d_vals, *d_out;
  CUDA_CHECK(cudaMalloc(&d_keys, n_max * sizeof(Key)));
  CUDA_CHECK(cudaMalloc(&d_vals, n_max * sizeof(Value)));
  CUDA_CHECK(cudaMalloc(&d_query, n_max * sizeof(Key)));
  CUDA_CHECK(cudaMalloc(&d_out, n_max * sizeof(Value)));
  CUDA_CHECK(cudaMemcpy(d_keys, keys.data(), n_max * sizeof(Key), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_vals, vals.data(), n_max * sizeof(Value), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_query, q.data(), n_max * sizeof(Key), cudaMemcpyHostToDevice));

  std::FILE* csv = std::fopen(out_path.c_str(), "w");
  std::fprintf(csv, "load_factor,impl,kernel_ms,mkeys_per_s\n");
  std::printf("\n=== Load-factor study (capacity=%zu fixed, uniform lookups, kernel-only) ===\n", cap);
  std::printf("%-12s %-13s %10s %10s\n", "load_factor", "impl", "kern_ms", "Mkeys/s");

  for (double lf : {0.50, 0.70, 0.85, 0.95}) {
    const size_t n = (size_t)(lf * cap);
    {
      LinearProbeTable t(cap);
      t.insert(d_keys, d_vals, n);
      CUDA_CHECK(cudaDeviceSynchronize());
      const double ms = time_lookup_kernel(t, d_query, d_out, n);
      std::printf("%-12.2f %-13s %10.3f %10.1f\n", lf, "gpu_linear", ms, mkeys_per_s(n, ms));
      std::fprintf(csv, "%.2f,gpu_linear,%.5f,%.3f\n", lf, ms, mkeys_per_s(n, ms));
    }
    {
      BucketedTable t(cap);
      t.insert(d_keys, d_vals, n);
      CUDA_CHECK(cudaDeviceSynchronize());
      const double ms = time_lookup_kernel(t, d_query, d_out, n);
      std::printf("%-12.2f %-13s %10.3f %10.1f\n", lf, "gpu_bucketed", ms, mkeys_per_s(n, ms));
      std::fprintf(csv, "%.2f,gpu_bucketed,%.5f,%.3f\n", lf, ms, mkeys_per_s(n, ms));
    }
  }
  std::fclose(csv);
  cudaFree(d_keys); cudaFree(d_vals); cudaFree(d_query); cudaFree(d_out);
  std::printf("Wrote %s\n", out_path.c_str());
}

}  // namespace

int main(int argc, char** argv) {
  const std::string out_path = argc > 1 ? argv[1] : "results/benchmark.csv";
  const std::vector<size_t> sweep = {1000,    10000,    100000,
                                     1000000, 4000000,  8000000};

  std::FILE* csv = std::fopen(out_path.c_str(), "w");
  if (!csv) {
    std::fprintf(stderr, "could not open %s for writing\n", out_path.c_str());
    return 1;
  }
  std::fprintf(csv, "distribution,n,impl,op,h2d_ms,kernel_ms,d2h_ms,total_ms,mkeys_per_s\n");

  cudaDeviceProp prop{};
  CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
  // memoryClockRate / memoryBusWidth were removed from cudaDeviceProp in CUDA
  // 13; query them as device attributes instead. Peak GB/s = 2 (DDR) * clock
  // (kHz) * bus bytes / 1e6.
  int mem_clock_khz = 0, bus_bits = 0;
  cudaDeviceGetAttribute(&mem_clock_khz, cudaDevAttrMemoryClockRate, 0);
  cudaDeviceGetAttribute(&bus_bits, cudaDevAttrGlobalMemoryBusWidth, 0);
  std::printf("Device: %s | sm_%d%d | %.0f GB/s peak DRAM\n\n", prop.name,
              prop.major, prop.minor,
              2.0 * mem_clock_khz * (bus_bits / 8) / 1.0e6);
  print_header();

  for (size_t n : sweep) {
    auto keys = make_keys(n, /*seed=*/123);
    auto vals = make_values(keys, /*seed=*/9);

    // ---- CPU baseline: build + lookup -------------------------------------
    std::unordered_map<Key, Value> ref;
    auto t0 = Clock::now();
    ref.reserve(n * 2);
    for (size_t i = 0; i < n; ++i) ref[keys[i]] = vals[i];
    Row cpu_insert{"uniform", n, "cpu", "insert", 0, ms_since(t0), 0, ms_since(t0)};

    auto q_uniform = lookup_uniform(keys, n, 5);
    auto q_zipf = lookup_zipf(keys, n, 0.99, 6);

    double cpu_uni = 0, cpu_zip = 0;
    for (int rep = 0; rep < kReps; ++rep) {
      volatile Value sink = 0;
      t0 = Clock::now();
      for (size_t i = 0; i < n; ++i) { auto it = ref.find(q_uniform[i]); sink += (it != ref.end()) ? it->second : 0; }
      cpu_uni += ms_since(t0);
      t0 = Clock::now();
      for (size_t i = 0; i < n; ++i) { auto it = ref.find(q_zipf[i]); sink += (it != ref.end()) ? it->second : 0; }
      cpu_zip += ms_since(t0);
      (void)sink;
    }
    cpu_uni /= kReps;
    cpu_zip /= kReps;

    // ---- GPU: linear + bucketed -------------------------------------------
    Key* d_keys = nullptr;
    Value* d_vals = nullptr;
    CUDA_CHECK(cudaMalloc(&d_keys, n * sizeof(Key)));
    CUDA_CHECK(cudaMalloc(&d_vals, n * sizeof(Value)));
    CUDA_CHECK(cudaMemcpy(d_keys, keys.data(), n * sizeof(Key), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_vals, vals.data(), n * sizeof(Value), cudaMemcpyHostToDevice));

    std::vector<Row> rows;
    rows.push_back(cpu_insert);
    rows.push_back(Row{"uniform", n, "cpu", "lookup", 0, cpu_uni, 0, cpu_uni});
    rows.push_back(Row{"zipfian", n, "cpu", "lookup", 0, cpu_zip, 0, cpu_zip});

    {
      LinearProbeTable lin(2 * n);
      lin.insert(d_keys, d_vals, n);
      CUDA_CHECK(cudaDeviceSynchronize());
      rows.push_back(time_gpu_insert(lin, "gpu_linear", n, keys, vals));
      lin.clear(); lin.insert(d_keys, d_vals, n); CUDA_CHECK(cudaDeviceSynchronize());
      rows.push_back(time_gpu_lookup(lin, "gpu_linear", "uniform", n, q_uniform));
      rows.push_back(time_gpu_lookup(lin, "gpu_linear", "zipfian", n, q_zipf));
    }
    {
      BucketedTable buck(2 * n);
      buck.insert(d_keys, d_vals, n);
      CUDA_CHECK(cudaDeviceSynchronize());
      rows.push_back(time_gpu_insert(buck, "gpu_bucketed", n, keys, vals));
      buck.clear(); buck.insert(d_keys, d_vals, n); CUDA_CHECK(cudaDeviceSynchronize());
      rows.push_back(time_gpu_lookup(buck, "gpu_bucketed", "uniform", n, q_uniform));
      rows.push_back(time_gpu_lookup(buck, "gpu_bucketed", "zipfian", n, q_zipf));
    }

    for (const Row& r : rows) { print_row(r); write_row(csv, r); }
    std::printf("\n");

    cudaFree(d_keys);
    cudaFree(d_vals);
  }

  std::fclose(csv);
  std::printf("Wrote %s\n", out_path.c_str());

  load_factor_study("results/loadfactor.csv");
  return 0;
}
