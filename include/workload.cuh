// workload.cuh — host-side data generation for tests and benchmarks.
//
// Provides distinct-key generation plus two lookup-key distributions:
//   * uniform  — every inserted key equally likely (classic throughput test)
//   * zipfian  — a few "hot" keys dominate, which concentrates atomic traffic
//                on the same slots and exposes contention behaviour.
#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <random>
#include <vector>

#include "gpu_hash_table.cuh"

namespace gpukv {

// SplitMix64 — fast bijective-ish mixer used to turn a counter into a pseudo
// random 64-bit key. Collisions over n << 2^64 keys are astronomically rare.
inline uint64_t splitmix64(uint64_t x) {
  x += 0x9e3779b97f4a7c15ULL;
  x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ULL;
  x = (x ^ (x >> 27)) * 0x94d049bb133111ebULL;
  return x ^ (x >> 31);
}

// n distinct keys, none equal to the reserved empty sentinel.
inline std::vector<Key> make_keys(size_t n, uint64_t seed = 1) {
  std::vector<Key> keys(n);
  for (size_t i = 0; i < n; ++i) {
    Key k = splitmix64(seed * 0x100000001b3ULL + i);
    if (k == kEmptyKey) k = 0;  // remap the one forbidden value
    keys[i] = k;
  }
  return keys;
}

// Arbitrary values paired with keys (here: value = derived from index).
inline std::vector<Value> make_values(const std::vector<Key>& keys,
                                      uint64_t seed = 2) {
  std::vector<Value> vals(keys.size());
  for (size_t i = 0; i < keys.size(); ++i)
    vals[i] = splitmix64(seed * 0x9e3779b1ULL + i) & 0x00FFFFFFFFFFFFFFULL;
  return vals;
}

// Uniformly sample `batch` lookup keys from the inserted set (all are hits).
inline std::vector<Key> lookup_uniform(const std::vector<Key>& keys,
                                       size_t batch, uint64_t seed = 3) {
  std::mt19937_64 rng(seed);
  std::uniform_int_distribution<size_t> pick(0, keys.size() - 1);
  std::vector<Key> out(batch);
  for (size_t i = 0; i < batch; ++i) out[i] = keys[pick(rng)];
  return out;
}

// Zipfian sampler over a bounded "hot" domain via an inverse-CDF table.
// Exponent s controls skew (s=0 -> uniform, s~1 -> heavy skew). The domain is
// capped so the CDF stays small even for huge key sets, while still producing a
// realistic hot-key pattern.
class ZipfSampler {
 public:
  ZipfSampler(size_t domain, double s, uint64_t seed)
      : cdf_(domain), rng_(seed) {
    double norm = 0.0;
    for (size_t i = 0; i < domain; ++i) norm += 1.0 / std::pow((double)(i + 1), s);
    double acc = 0.0;
    for (size_t i = 0; i < domain; ++i) {
      acc += (1.0 / std::pow((double)(i + 1), s)) / norm;
      cdf_[i] = acc;
    }
  }
  // Returns a rank in [0, domain).
  size_t next() {
    const double u = unif_(rng_);
    auto it = std::lower_bound(cdf_.begin(), cdf_.end(), u);
    size_t r = (size_t)(it - cdf_.begin());
    return r < cdf_.size() ? r : cdf_.size() - 1;
  }

 private:
  std::vector<double> cdf_;
  std::mt19937_64 rng_;
  std::uniform_real_distribution<double> unif_{0.0, 1.0};
};

// Zipfian-distributed lookup keys drawn from the inserted set.
inline std::vector<Key> lookup_zipf(const std::vector<Key>& keys, size_t batch,
                                    double s = 0.99, uint64_t seed = 4) {
  const size_t domain = std::min<size_t>(keys.size(), size_t(1) << 20);
  ZipfSampler zipf(domain, s, seed);
  std::vector<Key> out(batch);
  for (size_t i = 0; i < batch; ++i) out[i] = keys[zipf.next()];
  return out;
}

}  // namespace gpukv
