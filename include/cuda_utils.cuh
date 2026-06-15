// cuda_utils.cuh — tiny shared helpers: error checking + a CUDA event timer.
#pragma once

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>

namespace gpukv {

// Abort with file/line on any failed CUDA call. Wrap every runtime API call.
#define CUDA_CHECK(expr)                                                        \
  do {                                                                          \
    cudaError_t err__ = (expr);                                                 \
    if (err__ != cudaSuccess) {                                                 \
      std::fprintf(stderr, "CUDA error %s:%d: '%s' -> %s\n", __FILE__,          \
                   __LINE__, #expr, cudaGetErrorString(err__));                 \
      std::exit(EXIT_FAILURE);                                                  \
    }                                                                           \
  } while (0)

// Check for errors from the most recent kernel launch (async + sync).
inline void check_last_kernel(const char* what) {
  cudaError_t launch = cudaGetLastError();
  if (launch != cudaSuccess) {
    std::fprintf(stderr, "Kernel launch error (%s): %s\n", what,
                 cudaGetErrorString(launch));
    std::exit(EXIT_FAILURE);
  }
}

// RAII wall-clock timer for device work, measured with CUDA events (ms).
class GpuTimer {
 public:
  GpuTimer() {
    CUDA_CHECK(cudaEventCreate(&start_));
    CUDA_CHECK(cudaEventCreate(&stop_));
  }
  ~GpuTimer() {
    cudaEventDestroy(start_);
    cudaEventDestroy(stop_);
  }
  void start(cudaStream_t s = 0) { CUDA_CHECK(cudaEventRecord(start_, s)); }
  // Returns elapsed milliseconds between start() and this call.
  float stop(cudaStream_t s = 0) {
    CUDA_CHECK(cudaEventRecord(stop_, s));
    CUDA_CHECK(cudaEventSynchronize(stop_));
    float ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start_, stop_));
    return ms;
  }

 private:
  cudaEvent_t start_{};
  cudaEvent_t stop_{};
};

}  // namespace gpukv
