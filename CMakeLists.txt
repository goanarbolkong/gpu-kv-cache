cmake_minimum_required(VERSION 3.20)

# Target GPU architecture must be chosen BEFORE project() enables the CUDA
# language, otherwise CMake locks in its own default. Defaults to sm_89 (Ada
# Lovelace, e.g. RTX 4060 Ti). Override on a different GPU, e.g.
#   cmake -B build -DCMAKE_CUDA_ARCHITECTURES=86   (Ampere)
#   cmake -B build -DCMAKE_CUDA_ARCHITECTURES=native
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES 89)
endif()

project(gpu_kv_cache LANGUAGES CXX CUDA)

# --- Standards ---------------------------------------------------------------
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "" FORCE)
endif()

set(CMAKE_CUDA_FLAGS_RELEASE "-O3")
# -lineinfo keeps source<->SASS mapping so Nsight Compute can attribute metrics
# to lines without the overhead of full -G debug builds.
add_compile_options($<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>
                    $<$<COMPILE_LANGUAGE:CUDA>:-lineinfo>)

# --- Core library ------------------------------------------------------------
add_library(gpukv STATIC src/gpu_hash_table.cu)
target_include_directories(gpukv PUBLIC ${CMAKE_SOURCE_DIR}/include)
# Whole-program compilation: all device code lives in one TU, so we embed SASS
# for the target arch directly and avoid a relocatable-device-code link that
# would otherwise force a runtime PTX JIT.

# --- Executables -------------------------------------------------------------
add_executable(demo src/demo.cu)
target_link_libraries(demo PRIVATE gpukv)

add_executable(benchmark src/benchmark.cu)
target_link_libraries(benchmark PRIVATE gpukv)

add_executable(streams_benchmark src/streams_benchmark.cu)
target_link_libraries(streams_benchmark PRIVATE gpukv)

add_executable(profile_kernels src/profile_kernels.cu)
target_link_libraries(profile_kernels PRIVATE gpukv)

add_executable(unified_memory_benchmark src/unified_memory_benchmark.cu)
target_link_libraries(unified_memory_benchmark PRIVATE gpukv)

# --- Tests -------------------------------------------------------------------
enable_testing()
add_executable(test_correctness tests/test_correctness.cu)
target_link_libraries(test_correctness PRIVATE gpukv)
add_test(NAME correctness COMMAND test_correctness)
