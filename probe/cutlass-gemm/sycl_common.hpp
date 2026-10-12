// sycl_common.hpp (probe stand-in) - replaces examples/common/sycl_common.hpp for the 00_bmg_gemm build on this PC.
// Why (2026-10-10): the original pulls cutlass/util/reference/device/sycl_tensor_fill.h, which needs oneMKL's device RNG
// header (oneapi/mkl/rng/device.hpp); oneMKL is not installed here and the example only needs initialize_block().
// This fills with small integers (-2..2) from a hash of the index, deterministic, exact in bf16 and in fp32 sums, so the
// example's bitwise compare against its reference GEMM is meaningful. Found first on the include path.
#pragma once
#include <cstdint>
#include <cute/util/compat.hpp>
#include "cutlass/cutlass.h"
#include "cutlass/util/device_memory.h"

template <class, class> class probe_fill_name;

template <class Element>
bool initialize_block(Element* block, std::size_t size, uint64_t seed) {
  compat::get_default_queue().parallel_for<probe_fill_name<Element, void>>(sycl::range<1>(size), [=](sycl::id<1> idx) {
    uint64_t x = (static_cast<uint64_t>(idx[0]) + 1) * 0x9E3779B97F4A7C15ull ^ seed;
    x ^= x >> 29; x *= 0xBF58476D1CE4E5B9ull; x ^= x >> 32;
    block[idx[0]] = static_cast<Element>(static_cast<float>(static_cast<int>(x % 5) - 2));
  }).wait();
  return true;
}

template <class Element>
bool initialize_block(cutlass::DeviceAllocation<Element>& block, uint64_t seed) {
  return initialize_block<Element>(block.get(), block.size(), seed);
}
