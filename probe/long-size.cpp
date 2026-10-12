// long-size.cpp - how wide is `long` inside SYCL device code on this Windows toolchain? Answered at compile time, no GPU needed.
// Built by long-size.input.cmd. If the device pass sees a 4-byte long, the build fails on the static_assert below and
// says so; if it sees 8 bytes, the build succeeds. cute's new Xe 2D copy atoms pass GPU addresses as `long`.
#include <sycl/sycl.hpp>
#include <cstdio>

int main() {
  sycl::queue q;
  int* out = sycl::malloc_shared<int>(2, q);
  q.single_task([=] {
#ifdef __SYCL_DEVICE_ONLY__
    static_assert(sizeof(long) == 8, "DEVICE_LONG_IS_NOT_8_BYTES");
#endif
    out[0] = (int)sizeof(long);
    out[1] = (int)sizeof(void*);
  }).wait();
  std::printf("host sizeof(long) = %zu, device wrote sizeof(long) = %d, sizeof(void*) = %d\n", sizeof(long), out[0], out[1]);
  return 0;
}
