// poll-rate.cpp - how long does one device-memory poll take inside a kernel, i.e. what does a poll budget mean in seconds?
// Usage:  cmd /c probe\poll-rate.input.cmd      (writes probe\poll-rate.output.txt)
// Why (2026-10-11 08:55): the exchange's wait kernel gives up after N polls so it never runs into the ~2 s Windows TDR, but
// the budget was guessed (2,000,000) and the first vLLM run hit it during 14.7 MB exchanges. Measure a single lane polling
// a device int64 that never changes, for N polls, both as a single_task and as lane 0 of a 512-lane work-group.
#include <sycl/sycl.hpp>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>

template <class T>
using dev_atomic = sycl::atomic_ref<T, sycl::memory_order::relaxed, sycl::memory_scope::device,
                                    sycl::access::address_space::global_space>;

int main() {
  std::vector<sycl::device> cards;
  for (auto& p : sycl::platform::get_platforms())
    if (p.get_info<sycl::info::platform::name>().find("Level-Zero") != std::string::npos)
      for (auto& d : p.get_devices())
        if (d.is_gpu() && d.get_info<sycl::info::device::name>().find("B60") != std::string::npos) cards.push_back(d);
  if (cards.empty()) { std::printf("no B60\n"); return 3; }
  sycl::queue q{cards[0], sycl::property::queue::in_order{}};
  int64_t* flag = sycl::malloc_device<int64_t>(1, q);
  int64_t* out = sycl::malloc_device<int64_t>(1, q);
  q.memset(flag, 0, sizeof(int64_t)).wait();
  for (uint32_t n : {100000u, 1000000u, 4000000u}) {
    for (int mode = 0; mode < 2; ++mode) {
      // warm-up (JIT) then timed
      for (int rep = 0; rep < 2; ++rep) {
        auto t0 = std::chrono::steady_clock::now();
        if (mode == 0) {
          q.single_task([=]() {
            dev_atomic<int64_t> r(*flag); uint32_t polls = 0; int64_t v = 0;
            while (r.load() < 1) { v += 1; if (++polls >= n) break; }
            *out = v;
          });
        } else {
          q.parallel_for(sycl::nd_range<1>(512, 512), [=](sycl::nd_item<1> it) {
            if (it.get_local_id(0) == 0) {
              dev_atomic<int64_t> r(*flag); uint32_t polls = 0; int64_t v = 0;
              while (r.load() < 1) { v += 1; if (++polls >= n) break; }
              *out = v;
            }
            sycl::group_barrier(it.get_group());
          });
        }
        q.wait();
        double us = std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t0).count();
        if (rep == 1) std::printf("%-12s %9u polls: %10.0f us  -> %.1f ns per poll\n", mode == 0 ? "single_task" : "lane0/512", n, us, us * 1000.0 / n);
      }
    }
  }
  sycl::free(flag, q); sycl::free(out, q);
  return 0;
}
