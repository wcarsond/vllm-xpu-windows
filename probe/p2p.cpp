// p2p.cpp - what is the fastest path for a small tensor between the two B60s on this Windows driver? Built/run by p2p.input.cmd.
// Why (2026-10-10): a tensor split (both cards working every layer) needs ~56 exchanges of ~7 KB per token on the 7B; at
// today's relay cost (0.13 ms per hop over gloo) that eats the whole gain, at ~50 us it gives ~1.7x one card. This measures
// each candidate channel the driver offers: direct card-to-card copy (peer access), card->host->card through pinned host
// memory with no gloo in the way, and a kernel on card B reading host memory written by card A (zero-copy over PCIe).
// Numbers are median of repeats after a warm-up; sizes are decode (7 KB), small prefill (1 MB) and bulk (64 MB).
#include <sycl/sycl.hpp>
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

using clk = std::chrono::steady_clock;
static double med_us(std::vector<double>& v) { std::sort(v.begin(), v.end()); return v[v.size() / 2]; }

template <class F> static double time_us(F&& f, int reps) {
  f(); f();  // warm-up
  std::vector<double> t;
  for (int i = 0; i < reps; ++i) {
    auto a = clk::now(); f(); auto b = clk::now();
    t.push_back(std::chrono::duration<double, std::micro>(b - a).count());
  }
  return med_us(t);
}

int main(int argc, char** argv) {
  const std::string want = argc > 1 ? argv[1] : "B60";
  std::vector<sycl::device> cards;
  for (auto& p : sycl::platform::get_platforms())
    if (p.get_info<sycl::info::platform::name>().find("Level-Zero") != std::string::npos)
      for (auto& d : p.get_devices())
        if (d.is_gpu() && d.get_info<sycl::info::device::name>().find(want) != std::string::npos) cards.push_back(d);
  if (cards.size() < 2) { std::printf("FAIL: need two *%s* Level Zero devices, found %zu\n", want.c_str(), cards.size()); return 3; }
  sycl::device d0 = cards[0], d1 = cards[1];
  std::printf("card0 %s  card1 %s  driver %s\n", d0.get_info<sycl::info::device::name>().c_str(),
              d1.get_info<sycl::info::device::name>().c_str(), d0.get_info<sycl::info::device::driver_version>().c_str());

  // 1. does the driver say the cards can reach each other's memory?
  bool peer01 = false, peer10 = false, atomics01 = false;
  try {
    peer01 = d0.ext_oneapi_can_access_peer(d1, sycl::ext::oneapi::peer_access::access_supported);
    peer10 = d1.ext_oneapi_can_access_peer(d0, sycl::ext::oneapi::peer_access::access_supported);
    atomics01 = d0.ext_oneapi_can_access_peer(d1, sycl::ext::oneapi::peer_access::atomics_supported);
  } catch (const sycl::exception& e) { std::printf("peer query threw: %s\n", e.what()); }
  std::printf("peer access: card0->card1 %s, card1->card0 %s, atomics %s\n", peer01 ? "yes" : "no", peer10 ? "yes" : "no", atomics01 ? "yes" : "no");

  sycl::context ctx{{d0, d1}};
  sycl::queue q0{ctx, d0, sycl::property::queue::in_order{}}, q1{ctx, d1, sycl::property::queue::in_order{}};
  const size_t sizes[] = {7 * 1024, 1 << 20, 64 << 20};
  const char* names[] = {"7 KB (decode)", "1 MB", "64 MB"};
  for (int s = 0; s < 3; ++s) {
    size_t n = sizes[s];
    int reps = n > (8u << 20) ? 10 : 50;
    char* a0 = sycl::malloc_device<char>(n, q0);
    char* b1 = sycl::malloc_device<char>(n, q1);
    char* h = sycl::malloc_host<char>(n, ctx);
    q0.memset(a0, 1, n).wait(); q1.memset(b1, 0, n).wait(); std::memset(h, 0, n);
    std::printf("-- %s\n", names[s]);
    // 2. direct card0 -> card1 copy (needs peer access; otherwise the runtime stages or throws)
    if (peer01) {
      try {
        d0.ext_oneapi_enable_peer_access(d1);
        double us = time_us([&] { q0.memcpy(b1, a0, n).wait(); }, reps);
        std::printf("   card0->card1 direct copy:         %8.1f us  (%.2f GB/s)\n", us, n / us / 1e3);
      } catch (const sycl::exception& e) { std::printf("   direct copy threw: %s\n", e.what()); }
    } else {
      std::printf("   card0->card1 direct copy:         not offered by the driver\n");
    }
    // 3. card0 -> pinned host -> card1, two DMA copies, no gloo, no second process
    double us_stage = time_us([&] { q0.memcpy(h, a0, n).wait(); q1.memcpy(b1, h, n).wait(); }, reps);
    std::printf("   card0->host->card1 (two copies):  %8.1f us  (%.2f GB/s)\n", us_stage, n / us_stage / 1e3);
    // 4. zero-copy: a kernel on card1 reads host memory written by card0 (one DMA + one PCIe read, no second copy)
    double us_zc = time_us([&] {
      q0.memcpy(h, a0, n).wait();
      q1.parallel_for(sycl::range<1>(n / 4), [=](sycl::id<1> i) { ((int*)b1)[i] = ((const int*)h)[i]; }).wait();
    }, reps);
    std::printf("   card0->host, card1 kernel reads:  %8.1f us  (%.2f GB/s)\n", us_zc, n / us_zc / 1e3);
    // 5. the floor: one copy card0 -> host alone, and host -> card1 alone
    double us_d2h = time_us([&] { q0.memcpy(h, a0, n).wait(); }, reps);
    double us_h2d = time_us([&] { q1.memcpy(b1, h, n).wait(); }, reps);
    std::printf("   card0->host alone %7.1f us, host->card1 alone %7.1f us\n", us_d2h, us_h2d);
    // correctness of the last path
    std::vector<char> chk(std::min<size_t>(n, 4096));
    q1.memcpy(chk.data(), b1, chk.size()).wait();
    bool ok = std::all_of(chk.begin(), chk.end(), [](char c) { return c == 1; });
    std::printf("   data arrived intact: %s\n", ok ? "yes" : "NO");
    sycl::free(a0, ctx); sycl::free(b1, ctx); sycl::free(h, ctx);
  }
  return 0;
}
