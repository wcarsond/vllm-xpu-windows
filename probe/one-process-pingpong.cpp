// one-process-pingpong.cpp - in ONE process, can a kernel on B60 #1 see tagged words a kernel on B60 #0 writes into one host
// USM buffer, and how fast is the round trip? (The Linux side's idea 1 / item 4, 2026-10-13.)
// Both cards in one sycl::context, one sycl::malloc_host buffer in that context. Two persistent single-work-group kernels:
//   card 0: for s = 1..N: write the tagged words of round s (data + tag s) into ping[]; wait until every word of pong[]
//           carries tag s; check its data
//   card 1: for s = 1..N: wait until every word of ping[] carries tag s; check; write pong[] with tag s
// Writes use the uncached store hint (8-byte scalar), reads a 32-byte cache-bypassing load per lane + acquire fence (the pattern the exchange already
// relies on for host-written words). Each round is one full card0 -> card1 -> card0 trip of `words` tagged words.
// Usage:  one-process-pingpong.exe <words per message> <rounds>          (run by one-process-pingpong.input.cmd)
// Why: on Windows a kernel cannot touch another PROCESS's memory, so the decode-size exchange relays through a CPU helper
// in each process (28 us per all_reduce, oneCCL on Linux 13 us). If two cards in one process can poll each other through
// one host buffer, a helper-free decode exchange is possible (with a single-process two-card executor); if they cannot,
// the CPU relay is the floor on Windows.
#include <sycl/sycl.hpp>
#include <sycl/ext/oneapi/experimental/annotated_ptr/annotated_ptr.hpp>
#include <sycl/ext/intel/experimental/cache_control_properties.hpp>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

namespace iex = sycl::ext::intel::experimental;
namespace oex = sycl::ext::oneapi::experimental;
using wprops = decltype(oex::properties(
    iex::write_hint<iex::cache_control<iex::cache_mode::uncached, oex::cache_level::L1, oex::cache_level::L3>>));
using rprops = decltype(oex::properties(
    iex::read_hint<iex::cache_control<iex::cache_mode::uncached, oex::cache_level::L1, oex::cache_level::L2, oex::cache_level::L3>>));

using u64x4 = sycl::vec<uint64_t, 4>;
static inline uint64_t tagged(uint32_t d, uint32_t t) { return (uint64_t)d | ((uint64_t)t << 32); }

int main(int argc, char** argv) {
  std::setvbuf(stdout, nullptr, _IONBF, 0);
  const size_t words = argc > 1 ? std::atoll(argv[1]) : 896;    // 896 x 8 B = 7 KB of tagged words (3.5 KB data)
  const int rounds = argc > 2 ? std::atoi(argv[2]) : 2000;
  const int WG = 512;
  const uint32_t SPIN = 6000000;   // ~1 s at 150-180 ns a poll, under the 2 s Windows GPU timeout (TDR)

  std::vector<sycl::device> cards;
  for (auto& p : sycl::platform::get_platforms())
    if (p.get_info<sycl::info::platform::name>().find("Level-Zero") != std::string::npos)
      for (auto& d : p.get_devices())
        if (d.is_gpu() && d.get_info<sycl::info::device::name>().find("B60") != std::string::npos) cards.push_back(d);
  if (cards.size() < 2) { std::printf("need two B60s, found %zu\n", cards.size()); return 3; }
  sycl::context ctx({cards[0], cards[1]});
  sycl::queue q0(ctx, cards[0], sycl::property::queue::in_order{});
  sycl::queue q1(ctx, cards[1], sycl::property::queue::in_order{});
  std::printf("one context, two cards: %s | %s\n", cards[0].get_info<sycl::info::device::name>().c_str(),
              cards[1].get_info<sycl::info::device::name>().c_str());

  uint64_t* ping = sycl::malloc_host<uint64_t>(words, ctx);
  uint64_t* pong = sycl::malloc_host<uint64_t>(words, ctx);
  uint32_t* stat = sycl::malloc_host<uint32_t>(64, ctx);   // [0] card0 wrong, [1] card1 wrong, [2] card0 timeouts, [3] card1 timeouts
  for (size_t i = 0; i < words; ++i) { ping[i] = 0; pong[i] = 0; }
  for (int i = 0; i < 64; ++i) stat[i] = 0;

  auto t0 = std::chrono::steady_clock::now();
  sycl::event e1, e0;
  try {
    e1 = q1.parallel_for(sycl::nd_range<1>(WG, WG), [=](sycl::nd_item<1> it) {
      size_t lid = it.get_local_id(0);
      oex::annotated_ptr<const u64x4, rprops> rp4{reinterpret_cast<const u64x4*>(ping)};
      oex::annotated_ptr<uint64_t, wprops> wp{pong};
      for (int s = 1; s <= rounds; ++s) {
        for (size_t g = lid; g < words / 4; g += WG) {
          u64x4 v; uint32_t n = 0;
          for (;;) {
            { u64x4 t = rp4[g]; v = t; }
            sycl::atomic_fence(sycl::memory_order::acquire, sycl::memory_scope::system);
            if ((uint32_t)(v[0] >> 32) == (uint32_t)s && (uint32_t)(v[1] >> 32) == (uint32_t)s &&
                (uint32_t)(v[2] >> 32) == (uint32_t)s && (uint32_t)(v[3] >> 32) == (uint32_t)s) break;
            if (++n > SPIN) { sycl::atomic_ref<uint32_t, sycl::memory_order::relaxed, sycl::memory_scope::system> a(stat[3]); a++; break; }
          }
          for (int k = 0; k < 4; ++k)
            if ((uint32_t)v[k] != (uint32_t)((4 * g + k) * 7 + s)) { sycl::atomic_ref<uint32_t, sycl::memory_order::relaxed, sycl::memory_scope::system> a(stat[1]); a++; }
        }
        sycl::group_barrier(it.get_group());
        for (size_t i = lid; i < words; i += WG) wp[i] = tagged((uint32_t)(i * 11 + s), (uint32_t)s);
      }
    });
    e0 = q0.parallel_for(sycl::nd_range<1>(WG, WG), [=](sycl::nd_item<1> it) {
      size_t lid = it.get_local_id(0);
      oex::annotated_ptr<uint64_t, wprops> wp{ping};
      oex::annotated_ptr<const u64x4, rprops> rp4{reinterpret_cast<const u64x4*>(pong)};
      for (int s = 1; s <= rounds; ++s) {
        for (size_t i = lid; i < words; i += WG) wp[i] = tagged((uint32_t)(i * 7 + s), (uint32_t)s);
        for (size_t g = lid; g < words / 4; g += WG) {
          u64x4 v; uint32_t n = 0;
          for (;;) {
            { u64x4 t = rp4[g]; v = t; }
            sycl::atomic_fence(sycl::memory_order::acquire, sycl::memory_scope::system);
            if ((uint32_t)(v[0] >> 32) == (uint32_t)s && (uint32_t)(v[1] >> 32) == (uint32_t)s &&
                (uint32_t)(v[2] >> 32) == (uint32_t)s && (uint32_t)(v[3] >> 32) == (uint32_t)s) break;
            if (++n > SPIN) { sycl::atomic_ref<uint32_t, sycl::memory_order::relaxed, sycl::memory_scope::system> a(stat[2]); a++; break; }
          }
          for (int k = 0; k < 4; ++k)
            if ((uint32_t)v[k] != (uint32_t)((4 * g + k) * 11 + s)) { sycl::atomic_ref<uint32_t, sycl::memory_order::relaxed, sycl::memory_scope::system> a(stat[0]); a++; }
        }
        sycl::group_barrier(it.get_group());
      }
    });
    e0.wait(); e1.wait();
  } catch (const sycl::exception& e) {
    std::printf("SYCL exception: %s\n", e.what());
    return 5;
  }
  double sec = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
  std::printf("%zu tagged words (%zu B) x %d rounds: %.1f us per round trip (card0 -> card1 -> card0), %.1f us one way; "
              "wrong words card0 %u card1 %u, poll timeouts card0 %u card1 %u\n",
              words, words * 8, rounds, sec * 1e6 / rounds, sec * 1e6 / rounds / 2, stat[0], stat[1], stat[2], stat[3]);
  sycl::free(ping, ctx); sycl::free(pong, ctx); sycl::free(stat, ctx);
  return (stat[0] || stat[1] || stat[2] || stat[3]) ? 1 : 0;
}
