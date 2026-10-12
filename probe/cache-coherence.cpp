// cache-coherence.cpp - after the card has read a buffer (and cached it), which CPU-side ways of overwriting it does a later
// kernel see, and with which kind of load? Ground truth for the exchange design. Built/run by cache-coherence.input.cmd.
// Why (2026-10-11 09:20): the GPU-timeline exchange has read stale data three different ways (host flag polled from the
// card never updates; data staged into card memory by copy is stale two calls later; data in host memory read with
// "uncached" hints is still partly stale). Each row below: memory kind, how the CPU wrote it, how the kernel read it ->
// how many of 4096 words were stale. "busy" means the copy was enqueued behind a 30 ms kernel on the same queue.
#define NOMINMAX
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <sycl/sycl.hpp>
#include <sycl/ext/oneapi/experimental/annotated_ptr/annotated_ptr.hpp>
#include <sycl/ext/intel/experimental/cache_control_properties.hpp>
#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <thread>
#include <vector>

namespace iex = sycl::ext::intel::experimental;
namespace oex = sycl::ext::oneapi::experimental;
using uncached_rprops = decltype(oex::properties(
    iex::read_hint<iex::cache_control<iex::cache_mode::uncached, oex::cache_level::L1, oex::cache_level::L2, oex::cache_level::L3>>));
template <class T> using uncached_rptr = oex::annotated_ptr<T, uncached_rprops>;
template <class T>
using dev_atomic = sycl::atomic_ref<T, sycl::memory_order::relaxed, sycl::memory_scope::device, sycl::access::address_space::global_space>;
template <class T>
using sys_atomic = sycl::atomic_ref<T, sycl::memory_order::relaxed, sycl::memory_scope::system, sycl::access::address_space::global_space>;

static constexpr size_t N = 4096;  // words = 16 KB

enum Read { PLAIN, UNCACHED_HINT, DEV_ATOMIC, SYS_ATOMIC, SYS_ATOMIC_FENCE, RMW };
static const char* read_name(int r) {
  switch (r) { case PLAIN: return "plain load"; case UNCACHED_HINT: return "uncached hint"; case DEV_ATOMIC: return "device atomic load";
    case SYS_ATOMIC: return "system atomic load"; case SYS_ATOMIC_FENCE: return "sys fence + plain"; default: return "atomic fetch_or(0)"; }
}

// kernel: out[i] = buf[i] read the given way
static void read_kernel(sycl::queue& q, const uint32_t* buf, uint32_t* out, int how) {
  q.parallel_for(sycl::range<1>(N), [=](sycl::id<1> i) {
    uint32_t v;
    switch (how) {
      case PLAIN: v = buf[i]; break;
      case UNCACHED_HINT: { uncached_rptr<const uint32_t> p{buf}; v = p[i]; break; }
      case DEV_ATOMIC: v = dev_atomic<uint32_t>(const_cast<uint32_t&>(buf[i])).load(); break;
      case SYS_ATOMIC: v = sys_atomic<uint32_t>(const_cast<uint32_t&>(buf[i])).load(); break;
      case SYS_ATOMIC_FENCE: sycl::atomic_fence(sycl::memory_order::acquire, sycl::memory_scope::system); v = buf[i]; break;
      default: v = sys_atomic<uint32_t>(const_cast<uint32_t&>(buf[i])).fetch_or(0u); break;
    }
    out[i] = v;
  }).wait();
}

static void busy_kernel(sycl::queue& q, uint32_t* sink) {
  q.single_task([=]() { float a = 1.0f; for (int j = 0; j < 20000000; ++j) a = a * 1.0000001f + 1e-9f; *sink = a > 1e30f ? 1u : 0u; });
}

int main() {
  std::vector<sycl::device> cards;
  for (auto& p : sycl::platform::get_platforms())
    if (p.get_info<sycl::info::platform::name>().find("Level-Zero") != std::string::npos)
      for (auto& d : p.get_devices())
        if (d.is_gpu() && d.get_info<sycl::info::device::name>().find("B60") != std::string::npos) cards.push_back(d);
  if (cards.empty()) { std::printf("no B60\n"); return 3; }
  sycl::queue q{cards[0], sycl::property::queue::in_order{}};
  uint32_t* out = sycl::malloc_device<uint32_t>(N, q);
  uint32_t* sink = sycl::malloc_device<uint32_t>(1, q);
  uint32_t* host_src = sycl::malloc_host<uint32_t>(N, q);     // source for copies into card memory
  uint32_t* hbuf = sycl::malloc_host<uint32_t>(N, q);         // host USM read by the card
  uint32_t* dbuf = sycl::malloc_device<uint32_t>(N, q);       // card memory written by copies
  std::vector<uint32_t> chk(N);
  uint32_t gen = 1;

  auto check = [&](const char* mem, const char* write, int how) {
    q.memcpy(chk.data(), out, N * 4).wait();
    size_t stale = 0;
    for (size_t i = 0; i < N; ++i) stale += (chk[i] != gen);
    std::printf("  %-12s %-34s %-22s -> %s (%zu of %zu stale)\n", mem, write, read_name(how), stale ? "STALE" : "fresh", stale, N);
  };

  std::printf("1. host USM, CPU writes new values while the card holds the old ones in cache\n");
  for (int how = PLAIN; how <= RMW; ++how) {
    gen++; for (size_t i = 0; i < N; ++i) hbuf[i] = gen;            // old values
    read_kernel(q, hbuf, out, how);                                  // card reads (and caches) old values
    gen++; for (size_t i = 0; i < N; ++i) hbuf[i] = gen;            // CPU overwrites
    _mm_sfence();
    read_kernel(q, hbuf, out, how);                                  // card reads again
    check("host USM", "CPU store", how);
  }

  std::printf("2. card memory, CPU->card copy on an idle queue (16 KB) while the card holds the old lines\n");
  for (int how = PLAIN; how <= RMW; ++how) {
    gen++; for (size_t i = 0; i < N; ++i) host_src[i] = gen; q.memcpy(dbuf, host_src, N * 4).wait();
    read_kernel(q, dbuf, out, how);
    gen++; for (size_t i = 0; i < N; ++i) host_src[i] = gen;
    q.memcpy(dbuf, host_src, N * 4).wait();                          // idle queue: the runtime may write through the CPU
    read_kernel(q, dbuf, out, how);
    check("card memory", "copy, idle queue", how);
  }

  std::printf("3. card memory, CPU->card copy enqueued behind a busy kernel (copy engine path?)\n");
  for (int how = PLAIN; how <= RMW; ++how) {
    gen++; for (size_t i = 0; i < N; ++i) host_src[i] = gen; q.memcpy(dbuf, host_src, N * 4).wait();
    read_kernel(q, dbuf, out, how);
    gen++; for (size_t i = 0; i < N; ++i) host_src[i] = gen;
    busy_kernel(q, sink);
    q.memcpy(dbuf, host_src, N * 4);
    read_kernel(q, dbuf, out, how);
    check("card memory", "copy, behind a 30 ms kernel", how);
  }

  std::printf("4. card memory, small copy (8 B) on an idle queue: the flag case\n");
  for (int how = PLAIN; how <= RMW; ++how) {
    gen++; for (size_t i = 0; i < N; ++i) host_src[i] = gen; q.memcpy(dbuf, host_src, N * 4).wait();
    read_kernel(q, dbuf, out, how);
    gen++; host_src[0] = gen;
    q.memcpy(dbuf, host_src, 8).wait();
    read_kernel(q, dbuf, out, how);
    q.memcpy(chk.data(), out, N * 4).wait();
    std::printf("  %-12s %-34s %-22s -> %s (word 0 read %u, want %u)\n", "card memory", "8-byte copy, idle queue", read_name(how), chk[0] == gen ? "fresh" : "STALE", chk[0], gen);
  }

  // 5/6: the exchange's real situation. A kernel is already running and polling a flag in card memory; while it spins the
  // CPU writes the data (into host USM, or into card memory by a copy on a second queue) and then the flag by copy.
  // The kernel then reads the data in the given way. Before that, an earlier kernel read the old data, so stale lines
  // may sit in the card's caches.
  sycl::queue q2{cards[0], sycl::property::queue::in_order{}};
  int64_t* dflag = sycl::malloc_device<int64_t>(8, q);
  int64_t* hflag_src = sycl::malloc_host<int64_t>(8, q);
  q.memset(dflag, 0, 64).wait();
  int64_t fgen = 0;
  auto poll_read_kernel = [&](const uint32_t* buf, int how, int64_t want) {
    return q.parallel_for(sycl::nd_range<1>(N, 256), [=](sycl::nd_item<1> it) {
      size_t i = it.get_global_id(0);
      if (it.get_local_id(0) == 0) {
        dev_atomic<int64_t> f(*dflag);
        uint32_t polls = 0;
        while (f.load() < want) { if (++polls > 20000000u) break; }
      }
      sycl::group_barrier(it.get_group());
      sycl::atomic_fence(sycl::memory_order::acquire, sycl::memory_scope::system);
      uint32_t v;
      switch (how) {
        case PLAIN: v = buf[i]; break;
        case UNCACHED_HINT: { uncached_rptr<const uint32_t> p{buf}; v = p[i]; break; }
        case DEV_ATOMIC: v = dev_atomic<uint32_t>(const_cast<uint32_t&>(buf[i])).load(); break;
        case SYS_ATOMIC: v = sys_atomic<uint32_t>(const_cast<uint32_t&>(buf[i])).load(); break;
        case SYS_ATOMIC_FENCE: sycl::atomic_fence(sycl::memory_order::acquire, sycl::memory_scope::system); v = buf[i]; break;
        default: v = sys_atomic<uint32_t>(const_cast<uint32_t&>(buf[i])).fetch_or(0u); break;
      }
      out[i] = v;
    });
  };

  std::printf("5. kernel already polling a card-memory flag; CPU writes host USM data, then the flag by copy\n");
  for (int how = PLAIN; how <= RMW; ++how) {
    gen++; for (size_t i = 0; i < N; ++i) hbuf[i] = gen;
    read_kernel(q, hbuf, out, how);                                  // old values cached
    gen++; fgen++;
    auto ev = poll_read_kernel(hbuf, how, fgen);
    Sleep(20);                                                       // the kernel is spinning now
    for (size_t i = 0; i < N; ++i) hbuf[i] = gen;
    _mm_sfence();
    hflag_src[0] = fgen; q2.memcpy(dflag, hflag_src, 8).wait();
    ev.wait();
    check("host USM", "CPU store during the kernel", how);
  }

  std::printf("6. kernel already polling; CPU->card data copy on a second queue, then the flag by copy\n");
  for (int how = PLAIN; how <= RMW; ++how) {
    gen++; for (size_t i = 0; i < N; ++i) host_src[i] = gen; q.memcpy(dbuf, host_src, N * 4).wait();
    read_kernel(q, dbuf, out, how);
    gen++; fgen++;
    auto ev = poll_read_kernel(dbuf, how, fgen);
    Sleep(20);
    for (size_t i = 0; i < N; ++i) host_src[i] = gen;
    q2.memcpy(dbuf, host_src, N * 4).wait();
    hflag_src[0] = fgen; q2.memcpy(dflag, hflag_src, 8).wait();
    ev.wait();
    check("card memory", "copy during the kernel (q2)", how);
  }

  // 7: as 5, but the data and the flag copy come from a second thread (the exchange's helper), 512 bytes only, with and
  // without flushing the written lines out of the CPU cache before the flag copy.
  std::printf("7. kernel already polling; a second thread writes 512 B of host USM then copies the flag\n");
  for (int flush = 0; flush < 2; ++flush) {
    for (int how = PLAIN; how <= RMW; ++how) {
      gen++; for (size_t i = 0; i < N; ++i) hbuf[i] = gen;
      read_kernel(q, hbuf, out, how);
      gen++; fgen++;
      auto ev = poll_read_kernel(hbuf, how, fgen);
      Sleep(20);
      uint32_t g = gen; int64_t fg = fgen;
      std::thread th([&, g, fg]() {
        for (size_t i = 0; i < 128; ++i) hbuf[i] = g;
        if (flush) { for (size_t off = 0; off < 512; off += 64) _mm_clflush(reinterpret_cast<char*>(hbuf) + off); }
        _mm_mfence();
        hflag_src[0] = fg; q2.memcpy(dflag, hflag_src, 8).wait();
      });
      th.join();
      ev.wait();
      q.memcpy(chk.data(), out, N * 4).wait();
      size_t stale = 0; for (size_t i = 0; i < 128; ++i) stale += (chk[i] != gen);
      std::printf("  %-12s %-34s %-22s -> %s (%zu of 128 stale)\n", "host USM", flush ? "thread store + clflush" : "thread store, 512 B", read_name(how), stale ? "STALE" : "fresh", stale);
    }
  }

  // 8: the export direction, as K1 does it. The card stores N words into host USM with uncached stores, fences, and lane 0
  // stores a flag; the CPU (which holds the old lines in its cache) spins on the flag and then reads the words.
  // Counted over many iterations: words still old after the flag said new. With and without invalidating the CPU's
  // cache lines (clflush) before the read, and with the card idle (Sleep) or busy before the kernel.
  using uncached_wprops = decltype(oex::properties(
      iex::write_hint<iex::cache_control<iex::cache_mode::uncached, oex::cache_level::L1, oex::cache_level::L3>>));
  int64_t* hflag = sycl::malloc_host<int64_t>(8, q);
  hflag[0] = 0;
  std::printf("8. export: card stores words + flag into host USM, CPU polls the flag then reads the words\n");
  for (int idle = 0; idle < 2; ++idle) {
    for (int flush = 0; flush < 2; ++flush) {
      for (size_t words : {128u, 1792u}) {
        size_t stale_total = 0, iters = 300, stale_iters = 0;
        for (size_t it = 0; it < iters; ++it) {
          gen++;
          volatile uint32_t sum = 0;
          for (size_t i = 0; i < words; ++i) sum += hbuf[i];            // old lines into the CPU cache
          if (idle) Sleep(2);
          uint32_t g = gen; int64_t fg = (int64_t)gen;
          q.parallel_for(sycl::nd_range<1>(512, 512), [=](sycl::nd_item<1> item) {
            size_t lid = item.get_local_id(0);
            oex::annotated_ptr<uint32_t, uncached_wprops> m{hbuf};
            for (size_t i = lid; i < words; i += 512) m[i] = g;
            sycl::atomic_fence(sycl::memory_order::release, sycl::memory_scope::system);
            sycl::group_barrier(item.get_group());
            if (lid == 0) { oex::annotated_ptr<int64_t, uncached_wprops> f{hflag}; f[0] = fg; }
          });
          volatile int64_t* vf = hflag;
          auto t0 = std::chrono::steady_clock::now();
          while (*vf < fg) { if (std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count() > 2) break; }
          if (flush) { for (size_t off = 0; off < words * 4; off += 64) _mm_clflush(reinterpret_cast<char*>(hbuf) + off); _mm_mfence(); }
          size_t stale = 0;
          for (size_t i = 0; i < words; ++i) stale += (hbuf[i] != gen);
          stale_total += stale; stale_iters += (stale != 0);
          q.wait();
        }
        std::printf("  %-8s %-12s %5zu words: %zu stale words in %zu of %zu iterations\n", idle ? "idle" : "busy", flush ? "clflush" : "no flush", words, stale_total, stale_iters, iters);
      }
    }
  }
  // 9: can the card poll a flag in *host* memory and see the CPU's write? The card's loads of host memory were served
  // from a cache in the earlier exchange versions (the poll never ended). Three ways to read the flag in the loop; the
  // CPU writes it 20 ms after the kernel started; we measure whether and how soon the kernel finishes.
  std::printf("9. card polls a host-memory flag; CPU writes it 20 ms into the kernel (20 tries each)\n");
  {
    int64_t* hpoll = sycl::malloc_host<int64_t>(8, q);
    int64_t* dout = sycl::malloc_device<int64_t>(2, q);
    hpoll[0] = 0;
    int64_t pgen = 0;
    const char* names[] = {"atomic fetch_or(0)", "atomic load + acquire fence", "uncached hint + acquire fence", "plain volatile-like load"};
    for (int how = 0; how < 4; ++how) {
      int seen = 0; double worst_us = 0, total_us = 0;
      for (int it = 0; it < 20; ++it) {
        pgen++;
        int64_t want = pgen;
        q.memset(dout, 0, 16).wait();
        auto t_start = std::chrono::steady_clock::now();
        auto ev = q.single_task([=]() {
          uint32_t polls = 0; int64_t v = 0;
          for (;;) {
            if (how == 0) v = sys_atomic<int64_t>(*hpoll).fetch_or(0);
            else if (how == 1) { v = sys_atomic<int64_t>(*hpoll).load(); sycl::atomic_fence(sycl::memory_order::acquire, sycl::memory_scope::system); }
            else if (how == 2) { uncached_rptr<const int64_t> p{hpoll}; v = p[0]; sycl::atomic_fence(sycl::memory_order::acquire, sycl::memory_scope::system); }
            else { v = *reinterpret_cast<volatile int64_t*>(hpoll); }
            if (v >= want) { dout[0] = 1; dout[1] = polls; break; }
            if (++polls > 6000000u) { dout[0] = 2; dout[1] = polls; break; }   // ~1 s
          }
        });
        Sleep(20);
        auto t_write = std::chrono::steady_clock::now();
        *reinterpret_cast<volatile int64_t*>(hpoll) = want;
        _mm_mfence();
        ev.wait();
        double us = std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t_write).count();
        int64_t res[2]; q.memcpy(res, dout, 16).wait();
        if (res[0] == 1) { seen++; total_us += us; worst_us = std::max(worst_us, us); }
        (void)t_start;
      }
      std::printf("  %-32s seen %2d of 20, after the write: mean %.0f us, worst %.0f us\n", names[how], seen, seen ? total_us / seen : 0.0, worst_us);
    }
    sycl::free(hpoll, q); sycl::free(dout, q);
  }
  sycl::free(hflag, q);

  sycl::free(dflag, q); sycl::free(hflag_src, q);
  sycl::free(out, q); sycl::free(sink, q); sycl::free(host_src, q); sycl::free(hbuf, q); sycl::free(dbuf, q);
  return 0;
}
