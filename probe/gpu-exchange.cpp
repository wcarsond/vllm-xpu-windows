// gpu-exchange.cpp - a two-card all_reduce that keeps the GPU timeline flowing: the Python/main thread only enqueues, a
// helper thread moves the bytes between the processes, and the GPU waits on a flag in its own process's host memory.
// Built/run by gpu-exchange.input.cmd.
// Why (2026-10-11): the tensor split does ~58 all_reduces of 7 KB per token on the 7B. The host-side exchange costs ~258 us
// each in situ (33 us standalone): not the bytes, the pipeline bubble of stopping the GPU, running host code, restarting.
// 58 x 258 us = 15 ms of a 31 ms token, the whole gap between 32.5 tok/s and ~50. The cards have no peer access (p2p.cpp)
// and this driver will not let a kernel touch imported host memory (import-test.cpp: zexDriverImportExternalPointer says
// success, the kernel page-faults; the SYCL wrapper crashes), so no GPU ever sees the other process's memory. Instead:
//   main thread, per all_reduce s (in-order queue, nothing waits):
//     1. q.memcpy(mine[s&1] <- x)                       DMA into pinned host memory of this process (kernel-visible USM)
//     2. post job {s, that event} to the helper thread
//     3. kernel: lane 0 polls ready[s&1] >= s           a flag in this process's host USM, written by the helper
//                then x += theirs[s&1]                  the peer's half, also in this process's host USM
//   helper thread, per job: wait for the DMA event, memcpy mine -> my shared slot, publish seq in my shared flag, spin on
//     the peer's shared flag, memcpy the peer's shared slot -> theirs, store ready = s.
// Double-buffered by seq parity; the in-order queue gives the safety argument (job s+2 starts only after kernel s ran).
// Measured per iteration of "one fake layer + one all_reduce", in lockstep, checked on the device:
//   A  layer alone                     (the floor)
//   B  layer + helper exchange         (this design)
//   D  layer + host-sync exchange      (today's path: xpu_shm_exchange.py as C++)
//   E  helper exchange alone, back to back
// Usage: gpu-exchange.exe <rank> [iters] [layer_k] [words]   (words = floats per all_reduce; 1792 = 7 KB decode tensor)
#define NOMINMAX
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <sycl/sycl.hpp>
#include <sycl/ext/oneapi/experimental/annotated_ptr/annotated_ptr.hpp>
#include <sycl/ext/intel/experimental/cache_control_properties.hpp>
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <thread>
#include <vector>

using clk = std::chrono::steady_clock;

static constexpr size_t HEADER = 4096;
static constexpr size_t SLOT = 1 << 20;             // per rank, per buffer; 2 ranks x 2 buffers
static constexpr size_t FLAG_OFF[2] = {0, 64};      // shared: per-rank published sequence, one cache line each
static constexpr size_t PHASE_OFF[2] = {256, 320};  // shared: host handshake per test
static constexpr int WG = 512;
static constexpr uint32_t SPIN_LIMIT = 400000;      // GPU polls before giving up: well under the ~2 s Windows TDR
static constexpr int RING = 256;                    // jobs in flight between the main thread and the helper

template <class T>
using sys_atomic = sycl::atomic_ref<T, sycl::memory_order::relaxed, sycl::memory_scope::system,
                                    sycl::access::address_space::global_space>;
// Loads that bypass every GPU cache level. The first run (08:28) showed a relaxed system-scope atomic load of a host
// flag being served from cache: the poll never saw the host's write and only the acquire fence after it did.
namespace iex = sycl::ext::intel::experimental;
namespace oex = sycl::ext::oneapi::experimental;
using uncached_props = decltype(oex::properties(
    iex::read_hint<iex::cache_control<iex::cache_mode::uncached, oex::cache_level::L1, oex::cache_level::L2, oex::cache_level::L3>>));
template <class T> using uncached_ptr = oex::annotated_ptr<T, uncached_props>;
// Stores that go straight to memory (the host reads them): the card's half and the "it has landed" flag.
using uncached_wprops = decltype(oex::properties(
    iex::write_hint<iex::cache_control<iex::cache_mode::uncached, oex::cache_level::L1, oex::cache_level::L3>>));
template <class T> using uncached_wptr = oex::annotated_ptr<T, uncached_wprops>;
template <class T>
using dev_atomic = sycl::atomic_ref<T, sycl::memory_order::relaxed, sycl::memory_scope::device,
                                    sycl::access::address_space::global_space>;

static char* slot_ptr(char* base, int rank, int buf) { return base + HEADER + (size_t)(rank * 2 + buf) * SLOT; }

struct Job { int64_t seq; };

struct Ctx {
  explicit Ctx(const sycl::device& d) : q{d, sycl::property::queue::in_order{}}, q2{d, sycl::property::queue::in_order{}} {}
  int rank = 0, peer = 1;
  char* base = nullptr;          // the shared mapping (host only)
  sycl::queue q;                 // the compute queue (torch's stream in the real thing)
  sycl::queue q2;                // the helper's own queue for its two small host->device copies per exchange
  float* x = nullptr;            // device: the tensor being reduced
  float* mine[2] = {};           // host USM: my half, written by the export kernel, read by the helper
  float* theirs[2] = {};         // host USM: the peer's half, staging for the copy into theirs_dev
  float* theirs_dev[2] = {};     // device: the peer's half, read by the add kernel
  int64_t* ready_src = nullptr;  // host USM: ready_src[2], the helper's source for the 8-byte flag copy
  int64_t* ready_dev = nullptr;  // device: ready_dev[2], written by the helper's copy, polled by the kernel
  int64_t* dma_done = nullptr;   // host USM: dma_done[2], written by the export kernel, polled by the helper
  int64_t* err = nullptr;        // device: elements that came out wrong
  int64_t* spins = nullptr;      // device: total GPU polls
  int* timeouts = nullptr;       // device: exchanges whose GPU wait gave up
  size_t n = 1792;
  int layer_k = 12000;           // ~0.3 ms on a B60 (60000 measured 1568 us)
  int64_t seq = 0;
  // helper thread
  std::thread helper;
  Job ring[RING];
  std::atomic<int64_t> head{0}, tail{0};
  std::atomic<bool> stop{false};
  std::atomic<int64_t> helper_waits{0};   // helper spins on the peer flag (host polls)
};

static void layer(Ctx& c, int64_t seq, bool check_prev) {
  float* x = c.x; int64_t* err = c.err; size_t n = c.n; int k = c.layer_k; int rank = c.rank;
  c.q.parallel_for(sycl::nd_range<1>(WG, WG), [=](sycl::nd_item<1> it) {
    size_t lid = it.get_local_id(0);
    if (check_prev) {
      float expected_prev = 3.0f * (1 + ((seq - 1) % 7));
      int64_t bad = 0;
      for (size_t i = lid; i < n; i += WG) bad += (x[i] != expected_prev);
      if (bad) dev_atomic<int64_t>(*err).fetch_add(bad);
    }
    float v = (float)(rank + 1) * (1 + (seq % 7));
    float a = v;
    for (int j = 0; j < k; ++j) a = a * 1.000001f + 0.0000001f;
    float keep = a > 1e30f ? 1.0f : 0.0f;  // never true; keeps the chain alive
    for (size_t i = lid; i < n; i += WG) x[i] = v + keep;
  });
}

// ---- the helper thread ----------------------------------------------------------------------------------------------
static void helper_main(Ctx* c) {
  volatile int64_t* my_flag = reinterpret_cast<volatile int64_t*>(c->base + FLAG_OFF[c->rank]);
  volatile int64_t* peer_flag = reinterpret_cast<volatile int64_t*>(c->base + FLAG_OFF[c->peer]);
  size_t bytes = c->n * sizeof(float);
  while (!c->stop.load(std::memory_order_acquire)) {
    int64_t t = c->tail.load(std::memory_order_acquire);
    if (t == c->head.load(std::memory_order_acquire)) { _mm_pause(); continue; }
    Job job = c->ring[t % RING];
    int b = (int)(job.seq & 1);
    // 1. my half has landed in mine[b]: the export kernel says so in host memory (no runtime event involved; the first
    //    version polled the copy's SYCL event and never saw it complete while the next kernel was spinning)
    volatile int64_t* done = reinterpret_cast<volatile int64_t*>(c->dma_done) + b;
    while (*done < job.seq) _mm_pause();
    // 2. into the shared slot, publish
    std::memcpy(slot_ptr(c->base, c->rank, b), c->mine[b], bytes);
    MemoryBarrier();
    *my_flag = job.seq;
    // 3. the peer's half
    int64_t polls = 0;
    while (*peer_flag < job.seq) { ++polls; _mm_pause(); }
    c->helper_waits.fetch_add(polls, std::memory_order_relaxed);
    MemoryBarrier();
    std::memcpy(c->theirs[b], slot_ptr(c->base, c->peer, b), bytes);
    // 4. into the card: the peer's half, then the flag, on the helper's own in-order queue (the compute queue is busy
    //    spinning). The kernel polls device memory; no GPU read of host memory anywhere (the first versions showed a
    //    kernel never seeing a host write to host USM, cached or hoisted).
    c->q2.memcpy(c->theirs_dev[b], c->theirs[b], bytes);
    c->ready_src[b] = job.seq;
    c->q2.memcpy(c->ready_dev + b, c->ready_src + b, sizeof(int64_t));
    c->tail.store(t + 1, std::memory_order_release);
  }
}

static void post(Ctx& c, int64_t seq) {
  int64_t h = c.head.load(std::memory_order_relaxed);
  while (h - c.tail.load(std::memory_order_acquire) >= RING) _mm_pause();
  c.ring[h % RING] = Job{seq};
  c.head.store(h + 1, std::memory_order_release);
}

// K1: the card writes its half into mine[b] (host USM) and then says so in dma_done[b], uncached stores throughout.
static void export_half(Ctx& c, int64_t seq) {
  int b = (int)(seq & 1);
  float* x = c.x; size_t n = c.n; float* mine = c.mine[b]; int64_t* done = c.dma_done + b;
  c.q.parallel_for(sycl::nd_range<1>(WG, WG), [=](sycl::nd_item<1> it) {
    size_t lid = it.get_local_id(0);
    uncached_wptr<float> m{mine};
    for (size_t i = lid; i < n; i += WG) m[i] = x[i];
    sycl::atomic_fence(sycl::memory_order::release, sycl::memory_scope::system);
    sycl::group_barrier(it.get_group());
    if (lid == 0) { uncached_wptr<int64_t> d{done}; d[0] = seq; }
  });
}

// ---- the exchange as the main thread sees it: two kernel enqueues and a post, no waiting -------------------------------
static void helper_exchange(Ctx& c, int64_t seq) {
  int b = (int)(seq & 1);
  export_half(c, seq);
  post(c, seq);
  float* x = c.x; size_t n = c.n;
  float* theirs = c.theirs_dev[b];
  int64_t* ready = c.ready_dev + b;
  int64_t* spins = c.spins; int* timeouts = c.timeouts;
  c.q.parallel_for(sycl::nd_range<1>(WG, WG), [=](sycl::nd_item<1> it) {
    size_t lid = it.get_local_id(0);
    if (lid == 0) {
      dev_atomic<int64_t> r(*ready);
      uint32_t polls = 0;
      while (r.load() < seq) {
        if (++polls > SPIN_LIMIT) { dev_atomic<int>(*timeouts).fetch_add(1); break; }
      }
      dev_atomic<int64_t>(*spins).fetch_add(polls);
    }
    sycl::group_barrier(it.get_group());
    sycl::atomic_fence(sycl::memory_order::acquire, sycl::memory_scope::device);
    for (size_t i = lid; i < n; i += WG) x[i] += dev_atomic<float>(theirs[i]).load();   // through L3, past any stale L1 line
  });
}

// ---- today's path: stop the GPU, move the bytes with the host, add on the host, copy back --------------------------------
static void host_exchange(Ctx& c, int64_t seq, float* h_mine, float* h_sum) {
  int b = (int)(seq & 1);
  size_t bytes = c.n * sizeof(float);
  float* my_slot = reinterpret_cast<float*>(slot_ptr(c.base, c.rank, b));
  float* peer_slot = reinterpret_cast<float*>(slot_ptr(c.base, c.peer, b));
  volatile int64_t* my_flag = reinterpret_cast<volatile int64_t*>(c.base + FLAG_OFF[c.rank]);
  volatile int64_t* peer_flag = reinterpret_cast<volatile int64_t*>(c.base + FLAG_OFF[c.peer]);
  c.q.memcpy(h_mine, c.x, bytes).wait();
  std::memcpy(my_slot, h_mine, bytes);
  MemoryBarrier();
  *my_flag = seq;
  uint32_t spins = 0;
  while (*peer_flag < seq) { if (++spins > 2000) SwitchToThread(); }
  MemoryBarrier();
  for (size_t i = 0; i < c.n; ++i) h_sum[i] = h_mine[i] + peer_slot[i];
  c.q.memcpy(c.x, h_sum, bytes).wait();
}

static void handshake(Ctx& c, int64_t phase) {
  volatile int64_t* mine = reinterpret_cast<volatile int64_t*>(c.base + PHASE_OFF[c.rank]);
  volatile int64_t* theirs = reinterpret_cast<volatile int64_t*>(c.base + PHASE_OFF[c.peer]);
  *mine = phase;
  auto t0 = clk::now();
  while (*theirs < phase) {
    Sleep(1);
    if (std::chrono::duration<double>(clk::now() - t0).count() > 60) { std::printf("rank %d: peer never reached phase %lld\n", c.rank, (long long)phase); std::exit(4); }
  }
}

static void reset_counters(Ctx& c) {
  c.q.memset(c.err, 0, sizeof(int64_t)); c.q.memset(c.spins, 0, sizeof(int64_t)); c.q.memset(c.timeouts, 0, sizeof(int)); c.q.wait();
  c.helper_waits.store(0);
}

static void report(Ctx& c, const char* name, double wall_us, int iters) {
  int64_t err = 0, spins = 0; int to = 0;
  c.q.memcpy(&err, c.err, sizeof err).wait(); c.q.memcpy(&spins, c.spins, sizeof spins).wait(); c.q.memcpy(&to, c.timeouts, sizeof to).wait();
  std::printf("rank %d  %-40s %9.1f us/iter   wrong elements %lld   GPU polls/iter %.0f   helper polls/iter %.0f   timeouts %d\n",
              c.rank, name, wall_us / iters, (long long)err, iters ? (double)spins / iters : 0.0,
              iters ? (double)c.helper_waits.load() / iters : 0.0, to);
}

int main(int argc, char** argv) {
  std::setvbuf(stdout, nullptr, _IONBF, 0);
  if (argc < 2) { std::printf("usage: gpu-exchange <rank 0|1> [iters] [layer_k] [words]\n"); return 2; }
  int rank = std::atoi(argv[1]);
  int iters = argc > 2 ? std::atoi(argv[2]) : 5000;
  std::printf("rank %d: start\n", rank);

  std::vector<sycl::device> cards;
  for (auto& p : sycl::platform::get_platforms())
    if (p.get_info<sycl::info::platform::name>().find("Level-Zero") != std::string::npos)
      for (auto& d : p.get_devices())
        if (d.is_gpu() && d.get_info<sycl::info::device::name>().find("B60") != std::string::npos) cards.push_back(d);
  if (cards.size() < 2) { std::printf("rank %d: need two B60s over Level Zero, found %zu\n", rank, cards.size()); return 3; }
  Ctx c(cards[rank]);
  c.rank = rank; c.peer = 1 - rank;
  c.layer_k = argc > 3 ? std::atoi(argv[3]) : 12000;
  c.n = argc > 4 ? (size_t)std::atoll(argv[4]) : 1792;
  if (c.n * sizeof(float) > SLOT) { std::printf("words too large for a %zu-byte slot\n", SLOT); return 2; }
  std::printf("rank %d: card %s\n", c.rank, c.q.get_device().get_info<sycl::info::device::name>().c_str());

  // the shared block: rank 0 creates and zeroes it, rank 1 opens it
  const size_t total = HEADER + 4 * SLOT;
  const char* name = "Local\\vllm-gpu-exchange-probe";
  HANDLE hmap = NULL;
  if (c.rank == 0) {
    hmap = CreateFileMappingA(INVALID_HANDLE_VALUE, NULL, PAGE_READWRITE, (DWORD)(total >> 32), (DWORD)(total & 0xffffffffu), name);
  } else {
    for (int i = 0; i < 600 && !hmap; ++i) { hmap = OpenFileMappingA(FILE_MAP_ALL_ACCESS, FALSE, name); if (!hmap) Sleep(50); }
  }
  if (!hmap) { std::printf("rank %d: file mapping failed (error %lu)\n", c.rank, GetLastError()); return 2; }
  c.base = reinterpret_cast<char*>(MapViewOfFile(hmap, FILE_MAP_ALL_ACCESS, 0, 0, total));
  if (!c.base) { std::printf("rank %d: MapViewOfFile failed (error %lu)\n", c.rank, GetLastError()); return 2; }
  if (c.rank == 0) std::memset(c.base, 0, total);
  std::printf("rank %d: mapping at %p, %zu bytes\n", c.rank, (void*)c.base, total);
  handshake(c, 1);

  // device and host-USM state of this process
  c.x = sycl::malloc_device<float>(c.n, c.q);
  for (int b = 0; b < 2; ++b) {
    c.mine[b] = sycl::malloc_host<float>(c.n, c.q);
    c.theirs[b] = sycl::malloc_host<float>(c.n, c.q);
    c.theirs_dev[b] = sycl::malloc_device<float>(c.n, c.q);
  }
  c.ready_src = sycl::malloc_host<int64_t>(2, c.q); c.ready_src[0] = c.ready_src[1] = 0;
  c.ready_dev = sycl::malloc_device<int64_t>(2, c.q); c.q.memset(c.ready_dev, 0, 2 * sizeof(int64_t)).wait();
  c.dma_done = sycl::malloc_host<int64_t>(2, c.q); c.dma_done[0] = c.dma_done[1] = 0;
  c.err = sycl::malloc_device<int64_t>(1, c.q);
  c.spins = sycl::malloc_device<int64_t>(1, c.q);
  c.timeouts = sycl::malloc_device<int>(1, c.q);
  float* h_mine = sycl::malloc_host<float>(c.n, c.q);
  float* h_sum = sycl::malloc_host<float>(c.n, c.q);
  reset_counters(c);
  c.helper = std::thread(helper_main, &c);
  std::printf("rank %d: %d iterations, %zu floats (%zu bytes) per all_reduce, layer_k %d\n", c.rank, iters, c.n, c.n * sizeof(float), c.layer_k);

  // A. layer alone
  handshake(c, 2);
  { auto t0 = clk::now(); for (int i = 0; i < iters; ++i) { c.seq++; layer(c, c.seq, false); } c.q.wait();
    report(c, "A layer alone", std::chrono::duration<double, std::micro>(clk::now() - t0).count(), iters); }

  // smoke test 0: does the host see the export kernel's flag, and how soon? (no q.wait: the GPU must be the one to tell us)
  { c.seq++; layer(c, c.seq, false); export_half(c, c.seq);
    auto t0 = clk::now();
    volatile int64_t* done = reinterpret_cast<volatile int64_t*>(c.dma_done) + (c.seq & 1);
    while (*done < c.seq && std::chrono::duration<double>(clk::now() - t0).count() < 2.0) _mm_pause();
    double us = std::chrono::duration<double, std::micro>(clk::now() - t0).count();
    c.q.wait();
    std::printf("rank %d: export flag seen by the host after %.0f us (%s); mine[0] = %.0f (expect %.0f)\n", c.rank, us,
                *done >= c.seq ? "ok" : "NEVER", c.mine[c.seq & 1][0], (float)(c.rank + 1) * (1 + (c.seq % 7)));
    reset_counters(c); }

  // smoke test: one helper exchange, checked on the host
  handshake(c, 3);
  bool ok = true;
  { c.seq++; layer(c, c.seq, false); helper_exchange(c, c.seq); c.q.wait();
    std::vector<float> chk(c.n); c.q.memcpy(chk.data(), c.x, c.n * sizeof(float)).wait();
    float expect = 3.0f * (1 + (c.seq % 7));
    size_t bad = 0; for (float v : chk) bad += (v != expect);
    int to = 0; c.q.memcpy(&to, c.timeouts, sizeof to).wait();
    std::printf("rank %d: first helper exchange: %zu of %zu elements wrong (expected %.0f, got %.0f), timeouts %d\n", c.rank, bad, c.n, expect, chk[0], to);
    ok = !bad && !to;
    reset_counters(c); }

  if (ok) {
    // B. layer + helper exchange
    handshake(c, 4);
    { auto t0 = clk::now(); for (int i = 0; i < iters; ++i) { c.seq++; layer(c, c.seq, i > 0); helper_exchange(c, c.seq); } c.q.wait();
      report(c, "B layer + helper exchange", std::chrono::duration<double, std::micro>(clk::now() - t0).count(), iters); }
    reset_counters(c);
  } else {
    std::printf("rank %d: helper exchange failed its smoke test; B and E skipped\n", c.rank);
  }

  // D. layer + host-sync exchange (today's path)
  handshake(c, 5);
  { int it_h = std::min(iters, 2000);
    auto t0 = clk::now(); for (int i = 0; i < it_h; ++i) { c.seq++; layer(c, c.seq, i > 0); host_exchange(c, c.seq, h_mine, h_sum); } c.q.wait();
    report(c, "D layer + host-sync exchange (today)", std::chrono::duration<double, std::micro>(clk::now() - t0).count(), it_h); }
  reset_counters(c);

  if (ok) {
    // E. helper exchange alone, back to back (values grow, not checked)
    handshake(c, 6);
    c.seq++; layer(c, c.seq, false);
    { auto t0 = clk::now(); for (int i = 0; i < iters; ++i) { c.seq++; helper_exchange(c, c.seq); } c.q.wait();
      report(c, "E helper exchange alone, back to back", std::chrono::duration<double, std::micro>(clk::now() - t0).count(), iters); }
  }

  handshake(c, 7);
  c.stop.store(true); c.helper.join();
  sycl::free(c.x, c.q); sycl::free(c.err, c.q); sycl::free(c.spins, c.q); sycl::free(c.timeouts, c.q);
  c.q2.wait();
  for (int b = 0; b < 2; ++b) { sycl::free(c.mine[b], c.q); sycl::free(c.theirs[b], c.q); sycl::free(c.theirs_dev[b], c.q); }
  sycl::free(c.ready_src, c.q); sycl::free(c.ready_dev, c.q); sycl::free(c.dma_done, c.q); sycl::free(h_mine, c.q); sycl::free(h_sum, c.q);
  UnmapViewOfFile(c.base); CloseHandle(hmap);
  std::printf("rank %d exit 0\n", c.rank);
  return 0;
}
