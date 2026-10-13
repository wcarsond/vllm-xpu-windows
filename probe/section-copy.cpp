// section-copy.cpp - can each card's COPY ENGINE read and write one shared Windows section directly, so the exchange's big
// path needs no CPU memcpy? Two processes (rank 0 on B60 #0, rank 1 on B60 #1), one named file mapping, imported into
// each process with zexDriverImportExternalPointer (or not imported, for comparison). Per round s:
//   copy engine: my device half -> my slot in the section; publish my flag = s
//   wait for the peer's flag; copy engine: the peer's slot -> a device buffer; kernel: sum = mine + theirs (device memory)
//   publish done = s, and before the next round wait for the peer's done (so a slot is not overwritten while being read)
// Every round's sum is checked on the card against the expected pattern (4i + 2s); a mismatch count goes to the host.
// Usage:  section-copy.exe <rank 0|1> <zex|none> <bytes> <rounds>       (run by section-copy.input.cmd, both ranks)
// Why (2026-10-13): Windows 14.7 MB all_reduce = 4.2 ms (exchange-test, today), of which ~1.5 ms is the helper's CPU
// memcpy hop and ~1.1 ms the card copies. Linux's copy-engine probe through one shared region does 1.34 ms, oneCCL 2.7.
// import-test (2026-10-11) showed the section imports with result 0 but KERNELS fault on it; the copy engine was never
// tried. If copies work on the imported section at ~28 GB/s, the helper's memcpy disappears from the big path.
#define NOMINMAX
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <immintrin.h>
#include <sycl/sycl.hpp>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

typedef struct _ze_driver_handle_t* ze_driver_handle_t;
typedef uint32_t ze_result_t;
typedef ze_result_t (*pfn_zeInit)(uint32_t flags);
typedef ze_result_t (*pfn_zeDriverGet)(uint32_t* pCount, ze_driver_handle_t* phDrivers);
typedef ze_result_t (*pfn_zeDriverGetExtensionFunctionAddress)(ze_driver_handle_t, const char*, void**);
typedef ze_result_t (*pfn_zexDriverImportExternalPointer)(ze_driver_handle_t, void* ptr, size_t size);

static const size_t HEADER = 4096;   // flags: [rank*64] = published s, [128 + rank*64] = done s

int main(int argc, char** argv) {
  std::setvbuf(stdout, nullptr, _IONBF, 0);
  if (argc < 5) { std::printf("usage: section-copy <rank> <zex|none> <bytes> <rounds>\n"); return 2; }
  const int rank = std::atoi(argv[1]), peer = 1 - rank;
  const std::string method = argv[2];
  const size_t bytes = (size_t)std::atoll(argv[3]) / 64 * 64;
  const int rounds = std::atoi(argv[4]);
  const size_t slot = (bytes + 65535) / 65536 * 65536;
  const size_t total = HEADER + 2 * slot;
  const size_t n = bytes / 4;

  HANDLE hmap = CreateFileMappingA(INVALID_HANDLE_VALUE, NULL, PAGE_READWRITE, (DWORD)(total >> 32), (DWORD)(total & 0xffffffffu),
                                   "Local\\vllm-section-copy");
  if (!hmap) { std::printf("[%d] CreateFileMapping failed %lu\n", rank, GetLastError()); return 2; }
  char* base = (char*)MapViewOfFile(hmap, FILE_MAP_ALL_ACCESS, 0, 0, total);
  if (!base) { std::printf("[%d] MapViewOfFile failed %lu\n", rank, GetLastError()); return 2; }
  volatile int64_t* pub = (volatile int64_t*)base;
  auto flag = [&](int r) { return (volatile int64_t*)(base + r * 64); };
  auto done = [&](int r) { return (volatile int64_t*)(base + 128 + r * 64); };
  char* myslot = base + HEADER + rank * slot;
  char* peerslot = base + HEADER + peer * slot;
  (void)pub;

  std::vector<sycl::device> cards;
  for (auto& p : sycl::platform::get_platforms())
    if (p.get_info<sycl::info::platform::name>().find("Level-Zero") != std::string::npos)
      for (auto& d : p.get_devices())
        if (d.is_gpu() && d.get_info<sycl::info::device::name>().find("B60") != std::string::npos) cards.push_back(d);
  if ((int)cards.size() < 2) { std::printf("[%d] need two B60s over Level Zero, found %zu\n", rank, cards.size()); return 3; }
  sycl::queue q{cards[rank], sycl::property::queue::in_order{}};

  if (method == "zex") {
    HMODULE lib = LoadLibraryA("ze_loader.dll");
    auto zeInit = (pfn_zeInit)GetProcAddress(lib, "zeInit");
    auto zeDriverGet = (pfn_zeDriverGet)GetProcAddress(lib, "zeDriverGet");
    auto getExt = (pfn_zeDriverGetExtensionFunctionAddress)GetProcAddress(lib, "zeDriverGetExtensionFunctionAddress");
    zeInit(0);
    uint32_t nd = 0; zeDriverGet(&nd, nullptr);
    std::vector<ze_driver_handle_t> drivers(nd); zeDriverGet(&nd, drivers.data());
    int ok = 0;
    for (uint32_t d = 0; d < nd; ++d) {
      void* fn = nullptr; getExt(drivers[d], "zexDriverImportExternalPointer", &fn);
      if (!fn) continue;
      ze_result_t r = ((pfn_zexDriverImportExternalPointer)fn)(drivers[d], base, total);
      ok += (r == 0);
    }
    std::printf("[%d] imported the section into %d driver(s)\n", rank, ok);
  }

  uint32_t* mine = sycl::malloc_device<uint32_t>(n, q);
  uint32_t* theirs = sycl::malloc_device<uint32_t>(n, q);
  uint32_t* sum = sycl::malloc_device<uint32_t>(n, q);
  uint32_t* bad = sycl::malloc_shared<uint32_t>(1, q);
  const uint32_t mult = rank == 0 ? 1u : 3u;
  double copy_in_us = 0, wait_us = 0, copy_out_us = 0, add_us = 0, total_us = 0;
  uint64_t wrong = 0;
  auto now = [] { return std::chrono::steady_clock::now(); };
  auto us = [](auto a, auto b) { return std::chrono::duration<double, std::micro>(b - a).count(); };
  try {
    for (int s = 1; s <= rounds; ++s) {
      const uint32_t su = (uint32_t)s;
      q.parallel_for(sycl::range<1>(n), [=](sycl::id<1> i) { mine[i] = mult * (uint32_t)i + su; }).wait();
      while (*done(peer) < s - 1) _mm_pause();          // the peer has finished reading my previous half
      auto t0 = now();
      q.memcpy(myslot, mine, bytes).wait();             // copy engine: card -> shared section
      auto t1 = now();
      *flag(rank) = s;
      while (*flag(peer) < s) _mm_pause();
      auto t2 = now();
      q.memcpy(theirs, peerslot, bytes).wait();         // copy engine: shared section -> card
      auto t3 = now();
      *done(rank) = s;
      *bad = 0;
      q.parallel_for(sycl::range<1>(n), [=](sycl::id<1> i) {
        sum[i] = mine[i] + theirs[i];
        if (sum[i] != 4u * (uint32_t)i + 2u * su) sycl::atomic_ref<uint32_t, sycl::memory_order::relaxed, sycl::memory_scope::device>(*bad)++;
      }).wait();
      auto t4 = now();
      wrong += *bad;
      if (s > 3) { copy_in_us += us(t0, t1); wait_us += us(t1, t2); copy_out_us += us(t2, t3); add_us += us(t3, t4); total_us += us(t0, t4); }
    }
  } catch (const sycl::exception& e) {
    std::printf("[%d] %s %zu B: SYCL exception: %s\n", rank, method.c_str(), bytes, e.what());
    return 5;
  }
  const int m = rounds - 3;
  std::printf("[%d] %-4s %10zu B: %s; per round us: copy in %.0f (%.1f GB/s), wait peer %.0f, copy out %.0f (%.1f GB/s), add+check %.0f, total %.0f\n",
              rank, method.c_str(), bytes, wrong ? ("WRONG " + std::to_string(wrong) + " words").c_str() : "all sums correct",
              copy_in_us / m, bytes / (copy_in_us / m) / 1e3, wait_us / m, copy_out_us / m, bytes / (copy_out_us / m) / 1e3,
              add_us / m, total_us / m);
  sycl::free(mine, q); sycl::free(theirs, q); sycl::free(sum, q); sycl::free(bad, q);
  UnmapViewOfFile(base); CloseHandle(hmap);
  return wrong ? 1 : 0;
}
