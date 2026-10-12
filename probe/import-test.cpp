// import-test.cpp - can this driver import ordinary host memory so a kernel can read and write it? One variant per run,
// because a failing import takes the process down with an access violation instead of an error (gpu-exchange.cpp, 08:18).
// Usage:  import-test.exe <section|valloc|malloc> <sycl|zex|none>
//   memory: a named Windows file mapping (what two processes can share), VirtualAlloc'd pages, or 64 KB-aligned malloc
//   method: sycl::ext::oneapi::experimental::prepare_for_device_copy, the raw zexDriverImportExternalPointer through the
//           Level Zero loader, or no import at all (does the kernel fault, or does the driver map it on its own?)
// Then a kernel on B60 #0 writes a pattern into the memory and the host checks it; a second kernel reads it back into a
// device buffer and the host checks that. Unbuffered output so a crash shows the last step reached.
#define NOMINMAX
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <sycl/sycl.hpp>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

namespace syclex = sycl::ext::oneapi::experimental;

// minimal Level Zero declarations (no ze_api.h on this box): handles are opaque pointers, results are 32-bit codes
typedef struct _ze_driver_handle_t* ze_driver_handle_t;
typedef uint32_t ze_result_t;
typedef ze_result_t (*pfn_zeInit)(uint32_t flags);
typedef ze_result_t (*pfn_zeDriverGet)(uint32_t* pCount, ze_driver_handle_t* phDrivers);
typedef ze_result_t (*pfn_zeDriverGetExtensionFunctionAddress)(ze_driver_handle_t, const char*, void**);
typedef ze_result_t (*pfn_zexDriverImportExternalPointer)(ze_driver_handle_t, void* ptr, size_t size);
typedef ze_result_t (*pfn_zexDriverReleaseImportedPointer)(ze_driver_handle_t, void* ptr);

int main(int argc, char** argv) {
  std::setvbuf(stdout, nullptr, _IONBF, 0);
  if (argc < 3) { std::printf("usage: import-test <section|valloc|malloc> <sycl|zex|none>\n"); return 2; }
  std::string kind = argv[1], method = argv[2];
  const size_t total = 4u << 20;  // 4 MB, a multiple of 64 KB
  std::printf("[%s/%s] step 1: memory\n", kind.c_str(), method.c_str());
  char* base = nullptr;
  HANDLE hmap = NULL;
  if (kind == "section") {
    hmap = CreateFileMappingA(INVALID_HANDLE_VALUE, NULL, PAGE_READWRITE, 0, (DWORD)total, "Local\\vllm-import-test");
    if (!hmap) { std::printf("CreateFileMapping failed %lu\n", GetLastError()); return 2; }
    base = (char*)MapViewOfFile(hmap, FILE_MAP_ALL_ACCESS, 0, 0, total);
  } else if (kind == "valloc") {
    base = (char*)VirtualAlloc(NULL, total, MEM_COMMIT | MEM_RESERVE, PAGE_READWRITE);
  } else {
    base = (char*)_aligned_malloc(total, 65536);
  }
  if (!base) { std::printf("allocation failed %lu\n", GetLastError()); return 2; }
  std::memset(base, 0, total);
  std::printf("  memory at %p\n", (void*)base);

  std::printf("step 2: card\n");
  std::vector<sycl::device> cards;
  for (auto& p : sycl::platform::get_platforms())
    if (p.get_info<sycl::info::platform::name>().find("Level-Zero") != std::string::npos)
      for (auto& d : p.get_devices())
        if (d.is_gpu() && d.get_info<sycl::info::device::name>().find("B60") != std::string::npos) cards.push_back(d);
  if (cards.empty()) { std::printf("no B60 over Level Zero\n"); return 3; }
  sycl::queue q{cards[0], sycl::property::queue::in_order{}};
  std::printf("  %s\n", q.get_device().get_info<sycl::info::device::name>().c_str());

  std::printf("step 3: import (%s)\n", method.c_str());
  ze_driver_handle_t drv = nullptr;
  pfn_zexDriverReleaseImportedPointer release_fn = nullptr;
  if (method == "sycl") {
    try { syclex::prepare_for_device_copy(base, total, q.get_context()); std::printf("  prepare_for_device_copy ok\n"); }
    catch (const sycl::exception& e) { std::printf("  prepare_for_device_copy threw: %s\n", e.what()); }
  } else if (method == "zex") {
    HMODULE lib = LoadLibraryA("ze_loader.dll");
    if (!lib) { std::printf("  ze_loader.dll not loadable %lu\n", GetLastError()); return 2; }
    auto zeInit = (pfn_zeInit)GetProcAddress(lib, "zeInit");
    auto zeDriverGet = (pfn_zeDriverGet)GetProcAddress(lib, "zeDriverGet");
    auto getExt = (pfn_zeDriverGetExtensionFunctionAddress)GetProcAddress(lib, "zeDriverGetExtensionFunctionAddress");
    if (!zeInit || !zeDriverGet || !getExt) { std::printf("  loader entry points missing\n"); return 2; }
    ze_result_t r = zeInit(0);
    uint32_t n = 0; zeDriverGet(&n, nullptr);
    std::vector<ze_driver_handle_t> drivers(n); zeDriverGet(&n, drivers.data());
    std::printf("  zeInit %u, %u driver(s)\n", r, n);
    if (!n) return 2;
    // the loader lists every Level Zero driver on the box (the Core Ultra's NPU has one too): import into each one that
    // offers the extension, so the GPU's driver is certainly among them
    for (uint32_t d = 0; d < n; ++d) {
      void* fn = nullptr;
      r = getExt(drivers[d], "zexDriverImportExternalPointer", &fn);
      if (!fn) { std::printf("  driver %u: no zexDriverImportExternalPointer (lookup %u)\n", d, r); continue; }
      r = ((pfn_zexDriverImportExternalPointer)fn)(drivers[d], base, total);
      std::printf("  driver %u: import result %u (0 = success)\n", d, r);
      if (r == 0 && !drv) {
        drv = drivers[d];
        void* fn2 = nullptr; getExt(drv, "zexDriverReleaseImportedPointer", &fn2);
        release_fn = (pfn_zexDriverReleaseImportedPointer)fn2;
      }
    }
  } else {
    std::printf("  no import\n");
  }

  std::printf("step 4: kernel writes a pattern into the host memory\n");
  uint32_t* words = reinterpret_cast<uint32_t*>(base);
  const size_t nwords = total / sizeof(uint32_t);
  try {
    q.parallel_for(sycl::range<1>(nwords), [=](sycl::id<1> i) { words[i] = 0xA5000000u + (uint32_t)i; }).wait();
  } catch (const sycl::exception& e) { std::printf("  kernel threw: %s\n", e.what()); return 5; }
  size_t bad = 0;
  for (size_t i = 0; i < nwords; i += 1) bad += (words[i] != 0xA5000000u + (uint32_t)i);
  std::printf("  host sees %zu of %zu words wrong\n", bad, nwords);

  std::printf("step 5: host writes, kernel reads back into device memory\n");
  for (size_t i = 0; i < nwords; ++i) words[i] = 0x5A000000u + (uint32_t)i;
  uint32_t* dev = sycl::malloc_device<uint32_t>(nwords, q);
  try {
    q.parallel_for(sycl::range<1>(nwords), [=](sycl::id<1> i) { dev[i] = words[i]; }).wait();
  } catch (const sycl::exception& e) { std::printf("  kernel threw: %s\n", e.what()); return 5; }
  std::vector<uint32_t> chk(nwords);
  q.memcpy(chk.data(), dev, total).wait();
  bad = 0;
  for (size_t i = 0; i < nwords; ++i) bad += (chk[i] != 0x5A000000u + (uint32_t)i);
  std::printf("  device read %zu of %zu words wrong\n", bad, nwords);

  std::printf("step 6: release\n");
  if (method == "sycl") { try { syclex::release_from_device_copy(base, q.get_context()); } catch (...) {} }
  if (method == "zex" && release_fn) release_fn(drv, base);
  sycl::free(dev, q);
  if (kind == "section") { UnmapViewOfFile(base); CloseHandle(hmap); }
  else if (kind == "valloc") VirtualFree(base, 0, MEM_RELEASE);
  else _aligned_free(base);
  std::printf("[%s/%s] done exit 0\n", kind.c_str(), method.c_str());
  return 0;
}
