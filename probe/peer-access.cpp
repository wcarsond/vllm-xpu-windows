// peer-access.cpp - is "no peer access" a driver policy or a hardware fact on these two B60s under Windows?
// Usage:  peer-access.exe            (one process, both cards; writes nothing, prints everything)
// Why (2026-10-11 14:25, Carson: "pull REA and see if we can decomp intels drivers ... i cant look at what op codes
// someones software calls on my machine"): at 12:50 we read Intel's open source and found the Windows answer is a stub -
// shared/source/os_interface/query_peer_access_wddm.cpp returns false unconditionally, while the Linux file really probes
// by exporting and importing a shareable page. A stub tells us the DRIVER declines. It does not tell us the HARDWARE
// cannot. This probe asks the four questions that decide whether a workaround is physically possible, before anyone
// spends hours disassembling anything:
//   1. zeDeviceCanAccessPeer in both directions - confirm the stub's answer is what reaches user space.
//   2. Does the loader even expose the peer-to-peer entry points (zeDeviceGetP2PProperties)?
//   3. The Linux path: allocate on card 0, zeMemGetIpcHandle, then zeMemOpenIpcHandle on card 1's context. If the handle
//      exports and imports, the plumbing exists and only the policy bit says no. If the export itself fails, it does not.
//   4. Can a kernel on card 1 actually READ the pointer that came back? Everything above can "succeed" and still fault,
//      which is exactly what the host-memory import did at 08:20 (import-test.cpp) - so this is the only question that
//      counts, and it is asked last and guarded.
// Handles come from SYCL (sycl::get_native<level_zero>) so there are no Level Zero struct layouts to guess; only the
// struct-free entry points are called through the loader. Unbuffered output: step 4 may take the process down.
#define NOMINMAX
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <sycl/sycl.hpp>
#include <sycl/ext/oneapi/backend/level_zero.hpp>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

typedef struct _ze_device_handle_t* ze_device_handle_t;
typedef struct _ze_context_handle_t* ze_context_handle_t;
typedef uint32_t ze_result_t;
typedef uint8_t ze_bool_t;
struct ze_ipc_mem_handle_t { char data[64]; };   // ZE_MAX_IPC_HANDLE_SIZE

typedef ze_result_t (*pfn_canAccessPeer)(ze_device_handle_t, ze_device_handle_t, ze_bool_t*);
typedef ze_result_t (*pfn_getIpcHandle)(ze_context_handle_t, const void*, ze_ipc_mem_handle_t*);
typedef ze_result_t (*pfn_openIpcHandle)(ze_context_handle_t, ze_device_handle_t, ze_ipc_mem_handle_t, uint32_t, void**);
typedef ze_result_t (*pfn_closeIpcHandle)(ze_context_handle_t, const void*);

static const char* rc(ze_result_t r) {
  switch (r) {
    case 0x0: return "SUCCESS";
    case 0x78000001: return "ERROR_UNINITIALIZED";
    case 0x78000002: return "ERROR_DEVICE_LOST";
    case 0x78000003: return "ERROR_INVALID_ARGUMENT";
    case 0x70000001: return "ERROR_NOT_AVAILABLE";
    case 0x7800001c: return "ERROR_UNSUPPORTED_FEATURE";
    case 0x7800001e: return "ERROR_INVALID_NULL_HANDLE";
    case 0x78000013: return "ERROR_UNSUPPORTED_SIZE";
    default: return "(see ze_result_t)";
  }
}

int main() {
  std::setvbuf(stdout, nullptr, _IONBF, 0);
  namespace zel = sycl::ext::oneapi::level_zero;

  // 14:18: platform::get_backend() matched nothing here (0 devices) while compat-devices.cpp sees three level_zero GPUs.
  // Enumerate devices directly, the way that probe does, and filter on the DEVICE's backend. Discrete only: the iGPU is
  // also level_zero and would be picked up as a third "card".
  {
    auto all = sycl::device::get_devices(sycl::info::device_type::all);
    std::printf("diagnostic: %zu devices of any type, %zu platforms\n", all.size(),
                sycl::platform::get_platforms().size());
    for (auto& d : all)
      std::printf("   backend %d  gpu=%d  %s\n", (int)d.get_backend(), (int)d.is_gpu(),
                  d.get_info<sycl::info::device::name>().c_str());
    std::printf("   (level_zero enum is %d)\n", (int)sycl::backend::ext_oneapi_level_zero);
  }
  std::vector<sycl::device> gpus;
  for (auto& d : sycl::device::get_devices(sycl::info::device_type::gpu)) {
    if (d.get_backend() != sycl::backend::ext_oneapi_level_zero) continue;
    if (d.get_info<sycl::info::device::name>().find("Arc(TM) Pro") == std::string::npos) continue;
    gpus.push_back(d);
  }
  std::printf("Level Zero GPUs: %zu\n", gpus.size());
  for (size_t i = 0; i < gpus.size(); ++i)
    std::printf("  [%zu] %s, %.1f GB\n", i, gpus[i].get_info<sycl::info::device::name>().c_str(),
                gpus[i].get_info<sycl::info::device::global_mem_size>() / 1e9);
  if (gpus.size() < 2) { std::printf("need two cards\n"); return 2; }

  HMODULE loader = LoadLibraryA("ze_loader.dll");
  if (!loader) { std::printf("ze_loader.dll not loadable (%lu)\n", GetLastError()); return 2; }
  auto canAccessPeer  = (pfn_canAccessPeer)GetProcAddress(loader, "zeDeviceCanAccessPeer");
  auto getIpcHandle   = (pfn_getIpcHandle)GetProcAddress(loader, "zeMemGetIpcHandle");
  auto openIpcHandle  = (pfn_openIpcHandle)GetProcAddress(loader, "zeMemOpenIpcHandle");
  auto closeIpcHandle = (pfn_closeIpcHandle)GetProcAddress(loader, "zeMemCloseIpcHandle");
  std::printf("\nstep 2: entry points in ze_loader.dll\n");
  std::printf("  zeDeviceCanAccessPeer  %s\n", canAccessPeer ? "present" : "MISSING");
  std::printf("  zeMemGetIpcHandle      %s\n", getIpcHandle ? "present" : "MISSING");
  std::printf("  zeMemOpenIpcHandle     %s\n", openIpcHandle ? "present" : "MISSING");
  std::printf("  zeDeviceGetP2PProperties %s\n", GetProcAddress(loader, "zeDeviceGetP2PProperties") ? "present" : "MISSING");

  sycl::queue q0{gpus[0]}, q1{gpus[1]};
  auto d0 = sycl::get_native<sycl::backend::ext_oneapi_level_zero>(gpus[0]);
  auto d1 = sycl::get_native<sycl::backend::ext_oneapi_level_zero>(gpus[1]);
  auto c0 = sycl::get_native<sycl::backend::ext_oneapi_level_zero>(q0.get_context());
  auto c1 = sycl::get_native<sycl::backend::ext_oneapi_level_zero>(q1.get_context());

  std::printf("\nstep 1: zeDeviceCanAccessPeer (the stub's answer, as user space sees it)\n");
  if (canAccessPeer) {
    ze_bool_t a = 2, b = 2;
    ze_result_t r0 = canAccessPeer((ze_device_handle_t)d0, (ze_device_handle_t)d1, &a);
    ze_result_t r1 = canAccessPeer((ze_device_handle_t)d1, (ze_device_handle_t)d0, &b);
    std::printf("  0 -> 1: result %s, can access = %u\n", rc(r0), a);
    std::printf("  1 -> 0: result %s, can access = %u\n", rc(r1), b);
  }

  std::printf("\nstep 3: the Linux path - export a card-0 allocation, import it on card 1\n");
  const size_t N = 1u << 20;   // 1 MB
  uint32_t* p0 = sycl::malloc_device<uint32_t>(N / 4, q0);
  q0.fill(p0, 0xABCDu, N / 4).wait();
  std::printf("  allocated %zu KB on card 0 at %p and filled it\n", N / 1024, (void*)p0);

  void* imported = nullptr;
  if (getIpcHandle && openIpcHandle) {
    ze_ipc_mem_handle_t h{};
    ze_result_t re = getIpcHandle((ze_context_handle_t)c0, p0, &h);
    std::printf("  zeMemGetIpcHandle on card 0: %s\n", rc(re));
    if (re == 0) {
      ze_result_t ri = openIpcHandle((ze_context_handle_t)c1, (ze_device_handle_t)d1, h, 0, &imported);
      std::printf("  zeMemOpenIpcHandle on card 1: %s, pointer %p\n", rc(ri), imported);
      if (ri != 0) imported = nullptr;
    } else {
      std::printf("  -> the export itself is refused, so there is no handle to give the other card.\n");
    }
  }

  if (imported) {
    std::printf("\nstep 4: does a KERNEL on card 1 actually read card 0's memory through that pointer?\n");
    std::printf("  (if this is the last line you see, the kernel faulted - which is the real answer)\n");
    uint32_t* out = sycl::malloc_shared<uint32_t>(4, q1);
    out[0] = 0xDEAD;
    const uint32_t* src = (const uint32_t*)imported;
    q1.single_task([=] { out[0] = src[0]; out[1] = src[(N / 4) - 1]; }).wait();
    std::printf("  card 1 read 0x%X and 0x%X (expected 0xABCD twice)%s\n", out[0], out[1],
                (out[0] == 0xABCD && out[1] == 0xABCD) ? "  <-- PEER ACCESS WORKS" : "  <-- wrong data");
    sycl::free(out, q1);
    if (closeIpcHandle) closeIpcHandle((ze_context_handle_t)c1, imported);
  } else {
    std::printf("\nstep 4: skipped, nothing imported.\n");
  }
  sycl::free(p0, q0);
  std::printf("\ndone\n");
  return 0;
}
