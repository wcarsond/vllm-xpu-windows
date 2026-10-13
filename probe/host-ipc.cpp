// host-ipc.cpp - can one process export a HOST USM allocation as a Level Zero IPC handle, and can a kernel on the OTHER
// card in the OTHER process write into it? That is how oneCCL connects two cards without peer access on Linux (host USM
// shared by IPC handle, card -> the peer's host buffer directly), and it would give Windows a helper-free decode exchange
// with two processes, i.e. without restructuring vLLM. peer-access.cpp (2026-10-11) tried only DEVICE memory, in one
// process: zeMemGetIpcHandle -> ERROR_INVALID_ARGUMENT.
// Usage:  host-ipc.exe <0|1>         rank 0 exports (card 0), rank 1 imports (card 1); run both by host-ipc.input.cmd
// Steps (rank 0): malloc_host 1 MB, fill with zero, zeMemGetIpcHandle -> write the 64-byte handle and its pid into a named
// section, wait for rank 1's verdict, then check whether the pattern rank 1's kernel wrote is in its host memory.
// Steps (rank 1): read the handle; try zeMemOpenIpcHandle as given; if that fails, treat the first 8 bytes as a Windows
// handle in rank 0's process and DuplicateHandle it into this one, then try again. If a pointer comes back, a kernel on
// card 1 writes a pattern through it (this may fault: it is the step that counts).
#define NOMINMAX
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <sycl/sycl.hpp>
#include <sycl/ext/oneapi/backend/level_zero.hpp>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

typedef struct _ze_context_handle_t* ze_context_handle_t;
typedef struct _ze_device_handle_t* ze_device_handle_t;
typedef uint32_t ze_result_t;
struct ze_ipc_mem_handle_t { char data[64]; };
typedef ze_result_t (*pfn_getIpcHandle)(ze_context_handle_t, const void*, ze_ipc_mem_handle_t*);
typedef ze_result_t (*pfn_openIpcHandle)(ze_context_handle_t, ze_device_handle_t, ze_ipc_mem_handle_t, uint32_t, void**);

struct Shared {
  volatile LONG stage;          // 1: handle published, 2: rank 1 done, 3: rank 0 checked
  DWORD pid;
  ze_ipc_mem_handle_t handle;
  volatile LONG verdict;        // rank 1: 0 not opened, 1 opened as given, 2 opened after DuplicateHandle, 3 kernel ran
  char note[256];
};

static const char* rc(ze_result_t r) {
  switch (r) {
    case 0x0: return "SUCCESS";
    case 0x78000003: return "ERROR_INVALID_ARGUMENT";
    case 0x7800001c: return "ERROR_UNSUPPORTED_FEATURE";
    case 0x78000004: return "ERROR_INVALID_NULL_POINTER";
    case 0x70000001: return "ERROR_NOT_AVAILABLE";
    default: return "(other)";
  }
}

int main(int argc, char** argv) {
  std::setvbuf(stdout, nullptr, _IONBF, 0);
  const int rank = argc > 1 ? std::atoi(argv[1]) : 0;
  HANDLE hmap = CreateFileMappingA(INVALID_HANDLE_VALUE, NULL, PAGE_READWRITE, 0, sizeof(Shared), "Local\\vllm-host-ipc-probe");
  Shared* sh = (Shared*)MapViewOfFile(hmap, FILE_MAP_ALL_ACCESS, 0, 0, sizeof(Shared));
  std::vector<sycl::device> gpus;
  for (auto& d : sycl::device::get_devices(sycl::info::device_type::gpu))
    if (d.get_backend() == sycl::backend::ext_oneapi_level_zero &&
        d.get_info<sycl::info::device::name>().find("Arc(TM) Pro") != std::string::npos) gpus.push_back(d);
  if (gpus.size() < 2) { std::printf("[%d] need two cards\n", rank); return 2; }
  sycl::queue q{gpus[rank], sycl::property::queue::in_order{}};
  auto zctx = sycl::get_native<sycl::backend::ext_oneapi_level_zero>(q.get_context());
  auto zdev = sycl::get_native<sycl::backend::ext_oneapi_level_zero>(q.get_device());
  HMODULE loader = LoadLibraryA("ze_loader.dll");
  auto getIpc = (pfn_getIpcHandle)GetProcAddress(loader, "zeMemGetIpcHandle");
  auto openIpc = (pfn_openIpcHandle)GetProcAddress(loader, "zeMemOpenIpcHandle");
  const size_t bytes = 1 << 20, n = bytes / 4;

  if (rank == 0) {
    sh->stage = 0; sh->verdict = 0; sh->note[0] = 0;
    uint32_t* host = sycl::malloc_host<uint32_t>(n, q);
    std::memset(host, 0, bytes);
    ze_ipc_mem_handle_t h{};
    ze_result_t r = getIpc((ze_context_handle_t)zctx, host, &h);
    std::printf("[0] host USM %p, zeMemGetIpcHandle: %s (0x%x)\n", (void*)host, rc(r), r);
    if (r != 0) { std::printf("[0] -> host memory cannot be exported on this driver either\n"); sh->stage = 9; return 1; }
    std::printf("[0] handle bytes 0-15: ");
    for (int i = 0; i < 16; ++i) std::printf("%02x", (unsigned char)h.data[i]);
    std::printf("\n");
    sh->pid = GetCurrentProcessId();
    std::memcpy(&sh->handle, &h, sizeof h);
    MemoryBarrier();
    sh->stage = 1;
    for (int i = 0; i < 600 && sh->stage != 2; ++i) Sleep(50);
    if (sh->stage != 2) { std::printf("[0] rank 1 never finished (it may have crashed)\n"); return 3; }
    std::printf("[0] rank 1 says: verdict %ld, %s\n", sh->verdict, sh->note);
    size_t good = 0;
    for (size_t i = 0; i < n; ++i) good += host[i] == 0xC0DE0000u + (uint32_t)i;
    std::printf("[0] words in MY host memory carrying the pattern card 1 wrote: %zu of %zu\n", good, n);
    std::printf(good == n ? "[0] RESULT: a card in another process wrote straight into this process's host memory\n"
                          : "[0] RESULT: no cross-process card write\n");
    sh->stage = 3;
    return 0;
  }

  for (int i = 0; i < 600 && sh->stage != 1 && sh->stage != 9; ++i) Sleep(50);
  if (sh->stage != 1) { std::printf("[1] no handle from rank 0\n"); return 3; }
  void* p = nullptr;
  ze_ipc_mem_handle_t h = sh->handle;
  ze_result_t r = openIpc((ze_context_handle_t)zctx, (ze_device_handle_t)zdev, h, 0, &p);
  std::printf("[1] zeMemOpenIpcHandle as given: %s (0x%x) -> %p\n", rc(r), r, p);
  LONG verdict = r == 0 ? 1 : 0;
  if (r != 0) {
    HANDLE src = OpenProcess(PROCESS_DUP_HANDLE, FALSE, sh->pid);
    HANDLE theirs = nullptr, mine = nullptr;
    std::memcpy(&theirs, h.data, sizeof theirs);
    BOOL ok = src && DuplicateHandle(src, theirs, GetCurrentProcess(), &mine, 0, FALSE, DUPLICATE_SAME_ACCESS);
    std::printf("[1] DuplicateHandle of the first 8 bytes (%p) from pid %lu: %s\n", (void*)theirs, sh->pid, ok ? "ok" : "failed");
    if (ok) {
      std::memcpy(h.data, &mine, sizeof mine);
      r = openIpc((ze_context_handle_t)zctx, (ze_device_handle_t)zdev, h, 0, &p);
      std::printf("[1] zeMemOpenIpcHandle after DuplicateHandle: %s (0x%x) -> %p\n", rc(r), r, p);
      if (r == 0) verdict = 2;
    }
  }
  if (!p || r != 0) {
    std::snprintf(sh->note, sizeof sh->note, "could not open the handle (%s)", rc(r));
    sh->verdict = verdict; MemoryBarrier(); sh->stage = 2; return 1;
  }
  uint32_t* w = (uint32_t*)p;
  try {
    q.parallel_for(sycl::range<1>(n), [=](sycl::id<1> i) { w[i] = 0xC0DE0000u + (uint32_t)i; }).wait();
    verdict = 3;
    std::snprintf(sh->note, sizeof sh->note, "card 1 kernel wrote through the opened pointer without a fault");
  } catch (const sycl::exception& e) {
    std::snprintf(sh->note, sizeof sh->note, "card 1 kernel threw: %.200s", e.what());
  }
  std::printf("[1] %s\n", sh->note);
  sh->verdict = verdict; MemoryBarrier(); sh->stage = 2;
  for (int i = 0; i < 200 && sh->stage != 3; ++i) Sleep(50);
  return 0;
}
