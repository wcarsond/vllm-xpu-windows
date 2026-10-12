// thread-stack.cpp - print the native call stack of every thread of another process: module+offset per frame, and the
// symbol when the module has one (system DLLs do, Python and the Intel driver mostly do not).
// Usage:  probe\thread-stack.input.cmd <pid> [<pid> ...]        (build + run; output in probe\thread-stack.output.txt)
// Why (2026-10-11 03:50): the tensor-split workers finish their shutdown, Windows marks them exited, and one thread each
// keeps spinning (30% user, 70% kernel, last wait reason = yield) that no kill can reach. py-spy cannot read an exiting
// process ("Failed to find a python interpreter") and there is no debugger on the box. This takes a few samples of the
// thread's user-mode stack through SuspendThread + GetThreadContext + StackWalk64, which only needs the address space to
// still exist (it does: 3 GB resident). Three samples 200 ms apart show whether the frames change (a loop) or not.
#include <windows.h>
#include <dbghelp.h>
#include <tlhelp32.h>
#include <psapi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <vector>
#include <string>

#pragma comment(lib, "dbghelp.lib")
#pragma comment(lib, "psapi.lib")

struct Mod { ULONG64 base; ULONG64 size; std::string name; };

static std::vector<Mod> modules(HANDLE hp) {
    std::vector<Mod> out;
    std::vector<HMODULE> mods(4096);
    DWORD cb = 0;
    if (!EnumProcessModulesEx(hp, mods.data(), (DWORD)(mods.size() * sizeof(HMODULE)), &cb, LIST_MODULES_ALL)) {
        printf("  EnumProcessModulesEx failed (error %lu); frames will show raw addresses\n", GetLastError());
        return out;
    }
    size_t n = cb / sizeof(HMODULE);
    for (size_t i = 0; i < n; i++) {
        MODULEINFO mi{};
        char path[MAX_PATH] = "?";
        if (!GetModuleInformation(hp, mods[i], &mi, sizeof(mi))) continue;
        GetModuleFileNameExA(hp, mods[i], path, MAX_PATH);
        const char* slash = strrchr(path, '\\');
        out.push_back({ (ULONG64)mi.lpBaseOfDll, mi.SizeOfImage, slash ? slash + 1 : path });
    }
    return out;
}

static std::string where(const std::vector<Mod>& mods, HANDLE hp, ULONG64 addr) {
    char buf[512];
    for (const Mod& m : mods) {
        if (addr >= m.base && addr < m.base + m.size) {
            char symbuf[sizeof(SYMBOL_INFO) + 256] = {};
            SYMBOL_INFO* si = (SYMBOL_INFO*)symbuf;
            si->SizeOfStruct = sizeof(SYMBOL_INFO); si->MaxNameLen = 255;
            DWORD64 disp = 0;
            if (SymFromAddr(hp, addr, &disp, si))
                snprintf(buf, sizeof buf, "%s!%s+0x%llx", m.name.c_str(), si->Name, (unsigned long long)disp);
            else
                snprintf(buf, sizeof buf, "%s+0x%llx", m.name.c_str(), (unsigned long long)(addr - m.base));
            return buf;
        }
    }
    snprintf(buf, sizeof buf, "0x%llx (no module)", (unsigned long long)addr);
    return buf;
}

static void walk(HANDLE hp, HANDLE ht, const std::vector<Mod>& mods) {
    CONTEXT ctx{};
    ctx.ContextFlags = CONTEXT_FULL;
    if (!GetThreadContext(ht, &ctx)) { printf("    GetThreadContext failed (error %lu)\n", GetLastError()); return; }
    printf("    rip %s   rsp 0x%llx\n", where(mods, hp, ctx.Rip).c_str(), (unsigned long long)ctx.Rsp);
    STACKFRAME64 sf{};
    sf.AddrPC.Offset = ctx.Rip;    sf.AddrPC.Mode = AddrModeFlat;
    sf.AddrFrame.Offset = ctx.Rbp; sf.AddrFrame.Mode = AddrModeFlat;
    sf.AddrStack.Offset = ctx.Rsp; sf.AddrStack.Mode = AddrModeFlat;
    for (int i = 0; i < 48; i++) {
        if (!StackWalk64(IMAGE_FILE_MACHINE_AMD64, hp, ht, &sf, &ctx, NULL, SymFunctionTableAccess64, SymGetModuleBase64, NULL)) {
            printf("    [walk ended: error %lu]\n", GetLastError());
            break;
        }
        if (sf.AddrPC.Offset == 0) break;
        printf("    %2d  %s\n", i, where(mods, hp, sf.AddrPC.Offset).c_str());
    }
}

int main(int argc, char** argv) {
    if (argc < 2) { printf("usage: thread-stack <pid> [<pid> ...]\n"); return 2; }
    for (int a = 1; a < argc; a++) {
        DWORD pid = (DWORD)atoi(argv[a]);
        printf("== pid %lu\n", pid);
        HANDLE hp = OpenProcess(PROCESS_QUERY_INFORMATION | PROCESS_VM_READ, FALSE, pid);
        if (!hp) { printf("  OpenProcess failed (error %lu)\n", GetLastError()); continue; }
        DWORD code = 0;
        if (GetExitCodeProcess(hp, &code)) printf("  exit code %s\n", code == STILL_ACTIVE ? "STILL_ACTIVE" : std::to_string(code).c_str());
        SymSetOptions(SYMOPT_UNDNAME | SYMOPT_DEFERRED_LOADS);
        if (!SymInitialize(hp, NULL, TRUE)) printf("  SymInitialize failed (error %lu); module+offset only\n", GetLastError());
        std::vector<Mod> mods = modules(hp);
        printf("  %zu modules\n", mods.size());
        HANDLE snap = CreateToolhelp32Snapshot(TH32CS_SNAPTHREAD, 0);
        THREADENTRY32 te{}; te.dwSize = sizeof(te);
        int threads = 0;
        for (BOOL ok = Thread32First(snap, &te); ok; ok = Thread32Next(snap, &te)) {
            if (te.th32OwnerProcessID != pid) continue;
            threads++;
            printf("  thread %lu\n", te.th32ThreadID);
            HANDLE ht = OpenThread(THREAD_GET_CONTEXT | THREAD_SUSPEND_RESUME | THREAD_QUERY_INFORMATION, FALSE, te.th32ThreadID);
            if (!ht) { printf("    OpenThread failed (error %lu)\n", GetLastError()); continue; }
            for (int sample = 0; sample < 3; sample++) {
                printf("   sample %d\n", sample);
                DWORD prev = SuspendThread(ht);
                if (prev == (DWORD)-1) { printf("    SuspendThread failed (error %lu)\n", GetLastError()); }
                walk(hp, ht, mods);
                if (prev != (DWORD)-1) ResumeThread(ht);
                Sleep(200);
            }
            CloseHandle(ht);
        }
        if (!threads) printf("  no threads listed\n");
        CloseHandle(snap);
        SymCleanup(hp);
        CloseHandle(hp);
    }
    return 0;
}
