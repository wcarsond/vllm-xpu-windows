@echo off
rem build-the-failure.input.cmd - make the concurrent-load fault happen on purpose, then try to make it NOT happen.
rem Three vLLM runs back to back, load lock OFF for all, never two at once. Verdicts to probe\build-the-failure.output.txt.
rem   1. EXPOSURE: Qwen2.5-0.5B tensor split with the loader slowed to ~35 s per worker (VLLM_XPU_LOAD_SLEEP_MS). The
rem      0.5B holds ~0.5 GB per card, so if this faults the variable is how long both workers overlap in the loading
rem      state, not memory - and the 0.5B becomes a 40 s reproducer. If it stays clean, memory pressure is required.
rem   2. TRIM:  the 72B (faults every time) with the file pages released after every shard (VLLM_XPU_TRIM_WS=1).
rem   3. CACHE: the 72B with torch's device cache released after every shard (VLLM_XPU_EMPTY_CACHE=1).
rem Usage:  cmd /c probe\build-the-failure.input.cmd      (~5 min)
rem Why (2026-10-11 18:05, Carson: "again lets build the failure"): the spill theory died on the driver's own counters
rem (spill-B: 0.22 GB shared at the moment of the fault), the fault lands at a different shard every run (a race), and
rem the standalone probe only faults when it overflows the card, which vLLM never does. So the reproducer is vLLM.
setlocal
set HERE=%~dp0
set ROOT=%HERE%..
set LOG=%HERE%build-the-failure.output.txt
set VLLM_XPU_NO_LOAD_LOCK=1
echo run %DATE% %TIME%  load lock OFF throughout > "%LOG%"

echo. >> "%LOG%"
echo == 1 EXPOSURE: 0.5B tensor split, 120 ms sleep per tensor (~35 s load) >> "%LOG%"
set VLLM_XPU_LOAD_SLEEP_MS=120
call "%ROOT%\server\run-smoke-tp2.input.cmd" FLASH_ATTN Qwen2.5-0.5B-Instruct bfloat16 0.35 2048 > nul 2>&1
call :verdict "%ROOT%\server\smoke-tp2-Qwen2.5-0.5B-Instruct.console.log"
set VLLM_XPU_LOAD_SLEEP_MS=

echo. >> "%LOG%"
echo == 2 TRIM: 72B, EmptyWorkingSet after every shard >> "%LOG%"
set VLLM_XPU_TRIM_WS=1
call "%ROOT%\server\run-smoke-tp2.input.cmd" FLASH_ATTN Qwen2.5-72B-Instruct-AWQ float16 0.97 2048 > nul 2>&1
call :verdict "%ROOT%\server\smoke-tp2-Qwen2.5-72B-Instruct-AWQ.console.log"
set VLLM_XPU_TRIM_WS=

echo. >> "%LOG%"
echo == 3 CACHE: 72B, torch.xpu.empty_cache after every shard >> "%LOG%"
set VLLM_XPU_EMPTY_CACHE=1
call "%ROOT%\server\run-smoke-tp2.input.cmd" FLASH_ATTN Qwen2.5-72B-Instruct-AWQ float16 0.97 2048 > nul 2>&1
call :verdict "%ROOT%\server\smoke-tp2-Qwen2.5-72B-Instruct-AWQ.console.log"
set VLLM_XPU_EMPTY_CACHE=

type "%LOG%"
endlocal
exit /b 0

:verdict
findstr /c:"died before signalling" /c:"0xC0000005" /c:"engine up" /c:"median" /c:"Model loading took" %1 >> "%LOG%"
exit /b 0
