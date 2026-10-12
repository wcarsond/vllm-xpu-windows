@echo off
rem private-copy-test.input.cmd - Carson's "two cursors" in vLLM itself: both tensor-parallel workers read the shared file
rem pages, but each card DMAs only from a private heap copy of every tensor (VLLM_XPU_PRIVATE_COPY=1). 72B, lock OFF,
rem three loads back to back. Verdicts in probe\private-copy-test.output.txt.
rem Usage:  cmd /c probe\private-copy-test.input.cmd [repeats]      (default 3; ~3.5 min each if it loads, ~1 min if it faults)
rem Why (2026-10-11 19:00): memory pressure is out on both sides (driver counters, host sampler); tensor split faults
rem where pipeline split does not at the same memory per card, and the one thing tensor split alone does is feed the
rem same physical page-cache pages to two adapters at once. If a private copy per tensor stops the fault, this is the
rem fix: no lock, no second file, no stagger, one host memcpy per tensor.
setlocal enabledelayedexpansion
set HERE=%~dp0
set ROOT=%HERE%..
set LOG=%HERE%private-copy-test.output.txt
set N=%1
if "%N%"=="" set N=3
set VLLM_XPU_NO_LOAD_LOCK=1
set VLLM_XPU_WS_LIMIT_GB=
set VLLM_XPU_TRIM_WS=
set VLLM_XPU_RELEASE_TENSOR=
set VLLM_XPU_PRIVATE_COPY=1
echo run %DATE% %TIME%  72B tensor split, lock OFF, private copy per tensor ON, %N% repeats > "%LOG%"
for /l %%I in (1,1,%N%) do (
    echo. >> "%LOG%"
    echo == repeat %%I  !TIME! >> "%LOG%"
    call "%ROOT%\server\run-smoke-tp2.input.cmd" FLASH_ATTN Qwen2.5-72B-Instruct-AWQ float16 0.97 2048 > nul 2>&1
    findstr /c:"died before signalling" /c:"Model loading took" /c:"engine up" /c:"median" "%ROOT%\server\smoke-tp2-Qwen2.5-72B-Instruct-AWQ.console.log" >> "%LOG%"
)
type "%LOG%"
endlocal
