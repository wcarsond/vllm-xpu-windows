@echo off
rem release-test.input.cmd - Carson's release-on-use: drop each checkpoint tensor's pages from the working set the moment
rem the loader is done with it (VLLM_XPU_RELEASE_TENSOR=1). Three 72B tensor-split loads with the lock OFF, host RAM
rem sampled across the first. Verdicts in probe\release-test.output.txt, samples in probe\host-mem-release.output.txt.
rem Usage:  cmd /c probe\release-test.input.cmd [repeats]      (default 3; ~3-4 min each)
rem Why (2026-10-11 18:40, Carson: "like a web server not returning ports despite connection close ... we just close them
rem as we make them. Seems the most self limiting without cap issues"): the per-shard sweep was 1 of 3 (a shard's 3.7 GB
rem is resident again at once), the ceiling needs a number; this needs neither.
setlocal enabledelayedexpansion
set HERE=%~dp0
set ROOT=%HERE%..
set LOG=%HERE%release-test.output.txt
set N=%1
if "%N%"=="" set N=3
set VLLM_XPU_NO_LOAD_LOCK=1
set VLLM_XPU_WS_LIMIT_GB=
set VLLM_XPU_TRIM_WS=
set VLLM_XPU_RELEASE_TENSOR=1
echo run %DATE% %TIME%  72B tensor split, lock OFF, release-on-use ON, %N% repeats > "%LOG%"
for /l %%I in (1,1,%N%) do (
    echo. >> "%LOG%"
    echo == repeat %%I  !TIME! >> "%LOG%"
    if %%I==1 (
        del /q "%HERE%host-mem-release.output.txt" 2>nul
        start /b "" powershell -NoProfile -File "%HERE%host-mem-sample.input.ps1" 230 "%HERE%host-mem-release.output.txt"
        ping -n 3 127.0.0.1 > nul
    )
    call "%ROOT%\server\run-smoke-tp2.input.cmd" FLASH_ATTN Qwen2.5-72B-Instruct-AWQ float16 0.97 2048 > nul 2>&1
    findstr /c:"died before signalling" /c:"Model loading took" /c:"engine up" /c:"median" "%ROOT%\server\smoke-tp2-Qwen2.5-72B-Instruct-AWQ.console.log" >> "%LOG%"
)
:wait
if not exist "%HERE%host-mem-release.output.txt" (ping -n 3 127.0.0.1 > nul & goto wait)
findstr /c:"MIN free" "%HERE%host-mem-release.output.txt" >> "%LOG%"
type "%LOG%"
endlocal
