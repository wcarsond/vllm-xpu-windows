@echo off
rem ceiling-test.input.cmd - (0) the 72B lock-off fault with HOST memory sampled across it, then (1-3) the same load with a
rem hard 6 GiB working-set ceiling on each worker (VLLM_XPU_WS_LIMIT_GB=6), three times. Verdicts in probe\ceiling-test.output.txt,
rem host samples in probe\host-mem-fault.output.txt and probe\host-mem-ceiling.output.txt.
rem Usage:  cmd /c probe\ceiling-test.input.cmd      (~12 min)
rem Why (2026-10-11 18:30): the trim was 1 of 3, so the between-shard sweep is not enough; this is the OS-enforced version -
rem a worker can never hold more than 6 GiB resident at any instant. And the host side has never been measured at the fault.
setlocal enabledelayedexpansion
set HERE=%~dp0
set ROOT=%HERE%..
set LOG=%HERE%ceiling-test.output.txt
set VLLM_XPU_NO_LOAD_LOCK=1
set VLLM_XPU_WS_LIMIT_GB=
echo run %DATE% %TIME%  72B tensor split, lock OFF > "%LOG%"

echo. >> "%LOG%"
echo == 0 control: no ceiling, host memory sampled >> "%LOG%"
del /q "%HERE%host-mem-fault.output.txt" 2>nul
start /b "" powershell -NoProfile -File "%HERE%host-mem-sample.input.ps1" 110 "%HERE%host-mem-fault.output.txt"
ping -n 3 127.0.0.1 > nul
call "%ROOT%\server\run-smoke-tp2.input.cmd" FLASH_ATTN Qwen2.5-72B-Instruct-AWQ float16 0.97 2048 > nul 2>&1
call :verdict
:wait0
if not exist "%HERE%host-mem-fault.output.txt" (ping -n 3 127.0.0.1 > nul & goto wait0)
findstr /c:"MIN free" "%HERE%host-mem-fault.output.txt" >> "%LOG%"

set VLLM_XPU_WS_LIMIT_GB=6
for /l %%I in (1,1,3) do (
    echo. >> "%LOG%"
    echo == %%I ceiling 6 GiB per worker  !TIME! >> "%LOG%"
    if %%I==1 (
        del /q "%HERE%host-mem-ceiling.output.txt" 2>nul
        start /b "" powershell -NoProfile -File "%HERE%host-mem-sample.input.ps1" 230 "%HERE%host-mem-ceiling.output.txt"
        ping -n 3 127.0.0.1 > nul
    )
    call "%ROOT%\server\run-smoke-tp2.input.cmd" FLASH_ATTN Qwen2.5-72B-Instruct-AWQ float16 0.97 2048 > nul 2>&1
    call :verdict
)
:wait1
if not exist "%HERE%host-mem-ceiling.output.txt" (ping -n 3 127.0.0.1 > nul & goto wait1)
findstr /c:"MIN free" "%HERE%host-mem-ceiling.output.txt" >> "%LOG%"
type "%LOG%"
endlocal
exit /b 0

:verdict
findstr /c:"working-set ceiling" /c:"died before signalling" /c:"Model loading took" /c:"engine up" /c:"median" "%ROOT%\server\smoke-tp2-Qwen2.5-72B-Instruct-AWQ.console.log" >> "%LOG%"
exit /b 0
