@echo off
rem spill-chain.input.cmd - does the fault coincide with the driver spilling card memory into system RAM?
rem Runs four cases back to back, never two at once, each watched by probe\gpu-mem-sample.input.ps1 (the WDDM
rem per-adapter Dedicated/Shared Usage counters, i.e. the driver's own view, not torch's). Outputs beside this file:
rem   spill-A.output.txt  the probe variant that faulted at "~20 GB retained" (keeps every other tensor past the cap)
rem   spill-B.output.txt  vLLM 72B tensor split, load lock OFF (the real fault)
rem   spill-C.output.txt  vLLM 72B tensor split, load lock ON  (the working load)
rem   threshold-sweep.output.txt  retained-cap sweep 4..20 GiB and the EmptyWorkingSet mitigation
rem Usage:  cmd /c probe\spill-chain.input.cmd      (~10 min)
rem Why (2026-10-11 17:35): torch's max_memory_allocated says the 72B workers peak at 19.55 of 23.33 GiB, but torch only
rem knows what it asked for; this driver backs device allocations with RAM silently (30 GiB on a 23.3 GiB card, no OOM).
rem If Shared Usage rises just before the fault, the hazard IS the spill and loading can run in parallel up to the line
rem and serialize only past it (Carson's look-ahead carry). If it never rises, the spill story is dead for vLLM's case.
setlocal
set HERE=%~dp0
set ROOT=%HERE%..
set PY=%ROOT%\.venv312\Scripts\python.exe
set SAMPLER=%HERE%gpu-mem-sample.input.ps1

rem ---- A: the probe variant that faulted ------------------------------------------------------------------
del /q "%HERE%spill-A.output.txt" 2>nul
start /b "" powershell -NoProfile -File "%SAMPLER%" 70 "%HERE%spill-A.output.txt"
ping -n 3 127.0.0.1 > nul
set MMAP_DEVICE_PER_READER=1
set MMAP_NO_MASK=1
set MMAP_COPY_INTO=1
set MMAP_RETAIN_HALF=1
set MMAP_RETAIN_CAP_GB=
"%PY%" "%HERE%mmap-readers.input.py" Qwen2.5-72B-Instruct-AWQ 2 1 > "%HERE%spill-A.reader.log" 2>&1
set MMAP_RETAIN_HALF=
:waitA
if not exist "%HERE%spill-A.output.txt" (ping -n 3 127.0.0.1 > nul & goto waitA)

rem ---- B: vLLM, lock OFF (faults) -------------------------------------------------------------------------
del /q "%HERE%spill-B.output.txt" 2>nul
start /b "" powershell -NoProfile -File "%SAMPLER%" 150 "%HERE%spill-B.output.txt"
ping -n 3 127.0.0.1 > nul
set VLLM_XPU_NO_LOAD_LOCK=1
call "%ROOT%\server\run-smoke-tp2.input.cmd" FLASH_ATTN Qwen2.5-72B-Instruct-AWQ float16 0.97 2048 > nul 2>&1
copy /y "%ROOT%\server\smoke-tp2-Qwen2.5-72B-Instruct-AWQ.console.log" "%HERE%spill-B.vllm.log" > nul
set VLLM_XPU_NO_LOAD_LOCK=
:waitB
if not exist "%HERE%spill-B.output.txt" (ping -n 3 127.0.0.1 > nul & goto waitB)

rem ---- C: vLLM, lock ON (works) ---------------------------------------------------------------------------
del /q "%HERE%spill-C.output.txt" 2>nul
start /b "" powershell -NoProfile -File "%SAMPLER%" 240 "%HERE%spill-C.output.txt"
ping -n 3 127.0.0.1 > nul
call "%ROOT%\server\run-smoke-tp2.input.cmd" FLASH_ATTN Qwen2.5-72B-Instruct-AWQ float16 0.97 2048 > nul 2>&1
copy /y "%ROOT%\server\smoke-tp2-Qwen2.5-72B-Instruct-AWQ.console.log" "%HERE%spill-C.vllm.log" > nul
:waitC
if not exist "%HERE%spill-C.output.txt" (ping -n 3 127.0.0.1 > nul & goto waitC)

rem ---- D: threshold sweep + working-set trim ------------------------------------------------------------
call "%HERE%threshold-sweep.input.cmd" > nul 2>&1

echo CHAIN DONE
endlocal
