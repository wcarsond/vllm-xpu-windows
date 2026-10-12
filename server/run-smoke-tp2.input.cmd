@echo off
rem run-smoke-tp2.input.cmd - the two-card tensor-split run (Coder-7B, TP=2) inside the MSVC environment, console to a file.
rem Usage:  cmd /c server\run-smoke-tp2.input.cmd      (console -> smoke-tp2.console.log; result -> smoke-tp2.output.txt)
rem Why (2026-10-10): as run-smoke-pp2.input.cmd. Two workers, one per card, every layer split in half; the all_reduces go
rem through whatever exchange xpu_communicator.py selects (VLLM_XPU_EXCHANGE=shm once the shared-memory engine exists).
rem Do not run while the kernel build is compiling: the two workers load ~8 GB through RAM and the build peaks near 14 GB.
set HERE=%~dp0
call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat" > nul 2>&1
set CC=cl
rem Attention backend as the first argument; default FLASH_ATTN (cutlass, fixed 2026-10-11), TRITON_ATTN for an A/B.
rem 13:55, any model by name:  run-smoke-tp2.input.cmd FLASH_ATTN <model-folder> [dtype] [gpu_util] [max_model_len]
rem         e.g.  run-smoke-tp2.input.cmd FLASH_ATTN Qwen2.5-72B-Instruct-AWQ float16 0.95 4096
rem         Its files are then smoke-tp2-<model-folder>.console.log and .output.txt.
set BACKEND=%1
if "%BACKEND%"=="" set BACKEND=FLASH_ATTN
set SMOKE_ATTN_BACKEND=%BACKEND%
rem 15:10: print a Python stack when a worker takes an access violation. The 72B AWQ load kills a worker with
rem 0xC0000005 in native code, which leaves no Python traceback and, with Windows Error Reporting off on this box,
rem nothing in the event log either. faulthandler dumps every thread's Python stack as the process falls over, which
rem names the operation that crashed. Inherited by the spawned workers. Costs nothing when nothing crashes.
set PYTHONFAULTHANDLER=1
rem 03:20: zombie workers from an earlier run can hold the console log open ("being used by another process"), and then the
rem redirect fails and the run never starts. Write to a fresh name when the usual one is locked, and copy over afterwards.
set TAG=
if not "%2"=="" (set SMOKE_MODEL=C:\Users\claude\models-hf\%2& set TAG=-%2)
if not "%3"=="" set SMOKE_DTYPE=%3
if not "%4"=="" set SMOKE_GPU_UTIL=%4
if not "%5"=="" set SMOKE_MAX_LEN=%5
set CON=%HERE%smoke-tp2%TAG%.console.log
del /q "%CON%" 2>nul
if exist "%CON%" set CON=%HERE%smoke-tp2%TAG%.console.%RANDOM%.log
"%HERE%..\.venv312\Scripts\python.exe" -u "%HERE%smoke-tp2.input.py" > "%CON%" 2>&1
echo tp2 exit %ERRORLEVEL% >> "%CON%"
findstr /c:"engine up" /c:"run " /c:"median" /c:"Error" /c:"error:" /c:"tp2 exit" "%CON%"
if not "%CON%"=="%HERE%smoke-tp2%TAG%.console.log" copy /y "%CON%" "%HERE%smoke-tp2%TAG%.console.log" > nul 2>&1
