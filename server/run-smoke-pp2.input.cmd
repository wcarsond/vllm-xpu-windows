@echo off
rem run-smoke-pp2.input.cmd - the two-card run (Coder-7B, pipeline split) inside the MSVC environment, console to a file.
rem Usage:  cmd /c server\run-smoke-pp2.input.cmd      (console -> smoke-pp2.console.log; result -> smoke-pp2.output.txt)
rem Why (2026-10-10): as run-smoke.input.cmd. Two engine workers spawn here (one per card) and talk through the RAM relay;
rem their output is tagged by pid in the console log, which is the only place a worker's failure shows.
set HERE=%~dp0
call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat" > nul 2>&1
set CC=cl
rem Attention backend as the first argument; default FLASH_ATTN (cutlass, fixed 2026-10-11), TRITON_ATTN for an A/B.
set BACKEND=%1
if "%BACKEND%"=="" set BACKEND=FLASH_ATTN
set SMOKE_ATTN_BACKEND=%BACKEND%
"%HERE%..\.venv312\Scripts\python.exe" -u "%HERE%smoke-pp2.input.py" > "%HERE%smoke-pp2.console.log" 2>&1
echo pp2 exit %ERRORLEVEL% >> "%HERE%smoke-pp2.console.log"
