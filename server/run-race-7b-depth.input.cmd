@echo off
rem run-race-7b-depth.input.cmd - vLLM depth leg inside the MSVC environment, console to a file. Argument: attention backend.
rem Usage:  cmd /c server\run-race-7b-depth.input.cmd TRITON_ATTN      (console -> race-7b-depth.<backend>.console.log)
rem         cmd /c server\run-race-7b-depth.input.cmd FLASH_ATTN       (after the cutlass rebuild is installed)
rem Why (2026-10-10): as run-race-7b-vllm.input.cmd; one file per backend so the two depth tables sit side by side.
rem Not while the kernel build is compiling (RAM).
rem RACE_TP=2 in the environment runs the leg on both cards (tensor split); its files carry a -tp2 suffix (2026-10-11).
rem 13:40, by name instead of environment:  run-race-7b-depth.input.cmd FLASH_ATTN <model-folder> [dtype] [gpu_util] [tp2|pp2]
rem         e.g.  run-race-7b-depth.input.cmd FLASH_ATTN Qwen2.5-32B-Instruct-AWQ float16 0.85 tp2
rem         Files then carry the model folder name after the split suffix (race-7b-depth-tp2-<model>.<backend>.*).
set HERE=%~dp0
set BACKEND=%1
if "%BACKEND%"=="" set BACKEND=TRITON_ATTN
if not "%2"=="" set SMOKE_MODEL=C:\Users\claude\models-hf\%2
if not "%3"=="" set SMOKE_DTYPE=%3
if not "%4"=="" set SMOKE_GPU_UTIL=%4
if /i "%5"=="tp2" set RACE_TP=2
if /i "%5"=="pp2" set RACE_PP=2
set SUFFIX=
if "%RACE_TP%"=="2" set SUFFIX=-tp2
if "%RACE_PP%"=="2" set SUFFIX=-pp2
if defined SMOKE_MODEL for %%A in ("%SMOKE_MODEL%") do set SUFFIX=%SUFFIX%-%%~nxA
call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat" > nul 2>&1
set CC=cl
set SMOKE_ATTN_BACKEND=%BACKEND%
"%HERE%..\.venv312\Scripts\python.exe" -u "%HERE%race-7b-depth.input.py" > "%HERE%race-7b-depth%SUFFIX%.%BACKEND%.console.log" 2>&1
echo depth exit %ERRORLEVEL% >> "%HERE%race-7b-depth%SUFFIX%.%BACKEND%.console.log"
copy /y "%HERE%race-7b-depth%SUFFIX%.output.txt" "%HERE%race-7b-depth%SUFFIX%.%BACKEND%.output.txt" > nul
findstr /c:"engine up" /c:"depth " /c:"Error" /c:"error:" /c:"depth exit" "%HERE%race-7b-depth%SUFFIX%.%BACKEND%.console.log"
