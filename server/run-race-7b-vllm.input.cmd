@echo off
rem run-race-7b-vllm.input.cmd - vLLM side of the 7B race, run inside the MSVC environment Triton-XPU needs, console to a file.
rem Usage:  cmd /c server\run-race-7b-vllm.input.cmd      (console -> race-7b-vllm.console.log; result -> race-7b-vllm.output.txt)
rem Why (2026-10-10): same two lessons as run-smoke.input.cmd: Triton compiles launcher stubs with cl.exe at run time, and
rem a child process's death only shows if the console goes to a file as it happens. Run with the router's model unloaded.
set HERE=%~dp0
call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat" > nul 2>&1
set CC=cl
rem Attention backend as the first argument (TRITON_ATTN default; FLASH_ATTN = the cutlass kernels, valid once the
rem 64-bit-address rebuild is installed). One console log and one result copy per backend, so the two sit side by side.
set BACKEND=%1
rem 03:20: cutlass is the default again (1.7-4.5x Triton at depth once the 64-bit-address fix is in the kernels).
if "%BACKEND%"=="" set BACKEND=FLASH_ATTN
set SMOKE_ATTN_BACKEND=%BACKEND%
rem SMOKE_EAGER=0 in the environment compiles the model (torch.compile); its files carry a -compiled suffix (2026-10-11).
set SUFFIX=
if "%SMOKE_EAGER%"=="0" set SUFFIX=-compiled
rem SMOKE_MODEL set in the environment (a model other than the 7B) adds the model folder name to the file names so results never overwrite each other (2026-10-11 13:15).
if defined SMOKE_MODEL for %%A in ("%SMOKE_MODEL%") do set SUFFIX=%SUFFIX%-%%~nxA
if defined SMOKE_OFFLOAD_GB if not "%SMOKE_OFFLOAD_GB%"=="0" set SUFFIX=%SUFFIX%-offload%SMOKE_OFFLOAD_GB%
if defined SMOKE_TAG set SUFFIX=%SUFFIX%-%SMOKE_TAG%
"%HERE%..\.venv312\Scripts\python.exe" -u "%HERE%race-7b-vllm.input.py" > "%HERE%race-7b-vllm.%BACKEND%%SUFFIX%.console.log" 2>&1
echo race exit %ERRORLEVEL% >> "%HERE%race-7b-vllm.%BACKEND%%SUFFIX%.console.log"
copy /y "%HERE%race-7b-vllm.output.txt" "%HERE%race-7b-vllm.%BACKEND%%SUFFIX%.output.txt" > nul
findstr /c:"engine up" /c:"run " /c:"median" /c:"Error" /c:"error:" /c:"race exit" "%HERE%race-7b-vllm.%BACKEND%%SUFFIX%.console.log"
