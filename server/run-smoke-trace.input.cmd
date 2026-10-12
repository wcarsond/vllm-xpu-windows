@echo off
rem run-smoke-trace.input.cmd - the first-light run with the Unified Runtime call trace on, to see the last call before the failure.
rem Usage:  cmd /c server\run-smoke-trace.input.cmd      (console -> server\smoke-offline.trace.log, large)
rem Why (2026-10-10): the Level Zero validation layer printed nothing for the UR_RESULT_ERROR_OUT_OF_RESOURCES failure, so
rem the call is well-formed and the driver refused it. SYCL_UR_TRACE=2 logs every UR call with arguments and result; the
rem entry that returns UR_RESULT_ERROR_OUT_OF_RESOURCES names the kernel (urEnqueueKernelLaunch with its hKernel) or the
rem allocation. Diagnostics only: the trace is many MB and slows the run.
set HERE=%~dp0
call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat" > nul 2>&1
set CC=cl
set SYCL_UR_TRACE=2
"%HERE%..\.venv312\Scripts\python.exe" -u "%HERE%smoke-offline.input.py" > "%HERE%smoke-offline.trace.log" 2>&1
echo smoke exit %ERRORLEVEL% >> "%HERE%smoke-offline.trace.log"
