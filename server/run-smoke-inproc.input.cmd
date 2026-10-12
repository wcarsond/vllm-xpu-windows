@echo off
rem run-smoke-inproc.input.cmd - the first-light run with the engine core in the same process (no spawned child).
rem Usage:  cmd /c server\run-smoke-inproc.input.cmd      (console -> server\smoke-offline.inproc.log)
rem Why (2026-10-10): torch's F.linear passes every variant in a plain process (probe/linear-shapes*.py) but the same GEMM
rem launch fails with OUT_OF_DEVICE_MEMORY inside vLLM's spawned engine-core child. VLLM_ENABLE_V1_MULTIPROCESSING=0 keeps
rem the engine core in the launching process: if the light comes on this way, the difference is the child's process
rem environment on Windows (spawn, inherited variables, device context), not the kernels or the model.
set HERE=%~dp0
call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat" > nul 2>&1
set CC=cl
set VLLM_ENABLE_V1_MULTIPROCESSING=0
"%HERE%..\.venv312\Scripts\python.exe" -u "%HERE%smoke-offline.input.py" > "%HERE%smoke-offline.inproc.log" 2>&1
echo smoke exit %ERRORLEVEL% >> "%HERE%smoke-offline.inproc.log"
