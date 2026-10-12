@echo off
rem run-smoke-resident.input.cmd - the first-light run with Level Zero allocations made resident at creation.
rem Usage:  cmd /c server\run-smoke-resident.input.cmd      (console -> server\smoke-offline.resident.log)
rem Why (2026-10-10): the failing GEMM launch has USM_INDIRECT_ACCESS set, so Level Zero must reference every USM allocation
rem of the process per submission; Windows' GPU memory manager caps that list, and vLLM holds hundreds of allocations.
rem UR_L0_USM_RESIDENT (bit 0 = device allocations, 0x001; the older SYCL_PI_LEVEL_ZERO_USM_RESIDENT spelling is set too)
rem tells the Level Zero adapter to make allocations resident when they are created, so launches need no residency list.
set HERE=%~dp0
call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat" > nul 2>&1
set CC=cl
set UR_L0_USM_RESIDENT=0x001
set SYCL_PI_LEVEL_ZERO_USM_RESIDENT=0x001
"%HERE%..\.venv312\Scripts\python.exe" -u "%HERE%smoke-offline.input.py" > "%HERE%smoke-offline.resident.log" 2>&1
echo smoke exit %ERRORLEVEL% >> "%HERE%smoke-offline.resident.log"
