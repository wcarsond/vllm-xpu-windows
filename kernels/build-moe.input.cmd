@echo off
rem build-moe.input.cmd - the kernel build with the MoE kernels ON: the 36 split grouped-GEMM units, the _moe_C extension, the MoE ops.
rem Usage:  cmd /c kernels\build-moe.input.cmd      (same logs as build.input.cmd; detach it with build-detached.input.ps1 semantics if long)
rem Why (2026-10-11): MOE_KERNELS_ENABLED was forced OFF on 2026-10-10 because grouped_gemm_xe2.cpp needed >30 GB to compile.
rem After gen-moe-split.input.py each unit holds one or two kernels. Everything already built stays built (ninja), so this is
rem the MoE pieces only: roughly 36 x 1-3 min at one job. Run it after the attention rebuild and its tests, never beside them.
set MOE_KERNELS_ENABLED=ON
call "%~dp0build.input.cmd"
exit /b %ERRORLEVEL%
