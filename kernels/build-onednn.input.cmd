@echo off
rem build-onednn.input.cmd - the quantized-kernel build: oneDNN-backed GEMMs (int4 w4a16/w4a8, fp8 w8a16/w8a8, fp4 w4a4) plus
rem everything already built (attention, mhc, gdn, mqa, MoE), then the wheel into .venv312.
rem Usage:  cmd /c kernels\build-onednn.input.cmd      (hours; run detached via build-detached.input.ps1 style: see below)
rem         powershell -NoProfile -File kernels\build-detached.input.ps1 build-onednn.input.cmd   (detached + RAM watcher)
rem Why (2026-10-11, Carson: "yes if we need to compile again for different quants lets do it"): Q4/FP8-class models of the
rem size only two cards can hold need these GEMMs; they were built as stubs because oneDNN was not on the box.
rem cmake\Modules\FindoneDNN.cmake fetches oneDNN rls-v3.13 (0e2a5bf) and builds it STATIC with the SYCL GPU runtime and no CPU
rem runtime inside the kernel build. MAX_JOBS stays 1 (32 GB); the watcher kills the compiler under 2.5 GB free.
set VLLM_XPU_ENABLE_ONEDNN=ON
set MOE_KERNELS_ENABLED=ON
call "%~dp0build.input.cmd"
if errorlevel 1 exit /b 1
call "%~dp0wheel.input.cmd"
