@echo off
rem inspect-attn-dll.input.cmd - pull the device binaries out of attn_kernels_xe_2.dll and read what IGC recorded per kernel: GRF mode, spill, private memory, SLM.
rem Usage:  cmd /c kernels\inspect-attn-dll.input.cmd      (writes kernels\inspect-attn-dll.output.txt; scratch in kernels\inspect-attn-dll.work\)
rem Why (2026-10-10): every paged/causal chunk-prefill launch fails with OUT_OF_RESOURCES before any allocation, the no-window
rem non-paged one faults the device, and the base library (no 256-GRF link option) is fine. The zebin's .ze_info says whether
rem -cl-intel-256-GRF-per-thread really applied (grf_count) and how big the spills are; a kernel that needs gigabytes of scratch
rem across the device fails exactly like this. Reads the DLL from the build tree (same file the wheel installed).
setlocal
set HERE=%~dp0
set DLL=%HERE%..\src\vllm-xpu-kernels\build\temp\attn_kernels_xe_2.dll
set WORK=%HERE%inspect-attn-dll.work
set LOG=%HERE%inspect-attn-dll.output.txt
set ONEAPI=C:\Program Files (x86)\Intel\oneAPI
set OCLOC=%ONEAPI%\ocloc\latest\bin\ocloc.exe
set PATH=%ONEAPI%\compiler\2026.1\bin;%ONEAPI%\compiler\2026.1\bin\compiler;%ONEAPI%\ocloc\latest\bin;%PATH%
echo run %DATE% %TIME% > "%LOG%"
if not exist "%DLL%" ( echo DLL not found: %DLL% >> "%LOG%" & type "%LOG%" & exit /b 1 )
for %%F in ("%DLL%") do echo dll %%~fF  %%~zF bytes  %%~tF >> "%LOG%"
if exist "%WORK%" rmdir /s /q "%WORK%"
mkdir "%WORK%"
where clang-offload-extract.exe >> "%LOG%" 2>&1
echo -- extract device images >> "%LOG%"
clang-offload-extract.exe --output="%WORK%\img" "%DLL%" >> "%LOG%" 2>&1
dir /b "%WORK%" >> "%LOG%"
echo -- per image: kind, then .ze_info summary >> "%LOG%"
for %%I in ("%WORK%\img*") do call :one "%%~fI"
echo done %TIME% >> "%LOG%"
type "%LOG%"
exit /b 0

:one
echo == %~nx1  %~z1 bytes >> "%LOG%"
mkdir "%~1.dump" 2>nul
"%OCLOC%" disasm -file "%~1" -dump "%~1.dump" -device bmg >> "%LOG%" 2>&1
rem .ze_info is YAML: one kernel block per "- name:", with execution_env: grf_count, has_..., private_size, spill_size, slm_size
for /r "%~1.dump" %%Z in (*.ze_info*) do (
  echo    ze_info %%~nxZ >> "%LOG%"
  "%HERE%..\.venv312\Scripts\python.exe" "%HERE%ze-info-summary.input.py" "%%~fZ" >> "%LOG%" 2>&1
)
exit /b 0
