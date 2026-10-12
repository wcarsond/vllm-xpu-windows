@echo off
rem cutlass-example-run.input.cmd - run Intel's 00_bmg_gemm example (plain and 256-GRF links) on the B60 over torch's runtime set.
rem Usage:  cmd /c probe\cutlass-example-run.input.cmd      (writes probe\cutlass-example-run.output.txt; needs cutlass-example.input.cmd first)
rem Why (2026-10-10): see cutlass-example.input.cmd. ONEAPI_DEVICE_SELECTOR=level_zero:0 is the first Level Zero device, a B60
rem on this PC (the probe prints which device it got, as a check). Run only when nothing else is on the card.
setlocal
set HERE=%~dp0
set B=%HERE%cutlass-example.build
set LOG=%HERE%cutlass-example-run.output.txt
set ONEAPI=C:\Program Files (x86)\Intel\oneAPI
call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat" > nul 2>&1
set SETVARS_CALL=1
call "%ONEAPI%\compiler\2026.1\env\vars.bat" > nul 2>&1
set PATH=%HERE%..\.venv312\Library\bin;%PATH%
set ONEAPI_DEVICE_SELECTOR=level_zero:0
echo run %DATE% %TIME% > "%LOG%"
echo -- which device is level_zero:0 (grf256 probe prints it) >> "%LOG%"
"%HERE%grf256.build\aot-128.exe" check B60 >> "%LOG%" 2>&1
echo -- gemm-128 (default 5120x4096x4096, 100 iterations) >> "%LOG%"
"%B%\gemm-128.exe" --m=1024 --n=1024 --k=1024 --iterations=3 >> "%LOG%" 2>&1
echo    exit %ERRORLEVEL% >> "%LOG%"
echo -- gemm-256 >> "%LOG%"
"%B%\gemm-256.exe" --m=1024 --n=1024 --k=1024 --iterations=3 >> "%LOG%" 2>&1
echo    exit %ERRORLEVEL% >> "%LOG%"
echo -- gemm-plain (probe flags, 256 GRF) >> "%LOG%"
"%B%\gemm-plain.exe" --m=1024 --n=1024 --k=1024 --iterations=3 >> "%LOG%" 2>&1
echo    exit %ERRORLEVEL% >> "%LOG%"
echo -- gemm-legacy (old copy atoms, kernel-build flags, 256 GRF) >> "%LOG%"
"%B%\gemm-legacy.exe" --m=1024 --n=1024 --k=1024 --iterations=3 >> "%LOG%" 2>&1
echo    exit %ERRORLEVEL% >> "%LOG%"
echo done %TIME% >> "%LOG%"
type "%LOG%"
exit /b 0
