@echo off
rem cutlass-example-trace.input.cmd - run the 256-GRF 00_bmg_gemm example with the Unified Runtime trace on and show the last calls before it dies.
rem Usage:  cmd /c probe\cutlass-example-trace.input.cmd      (trace -> probe\cutlass-example.trace.log; summary printed and in cutlass-example-trace.output.txt)
rem Why (2026-10-10): the example exits with 0xC0000409 (fast-fail from an uncaught exception) and prints nothing. The trace names
rem the last UR call and its result; UR_LOG_LEVEL_ZERO adds the adapter's own error line. Small problem size so the trace stays short.
setlocal
set HERE=%~dp0
set B=%HERE%cutlass-example.build
set TRACE=%HERE%cutlass-example.trace.log
set LOG=%HERE%cutlass-example-trace.output.txt
set ONEAPI=C:\Program Files (x86)\Intel\oneAPI
call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat" > nul 2>&1
set SETVARS_CALL=1
call "%ONEAPI%\compiler\2026.1\env\vars.bat" > nul 2>&1
set PATH=%HERE%..\.venv312\Library\bin;%PATH%
set ONEAPI_DEVICE_SELECTOR=level_zero:0
set SYCL_UR_TRACE=2
set UR_LOG_LEVEL_ZERO=level:error;flush:error;output:stderr
echo run %DATE% %TIME% > "%LOG%"
"%B%\gemm-256.exe" --m=256 --n=256 --k=256 --iterations=0 > "%TRACE%" 2>&1
echo exit %ERRORLEVEL% >> "%LOG%"
echo -- kernel launches and errors in the trace >> "%LOG%"
findstr /n /c:"urEnqueueKernelLaunch" /c:"ERROR" /c:"error" /c:"Disposition" /c:"exception" "%TRACE%" | findstr /v "UR_RESULT_SUCCESS" >> "%LOG%"
echo -- last 25 lines >> "%LOG%"
powershell -NoProfile -Command "Get-Content -Tail 25 '%TRACE%' | ForEach-Object { if ($_.Length -gt 300) { $_.Substring(0,300) } else { $_ } }" >> "%LOG%"
type "%LOG%"
exit /b 0
