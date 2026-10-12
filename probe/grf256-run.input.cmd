@echo off
rem grf256-run.input.cmd - run the four grf256 builds on the B60 using torch's Unified Runtime set (the one that sees Level Zero).
rem Usage:  cmd /c probe\grf256-run.input.cmd      (writes probe\grf256-run.output.txt; needs probe\grf256.build from grf256.input.cmd)
rem Why (2026-10-10): with only the compiler's vars.bat on PATH the UR loader brings up the OpenCL adapter and never the Level
rem Zero ones (sycl-ls.input.cmd), so a bare run cannot find the B60s. torch's wheel carries a working set in
rem .venv312\Library\bin; putting it first on PATH makes the probe run on exactly the runtime vLLM runs on.
setlocal
set HERE=%~dp0
set B=%HERE%grf256.build
set LOG=%HERE%grf256-run.output.txt
set ONEAPI=C:\Program Files (x86)\Intel\oneAPI
call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat" > nul 2>&1
set SETVARS_CALL=1
call "%ONEAPI%\compiler\2026.1\env\vars.bat" > nul 2>&1
set PATH=%HERE%..\.venv312\Library\bin;%PATH%
set UR_LOG_LOADER=level:info;flush:info;output:stderr
echo run %DATE% %TIME% > "%LOG%"
echo -- run aot-128 on B60 >> "%LOG%"
"%B%\aot-128.exe" aot-128 B60 >> "%LOG%" 2>&1
echo    exit %ERRORLEVEL% >> "%LOG%"
echo -- run aot-256 on B60 >> "%LOG%"
"%B%\aot-256.exe" aot-256 B60 >> "%LOG%" 2>&1
echo    exit %ERRORLEVEL% >> "%LOG%"
echo -- run jit default GRF on B60 >> "%LOG%"
"%B%\jit.exe" jit-default B60 >> "%LOG%" 2>&1
echo    exit %ERRORLEVEL% >> "%LOG%"
echo -- run jit with SYCL_PROGRAM_COMPILE_OPTIONS=-ze-opt-large-register-file on B60 >> "%LOG%"
set SYCL_PROGRAM_COMPILE_OPTIONS=-ze-opt-large-register-file
"%B%\jit.exe" jit-largeGRF B60 >> "%LOG%" 2>&1
echo    exit %ERRORLEVEL% >> "%LOG%"
set SYCL_PROGRAM_COMPILE_OPTIONS=
echo done %TIME% >> "%LOG%"
type "%LOG%"
exit /b 0
