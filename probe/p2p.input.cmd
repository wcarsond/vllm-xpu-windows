@echo off
rem p2p.input.cmd - build probe\p2p.cpp (JIT, plain spir64) and run it over torch's runtime set on the two B60s.
rem Usage:  cmd /c probe\p2p.input.cmd      (writes probe\p2p.output.txt)
rem Why (2026-10-10): see p2p.cpp. Compile is one small file (safe beside the long kernel build); the run uses both cards for
rem a few seconds, so run it only when no vLLM process is on them (probe\gpu-holders.input.ps1).
setlocal
set HERE=%~dp0
set B=%HERE%p2p.build
set LOG=%HERE%p2p.output.txt
set ONEAPI=C:\Program Files (x86)\Intel\oneAPI
call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat" > nul 2>&1
set SETVARS_CALL=1
call "%ONEAPI%\compiler\2026.1\env\vars.bat" > nul 2>&1
if not exist "%B%" mkdir "%B%"
echo run %DATE% %TIME% > "%LOG%"
icx -fsycl "%HERE%p2p.cpp" /EHsc /O2 -c -o "%B%\p2p.obj" >> "%LOG%" 2>&1
icx -fsycl "%B%\p2p.obj" -o "%B%\p2p.exe" >> "%LOG%" 2>&1
echo build exit %ERRORLEVEL% >> "%LOG%"
set PATH=%HERE%..\.venv312\Library\bin;%PATH%
"%B%\p2p.exe" B60 >> "%LOG%" 2>&1
echo run exit %ERRORLEVEL% >> "%LOG%"
type "%LOG%"
exit /b 0
