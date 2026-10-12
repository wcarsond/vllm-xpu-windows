@echo off
rem poll-rate.input.cmd - build probe\poll-rate.cpp and run it on B60 #0.
rem Usage:  cmd /c probe\poll-rate.input.cmd      (writes probe\poll-rate.output.txt)
setlocal EnableDelayedExpansion
set HERE=%~dp0
set B=%HERE%poll-rate.build
set LOG=%HERE%poll-rate.output.txt
set ONEAPI=C:\Program Files (x86)\Intel\oneAPI
call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat" > nul 2>&1
set SETVARS_CALL=1
call "%ONEAPI%\compiler\2026.1\env\vars.bat" > nul 2>&1
if not exist "%B%" mkdir "%B%"
echo run %DATE% %TIME% > "%LOG%"
if not exist "%B%\poll-rate.exe" (
    icx -fsycl "%HERE%poll-rate.cpp" /EHsc /O2 -c -o "%B%\poll-rate.obj" >> "%LOG%" 2>&1
    icx -fsycl "%B%\poll-rate.obj" -o "%B%\poll-rate.exe" >> "%LOG%" 2>&1
    echo build exit !ERRORLEVEL! >> "%LOG%"
)
set PATH=%HERE%..\.venv312\Library\bin;%PATH%
set ONEAPI_DEVICE_SELECTOR=
"%B%\poll-rate.exe" >> "%LOG%" 2>&1
echo run exit !ERRORLEVEL! >> "%LOG%"
type "%LOG%"
exit /b 0
