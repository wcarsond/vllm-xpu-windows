@echo off
rem cache-coherence.input.cmd - build probe\cache-coherence.cpp and run it on B60 #0.
rem Usage:  cmd /c probe\cache-coherence.input.cmd [rebuild]      (writes probe\cache-coherence.output.txt)
setlocal EnableDelayedExpansion
set HERE=%~dp0
set B=%HERE%cache-coherence.build
set LOG=%HERE%cache-coherence.output.txt
set ONEAPI=C:\Program Files (x86)\Intel\oneAPI
call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat" > nul 2>&1
set SETVARS_CALL=1
call "%ONEAPI%\compiler\2026.1\env\vars.bat" > nul 2>&1
if not exist "%B%" mkdir "%B%"
echo run %DATE% %TIME% > "%LOG%"
if "%1"=="rebuild" del /q "%B%\cache-coherence.exe" 2>nul
if not exist "%B%\cache-coherence.exe" (
    icx -fsycl "%HERE%cache-coherence.cpp" /EHsc /O2 -c -o "%B%\cache-coherence.obj" >> "%LOG%" 2>&1
    icx -fsycl "%B%\cache-coherence.obj" -o "%B%\cache-coherence.exe" >> "%LOG%" 2>&1
    echo build exit !ERRORLEVEL! >> "%LOG%"
)
set PATH=%HERE%..\.venv312\Library\bin;%PATH%
set ONEAPI_DEVICE_SELECTOR=
"%B%\cache-coherence.exe" >> "%LOG%" 2>&1
echo run exit !ERRORLEVEL! >> "%LOG%"
type "%LOG%"
exit /b 0
