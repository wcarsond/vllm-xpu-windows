@echo off
rem peer-access.input.cmd - build and run probe\peer-access.cpp: is "no peer access" a driver policy or a hardware fact?
rem Usage:  cmd /c probe\peer-access.input.cmd [rebuild]      (writes probe\peer-access.output.txt)
rem Why (2026-10-11 14:25): the 12:50 source read found Intel's Windows peer-access query is a hard-coded "false". This
rem asks whether the plumbing underneath it works anyway: entry points, IPC export/import across the two cards, and
rem whether a kernel on card 1 can really read card 0's memory. Step 4 can take the process down; output is unbuffered
rem and the log keeps whatever was printed before the fall, so a crash is itself a result.
setlocal EnableDelayedExpansion
set HERE=%~dp0
set B=%HERE%peer-access.build
set LOG=%HERE%peer-access.output.txt
set ONEAPI=C:\Program Files (x86)\Intel\oneAPI
call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat" > nul 2>&1
set SETVARS_CALL=1
call "%ONEAPI%\compiler\2026.1\env\vars.bat" > nul 2>&1
if not exist "%B%" mkdir "%B%"
echo run %DATE% %TIME% > "%LOG%"
if "%1"=="rebuild" del /q "%B%\peer-access.exe" 2>nul
if not exist "%B%\peer-access.exe" (
    icx -fsycl "%HERE%peer-access.cpp" /EHsc /O2 /std:c++20 -w -c -o "%B%\peer-access.obj" >> "%LOG%" 2>&1
    icx -fsycl "%B%\peer-access.obj" -o "%B%\peer-access.exe" >> "%LOG%" 2>&1
    echo build exit !ERRORLEVEL! >> "%LOG%"
)
if not exist "%B%\peer-access.exe" (echo BUILD FAILED, see %LOG% & type "%LOG%" & exit /b 1)
rem 14:20: WITHOUT this the process sees only the OpenCL adapter (4 devices, backend 1) and zero Level Zero devices - the
rem L0 UR adapter DLLs live in the venv's Intel runtime, not in the oneAPI compiler environment. compat-devices.input.cmd
rem has had this line all along; copying it is what made this probe see the cards at all.
set PATH=%HERE%..\.venv312\Library\bin;%PATH%
"%B%\peer-access.exe" >> "%LOG%" 2>&1
echo run exit !ERRORLEVEL! >> "%LOG%"
type "%LOG%"
endlocal
