@echo off
rem one-process-pingpong.input.cmd - build and run one-process-pingpong.cpp: one process, both B60s, one host buffer, a
rem tagged-word ping-pong between two persistent kernels. Writes probe\one-process-pingpong.output.txt.
rem Usage:  cmd /c probe\one-process-pingpong.input.cmd [rebuild]
rem Why (2026-10-13): decides whether a helper-free decode exchange is possible on Windows (see the .cpp). Every run has a
rem hard 60 s timeout, and the round count is kept small first: a kernel that never sees its peer spins until the poll
rem limit, and a hung card can need a reset.
setlocal EnableDelayedExpansion
set HERE=%~dp0
set B=%HERE%one-process-pingpong.build
set LOG=%HERE%one-process-pingpong.output.txt
set ONEAPI=C:\Program Files (x86)\Intel\oneAPI
call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat" > nul 2>&1
set SETVARS_CALL=1
call "%ONEAPI%\compiler\2026.1\env\vars.bat" > nul 2>&1
if not exist "%B%" mkdir "%B%"
echo run %DATE% %TIME% > "%LOG%"
if "%1"=="rebuild" del /q "%B%\pp.exe" 2>nul
if not exist "%B%\pp.exe" (
    icx -fsycl "%HERE%one-process-pingpong.cpp" /EHsc /O2 -c -o "%B%\pp.obj" >> "%LOG%" 2>&1
    icx -fsycl "%B%\pp.obj" -o "%B%\pp.exe" >> "%LOG%" 2>&1
    echo build exit !ERRORLEVEL! >> "%LOG%"
)
if not exist "%B%\pp.exe" ( type "%LOG%" & exit /b 1 )
set PATH=%HERE%..\.venv312\Library\bin;%PATH%
set ZE_AFFINITY_MASK=
set ONEAPI_DEVICE_SELECTOR=
for %%A in ("896 20" "896 2000" "32 2000" "4096 500") do (
    for /f "tokens=1,2" %%W in (%%A) do (
        echo -- words %%W rounds %%X >> "%LOG%"
        powershell -NoProfile -Command "$p = Start-Process '%B%\pp.exe' -ArgumentList '%%W %%X' -NoNewWindow -PassThru -RedirectStandardOutput '%B%\out.txt'; if (-not $p.WaitForExit(60000)) { $p.Kill(); 'TIMEOUT after 60 s' }; 'exit ' + $p.ExitCode" >> "%LOG%" 2>&1
        type "%B%\out.txt" >> "%LOG%" 2>nul
    )
)
type "%LOG%"
exit /b 0
