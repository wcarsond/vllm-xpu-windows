@echo off
rem one-process-allreduce.input.cmd - build and run one-process-allreduce.cpp: the decode all_reduce between both B60s in
rem one process, by store width, work-groups and buffer place. Writes probe\one-process-allreduce.output.txt.
rem Usage:  cmd /c probe\one-process-allreduce.input.cmd [rebuild]
rem Why (2026-10-13): decides whether a one-process executor can beat the two-process exchange's 28 us (see the .cpp).
rem Every run has a hard 120 s timeout.
setlocal EnableDelayedExpansion
set HERE=%~dp0
set B=%HERE%one-process-allreduce.build
set LOG=%HERE%one-process-allreduce.output.txt
set ONEAPI=C:\Program Files (x86)\Intel\oneAPI
call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat" > nul 2>&1
set SETVARS_CALL=1
call "%ONEAPI%\compiler\2026.1\env\vars.bat" > nul 2>&1
if not exist "%B%" mkdir "%B%"
echo run %DATE% %TIME% > "%LOG%"
if "%1"=="rebuild" del /q "%B%\ar.exe" 2>nul
if not exist "%B%\ar.exe" (
    icx -fsycl "%HERE%one-process-allreduce.cpp" /EHsc /O2 -c -o "%B%\ar.obj" >> "%LOG%" 2>&1
    icx -fsycl "%B%\ar.obj" -o "%B%\ar.exe" >> "%LOG%" 2>&1
    echo build exit !ERRORLEVEL! >> "%LOG%"
)
if not exist "%B%\ar.exe" ( type "%LOG%" & exit /b 1 )
set PATH=%HERE%..\.venv312\Library\bin;%PATH%
set ZE_AFFINITY_MASK=
set ONEAPI_DEVICE_SELECTOR=
for %%A in ("1792 2000" "2048 2000" "4096 1000" "64 2000") do (
    for /f "tokens=1,2" %%W in (%%A) do (
        echo -- words %%W rounds %%X >> "%LOG%"
        powershell -NoProfile -Command "$p = Start-Process '%B%\ar.exe' -ArgumentList '%%W %%X' -NoNewWindow -PassThru -RedirectStandardOutput '%B%\out.txt'; if (-not $p.WaitForExit(120000)) { $p.Kill(); 'TIMEOUT after 120 s' }; 'exit ' + $p.ExitCode" >> "%LOG%" 2>&1
        type "%B%\out.txt" >> "%LOG%" 2>nul
    )
)
type "%LOG%"
exit /b 0
