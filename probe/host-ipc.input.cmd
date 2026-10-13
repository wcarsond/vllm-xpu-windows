@echo off
rem host-ipc.input.cmd - build host-ipc.cpp and run rank 0 (exporter, card 0) and rank 1 (importer, card 1) together, each
rem with a 60 s hard timeout. Writes probe\host-ipc.output.txt.
rem Usage:  cmd /c probe\host-ipc.input.cmd [rebuild]
rem Why (2026-10-13): if host USM exports as an IPC handle on Windows and the other process's card can write through it,
rem the decode exchange can drop its CPU helpers with two processes, as oneCCL does on Linux - no single-process rework.
setlocal EnableDelayedExpansion
set HERE=%~dp0
set B=%HERE%host-ipc.build
set LOG=%HERE%host-ipc.output.txt
set ONEAPI=C:\Program Files (x86)\Intel\oneAPI
call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat" > nul 2>&1
set SETVARS_CALL=1
call "%ONEAPI%\compiler\2026.1\env\vars.bat" > nul 2>&1
if not exist "%B%" mkdir "%B%"
echo run %DATE% %TIME% > "%LOG%"
if "%1"=="rebuild" del /q "%B%\host-ipc.exe" 2>nul
if not exist "%B%\host-ipc.exe" (
    icx -fsycl "%HERE%host-ipc.cpp" /EHsc /O2 /std:c++20 -w -c -o "%B%\host-ipc.obj" >> "%LOG%" 2>&1
    icx -fsycl "%B%\host-ipc.obj" -o "%B%\host-ipc.exe" >> "%LOG%" 2>&1
    echo build exit !ERRORLEVEL! >> "%LOG%"
)
if not exist "%B%\host-ipc.exe" ( type "%LOG%" & exit /b 1 )
set PATH=%HERE%..\.venv312\Library\bin;%PATH%
set ZE_AFFINITY_MASK=
set ONEAPI_DEVICE_SELECTOR=
powershell -NoProfile -Command "$a = Start-Process '%B%\host-ipc.exe' -ArgumentList '0' -NoNewWindow -PassThru -RedirectStandardOutput '%B%\r0.txt'; Start-Sleep -Milliseconds 500; $b = Start-Process '%B%\host-ipc.exe' -ArgumentList '1' -NoNewWindow -PassThru -RedirectStandardOutput '%B%\r1.txt'; if (-not $a.WaitForExit(60000)) { $a.Kill(); 'rank 0 TIMEOUT' }; if (-not $b.WaitForExit(60000)) { $b.Kill(); 'rank 1 TIMEOUT' }; 'exit codes ' + $a.ExitCode + ' ' + $b.ExitCode" >> "%LOG%" 2>&1
type "%B%\r0.txt" >> "%LOG%" 2>nul
type "%B%\r1.txt" >> "%LOG%" 2>nul
type "%LOG%"
exit /b 0
