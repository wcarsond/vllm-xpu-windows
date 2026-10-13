@echo off
rem section-copy.input.cmd - build section-copy.cpp and run both ranks (one per B60) over one shared section, imported
rem (zex) and not imported (none), at decode and prefill sizes. Writes probe\section-copy.output.txt.
rem Usage:  cmd /c probe\section-copy.input.cmd [rebuild]
rem Why (2026-10-13): see section-copy.cpp - can the copy engines use the shared section directly, so the exchange's big
rem path loses the helper's CPU memcpy (Linux's version of that design: 1.34 ms per 14.7 MB vs Windows' 4.2 ms today).
rem Each run gets a hard timeout (the Linux side's lesson: one hung probe left both cards needing a reset).
setlocal EnableDelayedExpansion
set HERE=%~dp0
set B=%HERE%section-copy.build
set LOG=%HERE%section-copy.output.txt
set ONEAPI=C:\Program Files (x86)\Intel\oneAPI
call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat" > nul 2>&1
set SETVARS_CALL=1
call "%ONEAPI%\compiler\2026.1\env\vars.bat" > nul 2>&1
if not exist "%B%" mkdir "%B%"
echo run %DATE% %TIME% > "%LOG%"
if "%1"=="rebuild" del /q "%B%\section-copy.exe" 2>nul
if not exist "%B%\section-copy.exe" (
    icx -fsycl "%HERE%section-copy.cpp" /EHsc /O2 -c -o "%B%\section-copy.obj" >> "%LOG%" 2>&1
    icx -fsycl "%B%\section-copy.obj" -o "%B%\section-copy.exe" >> "%LOG%" 2>&1
    echo build exit !ERRORLEVEL! >> "%LOG%"
)
if not exist "%B%\section-copy.exe" ( type "%LOG%" & exit /b 1 )
set PATH=%HERE%..\.venv312\Library\bin;%PATH%
set ZE_AFFINITY_MASK=
set ONEAPI_DEVICE_SELECTOR=
for %%M in (zex none) do (
    for %%S in (7168 1048576 3670016 14680064 33554432) do (
        echo -- %%M %%S >> "%LOG%"
        powershell -NoProfile -Command "$a = Start-Process '%B%\section-copy.exe' -ArgumentList '0 %%M %%S 40' -NoNewWindow -PassThru -RedirectStandardOutput '%B%\r0.txt'; $b = Start-Process '%B%\section-copy.exe' -ArgumentList '1 %%M %%S 40' -NoNewWindow -PassThru -RedirectStandardOutput '%B%\r1.txt'; if (-not $a.WaitForExit(60000)) { $a.Kill(); 'rank 0 TIMEOUT' }; if (-not $b.WaitForExit(60000)) { $b.Kill(); 'rank 1 TIMEOUT' }; 'exit codes ' + $a.ExitCode + ' ' + $b.ExitCode" >> "%LOG%" 2>&1
        type "%B%\r0.txt" >> "%LOG%" 2>nul
        type "%B%\r1.txt" >> "%LOG%" 2>nul
    )
)
type "%LOG%"
exit /b 0
