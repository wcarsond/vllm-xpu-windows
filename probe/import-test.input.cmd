@echo off
rem import-test.input.cmd - build probe\import-test.cpp and run every memory kind x import method, one process each.
rem Usage:  cmd /c probe\import-test.input.cmd [rebuild]      (writes probe\import-test.output.txt)
rem Why (2026-10-11 08:20): see import-test.cpp - the two-card exchange probe died inside the import of its shared mapping.
setlocal EnableDelayedExpansion
set HERE=%~dp0
set B=%HERE%import-test.build
set LOG=%HERE%import-test.output.txt
set ONEAPI=C:\Program Files (x86)\Intel\oneAPI
call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat" > nul 2>&1
set SETVARS_CALL=1
call "%ONEAPI%\compiler\2026.1\env\vars.bat" > nul 2>&1
if not exist "%B%" mkdir "%B%"
echo run %DATE% %TIME% > "%LOG%"
if "%1"=="rebuild" del /q "%B%\import-test.exe" 2>nul
if not exist "%B%\import-test.exe" (
    icx -fsycl "%HERE%import-test.cpp" /EHsc /O2 -c -o "%B%\import-test.obj" >> "%LOG%" 2>&1
    icx -fsycl "%B%\import-test.obj" -o "%B%\import-test.exe" >> "%LOG%" 2>&1
    echo build exit !ERRORLEVEL! >> "%LOG%"
)
if not exist "%B%\import-test.exe" ( type "%LOG%" & exit /b 1 )
set PATH=%HERE%..\.venv312\Library\bin;%PATH%
set ONEAPI_DEVICE_SELECTOR=
for %%K in (valloc malloc section) do (
    for %%M in (none sycl zex) do (
        echo -- %%K / %%M >> "%LOG%"
        "%B%\import-test.exe" %%K %%M >> "%LOG%" 2>&1
        echo exit !ERRORLEVEL! >> "%LOG%"
    )
)
type "%LOG%"
exit /b 0
