@echo off
rem gpu-exchange.input.cmd - build probe\gpu-exchange.cpp and run it as two processes, one per B60, in lockstep.
rem Usage:  cmd /c probe\gpu-exchange.input.cmd [iters] [layer_k] [words] [rebuild]     (writes probe\gpu-exchange.output.txt)
rem   defaults: 5000 iterations, layer_k 60000 (~0.3 ms fake layer), 1792 floats = the 7B's 7 KB decode tensor.
rem Why (2026-10-11): see gpu-exchange.cpp. Run only when no vLLM process is on the cards (probe\gpu-holders.input.ps1):
rem a spinning exchange kernel waiting for a peer that never comes gives up after ~0.5 s, but both cards are busy meanwhile.
setlocal EnableDelayedExpansion
set HERE=%~dp0
set B=%HERE%gpu-exchange.build
set LOG=%HERE%gpu-exchange.output.txt
set ITERS=%1
if "%ITERS%"=="" set ITERS=5000
set LAYERK=%2
if "%LAYERK%"=="" set LAYERK=12000
set WORDS=%3
if "%WORDS%"=="" set WORDS=1792
set ONEAPI=C:\Program Files (x86)\Intel\oneAPI
call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat" > nul 2>&1
set SETVARS_CALL=1
call "%ONEAPI%\compiler\2026.1\env\vars.bat" > nul 2>&1
if not exist "%B%" mkdir "%B%"
echo run %DATE% %TIME% iters %ITERS% layer_k %LAYERK% words %WORDS% > "%LOG%"
if "%4"=="rebuild" del /q "%B%\gpu-exchange.exe" 2>nul
if not exist "%B%\gpu-exchange.exe" (
    icx -fsycl "%HERE%gpu-exchange.cpp" /EHsc /O2 -c -o "%B%\gpu-exchange.obj" >> "%LOG%" 2>&1
    icx -fsycl "%B%\gpu-exchange.obj" -o "%B%\gpu-exchange.exe" >> "%LOG%" 2>&1
    echo build exit !ERRORLEVEL! >> "%LOG%"
)
if not exist "%B%\gpu-exchange.exe" ( type "%LOG%" & exit /b 1 )
set PATH=%HERE%..\.venv312\Library\bin;%PATH%
set ONEAPI_DEVICE_SELECTOR=
del /q "%HERE%gpu-exchange.rank0.log" "%HERE%gpu-exchange.rank1.log" 2>nul
start "" /b cmd /c ""%B%\gpu-exchange.exe" 0 %ITERS% %LAYERK% %WORDS% > "%HERE%gpu-exchange.rank0.log" 2>&1"
"%B%\gpu-exchange.exe" 1 %ITERS% %LAYERK% %WORDS% > "%HERE%gpu-exchange.rank1.log" 2>&1
echo rank 1 exit %ERRORLEVEL% >> "%HERE%gpu-exchange.rank1.log"
rem wait (up to ~60 s) for rank 0 to finish writing
set /a WAITED=0
:wait0
findstr /c:"rank 0 exit" "%HERE%gpu-exchange.rank0.log" > nul 2>&1
if not errorlevel 1 goto done0
if %WAITED% geq 60 goto done0
ping -n 2 127.0.0.1 > nul
set /a WAITED+=1
goto wait0
:done0
echo -- rank 0 >> "%LOG%"
type "%HERE%gpu-exchange.rank0.log" >> "%LOG%"
echo -- rank 1 >> "%LOG%"
type "%HERE%gpu-exchange.rank1.log" >> "%LOG%"
type "%LOG%"
exit /b 0
