@echo off
rem service-bench.input.cmd - run service-bench.input.py inside the MSVC environment the vLLM workers need (as
rem run-router.input.cmd does for the service). Usage:  cmd /c server\service-bench.input.cmd id1+id2 [new+old]
rem Why (2026-10-13): see the .py - old against new service setup, at the service's own settings.
setlocal
set HERE=%~dp0
call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat" > nul 2>&1
set CC=cl
set PATH=%HERE%..\.venv312\Library\bin;%PATH%
set PYTHONUNBUFFERED=1
set CONFIGS=%~2
if "%CONFIGS%"=="" set CONFIGS=new+old
"%HERE%..\.venv312\Scripts\python.exe" -u "%HERE%service-bench.input.py" "%~1" "%CONFIGS%"
