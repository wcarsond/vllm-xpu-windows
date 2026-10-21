@echo off
rem run-router.input.cmd - start the vLLM router (port 11434) inside the MSVC environment the vLLM workers need.
rem Usage:  cmd /c server\run-router.input.cmd [port]      - this is what the vllm-b60 service runs; console -> router.console.log
rem Models and their settings: server\models.input.ini.   Stop: stop-server.input.cmd (kills vllm.exe) or stop the service.
rem Why (2026-10-12): one named entry point for the service; vcvars64 so Triton can compile with cl.exe in the workers.
set HERE=%~dp0
call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat" > nul 2>&1
set CC=cl
set PATH=%HERE%..\.venv312\Library\bin;%PATH%
set PYTHONUNBUFFERED=1
set PORT=%1
if "%PORT%"=="" set PORT=11434
"%HERE%..\.venv312\Scripts\python.exe" -u "%HERE%vllm-router.input.py" %PORT% >> "%HERE%router.stdout.log" 2>&1
