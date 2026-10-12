@echo off
rem run-smoke.input.cmd - run the first-light test with its console written to a file as it happens.
rem Usage:  cmd /c server\run-smoke.input.cmd      (console -> server\smoke-offline.console.log; result -> smoke-offline.output.txt)
rem Why (2026-10-10): the engine core is a spawned child; when it dies at import the parent waits on the handshake and a
rem pipeline that only flushes at exit shows nothing. -u unbuffers Python; the redirect lives in this file because a quoted
rem command line handed to "cmd /c" from PowerShell lost its quotes and never started (20:58 attempt, zero lines).
set HERE=%~dp0
rem Triton-XPU compiles its kernel launcher stubs with a host C compiler at run time ("Failed to find C compiler", 21:02),
rem so the run needs MSVC on PATH: vcvars64 provides cl.exe plus INCLUDE/LIB, and CC names it for Triton.
call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat" > nul 2>&1
set CC=cl
"%HERE%..\.venv312\Scripts\python.exe" -u "%HERE%smoke-offline.input.py" > "%HERE%smoke-offline.console.log" 2>&1
echo smoke exit %ERRORLEVEL% >> "%HERE%smoke-offline.console.log"
