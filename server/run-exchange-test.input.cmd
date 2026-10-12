@echo off
rem run-exchange-test.input.cmd - run server\exchange-test.input.py inside the MSVC/torch environment, console to a file.
rem Usage:  cmd /c server\run-exchange-test.input.cmd [rounds]      (console -> exchange-test.console.log; result -> exchange-test.output.txt)
set HERE=%~dp0
call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat" > nul 2>&1
set CC=cl
"%HERE%..\.venv312\Scripts\python.exe" -u "%HERE%exchange-test.input.py" %1 > "%HERE%exchange-test.console.log" 2>&1
echo test exit %ERRORLEVEL% >> "%HERE%exchange-test.console.log"
findstr /c:"[rank" /c:"test exit" /c:"Error" /c:"error:" "%HERE%exchange-test.console.log"
