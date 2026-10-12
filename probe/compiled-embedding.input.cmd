@echo off
rem compiled-embedding.input.cmd - run probe\compiled-embedding.input.py inside the MSVC environment Inductor needs.
rem Usage:  cmd /c probe\compiled-embedding.input.cmd [rank 0|1]      (console -> compiled-embedding.console.log)
rem         set TORCH_LOGS=output_code first to get the generated Triton code in the console log.
set HERE=%~dp0
call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat" > nul 2>&1
set CC=cl
if not "%1"=="" set EMB_RANK=%1
set ZE_AFFINITY_MASK=0
"%HERE%..\.venv312\Scripts\python.exe" -u "%HERE%compiled-embedding.input.py" > "%HERE%compiled-embedding.console.log" 2>&1
echo probe exit %ERRORLEVEL% >> "%HERE%compiled-embedding.console.log"
findstr /c:"rank " /c:"eager" /c:"compiled" /c:"Error" /c:"probe exit" "%HERE%compiled-embedding.console.log"
