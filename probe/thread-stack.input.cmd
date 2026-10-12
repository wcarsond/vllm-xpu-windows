@echo off
rem thread-stack.input.cmd - build probe\thread-stack.cpp with MSVC (once) and print the native stacks of the given processes.
rem Usage:  cmd /c probe\thread-stack.input.cmd <pid> [<pid> ...]      (also writes probe\thread-stack.output.txt)
rem Why (2026-10-11 03:50): see thread-stack.cpp - the leftover tensor-split workers spin after their shutdown and nothing
rem else on the box can show what they are running. Plain Win32 + DbgHelp, no Intel toolchain involved.
setlocal
set TS_HERE=%~dp0
set TS_OUT=%TS_HERE%thread-stack.output.txt
if not exist "%TS_HERE%thread-stack.exe" (
    call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat" > nul 2>&1
    pushd "%TS_HERE%"
    cl /nologo /EHsc /O2 /std:c++17 thread-stack.cpp dbghelp.lib psapi.lib /Fe:thread-stack.exe > "%TS_HERE%thread-stack.build.log" 2>&1
    popd
    if not exist "%TS_HERE%thread-stack.exe" ( type "%TS_HERE%thread-stack.build.log" & exit /b 1 )
)
echo run %DATE% %TIME% pids %* > "%TS_OUT%"
"%TS_HERE%thread-stack.exe" %* >> "%TS_OUT%" 2>&1
type "%TS_OUT%"
endlocal
