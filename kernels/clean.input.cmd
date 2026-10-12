@echo off
rem clean.input.cmd - throw away the kernel repo's CMake build directory so the next build.input.cmd configures from nothing.
rem Usage:  cmd /c kernels\clean.input.cmd
rem Why (2026-10-10): attempt 6 left a CMakeCache.txt written by a configure that had lost its arguments (Debug, oneDNN ON,
rem no Python path). CMake reuses the cache, so a stale one poisons every later attempt. Run this after any change to
rem cmake\toolchain.cmake or to the -D arguments in setup.py; do not run it between ordinary retries, it costs the
rem incremental build. The fetched dependencies in .deps\ (cutlass-sycl) are kept.
set REPO=%~dp0..\src\vllm-xpu-kernels
if exist "%REPO%\build\temp" (
    rmdir /s /q "%REPO%\build\temp"
    echo removed %REPO%\build\temp
) else (
    echo nothing to remove at %REPO%\build\temp
)
