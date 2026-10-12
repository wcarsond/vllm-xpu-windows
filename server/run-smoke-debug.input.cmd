@echo off
rem run-smoke-debug.input.cmd - the first-light run with Level Zero's validation layer and error log on, to name the failing call.
rem Usage:  cmd /c server\run-smoke-debug.input.cmd      (console -> server\smoke-offline.debug.log)
rem Why (2026-10-10, attempt 5): generate() died with "level_zero backend failed with error: 40
rem (UR_RESULT_ERROR_OUT_OF_RESOURCES)" at a plain torch linear; Level Zero reports an earlier kernel's failure at the next
rem queue operation, so the Python frame does not name the culprit. ZE_ENABLE_VALIDATION_LAYER + PARAMETER_VALIDATION
rem (ze_validation_layer.dll, installed with the driver) checks every ze* call's arguments and says which one is wrong;
rem UR_LOG_LEVEL_ZERO=level:error prints the Unified Runtime adapter's error with the ze* entry point. Slower; diagnostics only.
set HERE=%~dp0
call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat" > nul 2>&1
set CC=cl
set ZE_ENABLE_VALIDATION_LAYER=1
set ZE_ENABLE_PARAMETER_VALIDATION=1
set UR_LOG_LEVEL_ZERO=level:error;flush:error;output:stderr
"%HERE%..\.venv312\Scripts\python.exe" -u "%HERE%smoke-offline.input.py" > "%HERE%smoke-offline.debug.log" 2>&1
echo smoke exit %ERRORLEVEL% >> "%HERE%smoke-offline.debug.log"
