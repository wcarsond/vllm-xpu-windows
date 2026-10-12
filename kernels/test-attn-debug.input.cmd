@echo off
rem test-attn-debug.input.cmd - one failing chunk-prefill case (vLLM's shape) with the compute runtime's and Unified Runtime's debug messages on.
rem Usage:  cmd /c kernels\test-attn-debug.input.cmd      (writes kernels\test-attn-debug.output.log)
rem Why (2026-10-10): the launch fails with OUT_OF_RESOURCES and no allocation in sight. NEO (the Intel compute runtime) can say
rem what it refused (scratch, SLM, work-group size) when NEOReadDebugKeys/PrintDebugMessages are set; UR_LOG_LEVEL_ZERO adds the
rem adapter's view. Only the last lines before the error matter; the whole log is kept for reading.
set HERE=%~dp0
set REPO=%HERE%..\src\vllm-xpu-kernels
set PY=%HERE%..\.venv312\Scripts\python.exe
set LOG=%HERE%test-attn-debug.output.log
call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat" > nul 2>&1
set CC=cl
set NEOReadDebugKeys=1
set PrintDebugMessages=1
set UR_LOG_LEVEL_ZERO=level:err;flush:err;output:stderr
set ZE_AFFINITY_MASK=0
echo run %DATE% %TIME% > "%LOG%"
cd /d "%REPO%"
"%PY%" -m pytest tests/flash_attn/test_flash_attn_varlen_func.py -q --tb=short -p no:cacheprovider -k "test_varlen_with_paged_kv and window_size3 and -16-128- and bfloat16 and not float8 and -True-True-False- and 0-_None_" >> "%LOG%" 2>&1
echo exit %ERRORLEVEL% %TIME% >> "%LOG%"
findstr /n /i /c:"scratch" /c:"fail" /c:"error" /c:"OUT_OF" /c:"passed" /c:"NEO" /c:"zeCommand" /c:"ZE_RESULT" "%LOG%"
exit /b 0
