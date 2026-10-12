@echo off
rem test-attn-cases.input.cmd - single chunk-prefill attention cases, one pytest process each, so a device loss cannot hide the next result.
rem Usage:  cmd /c kernels\test-attn-cases.input.cmd      (writes kernels\test-attn-cases.output.log)
rem Why (2026-10-10): SLIDING_WINDOWS[3] = (-1,-1) = no window. The suite lost the device on a no-window case while the three
rem windowed variants passed, and vLLM's Qwen prefill uses no window: same kernel configuration behind both failures. Test id
rem fields: stride_pad-fp8-is_paged-is_casual-is_sink-q_dtype-fa_version-num_blocks-soft_cap-dtype-window-block-head-heads-seqs.
set HERE=%~dp0
set REPO=%HERE%..\src\vllm-xpu-kernels
set PY=%HERE%..\.venv312\Scripts\python.exe
set LOG=%HERE%test-attn-cases.output.log
set T=tests/flash_attn/test_flash_attn_varlen_func.py
call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat" > nul 2>&1
set CC=cl
echo run %DATE% %TIME% > "%LOG%"
cd /d "%REPO%"
call :one "A vLLM 7B shape: paged, causal, no window, head 128, block 16, bf16" "test_varlen_with_paged_kv and window_size3 and -16-128- and bfloat16 and not float8 and -True-True-False-"
call :one "B vLLM 0.5B shape: paged, causal, no window, head 64, block 16, bf16"  "test_varlen_with_paged_kv and window_size3 and -16-64- and bfloat16 and not float8 and -True-True-False-"
call :one "C the suite's killer: non-paged, non-causal, no window, head 64, f16" "test_varlen_with_paged_kv and window_size3 and -16-64- and float16 and not float8 and -False-False-False-"
call :one "D contrast: paged, causal, window (127,-1), head 128, block 16, bf16" "test_varlen_with_paged_kv and window_size1 and -16-128- and bfloat16 and not float8 and -True-True-False-"
call :one "E contrast: paged, causal, no window, head 128, block 64, bf16"       "test_varlen_with_paged_kv and window_size3 and -64-128- and bfloat16 and not float8 and -True-True-False-"
call :one "F decode, paged, no window, head 128, block 16, bf16"                 "test_decode_with_paged_kv and window_size3 and -16-128- and bfloat16 and not float8"
echo tests done %TIME% >> "%LOG%"
type "%LOG%"
exit /b 0

:one
echo == %~1 >> "%LOG%"
"%PY%" -m pytest %T% -q --tb=line -p no:cacheprovider -k "%~2" 2>&1 | findstr /r /c:"passed" /c:"failed" /c:"error" /c:"Error" /c:"no tests" /c:"DEVICE_LOST" /c:"OUT_OF" >> "%LOG%"
exit /b 0
