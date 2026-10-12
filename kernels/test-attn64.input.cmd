@echo off
rem test-attn64.input.cmd - paged-KV flash attention cases for head size 64 only, bf16/f16 KV (no fp8), verbose, no early stop.
rem Usage:  cmd /c kernels\test-attn64.input.cmd      (writes kernels\test-attn64.output.log)
rem Why (2026-10-10): Qwen2.5-0.5B (first-light bulb) has head size 64; test-ops saw one head-64 case lose the device
rem (fp8 KV, sliding window). This runs every head-64 paged case with the KV dtypes vLLM uses, listing pass/fail per
rem parametrisation, so the shape that fails on Windows is named. Run while no model is on the card.
set HERE=%~dp0
set REPO=%HERE%..\src\vllm-xpu-kernels
set PY=%HERE%..\.venv312\Scripts\python.exe
set LOG=%HERE%test-attn64.output.log
call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat" > nul 2>&1
set CC=cl
echo run %DATE% %TIME% > "%LOG%"
cd /d "%REPO%"
rem cache / rotary / layernorm first: reshape_and_cache_flash is the one kernel the real step runs that the profiling
rem pass does not, so it is the prime suspect for poisoning the queue before the GEMM that reports the error.
echo == cache, rotary, layernorm (harness import fixed) >> "%LOG%"
"%PY%" -m pytest tests/test_cache.py tests/test_rotary_embedding.py tests/test_layernorm.py -q -x --tb=short -p no:cacheprovider >> "%LOG%" 2>&1
echo exit %ERRORLEVEL% >> "%LOG%"
echo == paged flash attention, head 64, no fp8 >> "%LOG%"
"%PY%" -m pytest tests/flash_attn/test_flash_attn_varlen_func.py -v --tb=line -p no:cacheprovider -k "(test_varlen_with_paged_kv or test_decode_with_paged_kv) and -64- and not float8" >> "%LOG%" 2>&1
echo exit %ERRORLEVEL% >> "%LOG%"
echo tests done %TIME% >> "%LOG%"
