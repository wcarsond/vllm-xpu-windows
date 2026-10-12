@echo off
rem test-ops.input.cmd - run the kernel repo's own pytest cases for the ops a decode step uses, against the installed wheel.
rem Usage:  cmd /c kernels\test-ops.input.cmd      (writes kernels\test-ops.output.log)
rem Why (2026-10-10, first light attempt 5): generate() died with UR_RESULT_ERROR_OUT_OF_RESOURCES surfacing at a plain torch
rem linear, which on Level Zero means an earlier kernel failed. These tests call each suspect on its own with known shapes:
rem KV-cache write (test_cache), rotary (test_rotary_embedding), the sampler (test_topk_topp_sampler), paged-KV flash
rem attention (flash_attn, paged cases, -x stops at the first failure). Runs in .venv312 so it tests the wheel vLLM uses.
set HERE=%~dp0
set REPO=%HERE%..\src\vllm-xpu-kernels
set PY=%HERE%..\.venv312\Scripts\python.exe
set LOG=%HERE%test-ops.output.log
call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat" > nul 2>&1
set CC=cl
echo run %DATE% %TIME% > "%LOG%"
"%PY%" -m pip install --quiet pytest >> "%LOG%" 2>&1
cd /d "%REPO%"
echo == test_cache >> "%LOG%"
"%PY%" -m pytest tests/test_cache.py -q -x --tb=short -p no:cacheprovider >> "%LOG%" 2>&1
echo exit %ERRORLEVEL% >> "%LOG%"
echo == test_rotary_embedding >> "%LOG%"
"%PY%" -m pytest tests/test_rotary_embedding.py -q -x --tb=short -p no:cacheprovider >> "%LOG%" 2>&1
echo exit %ERRORLEVEL% >> "%LOG%"
echo == test_topk_topp_sampler >> "%LOG%"
"%PY%" -m pytest tests/test_topk_topp_sampler.py -q -x --tb=short -p no:cacheprovider >> "%LOG%" 2>&1
echo exit %ERRORLEVEL% >> "%LOG%"
echo == test_layernorm >> "%LOG%"
"%PY%" -m pytest tests/test_layernorm.py -q -x --tb=short -p no:cacheprovider >> "%LOG%" 2>&1
echo exit %ERRORLEVEL% >> "%LOG%"
echo == flash_attn paged kv >> "%LOG%"
"%PY%" -m pytest tests/flash_attn/test_flash_attn_varlen_func.py -q -x --tb=short -p no:cacheprovider -k "test_varlen_with_paged_kv or test_decode_with_paged_kv" >> "%LOG%" 2>&1
echo exit %ERRORLEVEL% >> "%LOG%"
echo tests done %TIME% >> "%LOG%"
