@echo off
rem test-onednn.input.cmd - the kernel tests for the oneDNN-backed quantized GEMMs (int4 w4a16/w4a8, fp8, fp4) on card 0.
rem Usage:  cmd /c kernels\test-onednn.input.cmd      (one-line verdicts in kernels\test-onednn.output.log, full pytest output
rem         in kernels\test-onednn.pytest.log); run after build-onednn.input.cmd has installed the wheel.
rem Why (2026-10-11): these ops were stubs until the oneDNN build; a "pass" here is what lets a Q4 model run on the cards.
rem 13:20: the "int4 op present" line imports _xpu_C itself now; importing only the package left the ops unregistered and it said False.
setlocal
set HERE=%~dp0
set LOG=%HERE%test-onednn.output.log
set PY=%HERE%..\.venv312\Scripts\python.exe
call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat" > nul 2>&1
set CC=cl
set ZE_AFFINITY_MASK=0
echo run %DATE% %TIME% > "%LOG%"
"%PY%" -c "import vllm_xpu_kernels, vllm_xpu_kernels._xpu_C, torch; print('kernels', vllm_xpu_kernels.__file__); print('int4 op present:', hasattr(torch.ops._xpu_C, 'int4_gemm_w4a16'))" >> "%LOG%" 2>&1
for %%T in (test_int4_gemm_onednn.py test_fp8_gemm_onednn.py test_fp4_gemm_onednn.py) do (
    echo == %%T >> "%LOG%"
    "%PY%" -m pytest "%HERE%..\src\vllm-xpu-kernels\tests\%%T" -q --tb=line -p no:cacheprovider > "%HERE%test-onednn.pytest.log" 2>&1
    findstr /c:"passed" /c:"failed" /c:"error" /c:"skipped" "%HERE%test-onednn.pytest.log" >> "%LOG%"
)
type "%LOG%"
endlocal
