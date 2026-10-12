@echo off
rem after-build.input.cmd - everything that follows a finished kernel rebuild, in order, one-line verdicts in after-build.output.log.
rem Usage:  cmd /c kernels\after-build.input.cmd [start step 1-15]      (default 1; e.g. 7 resumes at the cutlass one-card race)
rem Why (2026-10-10 23:55): the rebuild with the cutlass fix ends while Carson sleeps; this is the night plan as one file.
rem   02:52 rewrite: called scripts set their own LOG/HERE and cmd shares variables with the caller, so the verdicts went into
rem   wheel.output.log and then test-attn-cases.output.log, and the gate read its own header sentence as a device loss.
rem   Now: a private log variable, setlocal/endlocal around every call, a gate that only reads the two test outputs, labels.
set AB_HERE=%~dp0
set AB_S=%AB_HERE%..\server
set AB_PY=%AB_HERE%..\.venv312\Scripts\python.exe
set AB_LOG=%AB_HERE%after-build.output.log
rem Every step that can run CMake must see this, or FetchContent checks the pinned cutlass tag back out over the fix (02:21).
set VLLM_CUTLASS_SRC_DIR=%AB_HERE%..\src\vllm-xpu-kernels\.deps\cutlass-sycl-src
set START=%1
if "%START%"=="" set START=1
if "%START%"=="1" (echo run %DATE% %TIME% > "%AB_LOG%") else (echo resume %DATE% %TIME% from step %START% >> "%AB_LOG%")
call :gate
goto step%START%

:step1
echo == 1. wheel into .venv312 >> "%AB_LOG%"
setlocal
call "%AB_HERE%wheel.input.cmd" > "%AB_HERE%after-build.wheel.log" 2>&1
endlocal
findstr /c:"install into .venv312 exit" /c:"import ok" "%AB_HERE%wheel.output.log" >> "%AB_LOG%"

:step2
echo == 2. attention cases that used to lose the device (one process each) >> "%AB_LOG%"
setlocal
call "%AB_HERE%test-attn-cases.input.cmd" > nul 2>&1
endlocal
type "%AB_HERE%test-attn-cases.output.log" >> "%AB_LOG%"

:step3
echo == 3. direct attention reproducer >> "%AB_LOG%"
"%AB_PY%" "%AB_HERE%..\probe\attn-direct.input.py" > nul 2>&1
type "%AB_HERE%..\probe\attn-direct.output.txt" >> "%AB_LOG%"

:step4
echo == 4. mhc tests (the cute split-K GEMM that died) >> "%AB_LOG%"
"%AB_PY%" -m pytest "%AB_HERE%..\src\vllm-xpu-kernels\tests\mhc\test_mhc.py" -q --tb=line -p no:cacheprovider 2>&1 | findstr /c:"passed" /c:"failed" /c:"error" >> "%AB_LOG%"

:step5
call :gate
echo == 5. gate: cutlass attention %CUTLASS_VERDICT% >> "%AB_LOG%"

:step6
echo == 6. one-card race, Triton (control) >> "%AB_LOG%"
setlocal
call "%AB_S%\run-race-7b-vllm.input.cmd" TRITON_ATTN > nul 2>&1
endlocal
findstr /c:"median" "%AB_S%\race-7b-vllm.TRITON_ATTN.output.txt" >> "%AB_LOG%"

:step7
if not "%CUTLASS_OK%"=="1" goto step8
echo == 7. one-card race, cutlass flash attention >> "%AB_LOG%"
setlocal
call "%AB_S%\run-race-7b-vllm.input.cmd" FLASH_ATTN > nul 2>&1
endlocal
findstr /c:"median" "%AB_S%\race-7b-vllm.FLASH_ATTN.output.txt" >> "%AB_LOG%"

:step8
echo == 8. depth leg, Triton >> "%AB_LOG%"
setlocal
call "%AB_S%\run-race-7b-depth.input.cmd" TRITON_ATTN > nul 2>&1
endlocal
findstr /c:"depth " /c:"Error" "%AB_S%\race-7b-depth.TRITON_ATTN.output.txt" "%AB_S%\race-7b-depth.TRITON_ATTN.console.log" | findstr /v /c:"depth exit" >> "%AB_LOG%"

:step9
if not "%CUTLASS_OK%"=="1" goto step10
echo == 9. depth leg, cutlass flash attention >> "%AB_LOG%"
setlocal
call "%AB_S%\run-race-7b-depth.input.cmd" FLASH_ATTN > nul 2>&1
endlocal
findstr /c:"depth " /c:"Error" "%AB_S%\race-7b-depth.FLASH_ATTN.output.txt" "%AB_S%\race-7b-depth.FLASH_ATTN.console.log" | findstr /v /c:"depth exit" >> "%AB_LOG%"

:step10
echo == 10. two cards, tensor split, shared-memory exchange >> "%AB_LOG%"
setlocal
call "%AB_S%\run-smoke-tp2.input.cmd" > nul 2>&1
endlocal
findstr /c:"engine up" /c:"median" /c:"Error" /c:"tp2 exit" "%AB_S%\smoke-tp2.console.log" >> "%AB_LOG%"

:step11
rem Router legs after the vLLM runs (its model must not sit on the cards while vLLM measures) and on a quiet machine.
echo == 11. router control, short race (16.5 tok/s at 22:46 on a quiet machine) >> "%AB_LOG%"
powershell -NoProfile -File "%AB_S%\race-7b-router.input.ps1" > nul 2>&1
findstr /c:"median" "%AB_S%\race-7b-router.output.txt" >> "%AB_LOG%"

:step12
echo == 12. router depth leg (cache_prompt off) >> "%AB_LOG%"
powershell -NoProfile -File "%AB_S%\race-7b-router-depth.input.ps1" > nul 2>&1
findstr /c:"depth " /c:"n_ctx" "%AB_S%\race-7b-router-depth.output.txt" >> "%AB_LOG%"

:step13
echo == 13. MoE build (split grouped GEMM, MOE_KERNELS_ENABLED=ON) %TIME% >> "%AB_LOG%"
setlocal
set MOE_KERNELS_ENABLED=ON
call "%AB_HERE%build-moe.input.cmd" > "%AB_HERE%after-build.moe.log" 2>&1
endlocal
findstr /c:"pip exit" "%AB_HERE%build.output.log" >> "%AB_LOG%"

:step14
echo == 14. wheel again (now with _moe_C) >> "%AB_LOG%"
setlocal
set MOE_KERNELS_ENABLED=ON
call "%AB_HERE%wheel.input.cmd" > "%AB_HERE%after-build.wheel2.log" 2>&1
endlocal
findstr /c:"install into .venv312 exit" /c:"import ok" /c:"_moe_C" "%AB_HERE%wheel.output.log" >> "%AB_LOG%"

:step15
echo == 15. MoE tests >> "%AB_LOG%"
"%AB_PY%" -m pytest "%AB_HERE%..\src\vllm-xpu-kernels\tests\fused_moe\test_grouped_gemm.py" -q --tb=line -p no:cacheprovider 2>&1 | findstr /c:"passed" /c:"failed" /c:"error" >> "%AB_LOG%"
"%AB_PY%" -m pytest "%AB_HERE%..\src\vllm-xpu-kernels\tests\fused_moe\test_fused_moe.py" -q --tb=line -p no:cacheprovider 2>&1 | findstr /c:"passed" /c:"failed" /c:"error" >> "%AB_LOG%"

echo done %TIME% >> "%AB_LOG%"
type "%AB_LOG%"
exit /b 0

:gate
rem CUTLASS_OK=1 unless the two attention test outputs mention a device loss. Those files hold only their own results.
findstr /c:"DEVICE_LOST" /c:"OUT_OF_RESOURCES" "%AB_HERE%test-attn-cases.output.log" "%AB_HERE%..\probe\attn-direct.output.txt" > nul 2>&1
if errorlevel 1 (set CUTLASS_OK=1& set "CUTLASS_VERDICT=no device loss in the tests, cutlass races run") else (set CUTLASS_OK=0& set "CUTLASS_VERDICT=STILL FAULTING, cutlass races skipped")
exit /b 0
