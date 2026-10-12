@echo off
rem moe-syntax.input.cmd - front-end only (-fsyntax-only) check of the split MoE grouped GEMM: the light dispatch, the selector, and one heavy variant.
rem Usage:  cmd /c kernels\moe-syntax.input.cmd      (writes kernels\moe-syntax.output.log; no objects, no GPU, a few GB RAM for the heavy one)
rem Why (2026-10-11): the real MoE compile is 36 translation units and must wait for the attention rebuild (RAM); this catches
rem typos and wrong casts in the refactor now. Flags and include paths are the kernel build's own (build\temp\build.ninja for a
rem cutlass TU), plus the grouped_gemm directories the target adds via add_xe2_kernel_library.
setlocal
set HERE=%~dp0
set K=%HERE%..\src\vllm-xpu-kernels
set CS=%K%\.deps\cutlass-sycl-src
set LOG=%HERE%moe-syntax.output.log
set ONEAPI=C:\Program Files (x86)\Intel\oneAPI
call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat" > nul 2>&1
set SETVARS_CALL=1
call "%ONEAPI%\compiler\2026.1\env\vars.bat" > nul 2>&1
set FLAGS=/DWIN32 /D_WINDOWS /EHsc "/FI%K%\csrc\sycl_first.h" /DNDEBUG -Qstd:c++20 -MD -fsycl -Wno-unused-command-line-argument -fsycl-targets=spir64_gen -DVLLM_XPU_ENABLE_XE2 -DCUTLASS_ENABLE_HEADERS_ONLY -DCUTLASS_ENABLE_SYCL -DSYCL_INTEL_TARGET -DCUTLASS_VERSIONS_GENERATED -fno-sycl-instrument-device-code /permissive- /bigobj -fsyntax-only
set INC=-I"%K%\csrc" -I"%K%\csrc\xpu\grouped_gemm\xe_2" -I"%K%\csrc\xpu\grouped_gemm" -I"%K%" -I"%CS%\include" -I"%CS%\tools\util\include" -I"%CS%\applications" -external:I"%K%\..\..\.venv\Lib\site-packages\torch\include" -external:I"%K%\..\..\.venv\Lib\site-packages\torch\include\torch\csrc\api\include" -external:I"%ONEAPI%\compiler\2026.1\include" -external:I"%ONEAPI%\compiler\2026.1\include\sycl" -external:W0
echo run %DATE% %TIME% > "%LOG%"
rem light = torch headers only (dispatch + selector). heavy = cutlass headers too, a few GB each: pass "heavy" as the first argument,
rem and not while the attention rebuild is running (its compiler peaks near 14 GB; the RAM guard kills every icx under 2.5 GB free).
set LIST="%K%\csrc\xpu\grouped_gemm\xe_2\grouped_gemm_xe2.cpp" "%K%\csrc\xpu\grouped_gemm\xe_2\split\select.cpp"
if "%1"=="heavy" set LIST=%LIST% "%K%\csrc\xpu\grouped_gemm\xe_2\split\launch_w16a16_m16_bf16.cpp" "%K%\csrc\xpu\grouped_gemm\xe_2\split\launch_w4a16_m8_bf16.cpp" "%K%\csrc\xpu\grouped_gemm\xe_2\split\launch_mxfp8_full_half.cpp"
for %%F in (%LIST%) do (
  echo -- %%~nxF >> "%LOG%"
  icx %FLAGS% %INC% "%%~F" >> "%LOG%" 2>&1
  echo    exit %ERRORLEVEL% %TIME% >> "%LOG%"
)
findstr /n /c:"error" /c:"exit" /c:"-- " "%LOG%" | findstr /v /c:"warning"
exit /b 0
