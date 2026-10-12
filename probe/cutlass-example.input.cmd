@echo off
rem cutlass-example.input.cmd - build Intel's own cutlass-sycl example 00_bmg_gemm with the exact flags the kernel build uses, two links: plain bmg and bmg + 256 GRF.
rem Usage:  cmd /c probe\cutlass-example.input.cmd      (writes probe\cutlass-example.output.txt; exes in probe\cutlass-example.build\; ~5 min CPU, no GPU)
rem Why (2026-10-10): every cutlass kernel in our build kills the device while every hand-written probe of the same instructions
rem passes. If Intel's reference GEMM example dies too, the fault reproduces outside vLLM with a file Intel maintains, and the
rem two links tell whether the 256-GRF link option is part of it. Run with cutlass-example-run.input.cmd when the card is free.
setlocal
set HERE=%~dp0
set K=%HERE%..\src\vllm-xpu-kernels
set CS=%K%\.deps\cutlass-sycl-src
set SRC=%CS%\examples\00_bmg_gemm\00_bmg_gemm.cpp
set B=%HERE%cutlass-example.build
set LOG=%HERE%cutlass-example.output.txt
set PY=%HERE%..\.venv312\Scripts\python.exe
set ONEAPI=C:\Program Files (x86)\Intel\oneAPI
call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat" > nul 2>&1
set SETVARS_CALL=1
call "%ONEAPI%\compiler\2026.1\env\vars.bat" > nul 2>&1
if exist "%ONEAPI%\ocloc\latest\env\vars.bat" call "%ONEAPI%\ocloc\latest\env\vars.bat" > nul 2>&1
if not exist "%B%" mkdir "%B%"
echo run %DATE% %TIME% > "%LOG%"
rem FLAGS/DEFINES/INCLUDES copied from build\temp\build.ninja for a cutlass TU (fmha_xe2.cpp), torch includes dropped (not needed).
set FLAGS=/DWIN32 /D_WINDOWS /EHsc "/FI%K%\csrc\sycl_first.h" /O2 /Ob2 /DNDEBUG -Qstd:c++20 -MD -fsycl -Wno-unused-command-line-argument -O3 -fsycl-targets=spir64_gen -DCUTLASS_ENABLE_HEADERS_ONLY -DCUTLASS_ENABLE_SYCL -DSYCL_INTEL_TARGET -DCUTLASS_VERSIONS_GENERATED -fno-sycl-instrument-device-code /permissive- /bigobj
rem probe\cutlass-gemm first: its sycl_common.hpp stands in for the example's (oneMKL RNG header is not installed here).
set INC=-I"%HERE%cutlass-gemm" -I"%CS%\include" -I"%CS%\tools\util\include" -I"%CS%\applications" -I"%CS%\examples\common"
set SPV=-Xspirv-translator -spirv-ext=+SPV_INTEL_2d_block_io,+SPV_INTEL_subgroup_matrix_multiply_accumulate,+SPV_INTEL_split_barrier
echo -- compile %SRC% >> "%LOG%"
icx %FLAGS% %INC% -c "%SRC%" -o "%B%\00_bmg_gemm.obj" >> "%LOG%" 2>&1
echo    exit %ERRORLEVEL% %TIME% >> "%LOG%"
echo -- link gemm-128 >> "%LOG%"
icx -fsycl "%B%\00_bmg_gemm.obj" %SPV% -fsycl-targets=spir64_gen -Xsycl-target-backend=spir64_gen "-device bmg" -fsycl-max-parallel-link-jobs=8 -o "%B%\gemm-128.exe" >> "%LOG%" 2>&1
echo    exit %ERRORLEVEL% %TIME% >> "%LOG%"
echo -- link gemm-256 >> "%LOG%"
icx -fsycl "%B%\00_bmg_gemm.obj" %SPV% -fsycl-targets=spir64_gen -Xsycl-target-backend=spir64_gen "-device bmg -internal_options -cl-intel-256-GRF-per-thread" -fsycl-max-parallel-link-jobs=8 -o "%B%\gemm-256.exe" >> "%LOG%" 2>&1
echo    exit %ERRORLEVEL% %TIME% >> "%LOG%"
rem Variants for bisecting (22:40: both links above die with 0xC0000409 on the B60):
rem  gemm-plain  = same source, the tiny probe's flags (no /FI sycl_first.h, no -fno-sycl-instrument-device-code, /O2 only), 256 GRF
rem  gemm-legacy = examples/00_bmg_gemm/legacy (old MainloopIntelXeXMX16 copy atoms), kernel-build flags, 256 GRF
set PLAIN=/EHsc /O2 -Qstd:c++20 -fsycl -Wno-unused-command-line-argument -fsycl-targets=spir64_gen -DCUTLASS_ENABLE_HEADERS_ONLY -DCUTLASS_ENABLE_SYCL -DSYCL_INTEL_TARGET -DCUTLASS_VERSIONS_GENERATED /permissive- /bigobj
set GRF256=-Xsycl-target-backend=spir64_gen "-device bmg -internal_options -cl-intel-256-GRF-per-thread"
echo -- compile gemm-plain (probe flags) >> "%LOG%"
icx %PLAIN% %INC% -c "%SRC%" -o "%B%\00_bmg_gemm_plain.obj" >> "%LOG%" 2>&1
echo    exit %ERRORLEVEL% %TIME% >> "%LOG%"
icx -fsycl "%B%\00_bmg_gemm_plain.obj" %SPV% -fsycl-targets=spir64_gen %GRF256% -fsycl-max-parallel-link-jobs=8 -o "%B%\gemm-plain.exe" >> "%LOG%" 2>&1
echo    link exit %ERRORLEVEL% %TIME% >> "%LOG%"
echo -- compile gemm-legacy (legacy example, kernel-build flags) >> "%LOG%"
icx %FLAGS% %INC% -c "%CS%\examples\00_bmg_gemm\legacy\00_bmg_gemm.cpp" -o "%B%\00_bmg_gemm_legacy.obj" >> "%LOG%" 2>&1
echo    exit %ERRORLEVEL% %TIME% >> "%LOG%"
icx -fsycl "%B%\00_bmg_gemm_legacy.obj" %SPV% -fsycl-targets=spir64_gen %GRF256% -fsycl-max-parallel-link-jobs=8 -o "%B%\gemm-legacy.exe" >> "%LOG%" 2>&1
echo    link exit %ERRORLEVEL% %TIME% >> "%LOG%"
echo -- what each exe carries >> "%LOG%"
"%PY%" "%HERE%image-kind.input.py" "%B%\gemm-128.exe" "%B%\gemm-256.exe" "%B%\gemm-plain.exe" "%B%\gemm-legacy.exe" >> "%LOG%" 2>&1
echo done %TIME% >> "%LOG%"
type "%LOG%"
exit /b 0
