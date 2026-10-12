@echo off
rem build.input.cmd - build vllm-xpu-kernels on Windows for the B60 with Intel's 2026.1 compiler and MSVC 2022.
rem Usage:  cmd /c kernels\build.input.cmd          (writes kernels\build.output.log; the full pip log is build.pip.log)
rem
rem Why (2026-10-10, rung 4): vLLM's xpu platform hard-imports vllm_xpu_kernels, which ships Linux wheels only.
rem This is the first attempt on the unmodified source: its error list is the task list. Settings, each a choice:
rem   VLLM_XPU_AOT_DEVICES / VLLM_XPU_XE2_AOT_DEVICES = bmg : compile card code ahead of time for the B60 only
rem       (torch names the card bmg; the repo default also builds pvc and two bmg steppings, hours of extra work)
rem   VLLM_XPU_ENABLE_XE3P=OFF : Xe3 kernels are for cards we do not have
rem   VLLM_XPU_ENABLE_ONEDNN=OFF : oneDNN is not installed; this swaps stubs in for the fp8/int4 GEMMs only
rem   MAX_JOBS=1 : was 4, then 2. Attempt 9 (16:40, 4 jobs): free RAM 15.5 GB -> 0.7 GB in two minutes, killed. Attempt 10
rem       (16:43, 2 jobs): 0.6 GB free at step 0, killed; the memory was in clang-cl.exe (the compiler icx.exe drives),
rem       one instance at 10.3 GB working set / 14 GB commit, another at 4.3 GB. With ~11 GB of other use on this 32 GB
rem       box, one such compile at a time is the safe number until the watcher has measured the peak.
rem   VLLM_CHUNK_PREFILL_CONFIG / VLLM_PAGED_DECODE_CONFIG = the *_default.conf presets: Llama/Qwen/DeepSeek shapes only.
rem       The CMake default is the *_full.conf pair, 240 + 384 attention kernel variants (the 732 steps), which Intel's
rem       KERNEL_CONFIGURATION.md puts at ~60 min against ~2 min for default. First light needs one model's shapes.
rem   setvars.bat vs2022 : Intel's environment script, which also runs vcvars64 for MSVC
set HERE=%~dp0
set REPO=%HERE%..\src\vllm-xpu-kernels
set LOG=%HERE%build.output.log
set PIPLOG=%HERE%build.pip.log
set PY=%HERE%..\.venv\Scripts\python.exe

echo run %DATE% %TIME% > "%LOG%"
rem Attempt 1 (16:26) used Intel's setvars.bat: it could not find Build Tools (it only looks for <Year>\<Edition>) and
rem its pushd-into-env-then-call-vars.bat loop ran nothing, so icx/CMPLR_ROOT/SYCL_HOME were all empty. Direct calls instead:
rem MSVC first (sets VSCMD_VER, VCToolsInstallDir), then the compiler's own vars.bat (sets CMPLR_ROOT, PATH), then ocloc and umf.
rem Attempt 2 (16:28): vcvars64 leaves a child holding its log open, so every later ">>" to the same file failed with
rem "being used by another process", and cmd skips a command whose redirect fails: vars.bat never ran. One log per call.
set "VS2022INSTALLDIR=C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools"
call "%VS2022INSTALLDIR%\VC\Auxiliary\Build\vcvars64.bat" > "%HERE%build.env-msvc.log" 2>&1
set SETVARS_CALL=1
set ONEAPI=C:\Program Files (x86)\Intel\oneAPI
call "%ONEAPI%\compiler\2026.1\env\vars.bat" > "%HERE%build.env-compiler.log" 2>&1
if exist "%ONEAPI%\ocloc\latest\env\vars.bat" call "%ONEAPI%\ocloc\latest\env\vars.bat" > "%HERE%build.env-ocloc.log" 2>&1
if exist "%ONEAPI%\umf\latest\env\vars.bat"   call "%ONEAPI%\umf\latest\env\vars.bat"   > "%HERE%build.env-umf.log" 2>&1
if exist "%ONEAPI%\tbb\latest\env\vars.bat"   call "%ONEAPI%\tbb\latest\env\vars.bat"   > "%HERE%build.env-tbb.log" 2>&1
echo == environment proof >> "%LOG%"
where icx icpx ocloc cl ninja cmake >> "%LOG%" 2>&1
echo CMPLR_ROOT=%CMPLR_ROOT% >> "%LOG%"
echo VCToolsInstallDir=%VCToolsInstallDir% >> "%LOG%"
if "%CMPLR_ROOT%"=="" ( echo compiler environment not set, see build.env-*.log >> "%LOG%" & type "%LOG%" & exit /b 1 )

set VLLM_XPU_AOT_DEVICES=bmg
set VLLM_XPU_XE2_AOT_DEVICES=bmg
set VLLM_XPU_ENABLE_XE3P=OFF
rem 2026-10-11: build-onednn.input.cmd sets ON before calling this file; oneDNN rls-v3.13 is then fetched and built from
rem source as a static library inside this build (cmake\Modules\FindoneDNN.cmake), hours of extra compile.
if not defined VLLM_XPU_ENABLE_ONEDNN set VLLM_XPU_ENABLE_ONEDNN=OFF
set MAX_JOBS=1
set VLLM_CHUNK_PREFILL_CONFIG=chunk_prefill_default.conf
set VLLM_PAGED_DECODE_CONFIG=paged_decode_default.conf
rem MOE_KERNELS_ENABLED=OFF (2026-10-10, attempt 11): compiling csrc\xpu\grouped_gemm\xe_2\grouped_gemm_xe2.cpp alone took
rem   clang-cl to 16.2 GB working set / 30.5 GB commit and was still growing at 0.4 GB free. That one file cannot compile on
rem   this 32 GB PC as written. The MoE grouped GEMM is only needed for MoE models; first light is a dense model. Open item.
rem   2026-10-11: the grouped GEMM is now split into 36 small translation units (kernels\gen-moe-split.input.py), so MoE can be
rem   built on 32 GB. Default stays OFF here; build-moe.input.cmd sets MOE_KERNELS_ENABLED=ON before calling this file.
if not defined MOE_KERNELS_ENABLED set MOE_KERNELS_ENABLED=OFF
rem VLLM_CUTLASS_SRC_DIR (2026-10-10 22:55): use the cutlass-sycl checkout already in .deps, which carries the Windows fix
rem   patches\cutlass-sycl-windows-64bit-addresses.patch (IGC block-2D address builtins declared with `long`, 32 bits here).
rem   Without this CMake's FetchContent would check the pinned tag out again and drop the fix.
set VLLM_CUTLASS_SRC_DIR=%REPO%\.deps\cutlass-sycl-src
set VLLM_TARGET_DEVICE=xpu
set VERBOSE=1
rem ze_loader.lib + headers made by toolchain\make-ze-loader-lib.input.cmd (CMakeLists.txt reads LEVEL_ZERO_ROOT on Windows)
set LEVEL_ZERO_ROOT=%HERE%..\toolchain\level-zero
rem Ninja lives in the venv; put it and the venv's python first.
set PATH=%HERE%..\.venv\Scripts;%PATH%

cd /d "%REPO%"
echo branch: >> "%LOG%"
git branch --show-current >> "%LOG%"
echo build start %TIME% >> "%LOG%"
"%PY%" -m pip install --no-build-isolation -e . -v > "%PIPLOG%" 2>&1
set RC=%ERRORLEVEL%
echo pip exit %RC% at %TIME% >> "%LOG%"
echo. >> "%LOG%"
echo == errors in the pip log >> "%LOG%"
findstr /i /c:"error" /c:"fatal" /c:"CMake Error" "%PIPLOG%" >> "%LOG%"
type "%LOG%"
exit /b %RC%
