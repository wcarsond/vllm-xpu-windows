@echo off
rem wheel.input.cmd - package the finished kernel build as a wheel and install it into .venv312.
rem Usage:  cmd /c kernels\wheel.input.cmd          (writes kernels\wheel.output.log; pip detail in wheel.pip.log)
rem
rem Why (2026-10-10): build.input.cmd compiles in .venv (Python 3.10) as an editable install; vLLM runs in .venv312
rem (Python 3.12, same torch 2.14.1+xpu). The extensions are built py_limited_api (abi3), so one wheel serves both.
rem "pip wheel" re-runs setup.py's build_ext in the same build\temp: CMake's cache and ninja's objects are current, so
rem this is a relink and a zip, not a second compile. Run only after build.output.log shows "pip exit 0".
set HERE=%~dp0
set REPO=%HERE%..\src\vllm-xpu-kernels
set LOG=%HERE%wheel.output.log
set PIPLOG=%HERE%wheel.pip.log
set PY310=%HERE%..\.venv\Scripts\python.exe
set PY312=%HERE%..\.venv312\Scripts\python.exe
set DIST=%HERE%dist

echo run %DATE% %TIME% > "%LOG%"
findstr /c:"pip exit 0" "%HERE%build.output.log" > nul || ( echo build.output.log does not say "pip exit 0": build first >> "%LOG%" & type "%LOG%" & exit /b 1 )

rem Same compiler environment and settings as build.input.cmd, or CMake would see changed options and reconfigure.
set "VS2022INSTALLDIR=C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools"
call "%VS2022INSTALLDIR%\VC\Auxiliary\Build\vcvars64.bat" > "%HERE%wheel.env-msvc.log" 2>&1
set SETVARS_CALL=1
set ONEAPI=C:\Program Files (x86)\Intel\oneAPI
call "%ONEAPI%\compiler\2026.1\env\vars.bat" > "%HERE%wheel.env-compiler.log" 2>&1
if exist "%ONEAPI%\ocloc\latest\env\vars.bat" call "%ONEAPI%\ocloc\latest\env\vars.bat" > "%HERE%wheel.env-ocloc.log" 2>&1
if exist "%ONEAPI%\umf\latest\env\vars.bat"   call "%ONEAPI%\umf\latest\env\vars.bat"   > "%HERE%wheel.env-umf.log" 2>&1
if exist "%ONEAPI%\tbb\latest\env\vars.bat"   call "%ONEAPI%\tbb\latest\env\vars.bat"   > "%HERE%wheel.env-tbb.log" 2>&1
set VLLM_XPU_AOT_DEVICES=bmg
set VLLM_XPU_XE2_AOT_DEVICES=bmg
set VLLM_XPU_ENABLE_XE3P=OFF
if not defined VLLM_XPU_ENABLE_ONEDNN set VLLM_XPU_ENABLE_ONEDNN=OFF
rem MOE_KERNELS_ENABLED: honour a preset (after-build.input.cmd turns it ON for the MoE wheel), else OFF like build.input.cmd.
if not defined MOE_KERNELS_ENABLED set MOE_KERNELS_ENABLED=OFF
set MAX_JOBS=1
set VLLM_CHUNK_PREFILL_CONFIG=chunk_prefill_default.conf
set VLLM_PAGED_DECODE_CONFIG=paged_decode_default.conf
set VLLM_TARGET_DEVICE=xpu
set LEVEL_ZERO_ROOT=%HERE%..\toolchain\level-zero
rem 2026-10-11 02:21: without this the setup.py CMake run re-fetched the pinned cutlass tag over the patched checkout and
rem ninja started recompiling every attention kernel against the broken headers. Same line as build.input.cmd.
set VLLM_CUTLASS_SRC_DIR=%REPO%\.deps\cutlass-sycl-src
set PATH=%HERE%..\.venv\Scripts;%PATH%

cd /d "%REPO%"
echo wheel start %TIME% >> "%LOG%"
rem bdist_wheel with --py-limited-api, as upstream's build_wheel_docker.sh does (it tags the wheel abi3; "pip wheel" cannot
rem pass that switch and tagged the first wheel cp310-cp310, which pip then refused for the 3.12 venv). xpumem_allocator
rem is cp310-bound by upstream design (it drops Py_LIMITED_API and so links python310.lib); the three kernel modules are abi3.
"%PY310%" setup.py bdist_wheel --dist-dir "%DIST%" --py-limited-api=cp310 > "%PIPLOG%" 2>&1
echo bdist_wheel exit %ERRORLEVEL% at %TIME% >> "%LOG%"
echo == kernel DLLs and modules in the package: >> "%LOG%"
dir /b "%REPO%\vllm_xpu_kernels\*.pyd" "%REPO%\vllm_xpu_kernels\*.dll" >> "%LOG%" 2>&1
dir /b "%DIST%\*.whl" >> "%LOG%" 2>&1

rem Install into the 3.12 venv and prove it imports there (the SYCL kernels load through torch's own runtime).
rem Only the abi3 wheel: the first run also left a cp310-cp310 one (from "pip wheel"), which pip rightly refuses for 3.12.
del /q "%DIST%\vllm_xpu_kernels-*-cp310-win_amd64.whl" 2>nul
for %%W in ("%DIST%\vllm_xpu_kernels-*-abi3-win_amd64.whl") do set WHL=%%W
echo installing %WHL% >> "%LOG%"
"%PY312%" -m pip install --no-deps --force-reinstall "%WHL%" >> "%PIPLOG%" 2>&1
echo install into .venv312 exit %ERRORLEVEL% >> "%LOG%"
rem _moe_C is not built while MOE_KERNELS_ENABLED=OFF; xpumem_allocator is cp310-only (full ABI by upstream design).
rem Import from a neutral folder so the installed wheel is what loads, not the source tree.
cd /d "%HERE%"
"%PY312%" -c "import torch, vllm_xpu_kernels, vllm_xpu_kernels._C, vllm_xpu_kernels._xpu_C, vllm_xpu_kernels._vllm_fa2_C; print('kernels import ok in 3.12 from', vllm_xpu_kernels.__file__)" >> "%LOG%" 2>&1
type "%LOG%"
exit /b 0
