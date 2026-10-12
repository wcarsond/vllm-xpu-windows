@echo off
rem grf256.input.cmd - build probe\grf256.cpp three ways and run each on the B60: AOT bmg normal, AOT bmg with 256 GRF, JIT (default and large GRF).
rem Usage:  cmd /c probe\grf256.input.cmd      (writes probe\grf256.output.txt; binaries in probe\grf256.build\; ~1 min, no GPU use until the runs)
rem Why (2026-10-10): the Xe2 kernel libraries are the only ones linked with "-internal_options -cl-intel-256-GRF-per-thread"
rem and the only ones whose kernels kill the device. Same toolchain and flags as kernels\build.input.cmd, one trivial kernel.
rem Compile and link are separate steps as in the CMake build: a one-step icx build drops the -Xsycl-target-backend option
rem ("argument unused", 22:18) and silently ships SPIR-V instead. image-kind.input.py confirms what each exe carries.
rem Read the run lines: only aot-256 failing = the driver cannot launch large-GRF kernels compiled by this ocloc;
rem jit-largeGRF failing too = the driver cannot run large-GRF kernels at all on Windows.
setlocal
set HERE=%~dp0
set SRC=%HERE%grf256.cpp
set B=%HERE%grf256.build
set LOG=%HERE%grf256.output.txt
set PY=%HERE%..\.venv312\Scripts\python.exe
set ONEAPI=C:\Program Files (x86)\Intel\oneAPI
call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat" > nul 2>&1
set SETVARS_CALL=1
call "%ONEAPI%\compiler\2026.1\env\vars.bat" > nul 2>&1
if exist "%ONEAPI%\ocloc\latest\env\vars.bat" call "%ONEAPI%\ocloc\latest\env\vars.bat" > nul 2>&1
if exist "%ONEAPI%\umf\latest\env\vars.bat" call "%ONEAPI%\umf\latest\env\vars.bat" > nul 2>&1
if exist "%ONEAPI%\tbb\latest\env\vars.bat" call "%ONEAPI%\tbb\latest\env\vars.bat" > nul 2>&1
if not exist "%B%" mkdir "%B%"
del /q "%B%\*" 2>nul
echo run %DATE% %TIME% > "%LOG%"

echo -- compile (device code for spir64_gen, like the kernel build) >> "%LOG%"
icx -fsycl "%SRC%" /EHsc /O2 -fsycl-targets=spir64_gen -c -o "%B%\grf256-gen.obj" >> "%LOG%" 2>&1
echo    exit %ERRORLEVEL% >> "%LOG%"
rem The SPIR-V extensions the kernel build enables at link (CMakeLists.txt): 2D block loads need the first one.
set SPV=-Xspirv-translator -spirv-ext=+SPV_INTEL_2d_block_io,+SPV_INTEL_subgroup_matrix_multiply_accumulate,+SPV_INTEL_split_barrier
echo -- link aot-128 >> "%LOG%"
icx -fsycl "%B%\grf256-gen.obj" %SPV% -fsycl-targets=spir64_gen -Xsycl-target-backend=spir64_gen "-device bmg" -o "%B%\aot-128.exe" >> "%LOG%" 2>&1
echo    exit %ERRORLEVEL% >> "%LOG%"
echo -- link aot-256 >> "%LOG%"
icx -fsycl "%B%\grf256-gen.obj" %SPV% -fsycl-targets=spir64_gen -Xsycl-target-backend=spir64_gen "-device bmg -internal_options -cl-intel-256-GRF-per-thread" -o "%B%\aot-256.exe" >> "%LOG%" 2>&1
echo    exit %ERRORLEVEL% >> "%LOG%"
echo -- build jit (plain spir64) >> "%LOG%"
icx -fsycl "%SRC%" /EHsc /O2 -c -o "%B%\grf256-jit.obj" >> "%LOG%" 2>&1
icx -fsycl "%B%\grf256-jit.obj" %SPV% -o "%B%\jit.exe" >> "%LOG%" 2>&1
echo    exit %ERRORLEVEL% >> "%LOG%"
echo -- what each exe carries >> "%LOG%"
"%PY%" "%HERE%image-kind.input.py" "%B%\aot-128.exe" "%B%\aot-256.exe" "%B%\jit.exe" >> "%LOG%" 2>&1

echo -- run aot-128 on B60 >> "%LOG%"
"%B%\aot-128.exe" aot-128 B60 >> "%LOG%" 2>&1
echo    exit %ERRORLEVEL% >> "%LOG%"
echo -- run aot-256 on B60 >> "%LOG%"
"%B%\aot-256.exe" aot-256 B60 >> "%LOG%" 2>&1
echo    exit %ERRORLEVEL% >> "%LOG%"
echo -- run jit default GRF on B60 >> "%LOG%"
"%B%\jit.exe" jit-default B60 >> "%LOG%" 2>&1
echo    exit %ERRORLEVEL% >> "%LOG%"
echo -- run jit with SYCL_PROGRAM_COMPILE_OPTIONS=-ze-opt-large-register-file on B60 >> "%LOG%"
set SYCL_PROGRAM_COMPILE_OPTIONS=-ze-opt-large-register-file
"%B%\jit.exe" jit-largeGRF B60 >> "%LOG%" 2>&1
echo    exit %ERRORLEVEL% >> "%LOG%"
set SYCL_PROGRAM_COMPILE_OPTIONS=
echo done %TIME% >> "%LOG%"
type "%LOG%"
exit /b 0
