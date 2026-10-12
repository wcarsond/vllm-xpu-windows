@echo off
rem long-size.input.cmd - compile probe\long-size.cpp (device pass included) and report whether device `long` is 4 or 8 bytes.
rem Usage:  cmd /c probe\long-size.input.cmd      (writes probe\long-size.output.txt; compile only, no GPU)
rem Why (2026-10-10): cute's rearchitected 2D copy atoms (copy_traits_xe_2d.hpp) pass the tensor base address to IGC's
rem block-2D address-payload builtins as `long`. MSVC's long is 32 bits; if the SYCL device pass follows the host, every
rem such address is cut in half on Windows, which is exactly a first-tile-load page fault.
setlocal
set HERE=%~dp0
set LOG=%HERE%long-size.output.txt
set ONEAPI=C:\Program Files (x86)\Intel\oneAPI
call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat" > nul 2>&1
set SETVARS_CALL=1
call "%ONEAPI%\compiler\2026.1\env\vars.bat" > nul 2>&1
echo run %DATE% %TIME% > "%LOG%"
icx -fsycl "%HERE%long-size.cpp" /EHsc -fsycl-targets=spir64_gen -c -o "%HERE%long-size.obj" >> "%LOG%" 2>&1
echo exit %ERRORLEVEL% (0 = device long is 8 bytes; a DEVICE_LONG_IS_NOT_8_BYTES error = 4 bytes) >> "%LOG%"
del /q "%HERE%long-size.obj" 2>nul
type "%LOG%"
exit /b 0
