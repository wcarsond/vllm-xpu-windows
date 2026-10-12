@echo off
rem sycl-ls.input.cmd - which devices does the freshly installed SYCL runtime see, and why is Level Zero missing?
rem Usage:  cmd /c probe\sycl-ls.input.cmd      (writes probe\sycl-ls.output.txt)
rem Why (2026-10-10): run bare, sycl-ls listed only OpenCL devices and chose the Core Ultra iGPU as default; torch's own
rem runtime talks to the B60s over Level Zero. The build does not care (ahead-of-time through ocloc) but every test
rem will. UR_LOG_LOADER makes the Unified Runtime loader say which adapter DLLs it tried and why one failed.
set HERE=%~dp0
set OUT=%HERE%sycl-ls.output.txt
echo run %DATE% %TIME% > "%OUT%"
rem setvars.bat set nothing on this PC (see kernels\build.input.cmd for why); call the compiler's vars.bat directly.
set SETVARS_CALL=1
call "C:\Program Files (x86)\Intel\oneAPI\compiler\2026.1\env\vars.bat" > nul 2>&1
if exist "C:\Program Files (x86)\Intel\oneAPI\umf\latest\env\vars.bat" call "C:\Program Files (x86)\Intel\oneAPI\umf\latest\env\vars.bat" > nul 2>&1
echo == with compiler vars.bat, loader log on >> "%OUT%"
where sycl-ls >> "%OUT%" 2>&1
set UR_LOG_LOADER=level:info;flush:info;output:stdout
sycl-ls --verbose >> "%OUT%" 2>&1
echo. >> "%OUT%"
echo == ze_loader.dll copies on this PC >> "%OUT%"
where ze_loader.dll >> "%OUT%" 2>&1
dir /b /s "C:\Windows\System32\ze_loader.dll" "C:\Program Files (x86)\Intel\oneAPI\compiler\2026.1\bin\ze_loader.dll" "%HERE%..\.venv\Lib\site-packages\intel_sycl_rt\bin\ze_loader.dll" >> "%OUT%" 2>&1
echo == UR adapters shipped with the compiler >> "%OUT%"
dir /b "C:\Program Files (x86)\Intel\oneAPI\compiler\2026.1\bin\ur_adapter_*.dll" >> "%OUT%" 2>&1
type "%OUT%"
