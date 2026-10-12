@echo off
rem compat-devices.input.cmd - build probe\compat-devices.cpp (host-only SYCL, no kernels) and run it over torch's runtime set.
rem Usage:  cmd /c probe\compat-devices.input.cmd [ONEAPI_DEVICE_SELECTOR value]      (writes probe\compat-devices.output.txt)
rem Why (2026-10-11 03:58): see compat-devices.cpp. With an argument the run repeats under that selector (e.g. level_zero:*),
rem to show what hiding OpenCL would do to the device list the cutlass kernels size themselves from.
setlocal
set HERE=%~dp0
set B=%HERE%compat-devices.build
set LOG=%HERE%compat-devices.output.txt
set ONEAPI=C:\Program Files (x86)\Intel\oneAPI
call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat" > nul 2>&1
set SETVARS_CALL=1
call "%ONEAPI%\compiler\2026.1\env\vars.bat" > nul 2>&1
if not exist "%B%" mkdir "%B%"
echo run %DATE% %TIME% > "%LOG%"
if not exist "%B%\compat-devices.exe" (
    icx -fsycl "%HERE%compat-devices.cpp" /EHsc /O2 -c -o "%B%\compat-devices.obj" >> "%LOG%" 2>&1
    icx -fsycl "%B%\compat-devices.obj" -o "%B%\compat-devices.exe" >> "%LOG%" 2>&1
    echo build exit %ERRORLEVEL% >> "%LOG%"
)
set PATH=%HERE%..\.venv312\Library\bin;%PATH%
echo -- as the workers see it (selector unset) >> "%LOG%"
set ONEAPI_DEVICE_SELECTOR=
"%B%\compat-devices.exe" >> "%LOG%" 2>&1
if not "%~1"=="" (
    echo -- with ONEAPI_DEVICE_SELECTOR=%~1 >> "%LOG%"
    set ONEAPI_DEVICE_SELECTOR=%~1
    "%B%\compat-devices.exe" >> "%LOG%" 2>&1
)
echo run exit %ERRORLEVEL% >> "%LOG%"
type "%LOG%"
exit /b 0
