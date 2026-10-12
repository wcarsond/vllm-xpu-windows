@echo off
rem build-exchange.input.cmd - build the two-card exchange extension (_xpu_exchange.pyd) with icx against the venv's torch.
rem Usage:  cmd /c kernels\build-exchange.input.cmd      (writes kernels\build-exchange.output.log; ~1 minute)
rem Why (2026-10-11 08:45): the all_reduce of the tensor split must be one C++ call that returns without waiting
rem (PORT-NOTES "GPU-TIMELINE EXCHANGE"). A standalone .pyd next to the Python module builds in a minute and leaves the
rem kernel wheel alone. JIT device code (no AOT flags): one small kernel family, compiled on first use.
setlocal EnableDelayedExpansion
set HERE=%~dp0
set LOG=%HERE%build-exchange.output.log
set B=%HERE%build-exchange.build
set PY=C:\Users\wcars\AppData\Local\Programs\Python\Python312
set TORCH=%HERE%..\.venv312\Lib\site-packages\torch
set SRC=%HERE%..\src\vllm\vllm\distributed\device_communicators\xpu_exchange_ext\xpu_exchange.cpp
set OUT=%HERE%..\src\vllm\vllm\distributed\device_communicators\_xpu_exchange.pyd
set ONEAPI=C:\Program Files (x86)\Intel\oneAPI
call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat" > nul 2>&1
set SETVARS_CALL=1
call "%ONEAPI%\compiler\2026.1\env\vars.bat" > nul 2>&1
if not exist "%B%" mkdir "%B%"
echo run %DATE% %TIME% > "%LOG%"
rem C++20: torch's headers use <=>, requires and string_view::starts_with. -w: the torch headers emit ~50 warnings under clang.
icx -fsycl /EHsc /O2 /MD /std:c++20 -w /DNOMINMAX /DWIN32_LEAN_AND_MEAN /I"%TORCH%\include" /I"%TORCH%\include\torch\csrc\api\include" /I"%PY%\include" -c "%SRC%" -o "%B%\xpu_exchange.obj" >> "%LOG%" 2>&1
echo compile exit !ERRORLEVEL! >> "%LOG%"
if not exist "%B%\xpu_exchange.obj" ( type "%LOG%" & exit /b 1 )
del /q "%OUT%" 2>nul
rem /LD (MSVC-style DLL), not -shared: icx in MSVC mode ignores -shared and demands an entry point
icx -fsycl /LD "%B%\xpu_exchange.obj" /Fe:"%OUT%" "%TORCH%\lib\torch.lib" "%TORCH%\lib\torch_cpu.lib" "%TORCH%\lib\torch_python.lib" "%TORCH%\lib\c10.lib" "%TORCH%\lib\c10_xpu.lib" "%TORCH%\lib\torch_xpu.lib" "%PY%\libs\python312.lib" >> "%LOG%" 2>&1
echo link exit !ERRORLEVEL! >> "%LOG%"
rem /LD leaves an import library and exports file next to the .pyd; nothing links against them
del /q "%OUT:.pyd=.lib%" "%OUT:.pyd=.exp%" 2>nul
if exist "%OUT%" (
    "%HERE%..\.venv312\Scripts\python.exe" -c "import torch, vllm.distributed.device_communicators._xpu_exchange as e; print('import ok:', e.__doc__)" >> "%LOG%" 2>&1
    echo import exit !ERRORLEVEL! >> "%LOG%"
)
type "%LOG%"
exit /b 0
