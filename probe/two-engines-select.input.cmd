@echo off
rem two-engines-select.input.cmd - two independent one-card engines, with the card chosen ABOVE the driver
rem (ONEAPI_DEVICE_SELECTOR=level_zero:N, SYCL runtime filter) instead of by the driver mask (ZE_AFFINITY_MASK=N).
rem Usage:  cmd /c probe\two-engines-select.input.cmd [oneapi|mask] [repeats]     (writes probe\two-engines-select.output.txt)
rem Why (2026-10-11 22:45): the masked pair faults (trigger 1) and the private-copy loader fix does not cover it
rem (two-engines-private-copy.output.txt). The clean cell in the record is "no mask, each process sees both cards and
rem picks one". If the SYCL-level selector lands in that cell, a job can run one independent worker per card.
set HERE=%~dp0
set SEL=%1
if "%SEL%"=="" set SEL=oneapi
set N=%2
if "%N%"=="" set N=2
set OUT=%HERE%two-engines-select.output.txt
call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat" > nul 2>&1
set CC=cl
set PATH=%HERE%..\.venv312\Library\bin;%PATH%
set TWO_ENGINES_SELECT=%SEL%
echo run %DATE% %TIME% select=%SEL% x%N% > "%OUT%"
for /l %%i in (1,1,%N%) do (
    echo --- repeat %%i >> "%OUT%"
    "%HERE%..\.venv312\Scripts\python.exe" -u "%HERE%two-engines.input.py" >> "%OUT%" 2>&1
)
findstr /c:"---" /c:"select=" /c:"engines" /c:"Error" "%OUT%"
