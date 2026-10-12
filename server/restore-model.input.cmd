@echo off
rem restore-model.input.cmd - copy a model back from F:\models-hf (the library) to C:\Users\claude\models-hf (what is served).
rem Usage:  cmd /c server\restore-model.input.cmd <model-folder>     e.g.  restore-model.input.cmd Qwen2.5-14B-Instruct
rem         cmd /c server\restore-model.input.cmd                    (no argument: list what F: has and what C: already has)
rem Why (2026-10-11 14:30, Carson: "yes you can move the uneeded models and dont delete them off F it has lots of space"
rem and "just re copy them"): the inverse of archive-model.input.cmd. F: is the library and nothing is ever removed from it,
rem so a model that was archived comes back with a copy, not a move. C: is the NVMe that models are actually served from;
rem never serve from F:, it is a spinning disk (~80 MB/s).
setlocal
set MODEL=%1
set SRC=F:\models-hf\%MODEL%
set DST=C:\Users\claude\models-hf\%MODEL%
if "%MODEL%"=="" (
    echo -- in the library on F: & dir /b F:\models-hf 2>nul
    echo.
    echo -- already on C: & dir /b C:\Users\claude\models-hf 2>nul
    echo.
    powershell -NoProfile -Command "'C: free {0:N1} GB   F: free {1:N1} GB' -f ((Get-PSDrive C).Free/1GB), ((Get-PSDrive F).Free/1GB)"
    exit /b 0
)
if not exist "%SRC%\config.json" (echo no model at %SRC% & exit /b 2)
if exist "%DST%\config.json" (echo %MODEL% is already on C: & exit /b 0)
rem refuse rather than fill the system drive: need the model's size plus 10 GB of headroom
for /f %%S in ('powershell -NoProfile -Command "[int]((Get-ChildItem -Recurse -File '%SRC%' | Measure-Object Length -Sum).Sum/1GB)"') do set NEED=%%S
for /f %%F in ('powershell -NoProfile -Command "[int]((Get-PSDrive C).Free/1GB)"') do set FREE=%%F
echo %MODEL% is %NEED% GB; C: has %FREE% GB free
set /a ROOM=%FREE%-%NEED%
if %ROOM% LSS 10 (echo that would leave only %ROOM% GB on C: - archive something first with archive-model.input.cmd & exit /b 2)
echo copying %SRC% -^> %DST%  %DATE% %TIME%
robocopy "%SRC%" "%DST%" /E /MT:8 /R:2 /W:5 /NFL /NDL /NJH /NP
if %ERRORLEVEL% GEQ 8 (echo robocopy failed with %ERRORLEVEL% & exit /b %ERRORLEVEL%)
echo restored  %DATE% %TIME%
exit /b 0
