@echo off
rem archive-model.input.cmd - move a model folder from C:\Users\claude\models-hf (NVMe, where models are served from) to
rem F:\models-hf (the 931 GB spinning disk), to make room on C:. Nothing is deleted; the model is restorable by moving it back.
rem Usage:  cmd /c server\archive-model.input.cmd <model-folder>      e.g.  archive-model.input.cmd Qwen2.5-14B-Instruct
rem Why (2026-10-11 13:50, Carson: "you can make space deleting an older model if needed, or push some stuff to F:/"): the
rem 72B-AWQ (41 GB) needs the room; F: is too slow to serve from (see memory: big-pc-f-is-a-spinning-disk) but fine to park on.
rem robocopy /MOVE copies then removes the source; /MT:8 keeps the spinning disk busy; exit codes 0-7 are success for robocopy.
set MODEL=%1
if "%MODEL%"=="" (echo model folder name required & exit /b 2)
set SRC=C:\Users\claude\models-hf\%MODEL%
set DST=F:\models-hf\%MODEL%
if not exist "%SRC%\config.json" (echo no model at %SRC% & exit /b 2)
if not exist F:\ (echo F: is not mounted & exit /b 2)
if not exist F:\models-hf mkdir F:\models-hf
echo moving %SRC% -^> %DST%  %DATE% %TIME%
robocopy "%SRC%" "%DST%" /E /MOVE /MT:8 /R:2 /W:5 /NFL /NDL /NJH /NP
set RC=%ERRORLEVEL%
if %RC% GEQ 8 (echo robocopy failed with %RC%, source left in place & exit /b %RC%)
if exist "%SRC%" rmdir /s /q "%SRC%"
echo moved; robocopy code %RC%  %DATE% %TIME%
exit /b 0
