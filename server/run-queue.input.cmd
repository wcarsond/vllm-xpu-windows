@echo off
rem run-queue.input.cmd - run several two-card model races back to back, ONE AT A TIME, waiting for RAM between them.
rem Usage:  cmd /c server\run-queue.input.cmd
rem Why (2026-10-11 14:15): the 72B's first attempt died during shard 1 of 11 because a 39 GB checkpoint was loading while a
rem download was still running and only 14.5 GiB of host RAM was free (PORT-NOTES "72B-AWQ, FIRST ATTEMPT FAILED ON HOST RAM").
rem Carson's standing rule is one heavy job at a time; this file is that rule written down so a queue cannot overlap again.
rem Edit the MODELS list to change what runs. Each line: <model-folder> <dtype> <gpu_util> <max_model_len>.
rem Each race writes smoke-tp2-<model-folder>.console.log and .output.txt beside this script; the summary lands in queue.output.txt.
setlocal enabledelayedexpansion
set HERE=%~dp0
set OUT=%HERE%queue.output.txt
echo queue started %DATE% %TIME% > "%OUT%"

rem --- the queue: smallest first, so a RAM problem shows on a cheap run rather than a 39 GB one -------------------------
rem 14:35: Granite done (33.1 tok/s). Qwen3.5-27B is the model Carson actually wants (newer than 2.5, vision built in for
rem 29 GB so it needs the pair); bfloat16 because float16 is reported to crash this checkpoint elsewhere.
rem Then the 72B again at 0.85 instead of 0.95 - test (1) of the three in PORT-NOTES, in case the silent teardown at
rem 0% of shard loading is a card-memory edge during load rather than anything to do with the host.
call :one Qwen3.5-27B-GPTQ-Int4     bfloat16 0.90 4096
call :one Qwen2.5-72B-Instruct-AWQ  float16  0.85 4096
goto :done

:one
set MODEL=%1
echo. >> "%OUT%"
echo == %MODEL%  %DATE% %TIME% >> "%OUT%"
rem wait for host RAM to come back (a finished run's pages, a finished download) before loading the next checkpoint
call :waitram
powershell -NoProfile -Command "$os=Get-CimInstance Win32_OperatingSystem; 'RAM free {0:N1} GB at start' -f ($os.FreePhysicalMemory/1MB)" >> "%OUT%"
call "%HERE%run-smoke-tp2.input.cmd" FLASH_ATTN %1 %2 %3 %4 >> "%OUT%" 2>&1
echo exit !ERRORLEVEL! >> "%OUT%"
exit /b 0

:waitram
rem up to 10 minutes for 18 GB of free RAM; proceed anyway after that so the queue never hangs forever
for /l %%I in (1,1,60) do (
    for /f %%R in ('powershell -NoProfile -Command "$os=Get-CimInstance Win32_OperatingSystem; [int]($os.FreePhysicalMemory/1MB)"') do set FREE=%%R
    if !FREE! GEQ 18 exit /b 0
    ping -n 11 127.0.0.1 > nul
)
echo WARNING: proceeding with only !FREE! GB free >> "%OUT%"
exit /b 0

:done
echo. >> "%OUT%"
echo queue finished %DATE% %TIME% >> "%OUT%"
findstr /c:"==" /c:"engine up" /c:"median" /c:"exit " /c:"RAM free" /c:"WARNING" "%OUT%"
endlocal
