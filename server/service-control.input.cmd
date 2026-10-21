@echo off
rem service-control.input.cmd - start, stop, restart or show the vllm-b60 service (the vLLM router on 11434). start/stop/restart
rem NEED ADMIN: run from an admin shell, or through the Pi's SSH session, which Windows gives an admin token. status needs nothing.
rem Usage:  cmd /c server\service-control.input.cmd start|stop|restart|status
rem Why (2026-10-12): one named command instead of nssm one-liners quoted through two SSH hops.
setlocal
set NSSM=C:\ollama\nssm.exe
if /i "%1"=="status" goto status
if /i "%1"=="stop" goto stop
if /i "%1"=="start" goto start
if /i "%1"=="restart" goto restart
echo usage: service-control.input.cmd start / stop / restart / status
exit /b 1

:stop
"%NSSM%" stop vllm-b60
ping -n 4 127.0.0.1 > nul
goto status

:start
"%NSSM%" start vllm-b60
ping -n 4 127.0.0.1 > nul
goto status

:restart
"%NSSM%" restart vllm-b60
ping -n 6 127.0.0.1 > nul
goto status

:status
"%NSSM%" status vllm-b60
tasklist /fi "imagename eq vllm.exe" /nh | findstr /i vllm > nul
if errorlevel 1 (echo no vllm.exe loaded) else (echo vllm.exe running)
powershell -NoProfile -Command "try { (Invoke-RestMethod http://localhost:11434/router/status -TimeoutSec 5) | ConvertTo-Json -Compress } catch { 'router not answering' }; 'free RAM GB: ' + [math]::Round((Get-CimInstance Win32_OperatingSystem).FreePhysicalMemory/1MB,1)"
endlocal
