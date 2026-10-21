@echo off
rem stop-server.input.cmd - stop the vLLM server started by start-server.input.cmd (the vllm.exe process and its two workers).
rem Usage:  cmd /c server\stop-server.input.cmd
rem Why (2026-10-11): the server runs in a hidden window; this is the one named way to end it and free both cards.
for /f "tokens=2 delims=," %%P in ('tasklist /fi "imagename eq vllm.exe" /fo csv /nh') do taskkill /pid %%~P /t /f > nul 2>&1
rem the workers are python.exe children of vllm.exe; /t above takes them, but a worker that outlived its parent would not be
for /f "tokens=2 delims=," %%P in ('wmic process where "name='python.exe' and commandline like '%%multiproc%%'" get processid /format:csv 2^>nul ^| findstr /r "[0-9]"') do taskkill /pid %%P /f > nul 2>&1
echo %DATE% %TIME% stopped >> "%~dp0server.console.log"
tasklist /fi "imagename eq vllm.exe" /nh | findstr /i vllm > nul && echo vllm.exe still running || echo vLLM server stopped
