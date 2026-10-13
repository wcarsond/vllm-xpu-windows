@echo off
rem oneproc-exchange-torch.input.cmd - run oneproc-exchange-torch.input.py under a 200 s hard timeout (it dumps its
rem threads' stacks at 150 s if stuck). Output: probe\oneproc-exchange-torch.output.txt (+ .log / .err.log beside it).
rem Usage:  cmd /c probe\oneproc-exchange-torch.input.cmd [eager|graph|both]
rem Why (2026-10-13): stage 2 of the one-process executor; see the .py.
setlocal
set HERE=%~dp0
set PATH=%HERE%..\.venv312\Library\bin;%PATH%
set ZE_AFFINITY_MASK=
set ONEAPI_DEVICE_SELECTOR=
set MODE=%~1
if "%MODE%"=="" set MODE=both
powershell -NoProfile -Command "$p = Start-Process '%HERE%..\.venv312\Scripts\python.exe' -ArgumentList '-u','%HERE%oneproc-exchange-torch.input.py','%MODE%' -NoNewWindow -PassThru -RedirectStandardOutput '%HERE%oneproc-exchange-torch.log' -RedirectStandardError '%HERE%oneproc-exchange-torch.err.log'; if (-not $p.WaitForExit(200000)) { Stop-Process -Id $p.Id -Force; 'TIMEOUT after 200 s (killed)' }"
type "%HERE%oneproc-exchange-torch.log"
powershell -NoProfile -Command "Get-Content '%HERE%oneproc-exchange-torch.err.log' | Select-String -Pattern 'Error|error|Traceback|File ' | Select-Object -Last 25"
