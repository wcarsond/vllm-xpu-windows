@echo off
rem threads-gil.input.cmd - run each case of threads-gil.input.py alone with a 180 s hard timeout, appending one line each to
rem probe\threads-gil.output.txt. A case that hangs is killed and recorded as TIMEOUT instead of stalling the rest.
rem Usage:  cmd /c probe\threads-gil.input.cmd
rem Why (2026-10-13): the all-in-one version hung with no output; this isolates which case hangs.
setlocal EnableDelayedExpansion
set HERE=%~dp0
set PATH=%HERE%..\.venv312\Library\bin;%PATH%
set OUT=%HERE%threads-gil.output.txt
echo run %DATE% %TIME%  28 layers, 200 steps, us per step (lower is better) > "%OUT%"
for %%N in (eager graph) do (
    for %%C in (A B C) do (
        powershell -NoProfile -Command "$p = Start-Process '%HERE%..\.venv312\Scripts\python.exe' -ArgumentList '-u','%HERE%threads-gil.input.py','%%N','%%C' -NoNewWindow -PassThru -RedirectStandardOutput '%HERE%threads-gil.%%N%%C.log' -RedirectStandardError '%HERE%threads-gil.%%N%%C.err.log'; if (-not $p.WaitForExit(180000)) { Get-CimInstance Win32_Process -Filter \"name='python.exe'\" | Where-Object { $_.CommandLine -match 'threads-gil' } | ForEach-Object { Stop-Process -Id $_.ProcessId -Force }; Add-Content '%OUT%' '%%N %%C: TIMEOUT after 180 s' }"
    )
)
type "%OUT%"
