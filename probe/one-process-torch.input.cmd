@echo off
rem one-process-torch.input.cmd - run T1, T2, T3 of one-process-torch.input.py, each alone under a 200 s hard timeout.
rem Usage:  cmd /c probe\one-process-torch.input.cmd           (writes probe\one-process-torch.output.txt)
rem Why (2026-10-13): see the .py - whether torch itself blocks one process from driving both cards.
setlocal
set HERE=%~dp0
set PATH=%HERE%..\.venv312\Library\bin;%PATH%
set OUT=%HERE%one-process-torch.output.txt
echo run %DATE% %TIME% > "%OUT%"
for %%T in (T1 T2 T3) do (
    powershell -NoProfile -Command "$p = Start-Process '%HERE%..\.venv312\Scripts\python.exe' -ArgumentList '-u','%HERE%one-process-torch.input.py','%%T' -NoNewWindow -PassThru -RedirectStandardOutput '%HERE%one-process-torch.%%T.log' -RedirectStandardError '%HERE%one-process-torch.%%T.err.log'; if (-not $p.WaitForExit(200000)) { Stop-Process -Id $p.Id -Force; Add-Content '%OUT%' '%%T: TIMEOUT after 200 s (killed)' } elseif ($p.ExitCode -ne 0) { Add-Content '%OUT%' ('%%T: exit ' + $p.ExitCode + ' - ' + ((Get-Content '%HERE%one-process-torch.%%T.err.log' | Where-Object { $_ -match 'Error' } | Select-Object -Last 1))) }"
)
type "%OUT%"
