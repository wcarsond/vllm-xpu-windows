@echo off
rem threads-hang.input.cmd - run every case of threads-hang.input.py alone under a 90 s hard timeout; a case that hangs
rem dumps its threads' stacks into probe\threads-hang.<case>.err.log at 60 s. Summary in probe\threads-hang.output.txt.
rem Usage:  cmd /c probe\threads-hang.input.cmd ["E M W S F"]
rem Why (2026-10-13): see the .py - a one-process executor needs one thread per card.
setlocal
set HERE=%~dp0
set PATH=%HERE%..\.venv312\Library\bin;%PATH%
set ZE_AFFINITY_MASK=
set ONEAPI_DEVICE_SELECTOR=
set OUT=%HERE%threads-hang.output.txt
set CASES=%~1
if "%CASES%"=="" set CASES=E M W S F
echo run %DATE% %TIME% > "%OUT%"
for %%T in (%CASES%) do (
    powershell -NoProfile -Command "$p = Start-Process '%HERE%..\.venv312\Scripts\python.exe' -ArgumentList '-u','%HERE%threads-hang.input.py','%%T' -NoNewWindow -PassThru -RedirectStandardOutput '%HERE%threads-hang.%%T.log' -RedirectStandardError '%HERE%threads-hang.%%T.err.log'; if (-not $p.WaitForExit(90000)) { Stop-Process -Id $p.Id -Force; Add-Content '%OUT%' '%%T: TIMEOUT after 90 s (killed)' } elseif ($p.ExitCode -ne 0) { Add-Content '%OUT%' ('%%T: exit ' + $p.ExitCode + ' (hung; stacks in threads-hang.%%T.err.log)') }"
)
type "%OUT%"
