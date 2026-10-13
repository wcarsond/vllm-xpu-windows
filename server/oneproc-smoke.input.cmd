@echo off
rem oneproc-smoke.input.cmd - run oneproc-smoke.input.py for the given executors under a 30 min timeout each, then compare
rem their token ids. Console logs: server\oneproc-smoke-<executor>-<mode>.console.log.
rem Usage:  cmd /c server\oneproc-smoke.input.cmd [eager|graphs] ["oneproc mp"]
rem Why (2026-10-13): stage 3 of the one-process executor; see the .py.
setlocal
set HERE=%~dp0
call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat" > nul 2>&1
set CC=cl
set PATH=%HERE%..\.venv312\Library\bin;%PATH%
set PYTHONFAULTHANDLER=1
set ZE_AFFINITY_MASK=
set ONEAPI_DEVICE_SELECTOR=
set MODE=%~1
if "%MODE%"=="" set MODE=eager
set EXECS=%~2
if "%EXECS%"=="" set EXECS=oneproc mp
for %%E in (%EXECS%) do (
    powershell -NoProfile -Command "$p = Start-Process '%HERE%..\.venv312\Scripts\python.exe' -ArgumentList '-u','%HERE%oneproc-smoke.input.py','%%E','%MODE%' -NoNewWindow -PassThru -RedirectStandardOutput '%HERE%oneproc-smoke-%%E-%MODE%.console.log' -RedirectStandardError '%HERE%oneproc-smoke-%%E-%MODE%.console.err.log'; if (-not $p.WaitForExit(1800000)) { taskkill /T /F /PID $p.Id | Out-Null; 'TIMEOUT %%E after 30 min (killed with its workers)' }"
    type "%HERE%oneproc-smoke-%%E-%MODE%.output.txt" 2>nul
)
"%HERE%..\.venv312\Scripts\python.exe" -c "import json,pathlib; h=pathlib.Path(r'%HERE%.'); r={e: [l for l in (h/f'oneproc-smoke-{e}-%MODE%.output.txt').read_text().splitlines() if l.startswith('ids ')] for e in '%EXECS%'.split()}; v=[json.loads(x[0][4:]) if x else None for x in r.values()]; print('token ids identical across executors:', len(v) > 1 and all(x == v[0] and x is not None for x in v))"
