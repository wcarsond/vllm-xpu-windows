@echo off
rem threshold-sweep.input.cmd - at what retained device volume does the mapped-read fault start, and does trimming the
rem working set between shards prevent it? Runs the standalone reproducer (probe\mmap-readers.input.py) back to back,
rem never two at once, because every case needs both cards to itself.
rem Usage:  cmd /c probe\threshold-sweep.input.cmd      (writes probe\threshold-sweep.output.txt, ~3 min)
rem Why (2026-10-11 17:25): vLLM's 72B workers fault at 19.55 GiB retained with the whole checkpoint mapped and no
rem overflow. If the fault only appears above some retained threshold, both workers can load in parallel up to it and
rem serialize only the remainder - Carson's look-ahead carry - instead of the whole load. And if dropping the file pages
rem between shards avoids it, that is a fix with no serialization at all.
setlocal
set HERE=%~dp0
set PY=%HERE%..\.venv312\Scripts\python.exe
set LOG=%HERE%threshold-sweep.output.txt
set MMAP_DEVICE_PER_READER=1
set MMAP_NO_MASK=1
set MMAP_COPY_INTO=1
echo run %DATE% %TIME%  two readers, one card each, whole checkpoint mapped > "%LOG%"
for %%C in (4 8 12 16 20) do (
    set MMAP_RETAIN_CAP_GB=%%C
    set MMAP_TRIM_WS=
    echo. >> "%LOG%"
    echo == retain cap %%C GiB, no trim >> "%LOG%"
    "%PY%" "%HERE%mmap-readers.input.py" Qwen2.5-72B-Instruct-AWQ 2 1 2>&1 | findstr /c:"reader" /c:"wall" >> "%LOG%"
)
set MMAP_RETAIN_CAP_GB=20
set MMAP_TRIM_WS=1
echo. >> "%LOG%"
echo == retain cap 20 GiB, EmptyWorkingSet after every shard >> "%LOG%"
"%PY%" "%HERE%mmap-readers.input.py" Qwen2.5-72B-Instruct-AWQ 2 1 2>&1 | findstr /c:"reader" /c:"wall" >> "%LOG%"
type "%LOG%"
endlocal
