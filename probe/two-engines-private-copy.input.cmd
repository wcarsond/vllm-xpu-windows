@echo off
rem two-engines-private-copy.input.cmd - does the private-copy loader fix also cover trigger 1 (two MASKED processes, one per
rem card, loading the same checkpoint)? Runs probe\two-engines.input.py twice: default, then VLLM_XPU_PRIVATE_COPY=1.
rem Usage:  cmd /c probe\two-engines-private-copy.input.cmd      (writes probe\two-engines-private-copy.output.txt)
rem Why (2026-10-11 22:35): per-item batch work wants one independent worker per card (ZE_AFFINITY_MASK=0 and =1) on a model
rem that fits one card. That is exactly the "two 0.5B engines, one card each" cell that faulted earlier today. A one-card
rem worker has tensor-parallel size 1, so the private copy is OFF by default; =1 forces it. If the forced run passes where
rem the default faults, the job sets it and the two workers may start together; if not, they start one after the other.
set HERE=%~dp0
set OUT=%HERE%two-engines-private-copy.output.txt
call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat" > nul 2>&1
set CC=cl
set PATH=%HERE%..\.venv312\Library\bin;%PATH%
echo run %DATE% %TIME% > "%OUT%"
echo --- default (private copy off for tensor-parallel size 1) >> "%OUT%"
set VLLM_XPU_PRIVATE_COPY=
"%HERE%..\.venv312\Scripts\python.exe" -u "%HERE%two-engines.input.py" >> "%OUT%" 2>&1
echo --- VLLM_XPU_PRIVATE_COPY=1 >> "%OUT%"
set VLLM_XPU_PRIVATE_COPY=1
"%HERE%..\.venv312\Scripts\python.exe" -u "%HERE%two-engines.input.py" >> "%OUT%" 2>&1
echo --- VLLM_XPU_PRIVATE_COPY=1 again >> "%OUT%"
"%HERE%..\.venv312\Scripts\python.exe" -u "%HERE%two-engines.input.py" >> "%OUT%" 2>&1
findstr /c:"---" /c:"engines" /c:"Error" "%OUT%"
