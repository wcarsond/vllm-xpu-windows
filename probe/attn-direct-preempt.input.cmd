@echo off
rem attn-direct-preempt.input.cmd - the known-failing kernels (attention, mhc) with the compute runtime's preemption mode forced, no rebuild.
rem Usage:  cmd /c probe\attn-direct-preempt.input.cmd      (writes probe\attn-direct-preempt.output.txt)
rem Why (2026-10-10): the only kernels that die are the ones with 256 GRF AND local memory/barriers (PORT-NOTES 22:25). That is
rem the thread state mid-thread preemption must save and restore; Windows keeps mid-thread preemption on for compute.
rem NEO reads ForcePreemptionMode when NEOReadDebugKeys=1: 1 disabled, 2 mid-batch, 3 thread-group, 4 mid-thread (default).
rem If mode 3 or 1 makes these kernels run, the fault is the driver's preemption handling and the fix is an environment
rem variable in every run script; if not, that theory is dead too.
set HERE=%~dp0
set PY=%HERE%..\.venv312\Scripts\python.exe
set T=%HERE%..\src\vllm-xpu-kernels\tests\mhc\test_mhc.py
set LOG=%HERE%attn-direct-preempt.output.txt
echo run %DATE% %TIME% > "%LOG%"
set NEOReadDebugKeys=1
for %%M in (3 1 2) do (
  set ForcePreemptionMode=%%M
  echo == ForcePreemptionMode=%%M  attention np_full_8 >> "%LOG%"
  "%PY%" "%HERE%attn-direct.input.py" np_full_8 >> "%LOG%" 2>&1
  echo == ForcePreemptionMode=%%M  attention pg64_causal_13 >> "%LOG%"
  "%PY%" "%HERE%attn-direct.input.py" pg64_causal_13 >> "%LOG%" 2>&1
  echo == ForcePreemptionMode=%%M  mhc tests >> "%LOG%"
  "%PY%" -m pytest "%T%" -q -x --tb=line -p no:cacheprovider 2>&1 | findstr /c:"passed" /c:"failed" /c:"Error" >> "%LOG%"
)
echo done %TIME% >> "%LOG%"
type "%LOG%"
exit /b 0
