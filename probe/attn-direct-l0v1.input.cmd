@echo off
rem attn-direct-l0v1.input.cmd - the same failing attention cases on the legacy Level Zero adapter instead of the V2 adapter torch picks by default.
rem Usage:  cmd /c probe\attn-direct-l0v1.input.cmd      (writes probe\attn-direct-l0v1.output.txt)
rem Why (2026-10-10): torch reports the platform as "Unified Runtime over Level-Zero V2". The V2 adapter is the newer code path for
rem immediate command lists, events and kernel arguments; if a faulting case passes on the legacy adapter the defect is in how the
rem launch is assembled, not in the kernel. Cheap: three processes, a few seconds each.
set HERE=%~dp0
set PY=%HERE%..\.venv312\Scripts\python.exe
set LOG=%HERE%attn-direct-l0v1.output.txt
set UR_LOADER_USE_LEVEL_ZERO_V2=0
set SYCL_UR_USE_LEVEL_ZERO_V2=0
echo run %DATE% %TIME%  UR_LOADER_USE_LEVEL_ZERO_V2=0 > "%LOG%"
"%PY%" -c "import torch; print('platform:', torch.xpu.get_device_properties(0).platform_name)" >> "%LOG%" 2>&1
for %%C in (np_win64_8 np_full_8 pg64_causal_13) do (
  echo == %%C >> "%LOG%"
  "%PY%" "%HERE%attn-direct.input.py" %%C >> "%LOG%" 2>&1
)
type "%LOG%"
exit /b 0
