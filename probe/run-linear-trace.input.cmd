@echo off
rem run-linear-trace.input.cmd - trace the Unified Runtime calls of ONE small bf16 F.linear with bias in a plain process.
rem Usage:  cmd /c probe\run-linear-trace.input.cmd      (writes probe\linear-trace.log and probe\linear-trace.summary.txt)
rem Why (2026-10-10): in vLLM the failing launch is oneDNN's "gemm_kernel" (17 args, one local-memory arg of size 1,
rem USM_INDIRECT_ACCESS set). The same shape passes in probe/linear-shapes2.input.py. This shows which kernel(s) oneDNN
rem launches for that shape in a plain process: same kernel passing here means the fault is the process context; a
rem different kernel means vLLM's tensor layout steers oneDNN to a variant that cannot launch on this driver.
set HERE=%~dp0
set SYCL_UR_TRACE=2
"%HERE%..\.venv312\Scripts\python.exe" -c "import torch, torch.nn.functional as F; d=torch.device('xpu:0'); dt=torch.bfloat16; w=torch.randn(896,896,dtype=dt,device=d); b=torch.randn(896,dtype=dt,device=d); x=torch.randn(13,896,dtype=dt,device=d); y=F.linear(x,w,b); torch.xpu.synchronize(); print('linear ok', tuple(y.shape))" > "%HERE%linear-trace.log" 2>&1
echo == kernel names launched: > "%HERE%linear-trace.summary.txt"
findstr /c:"UR_KERNEL_INFO_FUNCTION_NAME" "%HERE%linear-trace.log" | findstr /c:"pPropValue = " >> "%HERE%linear-trace.summary.txt"
echo == exec info set on kernels: >> "%HERE%linear-trace.summary.txt"
findstr /c:"urKernelSetExecInfo" "%HERE%linear-trace.log" | findstr /c:"<---" >> "%HERE%linear-trace.summary.txt"
echo == launches and their results: >> "%HERE%linear-trace.summary.txt"
findstr /c:"urEnqueueKernelLaunch" "%HERE%linear-trace.log" | findstr /c:"<---" >> "%HERE%linear-trace.summary.txt"
type "%HERE%linear-trace.summary.txt"
