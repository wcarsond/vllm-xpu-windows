@echo off
rem run-smoke-triton.input.cmd - the first-light smoke with vLLM's Triton attention backend instead of the cutlass flash-attention kernels.
rem Usage:  cmd /c server\run-smoke-triton.input.cmd      (console -> server\smoke-offline-triton.console.log; result -> smoke-offline.output.txt)
rem Why (2026-10-10): every compiled cutlass chunk-prefill kernel kills the device on its first launch on this Windows driver
rem (PORT-NOTES, 22:13). The Triton backend's attention kernels are compiled at run time by the driver's own IGC from
rem Triton-XPU's SPIR-V, the way the smoke step's Triton prep kernels already ran. KV writes still go through our
rem reshape_and_cache_flash (tests pass). If this lights up, the dual-card work can proceed while the cutlass fault is chased.
set HERE=%~dp0
call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat" > nul 2>&1
set CC=cl
rem VLLM_ATTENTION_BACKEND is ignored by this vLLM (22:33 run still said "Using Flash Attention backend");
rem smoke-offline.input.py reads SMOKE_ATTN_BACKEND and passes it as the attention_backend engine argument.
set SMOKE_ATTN_BACKEND=TRITON_ATTN
"%HERE%..\.venv312\Scripts\python.exe" -u "%HERE%smoke-offline.input.py" > "%HERE%smoke-offline-triton.console.log" 2>&1
echo smoke exit %ERRORLEVEL% >> "%HERE%smoke-offline-triton.console.log"
findstr /c:"Using" /c:"backend" /c:"engine up" /c:"generate:" /c:"text:" /c:"Error" /c:"error:" /c:"smoke exit" "%HERE%smoke-offline-triton.console.log"
