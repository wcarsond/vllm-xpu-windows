@echo off
rem graph-ab.input.cmd - one card, a small model, eager against compiled with graph capture, to measure how much of a decode
rem token is CPU-side overhead (Python, kernel launches, gaps between kernels) rather than card work.
rem Usage:  cmd /c server\graph-ab.input.cmd [model-folder]       (default Qwen2.5-0.5B-Instruct; writes graph-ab.output.txt)
rem Why (2026-10-13): the int4 kernel reads 88-100% of the card's bandwidth at real layer shapes (probe\int4-gemv), yet 4-bit
rem models decode at ~59% of the ceiling on two cards: with 4x less matrix work per layer, fixed per-kernel costs become most
rem of the token. A 0.5B model makes that regime extreme. If graphs cut its token time a lot, graph capture is the lever for
rem the 4-bit models (two-card capture is off on Windows today); if not, the cost is on the card, not the CPU.
setlocal
set HERE=%~dp0
set M=%1
if "%M%"=="" set M=Qwen2.5-0.5B-Instruct
set SMOKE_MODEL=C:\Users\claude\models-hf\%M%
set OUT=%HERE%graph-ab.output.txt
echo run %DATE% %TIME%  %M%, one card > "%OUT%"
set SMOKE_EAGER=1
call "%HERE%run-race-7b-vllm.input.cmd" FLASH_ATTN >> "%OUT%" 2>&1
set SMOKE_EAGER=0
call "%HERE%run-race-7b-vllm.input.cmd" FLASH_ATTN >> "%OUT%" 2>&1
set SMOKE_EAGER=1
call "%HERE%run-race-7b-vllm.input.cmd" FLASH_ATTN >> "%OUT%" 2>&1
set SMOKE_EAGER=0
call "%HERE%run-race-7b-vllm.input.cmd" FLASH_ATTN >> "%OUT%" 2>&1
type "%OUT%"
