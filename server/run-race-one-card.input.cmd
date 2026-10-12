@echo off
rem run-race-one-card.input.cmd - race any model folder under C:\Users\claude\models-hf on one B60 (card 0), console to a file.
rem Usage:  cmd /c serverun-race-one-card.input.cmd <model-folder> [offload_gb] [gpu_util] [dtype] [nopin]
rem         nopin (13:30) = spill without pinned memory and without the card view of host memory (vLLM copies each
rem         spilled weight to the card per forward); the pinned/UVA path dies on this driver, see PORT-NOTES.
rem         e.g.  run-race-one-card.input.cmd Qwen2.5-32B-Instruct-AWQ 0 0.95 float16
rem               run-race-one-card.input.cmd Qwen2.5-14B-Instruct 10 0.95
rem Why (2026-10-11 13:20): the one-card half of every "1 card = X, 2 cards = Y" line, by name, no environment to type.
rem Wraps run-race-7b-vllm.input.cmd, which names its files race-7b-vllm.FLASH_ATTN-<model-folder>[-offloadN].console.log
rem and .output.txt, so no result overwrites another. Offload = GB of weights spilled to pinned RAM (vLLM cpu_offload_gb).
set MODEL=%1
if "%MODEL%"=="" (echo model folder name required, e.g. Qwen2.5-32B-Instruct-AWQ & exit /b 2)
set SMOKE_MODEL=C:\Users\claude\models-hf\%MODEL%
if not exist "%SMOKE_MODEL%\config.json" (echo no model at %SMOKE_MODEL% & exit /b 2)
set SMOKE_OFFLOAD_GB=%2
if "%SMOKE_OFFLOAD_GB%"=="" set SMOKE_OFFLOAD_GB=0
set SMOKE_GPU_UTIL=%3
if "%SMOKE_GPU_UTIL%"=="" set SMOKE_GPU_UTIL=0.85
set SMOKE_DTYPE=%4
if "%SMOKE_DTYPE%"=="" set SMOKE_DTYPE=bfloat16
if /i "%5"=="nopin" (set VLLM_WEIGHT_OFFLOADING_DISABLE_PIN_MEMORY=1& set VLLM_WEIGHT_OFFLOADING_DISABLE_UVA=1& set SMOKE_TAG=nopin)
echo one card: %MODEL%  offload %SMOKE_OFFLOAD_GB% GB  util %SMOKE_GPU_UTIL%  dtype %SMOKE_DTYPE%  tag %SMOKE_TAG%
call "%~dp0run-race-7b-vllm.input.cmd"
