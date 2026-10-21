@echo off
rem pull-gemma4-big.input.cmd - 2026-10-16 (Carson: "let's try the last 2"): Gemma 4 26B-A4B (MoE, 4B active) and 31B
rem (dense), both with vision, Unsloth UD-Q4_K_XL + F16 projector, into models-b60\gemma4-26b-a4b and gemma4-31b.
rem ~17 + 1.2 and 18.8 + 1.2 GB. Resumable. Console -> pull-gemma4-big.console.log
set PY=C:\Users\claude\scripts\vllm-xpu-windows\.venv312\Scripts\python.exe
set HERE=%~dp0
"%PY%" -u "%HERE%pull-gemma4-big.input.py" > "%HERE%pull-gemma4-big.console.log" 2>&1
echo exit %ERRORLEVEL% >> "%HERE%pull-gemma4-big.console.log"
findstr /c:"pulling" /c:"done" /c:"Error" /c:"error" /c:"exit" "%HERE%pull-gemma4-big.console.log"
