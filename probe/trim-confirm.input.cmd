@echo off
rem trim-confirm.input.cmd - is "release the file pages after every shard" a real fix or one lucky pass?
rem Repeats the 72B tensor-split load with the lock OFF and VLLM_XPU_TRIM_WS=1, back to back. The fault fires on
rem nearly every lock-off load without the trim, so three clean loads in a row is strong; one fault means it is not a fix.
rem Usage:  cmd /c probe\trim-confirm.input.cmd [repeats]      (default 2; ~3 min each; verdicts in probe\trim-confirm.output.txt)
rem Why (2026-10-11 18:15): build-the-failure.input.cmd case 2 loaded the 72B in parallel for the first time ever with the
rem trim, 157 s to engine-up against 196 s serialized, while case 3 (empty_cache instead) still faulted.
setlocal enabledelayedexpansion
set HERE=%~dp0
set ROOT=%HERE%..
set LOG=%HERE%trim-confirm.output.txt
set N=%1
if "%N%"=="" set N=2
set VLLM_XPU_NO_LOAD_LOCK=1
set VLLM_XPU_TRIM_WS=1
echo run %DATE% %TIME%  72B tensor split, lock OFF, trim ON, %N% repeats > "%LOG%"
for /l %%I in (1,1,%N%) do (
    echo. >> "%LOG%"
    echo == repeat %%I  !TIME! >> "%LOG%"
    call "%ROOT%\server\run-smoke-tp2.input.cmd" FLASH_ATTN Qwen2.5-72B-Instruct-AWQ float16 0.97 2048 > nul 2>&1
    findstr /c:"died before signalling" /c:"0xC0000005" /c:"Model loading took" /c:"engine up" /c:"median" "%ROOT%\server\smoke-tp2-Qwen2.5-72B-Instruct-AWQ.console.log" >> "%LOG%"
)
type "%LOG%"
endlocal
