@echo off
rem start-server.input.cmd - vLLM OpenAI-compatible server on BOTH B60s (tensor split), reachable from the house LAN.
rem Usage:  cmd /c server\start-server.input.cmd [model-folder] [port] [max_len] [util] [tool-parser] [reasoning-parser] [think]
rem         model-folder under C:\Users\claude\models-hf (default Qwen3.5-27B-GPTQ-Int4);  port default 11434 (the port the
rem         llama.cpp router used, so clients need no change);  max_len default 65536;  util default 0.90;
rem         tool-parser default qwen3_xml (Qwen3.5; use hermes for Qwen2.5, granite4 for Granite);  reasoning-parser default
rem         qwen3 (- for none);  think 0|1 (default 0 = answers without a thinking block, faster for an editor).
rem         e.g.  start-server.input.cmd                                   (27B, port 11434)
rem               start-server.input.cmd Qwen2.5-32B-Instruct-AWQ 11434 32768 0.95 hermes -
rem The model is served under the fixed name "local", so VS Code's languageModels.json points at one id whatever is loaded:
rem         url http://192.168.0.21:11434/v1   model id "local"
rem Console: server\server.console.log (the window is hidden). Stop with server\stop-server.input.cmd.
rem Why (2026-10-11 23:45, Carson: "use them in VS Code ... the model shouldn't matter ... same port, if it's all fast we
rem won't need the old one"): vLLM on two cards measured 3-5x the llama.cpp router on the same 7B and holds depth to 32K.
set HERE=%~dp0
set MODEL=%1
if "%MODEL%"=="" set MODEL=Qwen3.5-27B-GPTQ-Int4
set PORT=%2
if "%PORT%"=="" set PORT=11434
set MAXLEN=%3
if "%MAXLEN%"=="" set MAXLEN=65536
set UTIL=%4
if "%UTIL%"=="" set UTIL=0.90
set TOOLP=%5
if "%TOOLP%"=="" set TOOLP=qwen3_xml
set REASP=%6
if "%REASP%"=="" set REASP=qwen3
set THINK=%7
if "%THINK%"=="" set THINK=0
call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat" > nul 2>&1
set CC=cl
set PATH=%HERE%..\.venv312\Library\bin;%PATH%
set VLLM_WORKER_MULTIPROC_METHOD=spawn
set PYTHONFAULTHANDLER=1
set ZE_AFFINITY_MASK=
set ONEAPI_DEVICE_SELECTOR=
set REAS=
if not "%REASP%"=="-" set REAS=--reasoning-parser %REASP%
set TMPL=
if "%THINK%"=="0" set TMPL=--default-chat-template-kwargs "{\"enable_thinking\": false}"
echo %DATE% %TIME% starting vLLM: %MODEL% on port %PORT%, max_len %MAXLEN%, util %UTIL%, tools %TOOLP%, reasoning %REASP%, think %THINK% > "%HERE%server.console.log"
"%HERE%..\.venv312\Scripts\vllm.exe" serve "C:\Users\claude\models-hf\%MODEL%" --host 0.0.0.0 --port %PORT% ^
  --served-model-name local --tensor-parallel-size 2 --distributed-executor-backend mp --enforce-eager ^
  --max-model-len %MAXLEN% --gpu-memory-utilization %UTIL% --limit-mm-per-prompt "{\"image\": 4}" ^
  --enable-auto-tool-choice --tool-call-parser %TOOLP% %REAS% %TMPL% >> "%HERE%server.console.log" 2>&1
echo server exit %ERRORLEVEL% >> "%HERE%server.console.log"
