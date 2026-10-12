@echo off
rem pull-model.input.cmd - download a Hugging Face model into C:\Users\claude\models-hf, console to a file.
rem Usage:  cmd /c server\pull-model.input.cmd Qwen/Qwen2.5-14B-Instruct      (console -> pull-model.console.log)
set HERE=%~dp0
"%HERE%..\.venv312\Scripts\python.exe" -u "%HERE%pull-model.input.py" %1 %2 > "%HERE%pull-model.console.log" 2>&1
echo pull exit %ERRORLEVEL% >> "%HERE%pull-model.console.log"
findstr /c:"pulling" /c:"done:" /c:"Error" /c:"error" /c:"pull exit" "%HERE%pull-model.console.log"
