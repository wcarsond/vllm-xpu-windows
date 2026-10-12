@echo off
rem publish.input.cmd - commit everything in this folder (scripts, probes, patches, PORT-NOTES) and push it to GitHub.
rem Usage:  cmd /c publish.input.cmd "commit message"      (first run creates the private repo wcarsond/vllm-xpu-windows)
rem Why (2026-10-11, Carson: "yeah we need to push this to github ... this is days of work"): the three source checkouts are
rem their own repos (pushed to wcarsond/vllm, wcarsond/vllm-xpu-kernels, wcarsond/sycl-tla); this folder holds the record and
rem every script, and it was in no repo at all. Author line as the other backups: Carson, with the assistant as co-author.
rem Runs with absolute paths on purpose: an earlier attempt with a relative cd re-initialised the vLLM worktree instead.
setlocal
set ROOT=C:\Users\claude\scripts\vllm-xpu-windows
set MSG=%~1
if "%MSG%"=="" set MSG=update the record and scripts
if not exist "%ROOT%\.git" (
    git -C "%ROOT%" init -q -b main
)
git -C "%ROOT%" add -A
git -C "%ROOT%" -c user.name=Carson -c user.email=wcarsond@gmail.com commit -q -m "%MSG%" -m "Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
git -C "%ROOT%" log --format="%%h %%s" -1
git -C "%ROOT%" remote get-url origin > nul 2>&1
if errorlevel 1 (
    gh repo create wcarsond/vllm-xpu-windows --private --description "vLLM on Windows for two Arc Pro B60s: scripts, probes, patches and the port record" --source "%ROOT%" --remote origin --push
) else (
    git -C "%ROOT%" push -u origin main
)
endlocal
