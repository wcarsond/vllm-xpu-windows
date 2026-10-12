@echo off
rem tidy-publish.input.cmd - re-apply .gitignore to everything already tracked, fold the result into the last commit, force-push.
rem Usage:  cmd /c tidy-publish.input.cmd      (only for this folder's own repo, wcarsond/vllm-xpu-windows)
rem Why (2026-10-11): the first publish tracked two 40 MB wheels and a folder of extracted DLL images from kernels\ before
rem the ignore list covered them. Rewriting the one commit keeps the repo small; this is the backup of the scripts and the
rem record, nobody has built on its history.
setlocal
set ROOT=C:\Users\claude\scripts\vllm-xpu-windows
git -C "%ROOT%" rm -r -q --cached .
git -C "%ROOT%" add -A
git -C "%ROOT%" -c user.name=Carson -c user.email=wcarsond@gmail.com commit -q --amend --no-edit
git -C "%ROOT%" ls-files | find /c /v ""
git -C "%ROOT%" count-objects -vH | findstr size-pack
git -C "%ROOT%" push --force -u origin main
endlocal
