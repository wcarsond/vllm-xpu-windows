# prep.input.ps1 - get src\vllm-xpu-kernels ready for a Windows build: a branch to hold the port, and the build tools in .venv.
# Usage:  powershell -NoProfile -File prep.input.ps1     (logs to prep.output.log)
#
# Why (2026-10-10, rung 4): every Windows edit to the kernel repo goes on branch `windows`, so `git diff main` is
# the port and nothing else. The repo's requirements.txt pins torch==2.14.0+xpu, which would REPLACE the 2.14.1
# already in .venv, so only the build tools are installed from it, by name.
$here = Split-Path -Parent $MyInvocation.MyCommand.Path
$log  = "$here\prep.output.log"
$repo = "$here\..\src\vllm-xpu-kernels"
$py   = "$here\..\.venv\Scripts\python.exe"
"run $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')" | Out-File $log -Encoding utf8

$branches = git -C $repo branch --list windows
if (-not $branches) { git -C $repo checkout -b windows | Out-File $log -Append -Encoding utf8 }
else { git -C $repo checkout windows | Out-File $log -Append -Encoding utf8 }
"branch: $(git -C $repo branch --show-current) at $(git -C $repo rev-parse --short HEAD)" | Out-File $log -Append -Encoding utf8

# Build tools from requirements.txt, minus torch/triton/tests/benchmarks.
& $py -m pip install --quiet regex 'cmake>=3.26' 'packaging>=24.2' scipy 'setuptools-scm>=8' 'setuptools>=77.0.3,<80.0.0' wheel build ninja psutil |
    ForEach-Object { "$_" } | Out-File $log -Append -Encoding utf8
& $py -m pip list 2>$null | Select-String -Pattern '^(cmake|ninja|setuptools|setuptools-scm|psutil|scipy|torch) ' | ForEach-Object { "$_" } | Out-File $log -Append -Encoding utf8
Get-Content $log
