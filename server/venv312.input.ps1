# venv312.input.ps1 - make .venv312 (Python 3.12) with torch 2.14.1+xpu, the venv rung 5 runs in.
# Usage:  powershell -NoProfile -File venv312.input.ps1     (logs to venv312.output.log)
#
# Why (2026-10-10, rung 5): vLLM main needs Python >= 3.11; the existing .venv is 3.10 and is busy building the kernels.
# This venv gets the same torch build (2.14.1+xpu, oneAPI 2026.1 runtimes) so the kernels compiled against the 3.10
# venv's torch link against identical DLLs here. Only torch is pinned; torchvision/torchaudio come later if an import
# asks for them, so nothing drifts in unasked.
$here = Split-Path -Parent $MyInvocation.MyCommand.Path
$log  = "$here\venv312.output.log"
$venv = "$here\..\.venv312"
$py   = "$venv\Scripts\python.exe"
"run $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')" | Out-File $log -Encoding utf8

if (-not (Test-Path $py)) {
    & py -3.12 -m venv $venv 2>&1 | ForEach-Object { "$_" } | Out-File $log -Append -Encoding utf8
    "venv created at $venv" | Out-File $log -Append -Encoding utf8
} else { "venv already at $venv" | Out-File $log -Append -Encoding utf8 }

$t0 = Get-Date
& $py -m pip install --quiet --upgrade pip 2>&1 | Out-Null
& $py -m pip install --quiet torch==2.14.1 --index-url https://download.pytorch.org/whl/xpu 2>&1 | ForEach-Object { "$_" } | Out-File $log -Append -Encoding utf8
"torch install exit $LASTEXITCODE in $([math]::Round(((Get-Date)-$t0).TotalSeconds)) s" | Out-File $log -Append -Encoding utf8
# torchvision: vLLM's multimodal code imports it unconditionally (first light attempt 4, 21:05, died on it after the
# profiling pass had already run on the card). The xpu index resolves the version that matches torch 2.14.1.
& $py -m pip install --quiet torchvision --index-url https://download.pytorch.org/whl/xpu 2>&1 | ForEach-Object { "$_" } | Out-File $log -Append -Encoding utf8
"torchvision install exit $LASTEXITCODE" | Out-File $log -Append -Encoding utf8
& $py -m pip list 2>$null | Select-String -Pattern '^torchvision ' | ForEach-Object { "$_" } | Out-File $log -Append -Encoding utf8

# Proof: version, both cards, the oneAPI runtime versions that the kernels must match.
& $py -c "import torch; print('torch', torch.__version__, 'xpu devices', torch.xpu.device_count(), 'version.xpu', torch.version.xpu)" 2>&1 | ForEach-Object { "$_" } | Out-File $log -Append -Encoding utf8
& $py -m pip list 2>$null | Select-String -Pattern 'intel-sycl-rt|intel-cmplr-lib-rt|^mkl |dpcpp-cpp-rt|^torch ' | ForEach-Object { "$_" } | Out-File $log -Append -Encoding utf8
Get-Content $log
