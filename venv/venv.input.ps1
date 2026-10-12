# venv.input.ps1 - build the project's Python environment with PyTorch for Intel cards (XPU).
# Usage:  powershell -File venv.input.ps1
# Creates C:\Users\claude\scripts\vllm-xpu-windows\.venv and logs to venv.output.log beside this file.
# Nothing is installed system-wide. Delete the .venv folder to undo.
$here = Split-Path -Parent $MyInvocation.MyCommand.Path
$root = Split-Path -Parent $here
$log  = "$here\venv.output.log"
$py   = "$root\.venv\Scripts\python.exe"

function Run($text, $block) {
    "== $text" | Out-File $log -Append -Encoding utf8
    & $block *>&1 | ForEach-Object { "$_" } | Out-File $log -Append -Encoding utf8
    if ($LASTEXITCODE -ne 0) { "FAILED: $text (exit $LASTEXITCODE)" | Out-File $log -Append -Encoding utf8; Write-Output "FAILED: $text"; exit 1 }
}

Run 'create venv'        { python -m venv "$root\.venv" }
Run 'upgrade pip'        { & $py -m pip install --no-cache-dir --upgrade pip }
Run 'install torch xpu'  { & $py -m pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/xpu }
Run 'install numpy'      { & $py -m pip install --no-cache-dir numpy }
Run 'show versions'      { & $py -m pip list }
Write-Output "venv ready: $py"
