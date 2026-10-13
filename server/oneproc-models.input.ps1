# oneproc-models.input.ps1 - the smoke test (full decode graphs) on other models, one-process executor against the
# two-process one: decode tok/s and whether the greedy token ids match.
# Usage:  powershell -File server\oneproc-models.input.ps1 ["Qwen3.5-27B-GPTQ-Int4,Granite-4.2-30B-GPTQ-INT4"] ["oneproc,mp"]
#         (writes server\oneproc-models.output.txt; per run: oneproc-smoke-<model>-<exec>-graphs.output.txt / .console.log)
# Model names are folders under C:\Users\claude\models-hf or full paths; a name ending in "@float16" runs in float16.
# Why (2026-10-13): the 7B gained 18% decode in one process; the service's everyday models are hybrids (Gated DeltaNet,
# Mamba) with their own module-level state, which is where a shared process can bite.
param([string]$Models = "Qwen3.5-27B-GPTQ-Int4,Granite-4.2-30B-GPTQ-INT4", [string]$Execs = "oneproc,mp")
$root = "C:\Users\claude\scripts\vllm-xpu-windows"
$out = "$root\server\oneproc-models.output.txt"
$py = "$root\.venv312\Scripts\python.exe"
$env:PATH = "$root\.venv312\Library\bin;$env:PATH"
$env:PYTHONFAULTHANDLER = "1"
Remove-Item Env:ZE_AFFINITY_MASK, Env:ONEAPI_DEVICE_SELECTOR -ErrorAction SilentlyContinue
$rows = @("run $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')  two cards, full decode graphs; decode tok/s over 256 tokens, greedy ids compared")
foreach ($m in $Models.Split(',')) {
    $name, $dtype = $m.Split('@')
    $env:SMOKE_MODEL = if ($name -match '^[A-Za-z]:\\') { $name } else { "C:\Users\claude\models-hf\$name" }
    $env:SMOKE_DTYPE = if ($dtype) { $dtype } else { "bfloat16" }
    $tag = Split-Path $env:SMOKE_MODEL -Leaf
    $ids = @{}
    foreach ($e in $Execs.Split(',')) {
        $res = "$root\server\oneproc-smoke-$tag-$e-graphs.output.txt"
        $con = "$root\server\oneproc-smoke-$tag-$e-graphs.console.log"
        Remove-Item $res -ErrorAction SilentlyContinue
        $p = Start-Process $py -ArgumentList '-u', "$root\server\oneproc-smoke.input.py", $e, 'graphs' -NoNewWindow -PassThru -RedirectStandardOutput $con -RedirectStandardError "$con.err"
        if (-not $p.WaitForExit(2400000)) { taskkill /T /F /PID $p.Id | Out-Null; $rows += "$tag ${e}: TIMEOUT after 40 min" }
        if ((Test-Path $res) -and (Select-String -Path $res -Pattern '^decode ' -Quiet)) {
            $rows += "$tag ${e}: " + (Select-String -Path $res -Pattern '^decode ').Line + "; " + (Select-String -Path $res -Pattern '^engine up').Line
            $ids[$e] = (Select-String -Path $res -Pattern '^ids ').Line
        } else {
            $rows += "$tag ${e}: FAILED - " + ((Select-String -Path "$con.err" -Pattern 'Error' | Select-Object -Last 1).Line)
        }
        $rows | Out-File $out -Encoding utf8
    }
    if ($ids.Count -eq 2) { $rows += "$tag token ids identical across executors: " + ($ids.Values | Select-Object -Unique).Count.Equals(1) }
    $rows | Out-File $out -Encoding utf8
}
Remove-Item Env:SMOKE_MODEL, Env:SMOKE_DTYPE -ErrorAction SilentlyContinue
$rows
