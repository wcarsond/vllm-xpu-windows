# oneproc-race.input.ps1 - the 7B depth leg on two cards, two-process executor against the one-process executor, both
# with whole-step decode graphs (no torch.compile), chunk 2048: prefill and decode tok/s at 0 / 4K / 16K / 32K tokens.
# Usage:  powershell -File server\oneproc-race.input.ps1 ["mp,oneproc"] ["0,4096,16384,32000"]
#         (writes server\oneproc-race.output.txt; the per-run tables are race-7b-depth-tp2-<tag>.output.txt)
# Why (2026-10-13): the one-process executor's real numbers, against the same mode on two processes, at the depths the
# earlier Windows/Linux tables used.
param([string]$Execs = "mp,oneproc", [string]$Depths = "0,4096,16384,32000")
$root = "C:\Users\claude\scripts\vllm-xpu-windows"
$out = "$root\server\oneproc-race.output.txt"
$py = "$root\.venv312\Scripts\python.exe"
$env:PATH = "$root\.venv312\Library\bin;$env:PATH"
$env:RACE_TP = "2"; $env:RACE_CHUNK = "2048"; $env:RACE_DEPTHS = $Depths; $env:RACE_GRAPHS = "full"
$env:SMOKE_ATTN_BACKEND = "FLASH_ATTN"; $env:SMOKE_GPU_UTIL = "0.88"; $env:PYTHONFAULTHANDLER = "1"
$env:VLLM_WORKER_MULTIPROC_METHOD = "spawn"
Remove-Item Env:ZE_AFFINITY_MASK, Env:ONEAPI_DEVICE_SELECTOR -ErrorAction SilentlyContinue
$rows = @("run $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')  Qwen2.5-Coder-7B-Instruct bf16, two cards, full decode graphs, chunk 2048; prefill / decode tok/s by depth")
foreach ($e in $Execs.Split(',')) {
    $env:RACE_EXECUTOR = $e
    $env:RACE_TAG = "graphs-$e"
    $res = "$root\server\race-7b-depth-tp2-graphs-$e.output.txt"
    $con = "$root\server\oneproc-race.$e.console.log"
    Remove-Item $res -ErrorAction SilentlyContinue
    $p = Start-Process $py -ArgumentList '-u', "$root\server\race-7b-depth.input.py" -NoNewWindow -PassThru -RedirectStandardOutput $con -RedirectStandardError "$con.err"
    if (-not $p.WaitForExit(2400000)) { taskkill /T /F /PID $p.Id | Out-Null; $rows += "${e}: TIMEOUT after 40 min" }
    if (Test-Path $res) {
        $rows += "${e}:"
        $rows += (Select-String -Path $res -Pattern 'engine up|depth ' | ForEach-Object { "  " + $_.Line })
    } else {
        $rows += "${e}: FAILED - " + ((Select-String -Path "$con.err" -Pattern 'Error' | Select-Object -Last 1).Line)
    }
    $rows | Out-File $out -Encoding utf8
}
Remove-Item Env:RACE_EXECUTOR, Env:RACE_TAG, Env:RACE_GRAPHS, Env:RACE_DEPTHS, Env:RACE_CHUNK, Env:RACE_TP -ErrorAction SilentlyContinue
$rows
