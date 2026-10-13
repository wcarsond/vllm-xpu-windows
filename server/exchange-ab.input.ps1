# exchange-ab.input.ps1 - run server\exchange-test.input.py once per exchange mode and keep each output, for an A/B of the
# small (decode-size) path. Modes are environment switches read by the exchange at construction.
# Usage:  powershell -File server\exchange-ab.input.ps1 [rounds]          (writes server\exchange-ab.output.txt)
# Why (2026-10-13): the decode-size all_reduce is 28 us against oneCCL's 13 on Linux; VLLM_XPU_EXCHANGE_FUSED=1 does the
# small path in one kernel instead of two. Both modes must pass the bit-exact check before any timing counts.
param([int]$Rounds = 3)
$root = "C:\Users\claude\scripts\vllm-xpu-windows"
$out = "$root\server\exchange-ab.output.txt"
$env:PATH = "$root\.venv312\Library\bin;$env:PATH"
$rows = @("run $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')  rounds $Rounds")
foreach ($mode in @(@{ name = "two kernels (default)"; fused = "0" }, @{ name = "fused, one kernel"; fused = "1" })) {
    $env:VLLM_XPU_EXCHANGE_FUSED = $mode.fused
    & "$root\.venv312\Scripts\python.exe" "$root\server\exchange-test.input.py" $Rounds 2>&1 | Out-Null
    $rows += "=== $($mode.name)"
    $rows += Get-Content "$root\server\exchange-test.output.txt" | Where-Object { $_ -match 'wrong|back to back|PASS|FAIL|big path' } |
        ForEach-Object { if ($_.Length -gt 220) { $_.Substring(0, 220) + ' ...' } else { $_ } }
}
Remove-Item Env:VLLM_XPU_EXCHANGE_FUSED -ErrorAction SilentlyContinue
$rows | Out-File $out -Encoding utf8
$rows
