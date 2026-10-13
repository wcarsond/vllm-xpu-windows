# ahead-ab.input.ps1 - does letting the Python thread run ahead of the cards speed up two-card decode? A/B of
# VLLM_XPU_EXCHANGE_AHEAD (1 = today's lockstep, K = up to K all_reduces queued ahead of the peer).
# Usage:  powershell -File server\ahead-ab.input.ps1 [K values, comma list] [repeats]     (writes server\ahead-ab.output.txt)
#         e.g.  ahead-ab.input.ps1 "1,8" 2   -> exchange-test at the largest K, then the 7B TP2 depth race K=1, K=8, K=1, K=8
# Why (2026-10-13): two-card 7B decode loses ~4.5 ms a token to the single-card ideal, of which only ~1.6 ms is the
# exchange itself; the guard keeps the CPU one all_reduce ahead at most, so its launch and Python time sit on the
# critical path. Decode noise run to run is a few percent, so every K runs `repeats` times, interleaved.
param([string]$Ks = "1,8", [int]$Repeats = 2)
$root = "C:\Users\claude\scripts\vllm-xpu-windows"
$out = "$root\server\ahead-ab.output.txt"
$env:PATH = "$root\.venv312\Library\bin;$env:PATH"
$kl = $Ks.Split(',') | ForEach-Object { [int]$_ }
$rows = @("run $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')  K = $Ks, repeats $Repeats")
function Save { $script:rows | Out-File $out -Encoding utf8 }
# correctness first, at the largest K: every all_reduce bit-exact or nothing else counts
$env:VLLM_XPU_EXCHANGE_AHEAD = ($kl | Measure-Object -Maximum).Maximum
& "$root\.venv312\Scripts\python.exe" "$root\server\exchange-test.input.py" 3 2>&1 | Out-Null
$rows += "exchange-test at AHEAD=$($env:VLLM_XPU_EXCHANGE_AHEAD):"
$rows += Get-Content "$root\server\exchange-test.output.txt" | Where-Object { $_ -match ' wrong;|7 KB back to back|PASS|FAIL' } |
    ForEach-Object { "  " + $(if ($_.Length -gt 140) { $_.Substring(0, 140) } else { $_ }) }
Save
if ($rows -match 'FAIL|[1-9]\d* wrong') { $rows += "STOP: not bit-exact"; Save; exit 1 }
$model = "C:\Users\claude\models-hf\Qwen2.5-Coder-7B-Instruct"
$res = "$root\server\race-7b-depth-tp2-Qwen2.5-Coder-7B-Instruct.output.txt"
for ($r = 1; $r -le $Repeats; $r++) {
    foreach ($k in $kl) {
        $env:VLLM_XPU_EXCHANGE_AHEAD = "$k"
        $con = "$root\server\ahead-ab.K$k.r$r.console.log"
        & cmd /c "`"C:\Users\claude\scripts\backend-race\vllm-cell.input.cmd`" `"$model`" bfloat16 0.88 2 FLASH_ATTN 32768 `"$con`""
        Get-Process vllm -ErrorAction SilentlyContinue | Stop-Process -Force
        $dec = Select-String -Path $res -Pattern 'depth\s+(\d+):.*prefill\s+([\d.]+) tok/s.*decode\s+([\d.]+) tok/s' |
            ForEach-Object { "{0}:{1}/{2}" -f $_.Matches[0].Groups[1].Value, $_.Matches[0].Groups[3].Value, [int][double]$_.Matches[0].Groups[2].Value }
        $rows += "K=$k run $r  depth:decode/prefill  " + ($dec -join '  ')
        Save
    }
}
Remove-Item Env:VLLM_XPU_EXCHANGE_AHEAD -ErrorAction SilentlyContinue
$rows
