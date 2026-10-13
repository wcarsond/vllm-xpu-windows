# compile-ab-tp2.input.ps1 - two cards, a 4-bit model, eager against torch.compile (no graph capture: it is off for two
# cards on Windows), depth race each way. Writes server\compile-ab-tp2.output.txt.
# Usage:  powershell -File server\compile-ab-tp2.input.ps1 [model-folder] [dtype]   (default Granite-4.2-30B-GPTQ-INT4 bfloat16)
# Why (2026-10-13): graph-ab showed eager decode wastes ~225 us per layer on CPU and launch overhead (0.5B: 118 -> 319 tok/s
# with compile + graphs). Two-card graphs need a graph-safe exchange (a day's work); torch.compile alone works on two cards
# today and fuses the small per-layer kernels, so it is the cheap half of the same win to measure first.
param([string]$Model = "Granite-4.2-30B-GPTQ-INT4", [string]$Dtype = "bfloat16")
$root = "C:\Users\claude\scripts\vllm-xpu-windows"
$out = "$root\server\compile-ab-tp2.output.txt"
$path = "C:\Users\claude\models-hf\$Model"
$res = "$root\server\race-7b-depth-tp2-$Model.output.txt"
$rows = @("run $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')  $Model $Dtype, two cards")
foreach ($mode in @(@{ n = "eager"; e = "1" }, @{ n = "compiled"; e = "0" })) {
    $env:SMOKE_EAGER = $mode.e
    $con = "$root\server\compile-ab-tp2.$($mode.n).console.log"
    & cmd /c "`"C:\Users\claude\scripts\backend-race\vllm-cell.input.cmd`" `"$path`" $Dtype 0.88 2 default 32768 `"$con`""
    Get-Process vllm -ErrorAction SilentlyContinue | Stop-Process -Force
    $ok = (Get-Content $con -Raw) -match 'cell exit 0'
    $up = (Select-String -Path $res -Pattern 'engine up in ([\d.]+) s' -ErrorAction SilentlyContinue | Select-Object -First 1).Matches.Groups[1].Value
    $dec = Select-String -Path $res -Pattern 'depth\s+(\d+):.*prefill\s+([\d.]+) tok/s.*decode\s+([\d.]+) tok/s' -ErrorAction SilentlyContinue |
        ForEach-Object { "{0}:{1}/{2}" -f $_.Matches[0].Groups[1].Value, $_.Matches[0].Groups[3].Value, [int][double]$_.Matches[0].Groups[2].Value }
    $why = if ($ok) { "" } else { "FAILED: " + ((Select-String -Path $con -Pattern 'Error|error:|0xC0000' | Select-Object -Last 2 | ForEach-Object { $_.Line.Trim() }) -join ' / ') }
    $rows += "$($mode.n): engine up $up s  depth:decode/prefill  " + ($dec -join '  ') + "  $why"
    $rows | Out-File $out -Encoding utf8
}
Remove-Item Env:SMOKE_EAGER -ErrorAction SilentlyContinue
$rows
