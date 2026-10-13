# graphs-tp2.input.ps1 - two-card depth race with torch.compile + graph capture on (SMOKE_EAGER=0), one model.
# Usage:  powershell -File server\graphs-tp2.input.ps1 [model-folder] [dtype] [attention]
#         default Qwen2.5-Coder-7B-Instruct bfloat16 FLASH_ATTN; writes server\graphs-tp2.output.txt (and keeps the console)
# Why (2026-10-13): the exchange is graph-safe for decode-size all_reduces and platforms/xpu.py now keeps the capture sizes
# whose all_reduce fits 64 KB. This measures what graphs buy two-card decode, against the eager runs of the same day.
param([string]$Model = "Qwen2.5-Coder-7B-Instruct", [string]$Dtype = "bfloat16", [string]$Attn = "FLASH_ATTN")
$root = "C:\Users\claude\scripts\vllm-xpu-windows"
$out = "$root\server\graphs-tp2.output.txt"
$con = "$root\server\graphs-tp2.$Model.console.log"
$res = "$root\server\race-7b-depth-tp2-$Model.output.txt"
Remove-Item $res -ErrorAction SilentlyContinue      # so a failed run cannot show an older table
$env:SMOKE_EAGER = "0"
& cmd /c "`"C:\Users\claude\scripts\backend-race\vllm-cell.input.cmd`" `"C:\Users\claude\models-hf\$Model`" $Dtype 0.88 2 $Attn 32768 `"$con`""
Remove-Item Env:SMOKE_EAGER
Get-Process vllm -ErrorAction SilentlyContinue | Stop-Process -Force
$rows = @("run $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')  $Model $Dtype $Attn, two cards, compile + graphs")
if (Test-Path $res) { $rows += Get-Content $res } else { $rows += "no result table" }
$rows += Select-String -Path $con -Pattern "graph capture limited|disabling cudagraph|Capturing CUDA graph|Graph capturing finished|Error|error:|cell exit" |
    Select-Object -First 10 | ForEach-Object { "  log: " + $_.Line.Substring(0, [Math]::Min(200, $_.Line.Length)) }
$rows | Out-File $out -Encoding utf8
$rows
