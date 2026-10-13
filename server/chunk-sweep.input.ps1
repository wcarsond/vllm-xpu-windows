# chunk-sweep.input.ps1 - two-card prefill at several chunk sizes (max_num_batched_tokens), same as the Ubuntu side's
# chunk-sweep, so Windows and Linux line up row for row. Prefill at 4K / 16K / 32K tokens; decode is printed too.
# Usage:  powershell -File server\chunk-sweep.input.ps1 ["512,1024,2048,4096"] [model-folder]   (writes chunk-sweep.output.txt)
# Why (2026-10-13): Linux's best two-card setting was chunk 2048 (6,019 / 5,472 / 4,729 tok/s), its sweep flat from 2048 up;
# Windows had only ever run 2048. The exchange's 32 MB slot holds a 4096-token chunk of the 7B (29 MB); 8192 would not fit.
param([string]$Chunks = "512,1024,2048,4096", [string]$Model = "Qwen2.5-Coder-7B-Instruct")
$root = "C:\Users\claude\scripts\vllm-xpu-windows"
$out = "$root\server\chunk-sweep.output.txt"
$rows = @("run $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')  $Model, two cards, eager; prefill tok/s at 4K / 16K / 32K, decode tok/s same depths")
$env:RACE_DEPTHS = "4096,16384,32000"
foreach ($c in $Chunks.Split(',')) {
    $env:RACE_CHUNK = $c
    $env:RACE_TAG = "chunk$c"
    $con = "$root\server\chunk-sweep.$c.console.log"
    $res = "$root\server\race-7b-depth-tp2-$Model-chunk$c.output.txt"
    Remove-Item $res -ErrorAction SilentlyContinue
    & cmd /c "`"C:\Users\claude\scripts\backend-race\vllm-cell.input.cmd`" `"C:\Users\claude\models-hf\$Model`" bfloat16 0.88 2 FLASH_ATTN 32768 `"$con`""
    Get-Process vllm -ErrorAction SilentlyContinue | Stop-Process -Force
    if (Test-Path $res) {
        $m = Select-String -Path $res -Pattern 'depth\s+(\d+):.*prefill\s+([\d.]+) tok/s.*decode\s+([\d.]+) tok/s'
        $p = ($m | ForEach-Object { "{0:N0}" -f [double]$_.Matches[0].Groups[2].Value }) -join ' / '
        $d = ($m | ForEach-Object { $_.Matches[0].Groups[3].Value }) -join ' / '
        $rows += "chunk {0,5}: prefill {1}   decode {2}" -f $c, $p, $d
    } else {
        $rows += "chunk {0,5}: FAILED - " -f $c + ((Select-String -Path $con -Pattern 'Error|error:' | Select-Object -Last 1).Line)
    }
    $rows | Out-File $out -Encoding utf8
}
Remove-Item Env:RACE_DEPTHS, Env:RACE_CHUNK, Env:RACE_TAG -ErrorAction SilentlyContinue
$rows
