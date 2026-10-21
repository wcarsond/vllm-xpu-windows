# router-test.input.ps1 - exercise the vLLM router on 11434: list models, one plain request, one STREAMING request that
# forces a model switch (counts the ": loading" heartbeats an editor would see), then status. Writes router-test.output.txt.
# Usage:  powershell -File server\router-test.input.ps1 [model-for-switch] [host]      defaults qwen3.5-4b, localhost
# Why (2026-10-12): the switch path (SSE headers first, heartbeat every 5 s, then the real stream) is the part VS Code
# depends on and the part a buffered HTTP client cannot show; curl -N with the body from a file keeps the JSON intact.
param([string]$Model = "qwen3.5-4b", [string]$HostName = "localhost")
$here = Split-Path -Parent $MyInvocation.MyCommand.Path
$out = "$here\router-test.output.txt"
$base = "http://${HostName}:11434"
$rows = @("run $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')  $base  switch target $Model")
function Say($s) { $script:rows += $s; Write-Host $s }
$m = Invoke-RestMethod "$base/v1/models" -TimeoutSec 10
Say ("models: " + (($m.data | ForEach-Object { if ($_.loaded) { "$($_.id)*" } else { $_.id } }) -join ', ') + "   (* = loaded)")
$loaded = ($m.data | Where-Object { $_.loaded } | Select-Object -First 1).id
if ($loaded) {
    $body = @{ model = $loaded; messages = @(@{ role = "user"; content = "Reply with exactly: router ok" }); max_tokens = 20; temperature = 0 } | ConvertTo-Json -Compress -Depth 5
    $t = [Diagnostics.Stopwatch]::StartNew()
    $c = Invoke-RestMethod "$base/v1/chat/completions" -Method Post -ContentType "application/json" -Body $body -TimeoutSec 300
    $t.Stop()
    Say ("plain request to {0}: '{1}' in {2} s" -f $loaded, $c.choices[0].message.content.Trim(), [math]::Round($t.Elapsed.TotalSeconds, 1))
}
$bodyFile = "$here\router-test.body.json"
@{ model = $Model; stream = $true; messages = @(@{ role = "user"; content = "Reply with exactly: switch ok" }); max_tokens = 20; temperature = 0 } |
    ConvertTo-Json -Compress -Depth 5 | Set-Content $bodyFile -Encoding ascii
$streamFile = "$here\router-test.stream.txt"
$t = [Diagnostics.Stopwatch]::StartNew()
& curl.exe -s -N -X POST "$base/v1/chat/completions" -H "Content-Type: application/json" --data-binary "@$bodyFile" -o $streamFile
$t.Stop()
$lines = Get-Content $streamFile
$beats = ($lines | Where-Object { $_ -like ': loading*' }).Count
$data = $lines | Where-Object { $_ -like 'data: {*' }
$text = ($data | ForEach-Object { try { ($_.Substring(6) | ConvertFrom-Json).choices[0].delta.content } catch { "" } }) -join ''
Say ("streaming request for {0}: {1} s wall, {2} heartbeats, {3} data chunks, text '{4}'" -f $Model, [math]::Round($t.Elapsed.TotalSeconds, 1), $beats, $data.Count, $text.Trim())
if ($data.Count -eq 0) { Say ("  raw reply: " + (($lines | Select-Object -First 3) -join ' | ')) }
$s = Invoke-RestMethod "$base/router/status" -TimeoutSec 10
Say ("status: loaded {0}, ready {1}, last switch {2} s" -f $s.loaded, $s.ready, $s.last_switch_s)
$rows | Out-File $out -Encoding utf8
