# switch-service-to-router.input.ps1 - point the vllm-b60 service at the router (run-router.input.cmd) instead of a single
# fixed model, then restart it and wait for the router to list its models. RUN ELEVATED.
# Usage (admin):  powershell -ExecutionPolicy Bypass -File C:\Users\claude\scripts\vllm-xpu-windows\server\switch-service-to-router.input.ps1
# Undo: nssm set vllm-b60 AppParameters "/c C:\Users\claude\scripts\vllm-xpu-windows\server\start-server.input.cmd"; nssm restart vllm-b60
# Why (2026-10-12, Carson: "switch models via VS Code"): the service runs the router; the router runs one vLLM at a time.
$nssm = "C:\ollama\nssm.exe"
$here = "C:\Users\claude\scripts\vllm-xpu-windows\server"
$out = "$here\switch-service-to-router.output.txt"
$rows = @("run $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')")
function Say($s) { $script:rows += $s; Write-Host $s }
if (-not ([Security.Principal.WindowsPrincipal][Security.Principal.WindowsIdentity]::GetCurrent()).IsInRole("Administrators")) {
    Say "NOT elevated"; $rows | Out-File $out -Encoding utf8; exit 1
}
& $nssm stop vllm-b60 2>$null | Out-Null
Start-Sleep -Seconds 3
foreach ($p in Get-Process -Name vllm -ErrorAction SilentlyContinue) { Stop-Process -Id $p.Id -Force; Say "stopped leftover vllm.exe pid $($p.Id)" }
& $nssm set vllm-b60 AppParameters "/c `"$here\run-router.input.cmd`"" | Out-Null
& $nssm set vllm-b60 Description "vLLM router on 11434: lists the models in server\models.input.ini, loads the one a request names, swaps on demand. Script: $here\run-router.input.cmd" | Out-Null
& $nssm set vllm-b60 AppStdout "$here\service.console.log" | Out-Null
& $nssm set vllm-b60 AppStderr "$here\service.console.log" | Out-Null
Say ("service now runs: " + (& $nssm get vllm-b60 Application) + " " + (& $nssm get vllm-b60 AppParameters))
& $nssm start vllm-b60 | Out-Null
$deadline = (Get-Date).AddSeconds(90); $up = $false
while ((Get-Date) -lt $deadline) {
    Start-Sleep -Seconds 5
    try { $r = Invoke-RestMethod "http://localhost:11434/v1/models" -TimeoutSec 5; $up = $true; break } catch {}
}
if ($up) { Say ("router UP; models: " + (($r.data | ForEach-Object { $_.id }) -join ', ')) } else { Say "router not answering after 90 s - read $here\router.stdout.log" }
Say ("service state: " + (Get-Service vllm-b60).Status)
$rows | Out-File $out -Encoding utf8
