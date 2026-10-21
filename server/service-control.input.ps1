# service-control.input.ps1 - start, stop, restart or show the vllm-b60 service (the vLLM router on 11434).
# start/stop/restart NEED ADMIN: an admin shell, or the Pi's SSH session (Windows gives it an admin token). status needs nothing.
# Usage:  powershell -File server\service-control.input.ps1 start|stop|restart|status
# Why (2026-10-12): the .cmd version broke when called over SSH; a .ps1 by path is the form proven to work through that route.
param([ValidateSet("start", "stop", "restart", "status")][string]$Action = "status")
$nssm = "C:\ollama\nssm.exe"
switch ($Action) {
    "stop"    { & $nssm stop vllm-b60;    Start-Sleep 3 }
    "start"   { & $nssm start vllm-b60;   Start-Sleep 3 }
    "restart" { & $nssm restart vllm-b60; Start-Sleep 5 }
}
"service: " + ((& $nssm status vllm-b60) -join '' -replace '\0', '')
"vllm.exe: " + $(if (Get-Process vllm -ErrorAction SilentlyContinue) { "running (a model is loaded or loading)" } else { "none (nothing loaded)" })
try { "router: " + ((Invoke-RestMethod http://localhost:11434/router/status -TimeoutSec 5) | ConvertTo-Json -Compress) } catch { "router: not answering" }
"free RAM GB: " + [math]::Round((Get-CimInstance Win32_OperatingSystem).FreePhysicalMemory / 1MB, 1)
