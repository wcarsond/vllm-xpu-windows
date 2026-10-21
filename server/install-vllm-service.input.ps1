# install-vllm-service.input.ps1 - replace the llama.cpp router with vLLM as the boot service on port 11434. RUN ELEVATED, once.
# Usage (admin PowerShell):  powershell -ExecutionPolicy Bypass -File C:\Users\claude\scripts\vllm-xpu-windows\server\install-vllm-service.input.ps1
# What it does, in order, printing each result:
#   1. kills any running llama-server (the router on 11434)
#   2. disables every scheduled task whose action runs llama-server / llama-router (what restarted it at boot)
#   3. stops and disables the old nssm services "llamacpp" and "Ollama"
#   4. installs the nssm service "vllm-b60": cmd /c start-server.input.cmd, auto start, restart on exit, LocalSystem
#   5. starts it and waits up to 6 minutes for http://localhost:11434/v1/models to answer (the 27B takes ~2.5 min to load)
# Undo: nssm stop vllm-b60; nssm remove vllm-b60 confirm; re-enable the task / service it disabled (names printed below).
# Why (2026-10-11 23:50, Carson: "Let's just replace with vLLM ... It's just a simple service entry").
$ErrorActionPreference = "Continue"
$nssm = "C:\ollama\nssm.exe"
$here = "C:\Users\claude\scripts\vllm-xpu-windows\server"
$out = "$here\install-vllm-service.output.txt"
$rows = @("run $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')")
function Say($s) { $script:rows += $s; Write-Host $s }
if (-not ([Security.Principal.WindowsPrincipal][Security.Principal.WindowsIdentity]::GetCurrent()).IsInRole("Administrators")) {
    Say "NOT elevated - open an admin PowerShell and run this file again"; $rows | Out-File $out -Encoding utf8; exit 1
}
if (-not (Test-Path $nssm)) { Say "nssm not found at $nssm"; $rows | Out-File $out -Encoding utf8; exit 1 }

# 1. the running router
foreach ($p in Get-Process -Name llama-server, llama-router -ErrorAction SilentlyContinue) {
    Stop-Process -Id $p.Id -Force; Say "1. stopped $($p.ProcessName) pid $($p.Id)"
}
if (-not (Get-Process -Name llama-server -ErrorAction SilentlyContinue)) { Say "1. no llama-server running" }

# 2. whatever starts it at boot
$found = $false
foreach ($t in Get-ScheduledTask) {
    foreach ($a in $t.Actions) {
        if (("$($a.Execute) $($a.Arguments)") -match 'llama') {
            Disable-ScheduledTask -TaskName $t.TaskName -TaskPath $t.TaskPath | Out-Null
            Say "2. disabled task $($t.TaskPath)$($t.TaskName)  [$($a.Execute) $($a.Arguments)]"; $found = $true
        }
    }
}
if (-not $found) { Say "2. no scheduled task runs llama (check Startup folders / registry Run keys if it comes back)" }

# 3. the old services
foreach ($s in "llamacpp", "Ollama") {
    $svc = Get-Service -Name $s -ErrorAction SilentlyContinue
    if ($svc) {
        if ($svc.Status -ne "Stopped") { Stop-Service -Name $s -Force }
        Set-Service -Name $s -StartupType Disabled
        Say "3. service $s stopped and disabled (was $($svc.Status), $($svc.StartType))"
    }
}

# 4. the new service
& $nssm stop vllm-b60 2>$null | Out-Null
& $nssm remove vllm-b60 confirm 2>$null | Out-Null
& $nssm install vllm-b60 "C:\Windows\System32\cmd.exe" "/c `"$here\start-server.input.cmd`"" | Out-Null
& $nssm set vllm-b60 AppDirectory $here | Out-Null
& $nssm set vllm-b60 DisplayName "vLLM on the B60s (port 11434)" | Out-Null
& $nssm set vllm-b60 Description "vLLM OpenAI-compatible server, Qwen3.5-27B tensor split over both Arc Pro B60s, model id 'local'. Script: $here\start-server.input.cmd" | Out-Null
& $nssm set vllm-b60 Start SERVICE_AUTO_START | Out-Null
& $nssm set vllm-b60 AppExit Default Restart | Out-Null
& $nssm set vllm-b60 AppRestartDelay 10000 | Out-Null
& $nssm set vllm-b60 AppKillProcessTree 1 | Out-Null
& $nssm set vllm-b60 AppStdout "$here\service.console.log" | Out-Null
& $nssm set vllm-b60 AppStderr "$here\service.console.log" | Out-Null
& $nssm set vllm-b60 AppRotateFiles 1 | Out-Null
& $nssm set vllm-b60 AppRotateBytes 10485760 | Out-Null
Say "4. service vllm-b60 installed: cmd /c $here\start-server.input.cmd (auto start, restart on exit, LocalSystem)"

# 5. start and wait for the model list
& $nssm start vllm-b60 | Out-Null
Say "5. started; waiting for http://localhost:11434/v1/models ..."
$deadline = (Get-Date).AddMinutes(6); $up = $false
while ((Get-Date) -lt $deadline) {
    Start-Sleep -Seconds 10
    try { $r = Invoke-RestMethod "http://localhost:11434/v1/models" -TimeoutSec 5; $up = $true; break } catch {}
}
if ($up) { Say ("5. UP in " + [math]::Round(((Get-Date) - $deadline.AddMinutes(-6)).TotalSeconds) + " s; models: " + (($r.data | ForEach-Object { $_.id }) -join ', ')) }
else { Say "5. not answering after 6 min - read $here\server.console.log" }
Say ("service state: " + (Get-Service vllm-b60).Status)
$rows | Out-File $out -Encoding utf8
