# router-restart.input.ps1 - stop the router's current llama-server so the next request starts a fresh one (fresh Vulkan context).
# Usage:  powershell -NoProfile -File server\router-restart.input.ps1     (writes server\router-restart.output.txt)
# Why (2026-10-11 03:15): the router's llama-server (pid from 22:46) measured 16.5 tok/s then, and 8.9 on a quiet machine at
# 03:13 after living through ~30 GPU resets caused by the cutlass crashes in between. vLLM on the same cards is at full
# speed, so the cards are fine; a Vulkan context that outlived device resets is the suspect. Carson OK'd stopping the
# router (22:48). This only stops llama-server; the router itself (elevated Python) stays up and respawns it on demand.
$here = Split-Path -Parent $MyInvocation.MyCommand.Path
$out  = "$here\router-restart.output.txt"
$rows = @("run $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')")
$servers = Get-Process -Name llama-server -ErrorAction SilentlyContinue
if (-not $servers) { $rows += "no llama-server running" }
foreach ($p in $servers) {
    # StartTime is unreadable for an elevated process from a normal shell (03:14: it came back null)
    $age = if ($p.StartTime) { [math]::Round(((Get-Date) - $p.StartTime).TotalHours, 1) } else { '?' }
    try {
        Stop-Process -Id $p.Id -Force -ErrorAction Stop
        $rows += "stopped llama-server pid $($p.Id) (alive $age h)"
    } catch {
        $rows += "could not stop llama-server pid $($p.Id): $($_.Exception.Message) (elevated; Carson can from an admin shell)"
    }
}
Start-Sleep -Seconds 3
$left = Get-Process -Name llama-server -ErrorAction SilentlyContinue
$rows += if ($left) { "still running: " + (($left | ForEach-Object { $_.Id }) -join ', ') } else { "no llama-server left; the router starts a new one on the next request" }
$rows | Out-File $out -Encoding utf8
Get-Content $out
