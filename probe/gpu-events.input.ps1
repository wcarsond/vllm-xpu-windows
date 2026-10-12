# gpu-events.input.ps1 - what Windows itself logged about the GPUs in the last hours: driver resets (TDR), display/driver errors, Level Zero.
# Usage:  powershell -NoProfile -File probe\gpu-events.input.ps1 [hours]     (writes probe\gpu-events.output.txt; default 3 hours)
# Why (2026-10-10): every attention test case suddenly failed on its first launch, including one that passed twice
# earlier tonight. Before blaming the kernel: did the driver reset the card (Event 4101 "stopped responding and has
# recovered"), and when, relative to our runs? A wedged card explains "everything fails instantly"; a kernel bug does not.
param([double]$hours = 3)
$here = Split-Path -Parent $MyInvocation.MyCommand.Path
$out  = Join-Path $here 'gpu-events.output.txt'
$since = (Get-Date).AddHours(-$hours)
$rows = @("run $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')   events since $($since.ToString('HH:mm:ss'))")
$ev = Get-WinEvent -FilterHashtable @{ LogName = 'System'; StartTime = $since } -ErrorAction SilentlyContinue |
      Where-Object { $_.Id -in 4101, 4102, 4103, 4104, 4105, 4106, 4107, 4108, 4109, 4110, 4111, 4112, 4113, 4114, 4115, 4116, 4117, 4118, 4119, 4120, 10110, 10111, 14 -or $_.ProviderName -match 'Display|igfx|Intel|nvlddmkm|dxgkrnl|WUDFRd' }
if (-not $ev) { $rows += "no display/driver events in the window" }
foreach ($e in ($ev | Sort-Object TimeCreated)) {
    $msg = ($e.Message -replace '\s+', ' ')
    $rows += ("{0:HH:mm:ss}  id {1,-5} {2,-28} {3}" -f $e.TimeCreated, $e.Id, $e.ProviderName, $msg.Substring(0, [math]::Min(170, $msg.Length)))
}
$rows += "-- application log: Level Zero / oneAPI / python crashes"
$app = Get-WinEvent -FilterHashtable @{ LogName = 'Application'; StartTime = $since } -ErrorAction SilentlyContinue |
       Where-Object { $_.Message -match 'python|ze_|level.zero|sycl|ur_|igd|intel' -and $_.LevelDisplayName -in 'Error', 'Critical', 'Warning' }
if (-not $app) { $rows += "none" }
foreach ($e in ($app | Sort-Object TimeCreated)) {
    $msg = ($e.Message -replace '\s+', ' ')
    $rows += ("{0:HH:mm:ss}  id {1,-5} {2,-28} {3}" -f $e.TimeCreated, $e.Id, $e.ProviderName, $msg.Substring(0, [math]::Min(170, $msg.Length)))
}
$rows | Out-File $out -Encoding utf8
Get-Content $out
