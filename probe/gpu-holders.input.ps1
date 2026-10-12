# gpu-holders.input.ps1 - who holds GPU memory on which adapter right now, plus every python/llama process with its command line.
# Usage:  powershell -NoProfile -File probe\gpu-holders.input.ps1     (writes probe\gpu-holders.output.txt)
# Why (2026-10-10): Carson asked whether an earlier server process was still sitting on a B60. Run this before any
# kernel test or smoke run; a card with someone else's gigabytes on it makes every OUT_OF_DEVICE_MEMORY result ambiguous.
# A B60 is the adapter with a ~24 GB dedicated limit; the iGPU has none. Elevated processes show an empty command line.
$here = Split-Path -Parent $MyInvocation.MyCommand.Path
$out  = Join-Path $here 'gpu-holders.output.txt'
$rows = @("run $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')")
$os = Get-CimInstance Win32_OperatingSystem
$rows += ("RAM free {0:N1} of {1:N1} GB" -f ($os.FreePhysicalMemory/1MB), ($os.TotalVisibleMemorySize/1MB))
# No "Dedicated Limit" counter exists on this driver, so the display adapter is the one dwm.exe draws on (the iGPU);
# every other adapter that any process has touched is a B60. On 2026-10-10 the B60s were luid ...13440 and ...14048.
$p = (Get-Counter '\GPU Process Memory(*)\Dedicated Usage','\GPU Process Memory(*)\Shared Usage','\GPU Process Memory(*)\Total Committed').CounterSamples | Where-Object { $_.CookedValue -gt 20MB }
$dwmPid = (Get-Process dwm -ErrorAction SilentlyContinue | Select-Object -First 1).Id
$display = ($p | Where-Object { $_.InstanceName -like "pid_${dwmPid}_*" } | Select-Object -First 1).InstanceName -replace '^pid_\d+_', ''
$rows += "-- adapters (MB)"
$a = (Get-Counter '\GPU Adapter Memory(*)\Dedicated Usage','\GPU Adapter Memory(*)\Total Committed').CounterSamples
foreach ($inst in ($a | Select-Object -ExpandProperty InstanceName -Unique)) {
    $use = ($a | Where-Object { $_.InstanceName -eq $inst -and $_.Path -like '*dedicated usage' }).CookedValue / 1MB
    $com = ($a | Where-Object { $_.InstanceName -eq $inst -and $_.Path -like '*total committed' }).CookedValue / 1MB
    $kind = if ($inst -eq $display) { 'display/iGPU' } elseif ($com -gt 20) { 'B60         ' } else { 'idle adapter' }
    $rows += ("{0} {1}  dedicated used {2,7:N0}  committed {3,7:N0}" -f $kind, $inst, $use, $com)
}
$rows += "-- processes holding more than 20 MB (MB)"
foreach ($s in ($p | Sort-Object InstanceName, Path)) {
    $pid_ = [regex]::Match($s.InstanceName, 'pid_(\d+)').Groups[1].Value
    $name = (Get-Process -Id $pid_ -ErrorAction SilentlyContinue).ProcessName
    $what = ($s.Path -split '\\')[-1]
    $rows += ("{0,8:N0}  {1,-15} pid {2,-6} {3,-16} {4}" -f ($s.CookedValue/1MB), $name, $pid_, $what, $s.InstanceName)
}
$rows += "-- python / llama / ollama processes"
Get-CimInstance Win32_Process | Where-Object { $_.Name -match '^(python|pythonw|llama-server|llama-router|ollama)' } | ForEach-Object {
    $owner = try { ($_ | Invoke-CimMethod -MethodName GetOwner).User } catch { '?' }
    $cmd = if ($_.CommandLine) { $_.CommandLine.Substring(0, [math]::Min(150, $_.CommandLine.Length)) } else { '(no command line visible: elevated or another session)' }
    $rows += ("pid {0,-6} parent {1,-6} {2,-10} started {3:MM-dd HH:mm:ss}  ws {4,5:N1} GB  {5}" -f $_.ProcessId, $_.ParentProcessId, $owner, $_.CreationDate, ($_.WorkingSetSize/1GB), $cmd)
}
$rows | Out-File $out -Encoding utf8
Get-Content $out
