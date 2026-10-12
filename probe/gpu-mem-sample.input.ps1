# gpu-mem-sample.input.ps1 - sample the DRIVER's view of card memory once a second: dedicated (on the card) and shared
# (spilled to system RAM) usage per adapter, from the same WDDM counters Task Manager draws.
# Usage:  powershell -File probe\gpu-mem-sample.input.ps1 <seconds> <out-file>
# Why (2026-10-11 17:30, Carson: "i thought you couldnt trust the system saying it wasnt spilling"): torch's
# max_memory_allocated() is what PyTorch asked for, not where the driver placed it, and this driver demonstrably backs
# device allocations with system RAM without any error. Only the driver's counter can say whether a load spilled.
param([int]$Seconds = 60, [string]$Out = "probe\gpu-mem-sample.output.txt")
$paths = @('\GPU Adapter Memory(*)\Dedicated Usage', '\GPU Adapter Memory(*)\Shared Usage')
$lines = @("sample start $(Get-Date -Format 'HH:mm:ss')  (bytes; one line per second per adapter instance with any usage)")
$peakShared = @{}
$peakDed = @{}
for ($i = 0; $i -lt $Seconds; $i++) {
    try {
        $c = Get-Counter -Counter $paths -ErrorAction Stop
    } catch {
        $lines += "counter read failed: $($_.Exception.Message)"
        break
    }
    $t = Get-Date -Format 'HH:mm:ss'
    foreach ($s in $c.CounterSamples) {
        if ($s.CookedValue -le 0) { continue }
        $inst = $s.InstanceName
        $kind = if ($s.Path -like '*Dedicated*') { 'dedicated' } else { 'shared' }
        $gb = [math]::Round($s.CookedValue / 1GB, 2)
        $lines += "$t  $inst  $kind  $gb GB"
        if ($kind -eq 'shared')    { if (-not $peakShared[$inst] -or $gb -gt $peakShared[$inst]) { $peakShared[$inst] = $gb } }
        else                       { if (-not $peakDed[$inst]    -or $gb -gt $peakDed[$inst])    { $peakDed[$inst] = $gb } }
    }
    Start-Sleep -Seconds 1
}
$lines += ""
$lines += "PEAKS over $Seconds s:"
foreach ($k in $peakDed.Keys)    { $lines += ("  {0}  dedicated peak {1} GB" -f $k, $peakDed[$k]) }
foreach ($k in $peakShared.Keys) { $lines += ("  {0}  SHARED peak {1} GB" -f $k, $peakShared[$k]) }
$lines | Set-Content -Path $Out -Encoding utf8
