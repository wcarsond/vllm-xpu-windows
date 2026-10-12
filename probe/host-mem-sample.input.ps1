# host-mem-sample.input.ps1 - the HOST side, once a second: free RAM, standby (file cache) and the working set of every
# python process, so the moment of the concurrent-load fault can be read against host pressure rather than guessed.
# Usage:  powershell -File probe\host-mem-sample.input.ps1 <seconds> <out-file>
# Why (2026-10-11 18:25): every memory measurement so far was of the cards (torch's counters, then the WDDM adapter
# counters, which showed no spill). The surviving explanation is host memory pressure from two workers' checkpoint pages,
# and nothing has measured that yet. If free RAM bottoms out right as a worker dies, the claim has its number.
param([int]$Seconds = 120, [string]$Out = "probe\host-mem-sample.output.txt")
$lines = @("sample start $(Get-Date -Format 'HH:mm:ss')  free RAM GB | standby (cache) GB | python working sets GB")
$minFree = 1e9
$peakWS = 0
for ($i = 0; $i -lt $Seconds; $i++) {
    $os = Get-CimInstance Win32_OperatingSystem
    $free = [math]::Round($os.FreePhysicalMemory / 1MB, 2)
    $standby = 0
    try { $standby = [math]::Round((Get-Counter '\Memory\Standby Cache Normal Priority Bytes').CounterSamples[0].CookedValue / 1GB, 2) } catch {}
    $ws = @()
    foreach ($p in (Get-Process python -ErrorAction SilentlyContinue)) {
        $g = [math]::Round($p.WorkingSet64 / 1GB, 2)
        if ($g -ge 0.5) { $ws += "$($p.Id):$g"; if ($g -gt $peakWS) { $peakWS = $g } }
    }
    if ($free -lt $minFree) { $minFree = $free }
    $lines += ("{0}  free {1,6} | standby {2,6} | {3}" -f (Get-Date -Format 'HH:mm:ss'), $free, $standby, ($ws -join ' '))
    Start-Sleep -Seconds 1
}
$lines += ""
$lines += ("MIN free RAM {0} GB   PEAK single python working set {1} GB" -f $minFree, $peakWS)
$lines | Set-Content -Path $Out -Encoding utf8
