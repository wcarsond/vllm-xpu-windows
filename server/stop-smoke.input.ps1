# stop-smoke.input.ps1 - stop every leftover Python process from this project's server\ scripts (smoke, race, pp2 runs) and nothing else.
# Usage:  powershell -NoProfile -File server\stop-smoke.input.ps1     (writes server\stop-smoke.output.txt)
# Why (2026-10-10): a failed smoke run can leave the EngineCore child alive (spawned, holding ~12 GB of VRAM and 14 GB of
# RAM) after the parent reported the error. Every later test on that card then runs against a half-full card. This matches
# on the command line, so the llama.cpp router and anything not launched from server\ is never touched.
$here = Split-Path -Parent $MyInvocation.MyCommand.Path
$out  = Join-Path $here 'stop-smoke.output.txt'
$rows = @("run $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')")
$all = Get-CimInstance Win32_Process | Where-Object { $_.Name -match '^python' }
$roots = $all | Where-Object { $_.CommandLine -like '*vllm-xpu-windows\server\*' }
# 03:19: the engine core and the per-card workers are spawned children whose command line is only
# "from multiprocessing.spawn import spawn_main ...", so they slipped past the first version and two of them
# (3.2 GB each, one per card) outlived their parent. Take the whole tree under a server\ root, plus any
# spawn_main python whose parent is already gone (an orphaned worker); the kernel build's pip never spawns those.
$alive = @{}; foreach ($p in $all) { $alive[$p.ProcessId] = $p }
$tree = @{}
$queue = New-Object System.Collections.Queue
foreach ($r in $roots) { $queue.Enqueue($r.ProcessId) }
while ($queue.Count -gt 0) {
    $pid_ = $queue.Dequeue()
    if ($tree.ContainsKey($pid_)) { continue }
    if ($alive.ContainsKey($pid_)) { $tree[$pid_] = $alive[$pid_] }
    foreach ($c in ($all | Where-Object { $_.ParentProcessId -eq $pid_ })) { $queue.Enqueue($c.ProcessId) }
}
$allProcIds = (Get-Process).Id
foreach ($p in ($all | Where-Object { $_.CommandLine -like '*multiprocessing.spawn*' -and -not ($allProcIds -contains $_.ParentProcessId) })) {
    $tree[$p.ProcessId] = $p
}
$mine = $tree.Values
if (-not $mine) { $rows += "nothing to stop" }
foreach ($p in ($mine | Sort-Object ParentProcessId -Descending)) {
    $cmd = if ($p.CommandLine) { $p.CommandLine.Substring(0, [math]::Min(120, $p.CommandLine.Length)) } else { '(no command line)' }
    $rows += ("stopping pid {0} (parent {1}, started {2:HH:mm:ss}, ws {3:N1} GB): {4}" -f $p.ProcessId, $p.ParentProcessId, $p.CreationDate, ($p.WorkingSetSize/1GB), $cmd)
    Stop-Process -Id $p.ProcessId -Force -ErrorAction SilentlyContinue
}
Start-Sleep -Seconds 2
$left = Get-CimInstance Win32_Process | Where-Object { $_.Name -match '^python' -and ($_.CommandLine -like '*vllm-xpu-windows\server\*' -or ($_.CommandLine -like '*multiprocessing.spawn*' -and -not ((Get-Process).Id -contains $_.ParentProcessId))) }
$rows += if ($left) { "STILL ALIVE: " + (($left | ForEach-Object { $_.ProcessId }) -join ', ') } else { "all stopped" }
$rows | Out-File $out -Encoding utf8
Get-Content $out
