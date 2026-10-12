# zombie-threads.input.ps1 - for processes that taskkill says do not exist but Get-Process still lists: what is their last thread doing?
# Usage:  powershell -NoProfile -File probe\zombie-threads.input.ps1 <pid> [<pid> ...]     (prints; also probe\zombie-threads.output.txt)
# Why (2026-10-11 03:22): the two tensor-split workers finished ("XPUWorker shutdown: done"), their parent exited, and they stayed
# behind with one thread each, 3.2 GB of card memory, a lock on the console log, and "no running instance" from taskkill.
# Kernel-mode time growing with the thread in a running state means the thread is inside a driver call that never returns
# (nothing short of the driver or a reboot frees it); user-mode time growing means a user spin that should be killable.
param([Parameter(ValueFromRemainingArguments=$true)][int[]]$pids)
$here = Split-Path -Parent $MyInvocation.MyCommand.Path
$out = "$here\zombie-threads.output.txt"
$rows = @("run $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')")
function Snap($p) {
    $pr = Get-Process -Id $p -ErrorAction SilentlyContinue
    if (-not $pr) { return $null }
    $th = Get-CimInstance Win32_Thread -Filter "ProcessHandle='$p'"
    return @{ user = $pr.UserProcessorTime.TotalSeconds; kernel = $pr.PrivilegedProcessorTime.TotalSeconds; threads = $th }
}
foreach ($p in $pids) {
    $a = Snap $p
    if (-not $a) { $rows += "pid ${p}: not found"; continue }
    Start-Sleep -Seconds 3
    $b = Snap $p
    $rows += ("pid {0}: user time {1:N1}s (+{2:N2}s in 3 s), kernel time {3:N1}s (+{4:N2}s in 3 s), threads {5}" -f $p, $b.user, ($b.user - $a.user), $b.kernel, ($b.kernel - $a.kernel), @($b.threads).Count)
    foreach ($t in $b.threads) {
        $state = @{0='init';1='ready';2='running';3='standby';4='terminated';5='waiting';6='transition';7='unknown'}[[int]$t.ThreadState]
        $rows += ("   thread {0}: state {1}, wait reason {2}, priority {3}, kernel ms {4}, user ms {5}" -f $t.Handle, $state, $t.ThreadWaitReason, $t.Priority, $t.KernelModeTime, $t.UserModeTime)
    }
}
$rows | Out-File $out -Encoding utf8
$rows
