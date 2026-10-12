# trim-ws.input.ps1 - page out the memory of processes that are dead but stuck in their exit (EmptyWorkingSet), so the RAM is usable again.
# Usage:  powershell -NoProfile -File probe\trim-ws.input.ps1 <pid> [<pid> ...]      (also writes probe\trim-ws.output.txt)
# Why (2026-10-11 03:55): four tensor-split workers are stuck inside ExitProcess (probe\thread-stack.cpp shows the Intel OpenCL
# runtime spinning in a DLL unload handler) holding 1.4-3.3 GB of RAM each until the next reboot. Nothing in that memory will
# ever be touched again, so push it to the page file and give the night's tests the RAM back. Locked (pinned) pages stay.
param([Parameter(ValueFromRemainingArguments=$true)][int[]]$pids)
$here = Split-Path -Parent $MyInvocation.MyCommand.Path
$out = "$here\trim-ws.output.txt"
Add-Type -Namespace TrimWs -Name Native -MemberDefinition @'
[DllImport("psapi.dll", SetLastError=true)] public static extern bool EmptyWorkingSet(IntPtr hProcess);
[DllImport("kernel32.dll", SetLastError=true)] public static extern IntPtr OpenProcess(uint access, bool inherit, int pid);
[DllImport("kernel32.dll")] public static extern bool CloseHandle(IntPtr h);
'@
function FreeGB { [math]::Round((Get-CimInstance Win32_OperatingSystem).FreePhysicalMemory / 1MB, 2) }
$rows = @("run $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss'); free RAM before $(FreeGB) GB")
foreach ($p in $pids) {
    $pr = Get-Process -Id $p -ErrorAction SilentlyContinue
    if (-not $pr) { $rows += "pid ${p}: not found"; continue }
    $before = $pr.WorkingSet64
    # PROCESS_QUERY_INFORMATION 0x0400 | PROCESS_SET_QUOTA 0x0100
    $h = [TrimWs.Native]::OpenProcess(0x0500, $false, $p)
    if ($h -eq [IntPtr]::Zero) { $rows += "pid ${p}: OpenProcess failed (error $([Runtime.InteropServices.Marshal]::GetLastWin32Error()))"; continue }
    $ok = [TrimWs.Native]::EmptyWorkingSet($h)
    $err = [Runtime.InteropServices.Marshal]::GetLastWin32Error()
    [void][TrimWs.Native]::CloseHandle($h)
    Start-Sleep -Milliseconds 800
    $after = (Get-Process -Id $p -ErrorAction SilentlyContinue).WorkingSet64
    $rows += ("pid {0}: EmptyWorkingSet {1}{2}, working set {3:N2} GB -> {4:N2} GB" -f $p, $ok, $(if ($ok) { "" } else { " (error $err)" }), ($before/1GB), ($after/1GB))
}
$rows += "free RAM after $(FreeGB) GB"
$rows | Out-File $out -Encoding utf8
$rows
