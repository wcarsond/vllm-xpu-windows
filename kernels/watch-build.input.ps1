# watch-build.input.ps1 - RAM guard and progress log for a running kernels\build.input.cmd; stops the compiler before the PC runs dry.
# Usage:  powershell -NoProfile -File kernels\watch-build.input.ps1     (loops until build.output.log says "pip exit"; log: watch-build.output.log)
# Why (2026-10-10): one clang-cl instance has reached 16 GB working set / 30 GB commit on this 32 GB PC; on 10-02 a job of
# mine ran the machine out of memory. Every 20 s: free RAM and the latest ninja step. Under 2.5 GB free it kills clang-cl,
# icx and ninja (the pip step then exits non-zero and the build stops cleanly; ninja resumes from the finished objects
# next time). Started detached by build-detached.input.ps1 so it outlives the tool call.
$here = Split-Path -Parent $MyInvocation.MyCommand.Path
$log  = "$here\watch-build.output.log"
$pip  = "$here\build.pip.log"
$out  = "$here\build.output.log"
$floorGB = 2.5
"watch start $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')  floor $floorGB GB" | Out-File $log -Encoding utf8
$lastStep = ""
while ($true) {
    Start-Sleep -Seconds 20
    $os = Get-CimInstance Win32_OperatingSystem
    $freeGB = [math]::Round($os.FreePhysicalMemory / 1MB, 1)
    $step = ""
    if (Test-Path $pip) {
        $m = Select-String -Path $pip -Pattern '^\s*\[(\d+)/(\d+)\]' | Select-Object -Last 1
        if ($m) { $step = $m.Matches[0].Value.Trim() }
    }
    $done = (Test-Path $out) -and (Select-String -Path $out -Pattern 'pip exit' -Quiet)
    if ($step -ne $lastStep -or $done -or $freeGB -lt $floorGB) {
        "$(Get-Date -Format 'HH:mm:ss')  free $freeGB GB  step $step" | Out-File $log -Append -Encoding utf8
        $lastStep = $step
    }
    if ($freeGB -lt $floorGB) {
        $victims = Get-Process clang-cl, icx, ninja -ErrorAction SilentlyContinue
        "$(Get-Date -Format 'HH:mm:ss')  FREE RAM $freeGB GB UNDER FLOOR: stopping $(($victims | ForEach-Object { $_.ProcessName + ':' + $_.Id }) -join ' ')" | Out-File $log -Append -Encoding utf8
        $victims | Stop-Process -Force -ErrorAction SilentlyContinue
    }
    if ($done) { "watch end $(Get-Date -Format 'HH:mm:ss'): build finished" | Out-File $log -Append -Encoding utf8; break }
}
