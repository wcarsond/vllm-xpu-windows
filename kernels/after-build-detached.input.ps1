# after-build-detached.input.ps1 - start kernels\after-build.input.cmd as its own process and return at once.
# Usage:  powershell -NoProfile -File kernels\after-build-detached.input.ps1     (writes after-build-detached.output.log with the PID)
# Why (2026-10-11): the chain (wheel, tests, four races, tensor split, router legs, MoE build and tests) runs for an hour or
# two, longer than a tool call may hold a process. Progress is the one-line verdicts in after-build.output.log.
param([int]$start = 1)   # first step to run (after-build.input.cmd labels 1-15)
$here = Split-Path -Parent $MyInvocation.MyCommand.Path
$log  = "$here\after-build-detached.output.log"
$p = Start-Process -FilePath 'cmd.exe' -ArgumentList '/c', "`"$here\after-build.input.cmd`" $start" -WindowStyle Hidden -PassThru
"started $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')  pid $($p.Id)  (cmd /c after-build.input.cmd $start)" | Out-File $log -Encoding utf8
Get-Content $log
