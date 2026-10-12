# build-detached.input.ps1 - start kernels\build.input.cmd (or another build script in this folder) as its own process and return at once.
# Usage:  powershell -NoProfile -File build-detached.input.ps1 [build-onednn.input.cmd]     (writes build-detached.output.log with the PID)
# Why (2026-10-10): a full compile of vllm-xpu-kernels (hundreds of SYCL files, ahead-of-time for the card) runs
# longer than the 10 minutes a tool call may hold a process, and a killed build leaves half-written objects. Started
# this way the build belongs to Windows, not to the tool call; progress is read from build.pip.log and build.output.log,
# and the exit is known when build.output.log gains its "pip exit" line.
param([string]$Script = 'build.input.cmd')
$here = Split-Path -Parent $MyInvocation.MyCommand.Path
$log  = "$here\build-detached.output.log"
$p = Start-Process -FilePath 'cmd.exe' -ArgumentList '/c', "`"$here\$Script`"" -WindowStyle Hidden -PassThru
"started $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')  pid $($p.Id)  (cmd /c $Script)" | Out-File $log -Encoding utf8
# The RAM guard runs beside it, also detached (watch-build.input.ps1: kills the compiler under 2.5 GB free, logs progress).
$w = Start-Process -FilePath 'powershell.exe' -ArgumentList '-NoProfile', '-File', "`"$here\watch-build.input.ps1`"" -WindowStyle Hidden -PassThru
"watcher pid $($w.Id)  (watch-build.input.ps1 -> watch-build.output.log)" | Out-File $log -Append -Encoding utf8
Get-Content $log
