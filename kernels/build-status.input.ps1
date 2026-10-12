# build-status.input.ps1 - one screen on a running kernel build: step, pace, ETA, free RAM, what is compiling, compiler processes.
# Usage:  powershell -NoProfile -File kernels\build-status.input.ps1     (prints; also writes build-status.output.txt)
# Why (2026-10-10): the build runs for hours detached; Carson checks from his phone. Pace is the average over the steps
# finished so far, which is pessimistic early (the first steps are the heaviest attention kernels) and settles later.
$here = Split-Path -Parent $MyInvocation.MyCommand.Path
$pip  = "$here\build.pip.log"
$out  = "$here\build.output.log"
$watch = "$here\watch-build.output.log"
$lines = @("status $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')")
$start = $null
if (Test-Path $out) {
    $m = Select-String -Path $out -Pattern 'build start (\d+:\d+:\d+)' | Select-Object -First 1
    if ($m) { $start = [datetime]::ParseExact($m.Matches[0].Groups[1].Value, 'H:mm:ss', $null) }
    if (Select-String -Path $out -Pattern 'pip exit' -Quiet) { $lines += (Select-String -Path $out -Pattern 'pip exit').Line }
}
$steps = @()
if (Test-Path $pip) { $steps = Select-String -Path $pip -Pattern '^\s*\[(\d+)/(\d+)\]' }
if ($steps.Count -gt 0) {
    $last = $steps[-1]; $n = [int]$last.Matches[0].Groups[1].Value; $total = [int]$last.Matches[0].Groups[2].Value
    $elapsed = if ($start) { (Get-Date) - $start } else { $null }
    $lines += "step $n of $total"
    if ($elapsed -and $n -gt 1) {
        $pace = $elapsed.TotalMinutes / ($n - 1)   # minutes per finished step
        $eta = (Get-Date).AddMinutes($pace * ($total - $n + 1))
        $lines += ("elapsed {0:N0} min, {1:N1} min/step so far, finish about {2:HH:mm} if the pace holds" -f $elapsed.TotalMinutes, $pace, $eta)
    }
    $file = ($last.Line | Select-String -Pattern '(csrc[^ ]+\.cpp)|([A-Za-z_0-9]+\.(dll|pyd))' -AllMatches).Matches | Select-Object -Last 1
    if ($file) { $lines += "now: $($file.Value)" }
}
$os = Get-CimInstance Win32_OperatingSystem
$lines += ("free RAM {0:N1} of {1:N1} GB" -f ($os.FreePhysicalMemory/1MB), ($os.TotalVisibleMemorySize/1MB))
$procs = Get-Process clang-cl, icx, ninja, ocloc -ErrorAction SilentlyContinue
if ($procs) { $lines += "compiler: " + (($procs | ForEach-Object { "{0} {1:N1} GB" -f $_.ProcessName, ($_.WorkingSet64/1GB) }) -join ', ') } else { $lines += "compiler: no clang-cl/icx/ninja running" }
if (Test-Path $watch) { $lines += "guard: " + (Get-Content $watch -Tail 1) }
$lines | Out-File "$here\build-status.output.txt" -Encoding utf8
$lines
